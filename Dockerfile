# syntax=docker/dockerfile:1
#
# ds4 production image — multi-GPU (mgpu) + CUDA SSD weight-streaming, merged with
# antirez/ds4:main. Built from the iSevenDays/ds4 feat/mgpu-ssd branch (commit 5dce0ac).
# Serves the 81GB DeepSeek-V4-Flash IQ2 model at 131k context across 2 GPUs by
# streaming expert weights from SSD, keeping a large per-device expert cache in VRAM.
#
# Target: 2x RTX 4090 D 48GB (Ada, sm_89).
#
# Build:  docker build -t ds4-2gpu-ssd --build-arg CUDA_ARCH=sm_89 .
# Run:    docker run -d --gpus all -p 8001:8000 -v $PWD/gguf:/models:ro \
#           -e DS4_MODEL=/models/<file>.gguf ds4-2gpu-ssd

############################
#  Stage 1: build (merged) #
############################
FROM nvcr.io/nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04 AS builder

# feat/mgpu-ssd @ 5dce0ac = antirez/main merged with mgpu + per-device SSD streaming.
ARG DS4_REPO=https://github.com/iSevenDays/ds4.git
ARG DS4_REF=feat/mgpu-ssd
ARG CUDA_ARCH=sm_89

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${DS4_REF}" "${DS4_REPO}" /tmp/ds4 \
    && cd /tmp/ds4 \
    && make -j"$(nproc)" cuda CUDA_ARCH="${CUDA_ARCH}" \
    && cp ds4 ds4-server ds4-bench ds4-eval ds4-agent /usr/local/bin/

############################
#  Stage 2: lean runtime   #
############################
FROM nvcr.io/nvidia/cuda:13.2.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/ds4 /usr/local/bin/ds4
COPY --from=builder /usr/local/bin/ds4-server /usr/local/bin/ds4-server
COPY --from=builder /usr/local/bin/ds4-bench /usr/local/bin/ds4-bench
COPY --from=builder /usr/local/bin/ds4-eval /usr/local/bin/ds4-eval
COPY --from=builder /usr/local/bin/ds4-agent /usr/local/bin/ds4-agent
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Defaults — override with `docker run -e`. Tune DS4_SSD_CACHE for your VRAM
# (per-device expert cache; bigger = fewer SSD reads = faster, up to free VRAM).
ENV DS4_MODEL=/models/model.gguf \
    DS4_GPU_VRAM=45,45 \
    DS4_SSD_CACHE=40GB \
    DS4_CTX=131072 \
    DS4_HOST=0.0.0.0 \
    DS4_PORT=8000 \
    DS4_EXTRA_ARGS=

WORKDIR /workspace
VOLUME ["/models"]
EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
