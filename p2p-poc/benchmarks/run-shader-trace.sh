#!/usr/bin/env bash
# Profiling recipe: capture a Nsight-Systems trace of a decode burst
# so the L2 cache's P2P transfers can be visually verified.
#
# Prereqs:
#   - nsight-systems installed (nsys)
#   - ds4-server built with -lineinfo (default for our Makefile)
#   - P2P enabled at the driver level (see ../analysis/P2P_ENABLEMENT.md)
#
# Output: /tmp/ds4_p2p.nsys-rep plus a stderr log of the OBS summary.

set -euo pipefail

PORT=${PORT:-8018}
DURATION_SEC=${DURATION_SEC:-30}
MODEL=${MODEL:-/root/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf}
MTP=${MTP:-/root/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support.gguf}
OUT=${OUT:-/tmp/ds4_p2p}

pkill -f "ds4-server --port ${PORT}" 2>/dev/null || true
sleep 2

# Launch under nsys. We don't need CUDA API traces (huge overhead); just
# the kernel + memcpy timeline is enough to see the L2 peer-copies.
nsys profile \
    --output="${OUT}" \
    --force-overwrite=true \
    --duration=${DURATION_SEC} \
    --trace=cuda,nvtx,osrt \
    --cuda-um-events=true \
    --cuda-memory-usage=true \
    --delay=20 \
    ./ds4-server \
        --cuda --cuda-tensor-parallel --ssd-streaming \
        --ssd-streaming-cache-experts 40GB --gpu-vram 45,45 --ctx 262144 \
        -m "${MODEL}" --mtp "${MTP}" --dspark \
        --host 0.0.0.0 --port ${PORT} \
        2>&1 | tee "${OUT}.log" &

NSYS_PID=$!
sleep 22   # nsys --delay=20 + slack

# Warm-up.
curl -s --max-time 120 -X POST http://localhost:${PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"ds4","messages":[{"role":"user","content":"warmup"}],"max_tokens":50}' \
    > /dev/null

# Generation burst — this is what we want the trace to capture.
for i in 1 2 3 4 5; do
    curl -s --max-time 60 -X POST http://localhost:${PORT}/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"ds4","messages":[{"role":"user","content":"Write a long essay."}],"max_tokens":200,"temperature":0}' \
        > /dev/null
done

wait $NSYS_PID 2>/dev/null || true
pkill -f "ds4-server --port ${PORT}" 2>/dev/null || true

echo ""
echo "Trace saved to ${OUT}.nsys-rep — open with `nsys-ui` or:"
echo "  nsys stats ${OUT}.nsys-rep"
echo ""
echo "Look for cudaEventRecord rows whose name contains 'peer' to confirm"
echo "L2 peer-copies are happening. Cross-reference with the OBS summary"
echo "line 'l2_hits=... l2_writes=...' in ${OUT}.log."
