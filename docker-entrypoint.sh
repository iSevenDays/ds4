#!/usr/bin/env bash
# ds4-server launcher: multi-GPU + CUDA SSD weight-streaming (merged build).
# Args passed -> forwarded straight to ds4-server (bypasses env defaults).
set -euo pipefail

if [ "$#" -gt 0 ]; then
  exec ds4-server "$@"
fi

: "${DS4_MODEL:?DS4_MODEL is required (e.g. -e DS4_MODEL=/models/your-model.gguf)}"

if [ ! -f "$DS4_MODEL" ]; then
  echo "ds4: model not found at '$DS4_MODEL'. Mount it: -v \$PWD/gguf:/models:ro" >&2
  exit 1
fi

GPU_VRAM="${DS4_GPU_VRAM:-45,45}"
SSD_CACHE="${DS4_SSD_CACHE:-40GB}"
CTX="${DS4_CTX:-131072}"
HOST="${DS4_HOST:-0.0.0.0}"
PORT="${DS4_PORT:-8000}"

echo "ds4: $DS4_MODEL | gpu-vram=$GPU_VRAM | ssd-cache=$SSD_CACHE | ctx=$CTX | http://$HOST:$PORT"

exec ds4-server \
  --cuda \
  --ssd-streaming \
  --ssd-streaming-cache-experts "$SSD_CACHE" \
  --gpu-vram "$GPU_VRAM" \
  -m "$DS4_MODEL" \
  -c "$CTX" \
  --host "$HOST" \
  --port "$PORT" \
  ${DS4_EXTRA_ARGS:-}
