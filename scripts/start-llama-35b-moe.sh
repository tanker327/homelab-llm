#!/bin/bash
# Start llama.cpp server with Qwen3.6-35B-A3B (MoE, MXFP4_MOE)
# API: http://0.0.0.0:5000/v1/chat/completions
# Web UI: http://localhost:5000
#
# Tuned for RTX PRO 6000 Blackwell (96GB): 524288 total ctx / 2 slots
# = full 262K native context per request. On a 24GB 4090 use
# --ctx-size 98304 instead (see docs/CONCURRENCY.md).

DIR="$(cd "$(dirname "$0")/.." && pwd)"

exec "$DIR/llama.cpp/build/bin/llama-server" \
  --model "$DIR/models/Qwen3.6-35B-A3B-MXFP4_MOE.gguf" \
  --host 0.0.0.0 \
  --port 5000 \
  --n-gpu-layers 99 \
  --ctx-size 524288 \
  --parallel 2 \
  --flash-attn on \
  --reasoning-format deepseek
