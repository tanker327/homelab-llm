#!/bin/bash
# Start llama.cpp server with Qwen3.5-9B
# API: http://0.0.0.0:5000/v1/chat/completions
# Web UI: http://localhost:5000

DIR="$(cd "$(dirname "$0")" && pwd)"

exec "$DIR/llama.cpp/build/bin/llama-server" \
  --model "$DIR/models/Qwen3.5-9B-Q4_K_M.gguf" \
  --host 0.0.0.0 \
  --port 5000 \
  --n-gpu-layers 99 \
  --ctx-size 131072 \
  --flash-attn on \
  --reasoning-format deepseek
