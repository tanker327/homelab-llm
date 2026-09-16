#!/bin/bash
# RTX 4090 (24GB): llama.cpp + Qwen3.5-9B Q4_K_M (dense)
# API: http://0.0.0.0:5000/v1/chat/completions   model name: "local"
#
# ~6GB VRAM, ~125 tok/s single-stream, 128K context. The pick when the GPU
# has to be shared with something else.

DIR="$(cd "$(dirname "$0")/../.." && pwd)"

exec "$DIR/llama.cpp/build/bin/llama-server" \
  --model "$DIR/models/Qwen3.5-9B-Q4_K_M.gguf" \
  --alias local \
  --host 0.0.0.0 \
  --port 5000 \
  --n-gpu-layers 99 \
  --ctx-size 131072 \
  --flash-attn on \
  --reasoning-format deepseek
