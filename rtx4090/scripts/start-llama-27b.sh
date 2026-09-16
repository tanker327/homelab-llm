#!/bin/bash
# RTX 4090 (24GB): llama.cpp + Qwen3.6-27B Q4_K_M (dense)
# API: http://0.0.0.0:5000/v1/chat/completions   model name: "local"
#
# ~23GB VRAM, 96K context. Dense, so every parameter activates per token —
# noticeably slower than the 35B MoE. Quality pick, not speed pick.
# The root scripts/start-llama-27b.sh (Q8_0 + MTP, 524K ctx) needs 96GB.

DIR="$(cd "$(dirname "$0")/../.." && pwd)"

exec "$DIR/llama.cpp/build/bin/llama-server" \
  --model "$DIR/models/Qwen3.6-27B-Q4_K_M.gguf" \
  --alias local \
  --host 0.0.0.0 \
  --port 5000 \
  --n-gpu-layers 99 \
  --ctx-size 98304 \
  --flash-attn on \
  --reasoning-format deepseek
