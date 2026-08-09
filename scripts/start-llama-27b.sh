#!/bin/bash
# Start llama.cpp server with Qwen3.6-27B (dense) — Q8_0 + MTP speculative decoding
# API: http://0.0.0.0:5000/v1/chat/completions
# Web UI: http://localhost:5000
#
# Tuned for RTX PRO 6000 Blackwell (96GB): near-lossless Q8_0 weights (29GB)
# and MTP self-speculation (~1.5-2x decode speedup, no quality change).
# 262144 ctx per slot x 2 slots. On a 24GB 4090 use the Q4_K_M non-MTP GGUF
# with --ctx-size 98304 and drop the --spec-* flags.

DIR="$(cd "$(dirname "$0")/.." && pwd)"

exec "$DIR/llama.cpp/build/bin/llama-server" \
  --model "$DIR/models/Qwen3.6-27B-MTP-Q8_0.gguf" \
  --alias Qwen3.6-27B \
  --host 0.0.0.0 \
  --port 5000 \
  --n-gpu-layers 99 \
  --ctx-size 524288 \
  --parallel 2 \
  --flash-attn on \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --reasoning-format deepseek
