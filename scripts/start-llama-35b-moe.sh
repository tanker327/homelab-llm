#!/bin/bash
# Start llama.cpp server with Qwen3.6-35B-A3B (MoE, MXFP4_MOE)
# API: http://0.0.0.0:5000/v1/chat/completions
# Web UI: http://localhost:5000
#
# Tuned for RTX PRO 6000 Blackwell (96GB): 524288 total ctx / 2 slots
# = full 262K native context per request, MTP GGUF + self-speculation
# (~1.5-2x decode speedup, no quality change). On a 24GB 4090 use the
# non-MTP GGUF with --ctx-size 98304 and drop the --spec-* flags
# (see docs/CONCURRENCY.md).

DIR="$(cd "$(dirname "$0")/.." && pwd)"

exec "$DIR/llama.cpp/build/bin/llama-server" \
  --model "$DIR/models/Qwen3.6-35B-A3B-MTP-MXFP4_MOE.gguf" \
  --host 0.0.0.0 \
  --port 5000 \
  --n-gpu-layers 99 \
  --ctx-size 524288 \
  --parallel 2 \
  --flash-attn on \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --reasoning-format deepseek
