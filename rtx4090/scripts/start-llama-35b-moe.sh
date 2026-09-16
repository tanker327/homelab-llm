#!/bin/bash
# RTX 4090 (24GB) DEFAULT: llama.cpp + Qwen3.6-35B-A3B MXFP4_MOE (MoE, 3B active)
# API: http://0.0.0.0:5000/v1/chat/completions   model name: "local"
# Web UI: http://localhost:5000
#
# Measured on the 4090: ~23.2GB VRAM, ~163 tok/s single-stream, ~300 tok/s
# aggregate ceiling. --ctx-size is the TOTAL KV cache shared by --parallel
# slots (2 slots -> 48K per request). No MTP: this is the non-MTP GGUF and
# the 96GB box's --spec-* flags do not apply here. See rtx4090/README.md.

DIR="$(cd "$(dirname "$0")/../.." && pwd)"

exec "$DIR/llama.cpp/build/bin/llama-server" \
  --model "$DIR/models/Qwen3.6-35B-A3B-MXFP4_MOE.gguf" \
  --alias local \
  --host 0.0.0.0 \
  --port 5000 \
  --n-gpu-layers 99 \
  --ctx-size 98304 \
  --parallel 2 \
  --flash-attn on \
  --reasoning-format deepseek
