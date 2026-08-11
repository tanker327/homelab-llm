#!/bin/bash
# Start llama.cpp server with Qwen3.5-122B-A10B (MoE, UD-Q4_K_XL) — experimental
# API: http://0.0.0.0:5000/v1/chat/completions
# Web UI: http://localhost:5000
#
# RTX PRO 6000 Blackwell (96GB): 78.7GB weights leave ~17GB for KV +
# buffers — the tightest fit on this box. 262144 ctx / 1 slot (full
# native context). MTP GGUF + self-speculation, n-max 3 (untested for
# this model — A/B against 2 and 4, see docs/BENCHMARKS.md).
#
# Stop the agent stack first and wait ~10s for VRAM release:
#   sudo systemctl stop llama-server
# If startup OOMs: halve --ctx-size, or offload experts with --n-cpu-moe 4
# (raise until it fits; costs decode speed).

DIR="$(cd "$(dirname "$0")/.." && pwd)"

exec "$DIR/llama.cpp/build/bin/llama-server" \
  --model "$DIR/models/Qwen3.5-122B-A10B-MTP-UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf" \
  --alias Qwen3.5-122B-A10B \
  --host 0.0.0.0 \
  --port 5000 \
  --n-gpu-layers 99 \
  --ctx-size 262144 \
  --parallel 1 \
  --flash-attn on \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --reasoning-format deepseek
