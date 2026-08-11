#!/bin/bash
# Start llama.cpp server with Qwen3-Coder-Next (80B-A3B MoE, UD-Q6_K_XL) — experimental
# API: http://0.0.0.0:5000/v1/chat/completions
# Web UI: http://localhost:5000
#
# RTX PRO 6000 Blackwell (96GB): 73.1GB weights, 262144 ctx / 1 slot
# (full native context). Non-thinking coding-agent model: no MTP GGUF
# exists (no --spec-* flags) and no <think> blocks (no --reasoning-format).
# --jinja enables the chat template's native tool-call parsing; sampling
# defaults are Qwen's recommendation for this model (temp 1.0!) — clients
# can still override per-request.
#
# Stop the agent stack first and wait ~10s for VRAM release:
#   sudo systemctl stop llama-server

DIR="$(cd "$(dirname "$0")/.." && pwd)"

exec "$DIR/llama.cpp/build/bin/llama-server" \
  --model "$DIR/models/Qwen3-Coder-Next-UD-Q6_K_XL/Qwen3-Coder-Next-UD-Q6_K_XL-00001-of-00003.gguf" \
  --alias Qwen3-Coder-Next \
  --host 0.0.0.0 \
  --port 5000 \
  --n-gpu-layers 99 \
  --ctx-size 262144 \
  --parallel 1 \
  --flash-attn on \
  --jinja \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 40 \
  --min-p 0.01 \
  --repeat-penalty 1.0
