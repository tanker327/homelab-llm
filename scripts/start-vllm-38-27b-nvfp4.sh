#!/bin/bash
# Start vLLM with Qwen3.8-27B NVFP4 (Inferact modelopt export) — MTP n=3 + FP8 KV
# API: http://0.0.0.0:5000/v1/chat/completions  (model name: "local")
#
# 2026-08-16 service-window bake-off vs the FP8 production config (identical
# flags; docs/BENCHMARKS.md addendum): +41% single-stream decode (128.9 vs
# 91.4 tok/s), +25% agg @ N=8 (563 vs 449), +14% @ N=16 (676 vs 592), MTP
# acceptance 0.89-0.97 (FP8: ~0.6), 10/10 quality smoke, coding-task success
# parity (16/18 vs FP8's 14-15/16), 1.65M KV tokens, ~88GB VRAM.
#
# TRADE-OFF: vLLM 0.27.1 runs this checkpoint TEXT-ONLY ("no registered
# multimodal processor") — no vision. The FP8 launcher keeps vision. That is
# the only reason this isn't the systemd default.
#
# 1M context (tested, both needles pass; ~14 min prefill at 980K): swap in
#   --max-model-len 1010000 --max-num-seqs 4 \
#   --hf-overrides '{"text_config":{"max_position_embeddings":1010000,"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}'

DIR="$(cd "$(dirname "$0")/.." && pwd)"

export PATH="$DIR/vllm-venv/bin:$PATH"
export NVCC_PREPEND_FLAGS="-I$DIR/toolchain-fix"

exec "$DIR/vllm-venv/bin/vllm" serve "$DIR/models/Qwen3.8-27B-NVFP4-Inferact" \
  --host 0.0.0.0 \
  --port 5000 \
  --served-model-name local \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
