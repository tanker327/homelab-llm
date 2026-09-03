#!/bin/bash
# Start vLLM with Qwen3.8-27B-FP8 — MTP speculative decoding + FP8 KV cache
# API: http://0.0.0.0:5000/v1/chat/completions  (model name: "local")
# PRODUCTION DEFAULT since 2026-08-15, replacing Qwen3.6-27B-FP8. Hybrid
# arch (48/64 Gated DeltaNet linear-attention layers, full attention every
# 4th) — KV cache holds ~1.55M tokens at 0.90 util, so 16 concurrent seqs
# fit easily. 2026-08-15 tuning (see docs/BENCHMARKS.md): ~592 agg tok/s
# @ N=16 agent load (3.6 record: ~530 @ N=8), ~444 @ N=8, ~90-95 tok/s
# single-stream, NIAH-correct to 261.7K, 10/10 quality smoke. MTP n=3 won
# (n=2 slower, n=4 erratic stalls — model has a single recursive MTP layer,
# acceptance decays per position). Sampling: thinking temp 1.0 / top_p 0.95
# / top_k 20 (changed from 3.6's 0.6/0.95); never temperature 0.
#
# Requirements learned the hard way (RTX PRO 6000 Blackwell, CUDA 13.1,
# Ubuntu 26.04): `ninja` must be on PATH and the glibc-2.43 rsqrt header
# shim must be injected for flashinfer's JIT — both handled below.

DIR="$(cd "$(dirname "$0")/.." && pwd)"

export PATH="$DIR/vllm-venv/bin:$PATH"
export NVCC_PREPEND_FLAGS="-I$DIR/toolchain-fix"

exec "$DIR/vllm-venv/bin/vllm" serve "$DIR/models/Qwen3.8-27B-FP8" \
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
  --default-chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
