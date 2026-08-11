#!/bin/bash
# Start vLLM with Qwen3.6-27B-FP8 — MTP speculative decoding + FP8 KV cache
# API: http://0.0.0.0:5000/v1/chat/completions  (model name: "local")
# 2026-08-12 bake-off winner: ~530 agg tok/s @ N=8 agent load (2.2x the
# llama.cpp Q8_0 incumbent), ~120 tok/s decode at 49K ctx, NIAH-correct to
# ~255K tokens, 10/10 quality smoke vs Q8_0 reference. ~86GB VRAM at 0.90
# utilization — does NOT coexist with the 35B agent stack; stop llama-server
# first. See docs/BENCHMARKS.md and benchmarks/results/ for the full record.
#
# Requirements learned the hard way (RTX PRO 6000 Blackwell, CUDA 13.1,
# Ubuntu 26.04): `ninja` must be on PATH and the glibc-2.43 rsqrt header
# shim must be injected for flashinfer's JIT — both handled below.

DIR="$(cd "$(dirname "$0")/.." && pwd)"

export PATH="$DIR/vllm-venv/bin:$PATH"
export NVCC_PREPEND_FLAGS="-I$DIR/toolchain-fix"

exec "$DIR/vllm-venv/bin/vllm" serve "$DIR/models/Qwen3.6-27B-FP8" \
  --host 0.0.0.0 \
  --port 5000 \
  --served-model-name local \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
