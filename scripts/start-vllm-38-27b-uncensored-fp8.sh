#!/bin/bash
# EXPERIMENTAL / TEMPORARY — vLLM with orcarouter/Qwen3.8-27B-Uncensored-FP8
# (abliterated community build of Qwen3.8-27B, FP8, vision tower intact).
# API: http://0.0.0.0:5000/v1/chat/completions  (model name: "local")
#
# Templated on start-vllm-38-27b-fp8.sh (the production FP8 launcher).
# Unlike the official build (separate mtp.safetensors), this repo embeds the
# MTP head in the main shards (mtp.* tensors, FP8) — spec decoding works.
#
# To make production temporarily serve this model:
#   echo start-vllm-38-27b-uncensored-fp8.sh > scripts/production-engine.conf  (edit, keep comments)
#   sudo systemctl restart llama-server
# To revert: point production-engine.conf back at start-vllm-38-27b-nvfp4.sh.

DIR="$(cd "$(dirname "$0")/.." && pwd)"

export PATH="$DIR/vllm-venv/bin:$PATH"
export NVCC_PREPEND_FLAGS="-I$DIR/toolchain-fix"

exec "$DIR/vllm-venv/bin/vllm" serve "$DIR/models/Qwen3.8-27B-Uncensored-FP8" \
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
