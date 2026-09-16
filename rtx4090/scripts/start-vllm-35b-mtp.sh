#!/bin/bash
# RTX 4090 (24GB) alternative: vLLM + Qwen3.6-35B-A3B GPTQ-Int4 + MTP n=5
# API: http://0.0.0.0:5000/v1/chat/completions   model name: "local"
#
# Verified on the previous 4090 host (vLLM 0.17 / torch cu128). UNTESTED on
# this Ubuntu 26.04 / driver 595 box — vllm-venv/ must be built first and is
# not created by setup.sh. GPTQ weights are ~22.7GB, hence --cpu-offload-gb 4
# and the 32K context. If startup OOMs, raise the offload or lower ctx.

DIR="$(cd "$(dirname "$0")/../.." && pwd)"

exec "$DIR/vllm-venv/bin/vllm" serve \
  "$DIR/models/Qwen3.6-35B-A3B-GPTQ-Int4" \
  --served-model-name local \
  --host 0.0.0.0 \
  --port 5000 \
  --quantization gptq \
  --dtype float16 \
  --gpu-memory-utilization 0.93 \
  --cpu-offload-gb 4 \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 5}'
