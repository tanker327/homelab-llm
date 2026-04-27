#!/bin/bash
# Start SGLang server with Qwen3.6-35B-A3B GPTQ-Int4 + MTP n=5 (NEXTN)
# API: http://0.0.0.0:5000/v1/chat/completions
# Web UI: http://localhost:5000

DIR="$(cd "$(dirname "$0")" && pwd)"

export SGLANG_ENABLE_SPEC_V2=1
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$DIR/sglang-venv/bin:$CUDA_HOME/bin:$PATH"
export NVCC_PREPEND_FLAGS="-ccbin=/usr/bin/g++-14 -Xcompiler=-D__THROW="

exec "$DIR/sglang-venv/bin/python" -m sglang.launch_server \
  --model-path "$DIR/models/Qwen3.6-35B-A3B-GPTQ-Int4" \
  --host 0.0.0.0 \
  --port 5000 \
  --quantization gptq_marlin \
  --dtype float16 \
  --mem-fraction-static 0.92 \
  --max-running-requests 4 \
  --context-length 32768 \
  --mamba-scheduler-strategy extra_buffer \
  --reasoning-parser qwen3 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6
