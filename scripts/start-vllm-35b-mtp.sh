#!/bin/bash
# Start vLLM server with Qwen3.6-35B-A3B GPTQ-Int4 + MTP n=5
# API: http://0.0.0.0:5000/v1/chat/completions
# Web UI: http://localhost:5000

DIR="$(cd "$(dirname "$0")/.." && pwd)"

exec "$DIR/vllm-venv/bin/vllm" serve \
  "$DIR/models/Qwen3.6-35B-A3B-GPTQ-Int4" \
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
