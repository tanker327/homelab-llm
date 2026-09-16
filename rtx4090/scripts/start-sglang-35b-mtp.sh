#!/bin/bash
# RTX 4090 (24GB) alternative: SGLang + Qwen3.6-35B-A3B GPTQ-Int4 + NEXTN n=5
# API: http://0.0.0.0:5000/v1/chat/completions   model name: "local"
#
# Verified on the previous 4090 host, where the CUDA 12.8 pin, g++-14 and the
# -D__THROW= workaround were required for glibc/CUDA header compatibility.
# UNTESTED on this Ubuntu 26.04 / driver 595 box — sglang-venv/ must be built
# first (not created by setup.sh) and the toolchain pins below likely need
# revisiting (the 96GB box found SPEC_V2 obsolete and plain pip wheels fine).

DIR="$(cd "$(dirname "$0")/../.." && pwd)"

export SGLANG_ENABLE_SPEC_V2=1
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$DIR/sglang-venv/bin:$CUDA_HOME/bin:$PATH"
export NVCC_PREPEND_FLAGS="-ccbin=/usr/bin/g++-14 -Xcompiler=-D__THROW="

exec "$DIR/sglang-venv/bin/python" -m sglang.launch_server \
  --model-path "$DIR/models/Qwen3.6-35B-A3B-GPTQ-Int4" \
  --served-model-name local \
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
