#!/bin/bash
# PRODUCTION (since 2026-08-28 via production-engine.conf):
# Qwen3.8-Flash-Next-Uncensored (orcarouter GGUF, IQ4_XS) on llama.cpp
# API: http://0.0.0.0:5000/v1/chat/completions
#
# 177B multimodal ultra-sparse MoE (125B transformer + 51B n-gram PLE table,
# 6B active/token, 512 experts 10+1 active). Needs llama.cpp master >= 2026-08-27
# (qwen4exp arch, PR #27742). No MTP head in the GGUF — no speculative decoding.
# The 51B PLE table is read via get_rows and mmap'd host-side; the GPU-resident
# part of the 97.5GB IQ4_XS is ~65-70GB. Native ctx 262144 (GDN linear-attn on
# 3/4 layers + QSA keeps KV growth slow); FLASHNEXT_CTX overrides for testing.
#
# Measured 2026-08-28 (IQ4_XS, full 262K ctx): 75.2GB VRAM (76.4GB with vision),
# ~109 tok/s single-stream decode, ~3000 tok/s prefill, NIAH-correct to 261K.
# Concurrency plateaus ~130 agg tok/s at N=4 — single-user engine, not a fleet
# engine (for agent fleets switch the conf back to start-vllm-38-27b-nvfp4.sh).
# Vision (mmproj, +~1GB) is ON by default; FLASHNEXT_VISION=0 disables.
# Template accepts per-request chat_template_kwargs {"reasoning_effort":"low"}.
# Cold-from-disk load can take ~10-25 min (page cache empty); ~20s warm.

DIR="$(cd "$(dirname "$0")/.." && pwd)"

CTX="${FLASHNEXT_CTX:-262144}"
MMPROJ_ARGS=()
if [ "${FLASHNEXT_VISION:-1}" = "1" ]; then
  MMPROJ_ARGS=(--mmproj "$DIR/models/Qwen3.8-Flash-Next-Uncensored-GGUF/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf")
fi

exec "$DIR/llama.cpp/build/bin/llama-server" \
  --model "$DIR/models/Qwen3.8-Flash-Next-Uncensored-GGUF/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf" \
  "${MMPROJ_ARGS[@]}" \
  --alias local \
  --host 0.0.0.0 \
  --port 5000 \
  --n-gpu-layers 999 \
  --ctx-size "$CTX" \
  --flash-attn on \
  --jinja \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --reasoning-format deepseek
