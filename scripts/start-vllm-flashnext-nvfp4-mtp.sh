#!/bin/bash
# PRODUCTION CANDIDATE — Qwen3.8-Flash-Next NVFP4 + INT4 PLE sidecar on vLLM,
# MTP n=3, 131K ctx. Selected 2026-09-05 as config "D" of the pilot bake-off.
# API: http://0.0.0.0:5000/v1/chat/completions   model name: "local"
#
# DEFAULT IS THE NO-SPECULATION CONFIG ("C"). MTP was measured FASTER but is
# NOT RELIABLE on this box - see below. Measured (agent workload, 12280 in /
# 2048 out, same harness as docs/BENCHMARKS.md):
#
#   C (default, no MTP, 262K ctx)      D (MTP n=3, 131K ctx)
#     N=1    88.6 agg /  92.1 dec        N=1   188.8 agg / 206.2 dec
#     N=4   249.4                        N=4   483.0
#     N=8   405.4                        N=8   376.3
#     N=16  614.9  <-- peak              N=16  479.6
#     N=32  594.0
#   Quality smoke 10/10 both. MTP acceptance 0.99. Boot ~250s. VRAM ~90-92.5GB.
#   Offload worker RSS 6.1GB; the 32GB INT4 table lives in page cache.
#
# WHY MTP IS OFF BY DEFAULT (measured 2026-09-05):
#   C: 15 runs, run-to-run spread 1.01-1.07x at every level. Zero collapses.
#   D: 4 of 19 runs at N=4 and 3 of 15 at N=8 collapsed to ~25% of nominal
#      (e.g. 124.7 agg / 36.4 decode vs 500 / 154), spread up to 2.65x.
#   The collapse signature: DECODE ONLY - prefill (5168 vs 5170 tok/s) and TTFT
#   (3004 vs 3006 ms) are untouched. Power RISES to 547-560W while the SM clock
#   FALLS to 2685-2745MHz (normal: 415-486W at 2812-2827MHz). Ruled out:
#     - thermal: no clocks_event_reasons ever active; one collapse at 78C while
#       81-83C runs were fine.
#     - PLE page-cache eviction: Cached steady at 49GB, pgmajfault ~0/s.
#     - competing load (open-webui points at :5000): server completed 25 of 28
#       requests sent, max running 4 / max waiting 1. No foreign traffic.
#     - concurrency: capping --max-num-seqs at 4 made it WORSE (2 of 4 collapsed).
#   Root cause not identified; it is specific to the speculative path.
#   To try MTP anyway (it is ~2x faster when it works):
#     FLASHNEXT_SPEC='{"method":"mtp","num_speculative_tokens":3}' \
#     FLASHNEXT_CTX=131072 FLASHNEXT_SEQS=64
#   MTP and 262K do not both fit in 96GB: at util 0.92 KV needs 7.57GiB and gets
#   3.45; at 0.97 KV fits but CUDA graph capture OOMs.
#
# REASONING EFFORT: pinned xhigh (2026-09-05, by request), deliberately
#   DIFFERENT from every other launcher here, which pin medium. xhigh is the
#   chat template's own default. The 2026-08-16 bench_effort.py sweep measured
#   xhigh at ~10x tokens and ~10-12x wall time with no coding pass-rate gain -
#   but that was on the 27B, NOT re-measured on this 177B MoE. Note it stacks
#   on this engine's tail latency (C TTFT p95 was 14s at N=16 on medium).
#   Valid values: xhigh, medium, low. "high" is NOT valid (HTTP 400).
#   Per-request chat_template_kwargs still overrides this.
#
# KNOWN LIMITS:
#   * TEXT-ONLY IN PRACTICE: vision was never tested on this checkpoint. If you
#     need images use start-llama-flashnext.sh.
#   * NOT ABLITERATED: this is base Flash-Next, unlike start-llama-flashnext.sh
#     and start-vllm-38-27b-uncensored-fp8.sh.
#   Benchmark record: benchmarks/results/flashnext-nvfp4-int4ple-*.jsonl
#
# HARD HOST DEPENDENCY: vm.overcommit_memory must be 1.
#   The PLE overlay creates the 95.4GiB BF16 table as untouched virtual memory
#   and stubs it out immediately; heuristic mode (0) refuses the allocation on
#   any host with less than ~96GB RAM and the worker dies with
#   "can't allocate memory: you tried to allocate 102400491520 bytes".
#   Persist it: /etc/sysctl.d/99-vllm-ple-overcommit.conf
#   This script checks and refuses to start rather than crash-looping.
#
# The three .py overlay mounts are MANDATORY - the stock image cannot read
# quantized PLE tables and silently falls back to the 95GB BF16 path.

set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"

MODEL_DIR="$DIR/models/Qwen3.8-Flash-Next-NVFP4"
PLE_REPO="$DIR/models/Qwen3.8-Flash-Next-PLE-quant"
PLE_DIR="$PLE_REPO/ples_int4"
SITE=/usr/local/lib/python3.12/dist-packages/vllm
NAME=flashnext-prod
PORT="${FLASHNEXT_PORT:-5000}"
CTX="${FLASHNEXT_CTX:-262144}"
SEQS="${FLASHNEXT_SEQS:-64}"
UTIL="${FLASHNEXT_UTIL:-0.92}"
SPEC="${FLASHNEXT_SPEC-}"

# --- host precondition ------------------------------------------------------
if [ "$(cat /proc/sys/vm/overcommit_memory)" != "1" ]; then
  echo "FATAL: vm.overcommit_memory=$(cat /proc/sys/vm/overcommit_memory), need 1." >&2
  echo "  sudo sysctl -w vm.overcommit_memory=1   (and persist in /etc/sysctl.d/)" >&2
  exit 1
fi

# --- assets -----------------------------------------------------------------
for p in "$MODEL_DIR/config.json" "$PLE_DIR/META.json" \
         "$PLE_REPO/worker_image_quant.py" "$PLE_REPO/ple_layer_quant.py" \
         "$PLE_REPO/connector_mrv2.py"; do
  [ -f "$p" ] || { echo "FATAL missing asset: $p" >&2; exit 1; }
done
n=$(ls "$PLE_DIR" | grep -c '^shard_')
[ "$n" -eq 128 ] || { echo "FATAL: expected 128 PLE shards, found $n" >&2; exit 1; }

# --- reclaim a stale container, then wait for the GPU to actually drain ------
# systemd restarts land here while the dead container may still hold VRAM.
docker rm -f "$NAME" >/dev/null 2>&1 || true
for i in $(seq 1 30); do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free_mib" -ge 94000 ] && break
  echo "waiting for VRAM release (${free_mib} MiB free)..." >&2
  sleep 2
done
if [ "$free_mib" -lt 94000 ]; then
  echo "FATAL: only ${free_mib} MiB VRAM free after 60s; another engine is running." >&2
  exit 1
fi

SPEC_ARGS=()
[ -n "$SPEC" ] && SPEC_ARGS=(--speculative-config "$SPEC")

exec docker run --rm --gpus all --ipc=host \
  --name "$NAME" \
  -p "${PORT}:8000" \
  -v "$MODEL_DIR:/model:ro" \
  -v "$PLE_DIR:/ples_int4:ro" \
  -v "$PLE_REPO/worker_image_quant.py:$SITE/v1/ple_offload/worker.py:ro" \
  -v "$PLE_REPO/ple_layer_quant.py:$SITE/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro" \
  -v "$PLE_REPO/connector_mrv2.py:$SITE/v1/ple_offload/connector.py:ro" \
  -e VLLM_PLE_QUANT_DIR=/ples_int4 \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
  -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600 \
  vllm/vllm-openai:qwen38-flash-next \
  --model /model \
  --served-model-name local \
  --host 0.0.0.0 \
  --distributed-executor-backend mp \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len "$CTX" \
  --max-num-seqs "$SEQS" \
  "${SPEC_ARGS[@]}" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"reasoning_effort":"xhigh"}'
