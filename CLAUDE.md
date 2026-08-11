# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local LLM inference server on an RTX PRO 6000 Blackwell (96GB VRAM, SM120, CUDA 13.1; originally built for an RTX 4090 — comments in some launchers note the old 24GB settings), exposing an OpenAI-compatible API on port 5000. The systemd default since 2026-08-12 is **vLLM serving dense `Qwen3.6-27B-FP8`** (`scripts/start-vllm-27b-fp8.sh`): MTP speculative decoding n=3 + FP8 KV cache, 262K max-model-len, 8 concurrent sequences, ~86–90GB VRAM — the winner of the three-engine bake-off recorded in `docs/BENCHMARKS.md` (~530 agg tok/s at N=8 agent load, ~118 tok/s single-stream, NIAH-correct to ~255K ctx). The former default two-model llama.cpp **agent stack** (`start-agent-stack.sh`: 35B MoE workers on 5000 + 27B Q8_0 on 5001) and all llama.cpp launchers remain for manual use. System-level changes (systemd, firewall, host config) are tracked in `CHANGELOG.md`.

## Layout

- `scripts/` — launchers (`setup.sh`, `start-*.sh`)
- `systemd/` — `llama-server.service`
- `clients/` — `chat.py`, `orchestrate.py`
- `benchmarks/` — `bench_serving.py` (main harness), `quality_smoke.py`, `bench_matrix.py`, `bench_concurrency.py`, `results/` (bake-off record)
- `tools/` — `probe_max_input.py`, `test_ctx*.py`
- `docs/` — `API.md`, `CONCURRENCY.md`, `BENCHMARKS.md`
- `CHANGELOG.md` — system-level change record (systemd, firewall, host config)
- `llama.cpp/`, `models/`, `venv/`, `vllm-venv/`, `sglang-venv/`, `toolchain-fix/` — git-ignored, stay at repo root

## Key Commands

```bash
./scripts/setup.sh                       # llama.cpp build + GGUF download + deps (idempotent; does NOT set up vllm-venv)
./scripts/start-vllm-27b-fp8.sh          # SYSTEMD DEFAULT: vLLM + Qwen3.6-27B-FP8 + MTP n=3 + FP8 KV (262K ctx, ~86GB)
./scripts/start-agent-stack.sh           # Manual: former default, two llama.cpp models (~77GB): 35B on 5000, 27B on 5001
./scripts/start-llama-27b.sh             # Manual: llama.cpp + 27B Q8_0 MTP (262K ctx/slot, ~64GB, ~139 tok/s single-stream)
./scripts/start-llama-35b-moe.sh         # Manual: llama.cpp + Qwen3.6-35B-A3B MTP MXFP4_MOE (MoE, ~418 tok/s)
./scripts/start-sglang-35b-mtp.sh        # Stale (4090-era flags); sglang-venv itself is current
./venv/bin/python clients/chat.py        # Interactive CLI chat client (commands: quit, clear)
./venv/bin/python clients/orchestrate.py "task"  # Plan -> N parallel workers -> judge, all on :5000; --mode bestof, --workers N
./venv/bin/python benchmarks/bench_serving.py --label x --port 5000 --workload agent --levels 1,4,8 --runs 3  # perf harness
sudo systemctl stop llama-server         # Stop the production service before running a manual launcher
```

Only one engine can bind port 5000 at a time — and the vLLM default plus any second model won't fit in VRAM anyway. Stop the service, wait ~10s for VRAM release, then launch a manual engine.

### Systemd Service (production)

```bash
sudo cp systemd/llama-server.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now llama-server
sudo systemctl status llama-server    # Check status
journalctl -u llama-server -f         # Live logs
```

`systemd/llama-server.service`'s `ExecStart` points at `scripts/start-vllm-27b-fp8.sh` (unit keeps its historical `llama-server` name). To switch the production engine, edit the unit, re-copy it, and `daemon-reload`. When restarting manually, leave ~10s between stop and start — the old process must release VRAM before the new one allocates, or it OOMs. vLLM takes ~2–3 min to become healthy (`/health` returns 200); llama.cpp takes ~5s.

## Architecture

- **vllm-venv/** (git-ignored): vLLM 0.27.1 + torch 2.13.0+cu130, installed 2026-08-11 via `uv` with plain pip wheels (SM120 works out of the box). Runs the production default. flashinfer JIT needs `ninja` on PATH and the `toolchain-fix` glibc-header shim — the launcher exports both.
- **llama.cpp** (git-ignored): compiled from source for SM120 (`-DCMAKE_CUDA_ARCHITECTURES=120` via setup.sh auto-detect) with flash attention. Binary at `llama.cpp/build/bin/llama-server`. Manual launchers only since 2026-08-12.
- **sglang-venv/** (git-ignored): SGLang 0.5.17 + torch 2.11.0+cu130. Evaluation only — best single-stream latency/prefill in the bake-off, but its scheduler collapses at 8-wide request bursts (see BENCHMARKS.md).
- **models/** (git-ignored): `Qwen3.6-27B-FP8/` safetensors (production, ~29GB, ships its MTP head in `mtp.safetensors`; the model is multimodal — bf16 vision tower, excluded from FP8 quant) plus GGUFs for llama.cpp (`Qwen3.6-27B-MTP-Q8_0.gguf`, `Qwen3.6-35B-A3B-MTP-MXFP4_MOE.gguf`, experimental 122B/Coder-Next shards).
- **venv/** (git-ignored): Python 3.12 venv with `openai`, `huggingface-hub` (provides `hf` CLI). Used by clients, benchmarks, and setup.sh downloads.
- **toolchain-fix/** (git-ignored): shimmed `bits/mathcalls.h` wrapping C23 `rsqrt` in `#ifndef __CUDACC__` — Ubuntu 26.04's glibc ≥2.42 breaks nvcc otherwise. Needed by setup.sh builds AND at runtime by flashinfer's JIT (injected via `NVCC_PREPEND_FLAGS`).
- **scripts/start-vllm-27b-fp8.sh**: the production launcher — see header comment for the bake-off numbers and required env.
- **scripts/start-agent-stack.sh**: former default, two llama.cpp instances; if either dies it kills the other and exits non-zero so systemd restarts the pair.
- **clients/chat.py / orchestrate.py**: OpenAI-SDK clients against `localhost:5000` with `model="local"`. orchestrate.py runs plan → N parallel workers → judge, all against the single production model since 2026-08-12.
- **benchmarks/bench_serving.py**: the serving benchmark harness (streaming TTFT p50/p95, token-exact prompts via `/tokenize` with calibration fallback, unique per-request prefixes to defeat prefix caching, GPU power/temp/util/VRAM sampling, tok/J, MTP acceptance). `--workload w1|agent|judge`, `--levels`, `--runs`, `--json`.
- **benchmarks/quality_smoke.py**: 10-task pass/fail quality gate (4 coding with asserts, 2 format, 2 JSON, 2 long-context NIAH/synthesis) at Qwen-recommended sampling.
- **benchmarks/results/**: bake-off record — `env.md` (environment + gate results), phase JSONLs, `RESUME-STATE.md` (final summary).
- **systemd/llama-server.service**: systemd unit (historical name; `ExecStart` = `start-vllm-27b-fp8.sh`, `RestartSec=15`). Auto-restart on crash.
- **docs/API.md** / **docs/CONCURRENCY.md** / **docs/BENCHMARKS.md**: API details; llama.cpp `--parallel` tuning (historical); full measurement record including the 2026-08-12 bake-off addendum.

## Important Details

- **The `model` field matters now**: vLLM validates it — clients must send `"local"` (launcher passes `--served-model-name local`). llama.cpp ignores the field (uses `--alias` for `/v1/models` display), which is why stale names went unnoticed before.
- **Reasoning/thinking mode**: llama.cpp `--reasoning-format deepseek`; vLLM/SGLang `--reasoning-parser qwen3`. Chain-of-thought arrives in `reasoning_content` (llama.cpp non-stream and stream; vLLM non-stream) or `reasoning` (vLLM stream deltas). Token counts include thinking tokens.
- **Never use temperature 0** with Qwen3.6 — greedy decoding produces unbounded thinking loops (54K+ reasoning tokens observed on a trivial task). Use Qwen's recommended temp 0.6 / top_p 0.95.
- **Speculative decoding is the whole ballgame for the dense 27B**: base decode is ~50 tok/s on every engine; MTP gives 2.4–2.6× (acceptance ~0.85–0.92) and holds under batching. vLLM: `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`; llama.cpp: `--spec-type draft-mtp --spec-draft-n-max 3` (needs an MTP GGUF); SGLang: `--speculative-algorithm NEXTN` (SPEC_V2 env var is obsolete — always on). n=3 beat other values on this GPU.
- **FP8 KV cache (`--kv-cache-dtype fp8_e4m3`) is a speed feature on this model**: +22% aggregate at N=8 and +79% decode at 49K ctx, no measurable quality loss. Part of the production config.
- **`max_tokens` on llama.cpp's `/v1/chat/completions` still truncates mid-reasoning** (returns empty content or 500s). Omit it there, or disable thinking. vLLM handles it correctly.
- llama.cpp builds target the GPU auto-detected by setup.sh (`compute_cap` → SM120 on this box; the old SM89 notes are obsolete).
- The old `start-vllm-35b-mtp.sh` / `start-sglang-35b-mtp.sh` carry 4090-era workarounds (CUDA 12.8 pin, g++-14, cpu-offload, 32K ctx) that are **obsolete on this box** — treat as templates only, don't copy their flags.
- The production server listens on `0.0.0.0:5000` (firewall allows `192.168.10.0/24`). Port 5001 is only used when manually running the agent stack; its ufw rule remains. vLLM serves no web UI at `/` (llama.cpp does, gzip-only — use `curl --compressed`).
- Ollama service was disabled (`systemctl disable ollama`) to avoid VRAM conflicts.
- Cold-start times: llama.cpp ~5s; vLLM ~2–3 min (weight load + JIT + CUDA graphs); SGLang ~2–3 min.
- The `hf` CLI (not `huggingface-cli`) is used for model downloads.
