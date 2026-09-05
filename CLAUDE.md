# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local LLM inference server on an RTX PRO 6000 Blackwell (96GB VRAM, SM120, CUDA 13.1; originally built for an RTX 4090 — comments in some launchers note the old 24GB settings), exposing an OpenAI-compatible API on port 5000. The systemd default since 2026-09-05 is **vLLM (in docker) serving the 177B `Qwen3.8-Flash-Next`** as `primitive-ai/Qwen3.8-Flash-Next-NVFP4` + the 32GB INT4 PLE sidecar (262K ctx, no speculation, 64 seqs, ~90GB VRAM, `reasoning_effort` xhigh, **text-only, not abliterated**) via `scripts/start-production.sh`, which execs whichever launcher `scripts/production-engine.conf` names (currently `start-vllm-flashnext-nvfp4-mtp.sh`); start-up is ~190-250s before `/health` is up. It needs **`vm.overcommit_memory=1`** (see CHANGELOG 2026-09-05) and the pinned `vllm/vllm-openai:qwen38-flash-next` image with three bind-mounted `.py` overlays — the launcher checks and refuses to start otherwise. 615 agg tok/s @ N=16, vs ~130 plateauing at N=4 on the llama.cpp Flash-Next path. **For vision or uncensored single-stream work, switch the conf line to `start-llama-flashnext.sh`** (llama.cpp + `Qwen3.8-Flash-Next-Uncensored` IQ4_XS, orcarouter abliterated 177B MoE / 6B active, full 262K context, vision on via mmproj, ~75-76GB VRAM with the 51B PLE n-gram table mmap'd host-side, ~109 tok/s single-stream with no MTP head, ~3000 tok/s prefill but aggregate plateauing ~130 tok/s at N=4; cold-from-disk start ~10-25 min, ~20s warm) **and `sudo systemctl restart llama-server`.** `start-vllm-38-27b-nvfp4.sh` (text-only, ~676 tok/s aggregate at N=16 — the agent-fleet engine), `start-vllm-38-27b-fp8.sh` (vision-capable 27B) and `scripts/start-vllm-27b-fp8.sh` (Qwen3.6, 2026-08-12 bake-off winner) remain as rollbacks; see the 2026-08-28 addendum in `docs/BENCHMARKS.md`. The former default two-model llama.cpp **agent stack** (`start-agent-stack.sh`: 35B MoE workers on 5000 + 27B Q8_0 on 5001) and all llama.cpp launchers remain for manual use. System-level changes (systemd, firewall, host config) are tracked in `CHANGELOG.md`.

## Layout

- `scripts/` — launchers (`setup.sh`, `start-*.sh`)
- `systemd/` — `llama-server.service`
- `clients/` — `chat.py`, `orchestrate.py`
- `benchmarks/` — `bench_serving.py` (main harness), `quality_smoke.py`, `bench_effort.py` (reasoning-effort sweep), `bench_matrix.py`, `bench_concurrency.py`, `results/` (bake-off record)
- `tools/` — `probe_max_input.py`, `test_ctx*.py`
- `docs/` — `API.md`, `CONCURRENCY.md`, `BENCHMARKS.md`
- `CHANGELOG.md` — system-level change record (systemd, firewall, host config)
- `llama.cpp/`, `models/`, `venv/`, `vllm-venv/`, `sglang-venv/`, `toolchain-fix/` — git-ignored, stay at repo root

## Key Commands

```bash
./scripts/setup.sh                       # llama.cpp build + GGUF download + deps (idempotent; does NOT set up vllm-venv)
./scripts/start-production.sh            # SYSTEMD DEFAULT: runs the launcher named in production-engine.conf (currently Flash-Next NVFP4)
./scripts/start-vllm-flashnext-nvfp4-mtp.sh    # Production engine: Flash-Next 177B NVFP4 + INT4 PLE sidecar, 262K, ~90GB, TEXT-ONLY, needs vm.overcommit_memory=1
./scripts/start-vllm-38-27b-nvfp4.sh     # Production engine: vLLM + Qwen3.8-27B NVFP4 + MTP n=3 + FP8 KV (262K, 16 seqs, ~88GB, TEXT-ONLY)
./scripts/start-llama-flashnext.sh       # Vision/uncensored engine: llama.cpp + Flash-Next-Uncensored 177B IQ4_XS (262K ctx, vision, ~76GB, ~109 tok/s)
./scripts/start-vllm-38-27b-fp8.sh       # Vision fallback: vLLM + Qwen3.8-27B-FP8 + MTP n=3 + FP8 KV (262K ctx, 16 seqs, ~88GB)
./scripts/start-vllm-27b-fp8.sh          # Rollback: vLLM + Qwen3.6-27B-FP8 + MTP n=3 + FP8 KV (262K ctx, ~86GB)
./scripts/start-vllm-38-27b-uncensored-fp8.sh  # On-demand: orcarouter Qwen3.8-27B-Uncensored-FP8 (abliterated, vision + MTP work; same flags as FP8 launcher)
./scripts/start-agent-stack.sh           # Manual: former default, two llama.cpp models (~77GB): 35B on 5000, 27B on 5001
./scripts/start-llama-27b.sh             # Manual: llama.cpp + 27B Q8_0 MTP (262K ctx/slot, ~64GB, ~139 tok/s single-stream)
./scripts/start-llama-35b-moe.sh         # Manual: llama.cpp + Qwen3.6-35B-A3B MTP MXFP4_MOE (MoE, ~418 tok/s)
./scripts/start-sglang-35b-mtp.sh        # Stale (4090-era flags); sglang-venv itself is current
./venv/bin/python clients/chat.py        # Interactive CLI chat client (commands: quit, clear)
./venv/bin/python clients/orchestrate.py "task"  # Plan -> N parallel workers -> judge, all on :5000; --mode bestof, --workers N
./venv/bin/python benchmarks/bench_serving.py --label x --port 5000 --workload agent --levels 1,4,8 --runs 3  # perf harness
sudo systemctl stop llama-server         # Stop the production service before running a manual launcher
```

Only one engine can bind port 5000 at a time — and the NVFP4 default (~88GB) plus any second model won't fit in VRAM anyway. Stop the service, wait ~10s for VRAM release, then launch a manual engine.

### Systemd Service (production)

```bash
sudo cp systemd/llama-server.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now llama-server
sudo systemctl status llama-server    # Check status
journalctl -u llama-server -f         # Live logs
```

`systemd/llama-server.service`'s `ExecStart` points at `scripts/start-production.sh` (unit keeps its historical `llama-server` name), which execs the launcher named in `scripts/production-engine.conf`. **To switch the production engine (e.g. Flash-Next ↔ NVFP4 for agent fleets): edit that one conf line, then `sudo systemctl restart llama-server`** — no unit copy or daemon-reload. When restarting manually, leave ~10s between stop and start — the old process must release VRAM before the new one allocates, or it OOMs. Until `/health` returns 200: Flash-Next ~20s–1 min warm page cache but ~10–25 min cold from disk; vLLM ~2–3 min; small llama.cpp GGUFs ~5s.

## Architecture

- **vllm-venv/** (git-ignored): vLLM 0.27.1 + torch 2.13.0+cu130, installed 2026-08-11 via `uv` with plain pip wheels (SM120 works out of the box). Runs the production default. flashinfer JIT needs `ninja` on PATH and the `toolchain-fix` glibc-header shim — the launcher exports both.
- **llama.cpp** (git-ignored): compiled from source for SM120 (`-DCMAKE_CUDA_ARCHITECTURES=120` via setup.sh auto-detect) with flash attention. Binary at `llama.cpp/build/bin/llama-server`. Manual launchers only since 2026-08-12.
- **sglang-venv/** (git-ignored): SGLang 0.5.17 + torch 2.11.0+cu130. Evaluation only — best single-stream latency/prefill in the bake-off, but its scheduler collapses at 8-wide request bursts (see BENCHMARKS.md).
- **models/** (git-ignored): `Qwen3.8-27B-FP8/` safetensors (production, ~29GB, MTP head in `mtp.safetensors`, bf16 vision tower; hybrid arch — 48/64 layers Gated DeltaNet linear attention, KV cache only on the 16 full-attention layers → ~1.55M KV tokens at 0.90 util); `Qwen3.6-27B-FP8/` (rollback); `Qwen3.8-27B-Uncensored-FP8/` (~29GB, orcarouter abliterated build, MTP head embedded in the main shards rather than a separate `mtp.safetensors` — kept for on-demand use via `production-engine.conf`); `Qwen3.8-Flash-Next-NVFP4/` (174GB, production weights: NVFP4 experts + BF16 tail) and `Qwen3.8-Flash-Next-PLE-quant/ples_int4/` (30GB, 128-shard INT4 n-gram table + the three `.py` overlays the pinned vLLM image needs); GGUFs for llama.cpp (`Qwen3.6-27B-MTP-Q8_0.gguf`, `Qwen3.6-35B-A3B-MTP-MXFP4_MOE.gguf`, experimental 122B/Coder-Next shards).
- **venv/** (git-ignored): Python 3.12 venv with `openai`, `huggingface-hub` (provides `hf` CLI). Used by clients, benchmarks, and setup.sh downloads.
- **toolchain-fix/** (git-ignored): shimmed `bits/mathcalls.h` wrapping C23 `rsqrt` in `#ifndef __CUDACC__` — Ubuntu 26.04's glibc ≥2.42 breaks nvcc otherwise. Needed by setup.sh builds AND at runtime by flashinfer's JIT (injected via `NVCC_PREPEND_FLAGS`).
- **scripts/start-production.sh + production-engine.conf**: systemd entry point; the conf names the active launcher. **start-vllm-38-27b-nvfp4.sh** (production, text-only) and **start-vllm-38-27b-fp8.sh** (vision fallback) both carry tuning numbers in their headers. `start-vllm-27b-fp8.sh` is the Qwen3.6 rollback launcher.
- **scripts/start-agent-stack.sh**: former default, two llama.cpp instances; if either dies it kills the other and exits non-zero so systemd restarts the pair.
- **clients/chat.py / orchestrate.py**: OpenAI-SDK clients against `localhost:5000` with `model="local"`. orchestrate.py runs plan → N parallel workers → judge, all against the single production model since 2026-08-12.
- **benchmarks/bench_serving.py**: the serving benchmark harness (streaming TTFT p50/p95, token-exact prompts via `/tokenize` with calibration fallback, unique per-request prefixes to defeat prefix caching, GPU power/temp/util/VRAM sampling, tok/J, MTP acceptance). `--workload w1|agent|judge`, `--levels`, `--runs`, `--json`.
- **benchmarks/quality_smoke.py**: 10-task pass/fail quality gate (4 coding with asserts, 2 format, 2 JSON, 2 long-context NIAH/synthesis) at Qwen-recommended sampling.
- **benchmarks/results/**: bake-off record — `env.md` (environment + gate results), phase JSONLs, `RESUME-STATE.md` (final summary).
- **systemd/llama-server.service**: systemd unit (historical name; `ExecStart` = `start-production.sh`, `RestartSec=15`). Auto-restart on crash.
- **docs/API.md** / **docs/CONCURRENCY.md** / **docs/BENCHMARKS.md**: API details; llama.cpp `--parallel` tuning (historical); full measurement record including the 2026-08-12 bake-off addendum.

## Important Details

- **The `model` field matters now**: vLLM validates it — clients must send `"local"` (launcher passes `--served-model-name local`). llama.cpp ignores the field (uses `--alias` for `/v1/models` display), which is why stale names went unnoticed before.
- **Reasoning/thinking mode**: llama.cpp `--reasoning-format deepseek`; vLLM/SGLang `--reasoning-parser qwen3`. Chain-of-thought arrives in `reasoning_content` (llama.cpp non-stream and stream) or `reasoning` (vLLM, both non-stream `message.reasoning` and stream deltas — verified on vLLM 0.27.1; `reasoning_content` is absent/empty there). To carry thinking across turns (`preserve_thinking`, template default on), the client must echo it back as `reasoning_content` on prior assistant messages — the OpenAI SDK drops it silently otherwise.
- **Never use temperature 0** with Qwen3.6/3.8 — greedy decoding produces unbounded thinking loops (54K+ reasoning tokens observed on a trivial task). Recommended thinking-mode sampling: **Qwen3.8: temp 1.0 / top_p 0.95 / top_k 20**; Qwen3.6: temp 0.6 / top_p 0.95. Qwen3.8 non-thinking: temp 0.7 / top_p 0.8 / presence_penalty 1.5.
- **Reasoning effort**: `start-vllm-flashnext-nvfp4-mtp.sh` pins `xhigh` (2026-09-05, by request); every other launcher pins `medium` since 2026-09-03. The three Qwen3.8 **vLLM** launchers pass `--default-chat-template-kwargs '{"reasoning_effort":"medium"}'` and `start-llama-flashnext.sh` passes llama.cpp's `--chat-template-kwargs '{"reasoning_effort":"medium"}'`, both overriding the chat template's `xhigh` default (which burns ~10× tokens and ~10–12× wall time for no measured pass-rate gain on coding tasks — 2026-08-16 sweep, `benchmarks/bench_effort.py`, results in BENCHMARKS.md; launchers served `low` from 2026-08-26 until the 2026-09-03 change to `medium`). Per-request `chat_template_kwargs: {"reasoning_effort": ...}` still overrides it (vLLM merges request kwargs over the default). **The template accepts only `xhigh`, `medium`, `low`** — anything else (`high`, `minimal`) raises a template exception. Thinking fully off is measurably weaker. Tool calling works on the production config as-is; **vision does not** — NVFP4 is text-only on vLLM 0.27.1, so use the FP8 or Flash-Next launcher for images.
- **Speculative decoding is the whole ballgame for the dense 27B**: base decode is ~50 tok/s on every engine; MTP gives 2.4–2.6× (acceptance ~0.85–0.92) and holds under batching. vLLM: `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`; llama.cpp: `--spec-type draft-mtp --spec-draft-n-max 3` (needs an MTP GGUF); SGLang: `--speculative-algorithm NEXTN` (SPEC_V2 env var is obsolete — always on). n=3 beat other values on this GPU.
- **Qwen3.8's MTP head is a single layer applied recursively** for n>1: per-position acceptance decays (~0.83/0.71/0.61), overall ~0.6. n=3 is still the throughput winner; n=4 benched faster single-stream but produced erratic multi-second stalls — do not use.
- **FP8 KV cache (`--kv-cache-dtype fp8_e4m3`) is a speed feature on this model**: +22% aggregate at N=8 and +79% decode at 49K ctx, no measurable quality loss. Part of the production config.
- **`max_tokens` on llama.cpp's `/v1/chat/completions` still truncates mid-reasoning** (returns empty content or 500s). Omit it there, or disable thinking. vLLM handles it correctly.
- llama.cpp builds target the GPU auto-detected by setup.sh (`compute_cap` → SM120 on this box; the old SM89 notes are obsolete).
- The old `start-vllm-35b-mtp.sh` / `start-sglang-35b-mtp.sh` carry 4090-era workarounds (CUDA 12.8 pin, g++-14, cpu-offload, 32K ctx) that are **obsolete on this box** — treat as templates only, don't copy their flags.
- The production server listens on `0.0.0.0:5000` (firewall allows `192.168.10.0/24`). Port 5001 is only used when manually running the agent stack; its ufw rule remains. vLLM serves no web UI at `/` (llama.cpp does, gzip-only — use `curl --compressed`).
- Ollama service was disabled (`systemctl disable ollama`) to avoid VRAM conflicts.
- Cold-start times: vLLM (production) ~2–3 min (weight load + JIT + CUDA graphs); Flash-Next ~20s–1 min with warm page cache but ~10–25 min cold from disk; small llama.cpp GGUFs ~5s; SGLang ~2–3 min.
- The `hf` CLI (not `huggingface-cli`) is used for model downloads.
