# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local LLM inference server on an RTX 4090 (24GB VRAM), exposing an OpenAI-compatible API on port 5000. The systemd default is **llama.cpp** serving `Qwen3.6-35B-A3B-MXFP4_MOE.gguf` (via `scripts/start-llama-35b-moe.sh`). vLLM and SGLang launchers are kept on disk as alternatives that can run the GPTQ-Int4 weights with MTP speculative decoding.

## Layout

- `scripts/` — launchers (`setup.sh`, `start-*.sh`)
- `systemd/` — `llama-server.service`
- `clients/` — `chat.py`
- `benchmarks/` — `bench_concurrency.py` (and gitignored `benchmark.py`, `bench_separate.py`)
- `tools/` — `probe_max_input.py`, `test_ctx*.py`
- `docs/` — `API.md`, `CONCURRENCY.md`
- `llama.cpp/`, `models/`, `venv/`, `vllm-venv/`, `sglang-venv/` — git-ignored, stay at repo root

## Key Commands

```bash
./scripts/setup.sh                       # Full setup: build llama.cpp, download model, install deps (idempotent)
./scripts/start-llama-35b-moe.sh         # Systemd default: llama.cpp + Qwen3.6-35B-A3B MXFP4_MOE (MoE, 96K ctx)
./scripts/start-vllm-35b-mtp.sh          # Alt: vLLM + Qwen3.6-35B-A3B GPTQ-Int4 + MTP n=5
./scripts/start-sglang-35b-mtp.sh        # Alt: SGLang + Qwen3.6-35B-A3B GPTQ-Int4 + NEXTN n=5
./scripts/start-llama.sh                 # Alt: llama.cpp + Qwen3.5-35B-A3B Q4_K_M (legacy MoE)
./scripts/start-llama-9b.sh              # Alt: llama.cpp + dense 9B (128K ctx, ~6GB VRAM)
./scripts/start-llama-27b.sh             # Alt: llama.cpp + dense Qwen3.6-27B Q4_K_M (96K ctx, ~23GB VRAM)
./venv/bin/python clients/chat.py        # Interactive CLI chat client (commands: quit, clear)
sudo systemctl stop llama-server         # Stop the production service before running a manual launcher
```

Only one engine can run at a time (all bind port 5000); stop the running one before switching.

### Systemd Service (production)

```bash
sudo cp systemd/llama-server.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now llama-server
sudo systemctl status llama-server    # Check status
journalctl -u llama-server -f         # Live logs
```

`systemd/llama-server.service`'s `ExecStart` points at `scripts/start-llama-35b-moe.sh`. To switch the production engine, edit the unit (or change the symlink target) and `daemon-reload`.

## Architecture

- **llama.cpp** (git-ignored): Compiled from source with CUDA SM89 and flash attention. Binary at `llama.cpp/build/bin/llama-server`. Used by the systemd default and the manual `start-llama*.sh` launchers.
- **vllm-venv/** (git-ignored): vLLM 0.17.0 + flashinfer 0.6.4 + torch 2.10.0+cu128. Used only by `start-vllm-35b-mtp.sh`.
- **sglang-venv/** (git-ignored): SGLang environment. Used only by `start-sglang-35b-mtp.sh`.
- **models/** (git-ignored): GGUF files for llama.cpp (`Qwen3.6-35B-A3B-MXFP4_MOE.gguf`, `Qwen3.5-35B-A3B-Q4_K_M.gguf`, `Qwen3.6-27B-Q4_K_M.gguf`, `Qwen3.5-9B-Q4_K_M.gguf`) and the GPTQ safetensors directory `Qwen3.6-35B-A3B-GPTQ-Int4/` (~22.74GB) shared by vLLM and SGLang.
- **venv/** (git-ignored): Python 3.12 venv with `openai`, `huggingface-hub` (provides `hf` CLI for downloads). Used by `chat.py` and the model downloader in `setup.sh`.
- **chat.py**: Streaming multi-turn chat client using OpenAI SDK against localhost:5000 (works against any engine — they all expose the same API shape).
- **scripts/start-llama-35b-moe.sh**: llama.cpp launcher for the MXFP4 MoE GGUF (`--n-gpu-layers 99 --ctx-size 98304 --parallel 8 --flash-attn on --reasoning-format deepseek`). This is what the systemd unit runs. `--parallel 8` is tuned — see `docs/CONCURRENCY.md`.
- **scripts/start-vllm-35b-mtp.sh**: vLLM launcher with `--quantization gptq`, `--reasoning-parser qwen3` (≡ llama.cpp's deepseek format — populates the same `reasoning_content` field), `--speculative-config '{"method": "mtp", "num_speculative_tokens": 5}'` for MTP n=5, and `--cpu-offload-gb 4` because GPTQ weights are tight on a 24GB GPU.
- **scripts/start-sglang-35b-mtp.sh**: SGLang launcher with `--quantization gptq_marlin`, `--reasoning-parser qwen3`, and the NEXTN speculative algorithm (`--speculative-algorithm NEXTN --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6`). Sets `SGLANG_ENABLE_SPEC_V2=1` and pins CUDA 12.8 + g++-14.
- **scripts/start-llama.sh** / **scripts/start-llama-9b.sh** / **scripts/start-llama-27b.sh**: llama.cpp launchers using `--reasoning-format deepseek`. The 9B variant uses 128K context / ~6GB VRAM. The 27B is dense Qwen3.6 — slower than the MoE since all params activate per token.
- **systemd/llama-server.service**: Systemd unit (kept under the historical name even though `ExecStart` is now `scripts/start-llama-35b-moe.sh`). Auto-restart on crash.
- **docs/API.md**: Full API documentation with endpoint details, streaming format, and client examples.
- **docs/CONCURRENCY.md**: Concurrency / `--parallel` tuning results and recommendation.
- **benchmarks/benchmark.py** / **benchmarks/bench_separate.py** (git-ignored): Benchmark scripts.
- **benchmarks/bench_concurrency.py**: Fires N parallel requests and reports aggregate + per-request tok/s. Used to pick `--parallel`.

## Important Details

- **Reasoning/thinking mode**: llama.cpp uses `--reasoning-format deepseek`; vLLM and SGLang both use `--reasoning-parser qwen3`. All three split responses into `reasoning_content` (chain-of-thought) and `content` (final answer). Token counts include both.
- **Speculative decoding (n=5) is engine-specific**:
  - vLLM: `--speculative-config '{"method": "mtp", "num_speculative_tokens": 5}'` (native MTP).
  - SGLang: `--speculative-algorithm NEXTN --speculative-num-steps 5` (also requires `SGLANG_ENABLE_SPEC_V2=1`).
  - llama.cpp does **not** support native MTP — only generic draft-model speculative decoding via `--model-draft`. The systemd default runs without speculation.
  - If the model's MTP module supports fewer than 5 heads, vLLM silently clamps; check logs for the actual acceptance rate.
- **`max_tokens` is broken on llama.cpp's `/v1/chat/completions`** (truncating mid-reasoning causes 500 errors). Omit it or use `stop` sequences. vLLM and SGLang do not have this bug.
- **VRAM is tight under vLLM/SGLang.** GPTQ weights are ~22.7GB. The vLLM launcher uses `--gpu-memory-utilization 0.93` and `--cpu-offload-gb 4`; SGLang uses `--mem-fraction-static 0.92`. If startup OOMs, raise the offload knob or reduce `--max-model-len` / `--context-length` (currently 32768, well below the model's 262K native limit, traded for KV cache room).
- llama.cpp build targets **SM89** (Ada Lovelace / RTX 4090). Change `-DCMAKE_CUDA_ARCHITECTURES=89` in setup.sh for other GPUs.
- SGLang's launcher pins `CUDA_HOME=/usr/local/cuda-12.8` and forces g++-14 with a `-D__THROW=` workaround — this is required because of glibc/CUDA header incompatibilities; do not remove without testing.
- Server listens on `0.0.0.0:5000` with a web UI at the root. Firewall rule allows `192.168.10.0/24`.
- Ollama service was disabled (`systemctl disable ollama`) to avoid VRAM conflicts.
- Cold-start times: llama.cpp ~5s, vLLM ~30–60s, SGLang ~30–90s.
- The `hf` CLI (not `huggingface-cli`) is used for model downloads.
- Only one engine loaded at a time; the `model` field in API requests is accepted but ignored.
