# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local LLM inference server running **Qwen3.5-35B-A3B** (MoE: 35B total, 3B active) on an RTX 4090 (24GB VRAM) using llama.cpp with an OpenAI-compatible API.

## Key Commands

```bash
./setup.sh                  # Full setup: build llama.cpp, download model, install deps (idempotent)
./start-llama.sh            # Start the API server on port 5000
./venv/bin/python chat.py   # Interactive CLI chat client (commands: quit, clear)
pkill -f llama-server       # Stop the server
```

### Systemd Service (production)

```bash
sudo cp llama-server.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now llama-server
sudo systemctl status llama-server    # Check status
journalctl -u llama-server -f         # Live logs
```

## Architecture

- **llama.cpp** (git-ignored): Compiled from source with CUDA SM89 and flash attention. Binary at `llama.cpp/build/bin/llama-server`.
- **models/** (git-ignored): Contains `Qwen3.5-35B-A3B-Q4_K_M.gguf` (20.5GB, Q4_K_M from `unsloth/Qwen3.5-35B-A3B-GGUF`).
- **venv/** (git-ignored): Python 3.12 venv with `openai` and `huggingface-hub` packages.
- **chat.py**: Streaming multi-turn chat client using OpenAI SDK against localhost:5000.
- **start-llama.sh**: Launches llama-server with `--reasoning-format deepseek` for thinking mode support.
- **llama-server.service**: Systemd unit file for production deployment (auto-restart on crash).
- **API.md**: Full API documentation with endpoint details, streaming format, and client examples.
- **benchmark.py** / **bench_separate.py** (git-ignored): Benchmark scripts comparing llama.cpp vs Ollama performance.

## Important Details

- **Reasoning/thinking mode**: Server uses `--reasoning-format deepseek`, which splits responses into `reasoning_content` (chain-of-thought) and `content` (final answer). Token counts include both.
- **`max_tokens` is broken on `/v1/chat/completions`**: Truncating mid-reasoning causes 500 errors. Omit the field or use `stop` sequences. Works fine on `/v1/completions`.
- The build targets **SM89** (Ada Lovelace / RTX 4090). Change `-DCMAKE_CUDA_ARCHITECTURES=89` in setup.sh for other GPUs.
- Server listens on `0.0.0.0:5000` with a web UI at the root. Firewall rule allows `192.168.10.0/24`.
- Model uses ~23.5GB VRAM with 96K context (98,304 tokens). Max tested is 112K (OOM at 128K). Reduce `--ctx-size` in start-llama.sh if VRAM is tight.
- Ollama service was disabled (`systemctl disable ollama`) to avoid VRAM conflicts.
- The `hf` CLI (not `huggingface-cli`) is used for model downloads in newer huggingface-hub versions.
- Only one model loaded at a time; the `model` field in API requests is accepted but ignored.
