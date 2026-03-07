# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local LLM inference server running **Qwen3.5-35B-A3B** (MoE: 35B total, 3B active) on an RTX 4090 (24GB VRAM) using llama.cpp with an OpenAI-compatible API.

## Key Commands

```bash
./setup.sh          # Full setup: build llama.cpp, download model, install deps
./start-llama.sh    # Start the API server on port 5000
./venv/bin/python chat.py   # Interactive CLI chat client
pkill -f llama-server       # Stop the server
```

## Architecture

- **llama.cpp** (git-ignored): Compiled from source with CUDA SM89 and flash attention. Binary at `llama.cpp/build/bin/llama-server`.
- **models/** (git-ignored): Contains `Qwen3.5-35B-A3B-Q4_K_M.gguf` (20.5GB, from `unsloth/Qwen3.5-35B-A3B-GGUF`).
- **venv/** (git-ignored): Python 3.12 venv with `openai` and `huggingface-hub` packages.
- **chat.py**: Streaming chat client using OpenAI SDK against localhost:5000.
- **setup.sh**: Idempotent setup script — safe to re-run, skips completed steps.
- **start-llama.sh**: Launches llama-server with tuned flags for RTX 4090.

## Important Details

- The build targets **SM89** (Ada Lovelace / RTX 4090). Change `-DCMAKE_CUDA_ARCHITECTURES=89` in setup.sh for other GPUs.
- Server listens on `0.0.0.0:5000`. Firewall rule allows access from `192.168.10.0/24`.
- Model uses ~22GB VRAM with 8192 context. Reduce `--ctx-size` in start-llama.sh if VRAM is tight.
- Ollama service was disabled (`systemctl disable ollama`) to avoid VRAM conflicts.
- The `hf` CLI (not `huggingface-cli`) is used for downloads in newer huggingface-hub versions.
