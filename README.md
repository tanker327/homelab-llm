# Homelab LLM Server (Qwen3.6 on RTX PRO 6000)

Local LLM inference on an RTX PRO 6000 Blackwell (96GB VRAM) via llama.cpp, exposing an OpenAI-compatible API. The production default is the **agent stack** — both Qwen3.6 models resident at once:

| Port | Model | Role | Speed | Slots × ctx |
|---|---|---|---|---|
| 5000 | `Qwen3.6-35B-A3B` (MoE, MXFP4, MTP) | workers / general chat | ~418 tok/s | 3 × 128K |
| 5001 | `Qwen3.6-27B` (dense, Q8_0, MTP) | planner / judge, best quality | ~139 tok/s | 2 × 128K |

Total footprint ~77GB. Both use MTP speculative decoding (`--spec-type draft-mtp`). Full measurements are in [docs/BENCHMARKS.md](./docs/BENCHMARKS.md); system-level changes are tracked in [CHANGELOG.md](./CHANGELOG.md).

## Quick Start

```bash
# One-time setup (builds llama.cpp, downloads models)
./scripts/setup.sh

# Start the agent stack (what systemd runs in production)
./scripts/start-agent-stack.sh

# Or a single model with the full 262K context per request:
./scripts/start-llama-27b.sh        # dense 27B Q8_0, best quality
./scripts/start-llama-35b-moe.sh    # 35B MoE, fastest
```

Only one launcher can own port 5000 at a time; stop the systemd service first (`sudo systemctl stop llama-server`) before running one manually, and leave ~10s between stop and start so VRAM is released.

## Usage

### Web UI

- http://192.168.10.106:5000 — 35B workers
- http://192.168.10.106:5001 — 27B planner/judge

(Port 5001 needs its own firewall rule — see [CHANGELOG.md](./CHANGELOG.md).)

### CLI Chat

```bash
./venv/bin/python clients/chat.py
```

### Orchestrator (plan → parallel workers → judge)

```bash
./venv/bin/python clients/orchestrate.py "your task"   # --mode bestof, --workers N
```

Plans on the 27B, fans out to 3 parallel 35B workers, judges on the 27B.

### API (OpenAI-compatible)

```bash
curl http://192.168.10.106:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.6-35B-A3B",
       "messages": [{"role": "user", "content": "Hello"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://192.168.10.106:5000/v1", api_key="none")
response = client.chat.completions.create(
    model="Qwen3.6-35B-A3B",   # accepted but ignored — each port serves one model
    messages=[{"role": "user", "content": "Hello"}]
)
```

`/v1/models` reports clean alias names (`Qwen3.6-35B-A3B`, `Qwen3.6-27B`), not file paths. Note: `max_tokens` is unreliable on llama.cpp's chat endpoint with thinking models — omit it or use `stop` sequences. See [docs/API.md](./docs/API.md) for full endpoint documentation.

### Benchmark concurrency

```bash
./venv/bin/python benchmarks/bench_concurrency.py
```

Fires 1, 2, 4, 6, 8, 10 parallel requests and reports aggregate + per-request tok/s. See [docs/CONCURRENCY.md](./docs/CONCURRENCY.md) for the tuning results.

## Production Service (systemd)

Install and enable:
```bash
sudo cp systemd/llama-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server
```

Manage:
```bash
sudo systemctl status llama-server    # Check status
sudo systemctl restart llama-server   # Restart (both models)
journalctl -u llama-server -f         # Live logs
```

The service auto-starts on boot and auto-restarts on crash (5s delay). It runs `start-agent-stack.sh`; if either of the two servers dies, the stack exits and systemd relaunches the pair together. Alternative engines (vLLM, SGLang) and other model launchers are documented in [CLAUDE.md](./CLAUDE.md).

## Context Window vs Concurrency

llama.cpp's `--ctx-size` is the **total** KV cache, divided across `--parallel` slots — the stack's 128K per request comes from that split. The models natively support 262K; the single-model `start-llama-27b.sh` provides it (2 × 262K, ~32GB KV cache). To get 262K per slot inside the stack, trade concurrency: 35B `--ctx-size 524288 --parallel 2`, 27B `--parallel 1` (~+3GB).

## Requirements

- NVIDIA GPU with 96GB VRAM for the agent stack (single 35B fits in 24GB — the repo originally targeted an RTX 4090, and the launchers note the old settings)
- CUDA toolkit, cmake, build-essential
- ~60GB disk for the two production GGUFs + llama.cpp build

## Models

- [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) — MXFP4_MOE with MTP head, 35B total / 3B active per token, ~20GB
- Qwen3.6-27B Q8_0 with MTP head — dense, near-lossless quant, ~29GB

Both run in thinking mode (`--reasoning-format deepseek`): responses split into `reasoning_content` and `content`.
