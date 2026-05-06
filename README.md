# Qwen3.6-35B-A3B Local LLM Server

Run Qwen3.6-35B-A3B locally on an RTX 4090 via llama.cpp with an OpenAI-compatible API. Default config is tuned for **single-user max context** — the full 96K window is available per request.

## Performance

| Metric | Value |
|---|---|
| Single-request generation | ~163 tok/s |
| Prompt processing (1K tokens) | ~5,500 tok/s |
| VRAM usage | ~23.2 GB |
| Total context window | 96K tokens (98,304) |
| Per-request context (`--parallel 1`) | 96K tokens |

## Context Window vs Concurrency

llama.cpp's `--ctx-size` is the **total** KV cache, divided across `--parallel` slots. The default `--parallel 1` gives one user the full 96K. Switch to higher `--parallel` if you need to serve concurrent requests, at the cost of per-request context:

| `--parallel` | Per-request context | Concurrent requests | Use case |
|---|---|---|---|
| **1 (default)** | **96K** | 1 (rest queue) | Solo use, long docs/code/chats |
| 2 | 48K | 2 | You + a background agent |
| 4 | 24K | 4 | Small team / multi-agent |
| 8 | 12K | 8 | Max throughput @ ~300 tok/s aggregate |

To change, edit `--parallel` in `scripts/start-llama-35b-moe.sh` and restart the service. See [docs/CONCURRENCY.md](./docs/CONCURRENCY.md) for the throughput curve across `--parallel 4 / 8 / 10`.

### Going beyond 96K

VRAM is the bottleneck on a 24GB 4090 — the model alone is ~21.7 GB. To push context higher you'd need to free KV-cache memory, e.g. add `--cache-type-k q8_0 --cache-type-v q8_0` (roughly halves KV memory) and bump `--ctx-size` toward the model's 262K native max. YaRN extends the model to ~1M but requires KV quantization to fit.

### Model Comparison (RTX 4090, llama.cpp)

| Model | Gen speed | VRAM | Context |
|---|---|---|---|
| **Qwen3.6-35B-A3B** (MoE, 3B active, MXFP4) | **163 tok/s** | ~23.5 GB | 96K |
| **Qwen3.5-9B** (dense, Q4_K_M) | 125 tok/s | ~6 GB | 128K+ |

The 35B MoE is faster than the dense 9B at generation despite being larger — only 3B parameters activate per token. The 9B wins on VRAM and supports a larger per-request context.

## Quick Start

```bash
# One-time setup (builds llama.cpp, downloads model)
./scripts/setup.sh

# Start the server
./scripts/start-llama-35b-moe.sh
```

## Usage

### Web UI
Open http://localhost:5000

### CLI Chat
```bash
./venv/bin/python clients/chat.py
```

### API (OpenAI-compatible)
```bash
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.6-35B-A3B-MXFP4_MOE.gguf",
       "messages": [{"role": "user", "content": "Hello"}]}'
```

### Python SDK
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:5000/v1", api_key="none")
response = client.chat.completions.create(
    model="Qwen3.6-35B-A3B-MXFP4_MOE.gguf",
    messages=[{"role": "user", "content": "Hello"}]
)
```

Network access: http://192.168.10.124:5000 (firewall rule added by `setup.sh`).

### Benchmark concurrency

```bash
./venv/bin/python benchmarks/bench_concurrency.py
```

Fires 1, 2, 4, 6, 8, 10 parallel requests and reports aggregate + per-request tok/s.

## Production Service (systemd)

Install and enable:
```bash
sudo cp systemd/llama-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable llama-server
sudo systemctl start llama-server
```

Manage:
```bash
sudo systemctl start llama-server     # Start
sudo systemctl stop llama-server      # Stop
sudo systemctl restart llama-server   # Restart
sudo systemctl status llama-server    # Check status
journalctl -u llama-server -f         # View live logs
```

The service auto-starts on boot and auto-restarts on crash (5s delay). It runs `start-llama-35b-moe.sh`; alternative engines (vLLM, SGLang) and other model launchers are documented in [CLAUDE.md](./CLAUDE.md).

## Requirements

- NVIDIA GPU with 24GB VRAM (RTX 4090)
- CUDA toolkit
- cmake, build-essential
- ~25GB disk space (model + llama.cpp build)

## Model

[unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) (MXFP4_MOE quantization)

- 35B total params, 3B active per token (MoE architecture)
- MXFP4 quantization, ~20.2 GB file size
- Reasoning/thinking mode enabled via `--reasoning-format deepseek` (responses split into `reasoning_content` and `content`)
