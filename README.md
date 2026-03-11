# Qwen3.5-35B-A3B Local LLM Server

Run Qwen3.5-35B-A3B locally on an RTX 4090 via llama.cpp with an OpenAI-compatible API.

## Performance

| Metric | Value |
|---|---|
| Generation speed | ~169 tok/s |
| Prompt processing | ~849 tok/s |
| VRAM usage | ~23.5 GB |
| Context window | 96K tokens (98,304) |
| Max tested context | 112K tokens (OOM at 128K) |

### Model Comparison (RTX 4090, Q4_K_M, llama.cpp)

| Model | Gen speed | Prompt speed | VRAM | Context |
|---|---|---|---|---|
| **Qwen3.5-35B-A3B** (MoE, 3B active) | **169 tok/s** | 849 tok/s | ~23.5 GB | 96K |
| **Qwen3.5-9B** (dense) | 125 tok/s | 1,260 tok/s | ~6 GB | 128K+ |

The 35B MoE model is 35% faster at generation despite being larger, since only 3B parameters are active per token. The 9B model wins on prompt processing and VRAM usage.

## Quick Start

```bash
# One-time setup (builds llama.cpp, downloads model ~20.5GB)
./setup.sh

# Start the server
./start-llama.sh
```

## Usage

### Web UI
Open http://localhost:5000

### CLI Chat
```bash
./venv/bin/python chat.py
```

### API (OpenAI-compatible)
```bash
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
       "messages": [{"role": "user", "content": "Hello"}]}'
```

### Python SDK
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:5000/v1", api_key="none")
response = client.chat.completions.create(
    model="Qwen3.5-35B-A3B-Q4_K_M.gguf",
    messages=[{"role": "user", "content": "Hello"}]
)
```

Network access: http://192.168.10.124:5000 (firewall rule added by setup.sh)

## Production Service (systemd)

Install and enable:
```bash
sudo cp llama-server.service /etc/systemd/system/
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

The service auto-starts on boot and auto-restarts on crash (5s delay).

## Requirements

- NVIDIA GPU with 24GB VRAM (RTX 4090)
- CUDA toolkit
- cmake, build-essential
- ~25GB disk space (model + llama.cpp build)

## Model

[unsloth/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) (Q4_K_M quantization)

- 35B total params, 3B active (MoE architecture)
- 4-bit quantization, 20.5GB file size
