# Qwen3.5-35B-A3B Local LLM Server

Run Qwen3.5-35B-A3B locally on an RTX 4090 via llama.cpp with an OpenAI-compatible API.

## Performance

| Metric | Value |
|---|---|
| Generation speed | ~167 tok/s |
| Prompt processing | ~700 tok/s |
| VRAM usage | ~22 GB |
| Context window | 8192 tokens |

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
