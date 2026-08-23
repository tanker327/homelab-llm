# Homelab LLM Server (Qwen3.8 on RTX PRO 6000)

Local LLM inference on an RTX PRO 6000 Blackwell (96GB VRAM, SM120) exposing an OpenAI-compatible API on port 5000. Production is **vLLM 0.27.1 serving Qwen3.8-27B-NVFP4** with MTP speculative decoding (n=3) and FP8 KV cache — 262K context per request, 16 concurrent sequences, ~88GB VRAM.

| Engine (via `scripts/production-engine.conf`) | Model | Notes | Agg @ N=16 | Single-stream |
|---|---|---|---|---|
| `start-vllm-38-27b-nvfp4.sh` **(production)** | Qwen3.8-27B-NVFP4-Inferact | fastest; **text-only** | ~676 tok/s | ~129 tok/s |
| `start-vllm-38-27b-fp8.sh` | Qwen3.8-27B-FP8 | vision-capable fallback | ~592 tok/s | ~91 tok/s |
| `start-vllm-38-27b-uncensored-fp8.sh` | Qwen3.8-27B-Uncensored-FP8 (orcarouter) | abliterated build, on-demand | — | — |
| `start-vllm-27b-fp8.sh` | Qwen3.6-27B-FP8 | deeper rollback (2026-08-12 bake-off winner) | ~530 @ N=8 | ~118 tok/s |

**To switch engines** (e.g. NVFP4 ↔ FP8 when vision is needed): edit the one non-comment line in `scripts/production-engine.conf`, then `sudo systemctl restart llama-server`. Full measurements are in [docs/BENCHMARKS.md](./docs/BENCHMARKS.md); system-level changes are tracked in [CHANGELOG.md](./CHANGELOG.md).

The former llama.cpp production setups (two-model agent stack, single 27B/35B) remain as manual launchers — see [CLAUDE.md](./CLAUDE.md).

## Quick Start

```bash
# One-time setup (builds llama.cpp, downloads GGUFs; vLLM lives in vllm-venv/, set up separately)
./scripts/setup.sh

# Start production manually (what systemd runs — execs the launcher named in production-engine.conf)
./scripts/start-production.sh

# Manual llama.cpp alternatives:
./scripts/start-agent-stack.sh      # former default: 35B MoE on 5000 + 27B Q8_0 on 5001 (~77GB)
./scripts/start-llama-27b.sh        # dense 27B Q8_0 MTP, 262K ctx (~64GB)
./scripts/start-llama-35b-moe.sh    # 35B-A3B MoE, fastest llama.cpp option
```

Only one engine can own port 5000 at a time (and production plus a second model won't fit in VRAM). Stop the service first (`sudo systemctl stop llama-server`), wait ~10s for VRAM release, then launch. vLLM takes ~2–3 min to become healthy (`curl localhost:5000/health` → 200); llama.cpp ~5s.

## Usage

### CLI Chat

```bash
./venv/bin/python clients/chat.py
```

### Orchestrator (plan → parallel workers → judge)

```bash
./venv/bin/python clients/orchestrate.py "your task"   # --mode bestof, --workers N
```

All stages run against the single production model on :5000.

### API (OpenAI-compatible)

```bash
curl http://192.168.10.106:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "local",
       "messages": [{"role": "user", "content": "Hello"}],
       "chat_template_kwargs": {"reasoning_effort": "medium"}}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://192.168.10.106:5000/v1", api_key="none")
response = client.chat.completions.create(
    model="local",   # required — vLLM validates the model name
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={"chat_template_kwargs": {"reasoning_effort": "medium"}},
)
```

Things that matter:

- **`model` must be `"local"`** — vLLM rejects other names (llama.cpp ignored the field, which is why stale names used to go unnoticed).
- **Send `reasoning_effort: "medium"`** (or `low`) — the chat-template default is `xhigh`, which burns ~10× tokens for no measured quality gain.
- **Never use temperature 0** — greedy decoding produces unbounded thinking loops. Qwen3.8 thinking mode: temp 1.0 / top_p 0.95 / top_k 20.
- Chain-of-thought arrives in the **`reasoning`** field on vLLM (not `reasoning_content`).
- Vision (base64 `image_url`, FP8 engine only) and tool calling work as-is.
- vLLM serves no web UI at `/` (the old llama.cpp UI only exists on the manual launchers).

See [docs/API.md](./docs/API.md) for full endpoint documentation.

### Benchmarks

```bash
./venv/bin/python benchmarks/bench_serving.py --label x --port 5000 --workload agent --levels 1,4,8 --runs 3
./venv/bin/python benchmarks/quality_smoke.py     # 10-task pass/fail quality gate
```

`bench_serving.py` is the main harness (streaming TTFT p50/p95, GPU power/VRAM sampling, tok/J, MTP acceptance). The 2026-08 bake-off record lives in `benchmarks/results/` and [docs/BENCHMARKS.md](./docs/BENCHMARKS.md).

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
sudo systemctl restart llama-server   # Restart (also how you apply an engine switch)
journalctl -u llama-server -f         # Live logs
```

The unit keeps its historical `llama-server` name but runs `scripts/start-production.sh`, which execs whichever launcher `scripts/production-engine.conf` names. Auto-starts on boot, auto-restarts on crash.

## Context Window vs Concurrency

Each request gets up to **262,144 tokens** (prompt + reasoning + output combined — the model's native window). Qwen3.8's hybrid architecture (48 of 64 layers are Gated-DeltaNet linear attention; only 16 full-attention layers hold KV) means the FP8 KV cache pool holds ~1.55M tokens, so 16 concurrent sequences are practical. Requests beyond 16 queue and are admitted as slots free up. A 1M-context YaRN config was probed and works (needle-pass at 980K) but is prefill-bound — special occasions only; flags in BENCHMARKS.md.

## Requirements

- NVIDIA GPU with 96GB VRAM for the production config (the repo originally targeted an RTX 4090 — a ~4-bit 27B at reduced context is the realistic 24GB deployment)
- CUDA 13.1 toolkit, cmake, build-essential; `ninja` for flashinfer JIT
- ~90GB disk for the vLLM safetensors models + GGUFs + llama.cpp build

## Models

- [Qwen3.8-27B](https://huggingface.co/Qwen) — hybrid GDN/full-attention 27B, Apache 2.0, MTP head for speculative decoding: NVFP4 (Inferact modelopt, production), FP8 (official, vision fallback), Uncensored-FP8 (orcarouter community build)
- Qwen3.6-27B-FP8 — rollback; Qwen3.6 GGUFs (27B Q8_0, 35B-A3B MXFP4_MOE) for the llama.cpp launchers

All run in thinking mode: MTP speculative decoding is the whole ballgame for the dense 27B (~50 tok/s base decode → 2.4–2.6× with n=3).
