# Homelab LLM Server (Qwen3.8 on RTX PRO 6000)

Local LLM inference on an RTX PRO 6000 Blackwell (96GB VRAM, SM120) exposing an OpenAI-compatible API on port 5000. Production since 2026-08-28 is **llama.cpp serving Qwen3.8-Flash-Next-Uncensored IQ4_XS** (orcarouter abliterated 177B ultra-sparse MoE, 6B active/token) — full 262K native context, vision enabled, ~76GB VRAM. The model's 51B n-gram (PLE) table is mmap'd host-side, which is how a 97.5GB GGUF fits.

| Engine (via `scripts/production-engine.conf`) | Model | Notes | Concurrency | Single-stream |
|---|---|---|---|---|
| `start-llama-flashnext.sh` **(production)** | Qwen3.8-Flash-Next-Uncensored IQ4_XS | 262K ctx, **vision**, uncensored; single-user profile | ~130 agg @ N=4 (plateau) | ~109 tok/s |
| `start-vllm-38-27b-nvfp4.sh` | Qwen3.8-27B-NVFP4-Inferact | **agent-fleet engine**; text-only | ~676 agg @ N=16 | ~129 tok/s |
| `start-vllm-38-27b-fp8.sh` | Qwen3.8-27B-FP8 | vision-capable 27B | ~592 agg @ N=16 | ~91 tok/s |
| `start-vllm-38-27b-uncensored-fp8.sh` | Qwen3.8-27B-Uncensored-FP8 (orcarouter) | abliterated 27B, on-demand | — | — |
| `start-vllm-27b-fp8.sh` | Qwen3.6-27B-FP8 | deeper rollback (2026-08-12 bake-off winner) | ~530 @ N=8 | ~118 tok/s |

**To switch engines** (e.g. Flash-Next ↔ NVFP4 when running parallel agent fleets): edit the one non-comment line in `scripts/production-engine.conf`, then `sudo systemctl restart llama-server`. Full measurements are in [docs/BENCHMARKS.md](./docs/BENCHMARKS.md) (2026-08-28 addendum for Flash-Next); system-level changes are tracked in [CHANGELOG.md](./CHANGELOG.md).

The former llama.cpp setups (two-model agent stack, single 27B/35B) remain as manual launchers — see [CLAUDE.md](./CLAUDE.md).

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

Only one engine can own port 5000 at a time (and production plus a second model won't fit in VRAM). Stop the service first (`sudo systemctl stop llama-server`), wait ~10s for VRAM release, then launch. Startup until `curl localhost:5000/health` → 200: Flash-Next ~20s–1 min with warm page cache but **~10–25 min cold from disk** (97.5GB streams in); vLLM engines ~2–3 min; small llama.cpp GGUFs ~5s.

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
       "messages": [{"role": "user", "content": "Hello"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://192.168.10.106:5000/v1", api_key="none")
response = client.chat.completions.create(
    model="local",   # llama.cpp ignores the field, but vLLM fallback engines validate it — always send "local"
    messages=[{"role": "user", "content": "Hello"}],
    # optional: clamp thinking on a per-request basis ("minimal" is not a valid value)
    extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
)
```

Things that matter:

- **Always send `model: "local"`** — the llama.cpp production engine ignores the field, but the vLLM fallback engines reject other names, so hardcoding `"local"` works across every engine.
- **Omit `max_tokens` on the production engine** (or disable thinking) — llama.cpp still truncates mid-reasoning and returns empty content or 500s. The vLLM engines handle it correctly.
- **Reasoning effort**: all launchers pin `medium` server-side (vLLM `--default-chat-template-kwargs`, llama.cpp `--chat-template-kwargs`), overriding the template's `xhigh` default. Send `chat_template_kwargs: {"reasoning_effort": "low"}` per request to clamp further; only `xhigh`, `medium`, `low` are valid.
- **Never use temperature 0** — greedy decoding produces unbounded thinking loops. Qwen3.8 thinking mode: temp 1.0 / top_p 0.95 / top_k 20 (the launcher sets these server-side).
- Chain-of-thought arrives in **`reasoning_content`** on the llama.cpp production engine; the vLLM fallbacks use **`reasoning`** instead.
- **Vision works on production** (base64 `image_url` — the mmproj loads by default; `FLASHNEXT_VISION=0` disables). Tool calling works as-is.
- llama.cpp serves its web UI at `/` (gzip-only — use `curl --compressed`); the vLLM engines serve none.

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

Each request gets up to **262,144 tokens** (prompt + reasoning + output combined — Flash-Next's native window; needle-retrieval verified to 261K). The hybrid architecture (3 of 4 layers are Gated-DeltaNet linear attention with constant-size state, plus query-sparse attention on the full-attention layers) keeps KV growth slow — the full 262K window costs only ~8GB VRAM over a 16K config. The server runs 4 slots over a unified KV pool; it's tuned for 1–4 concurrent users, and aggregate throughput plateaus around ~130 tok/s at N=4. **For wide agent fleets** (orchestrate.py with many workers), switch to the NVFP4 27B engine (~676 agg tok/s @ N=16, ~1.55M-token FP8 KV pool) for the session.

## Requirements

- NVIDIA GPU with 96GB VRAM for the production config; ≥48GB host RAM (Flash-Next's 51B PLE table lives in page cache)
- CUDA 13.1 toolkit, cmake, build-essential; `ninja` for the vLLM engines' flashinfer JIT
- ~350GB disk for the model zoo (Flash-Next GGUFs + vLLM safetensors + llama.cpp GGUFs + builds)

## Models

- [Qwen3.8-Flash-Next-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF) (**production**) — 177B multimodal ultra-sparse MoE (125B transformer + 51B PLE n-gram table, 6B active/token, 512 experts 10+1 active), orcarouter abliterated build, IQ4_XS + F16 mmproj; gated repo. Needs llama.cpp ≥ 2026-08-27 (`qwen4exp`). No MTP head in the GGUF — speed comes from sparsity (~109 tok/s single-stream at 4-bit).
- [Qwen3.8-27B](https://huggingface.co/Qwen) — hybrid GDN/full-attention dense 27B, Apache 2.0, MTP head for speculative decoding: NVFP4 (Inferact modelopt, agent-fleet engine), FP8 (official, vision-capable), Uncensored-FP8 (orcarouter community build)
- Qwen3.6-27B-FP8 — rollback; Qwen3.6 GGUFs (27B Q8_0, 35B-A3B MXFP4_MOE) for the manual llama.cpp launchers

All run in thinking mode. For the dense 27Bs, MTP speculative decoding is the whole ballgame (~50 tok/s base decode → 2.4–2.6× with n=3); Flash-Next has no MTP and doesn't need it.
