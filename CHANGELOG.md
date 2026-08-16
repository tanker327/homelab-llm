# Changelog

System-level changes to the inference server (host config, systemd, firewall)
are recorded here — code changes are tracked by git history (`git log`).

## 2026-08-15 — Qwen3.8-27B-FP8 replaces Qwen3.6-27B-FP8 as production default

Qwen3.8-27B went open-weights 2026-08-13 (Apache 2.0) — a large quality jump
over 3.6 (SWE-bench Pro 61.7 vs 53.5, Terminal Bench 73.0 vs 63.4) on a new
hybrid backbone (48/64 Gated DeltaNet linear-attention layers). Validated and
tuned 2026-08-15; record in `docs/BENCHMARKS.md` (Qwen3.8 addendum) and
`benchmarks/results/qwen38_*.jsonl`.

- **systemd unit `llama-server.service` now runs `scripts/start-vllm-38-27b-fp8.sh`**:
  same vLLM 0.27.1 (no engine upgrade needed — arch `Qwen3_5ForConditionalGeneration`
  was already supported), MTP n=3, FP8 KV cache, 262K max-model-len,
  **16 concurrent sequences** (up from 8 — the hybrid KV cache holds ~1.55M
  tokens, vs far less on 3.6's dense attention). ~88GB VRAM.
- **Numbers**: ~592 agg tok/s @ N=16 agent load (3.6 record: ~530 @ N=8),
  ~444 @ N=8, ~90–95 tok/s single-stream (3.6: ~118 — 3.8's single recursive
  MTP layer accepts fewer drafts), 1.0 tok/J @ N=16, NIAH-correct to 261.7K
  (3.6: ~255K), 10/10 quality smoke. MTP n=4 was faster single-stream but
  showed erratic multi-second stalls — rejected.
- **Sampling changed**: Qwen3.8 thinking-mode recommendation is temp 1.0 /
  top_p 0.95 / top_k 20 (3.6 was 0.6/0.95). `quality_smoke.py` default
  updated; temp 0 still forbidden.
- **Rollback**: Qwen3.6-27B-FP8 weights and `start-vllm-27b-fp8.sh` kept —
  point `ExecStart` back and restart.

## 2026-08-12 — vLLM Qwen3.6-27B-FP8 replaces the agent stack as production default

Decision from the three-engine bake-off recorded in `docs/BENCHMARKS.md`
(addendum) and `benchmarks/results/`: the vLLM winner more than doubles the
stack's 27B agent throughput (530 vs 244 agg tok/s at N=8) with 2× the energy
efficiency, 10/10 quality parity, and usable context to ~255K tokens.

- **systemd unit `llama-server.service` now runs `scripts/start-vllm-27b-fp8.sh`**
  (unit keeps its historical name): vLLM 0.27.1, `Qwen/Qwen3.6-27B-FP8`
  safetensors, MTP speculative decoding n=3, FP8 KV cache, 262K max-model-len,
  8 concurrent sequences, ~86–90GB VRAM. `RestartSec` raised 5→15s (vLLM needs
  the old process's VRAM released before reallocating; boots in ~2–3 min vs
  llama.cpp's 5s).
- **The two-model agent stack is retired from systemd** (35B MoE workers +
  27B planner). `start-agent-stack.sh` and all llama.cpp launchers remain on
  disk for manual use — stop the service first. Port 5001 is no longer served;
  the ufw rule for it is harmless and was left in place.
- **API compatibility**: same OpenAI-compatible surface on port 5000, but vLLM
  *validates* the `model` field — clients must send `"local"`
  (`--served-model-name local`). `chat.py` fixed (sent a stale GGUF name that
  llama.cpp ignored); `orchestrate.py` now points planner/judge at 5000 too.
  `max_tokens` works correctly (the llama.cpp truncation bug does not apply),
  reasoning arrives in `reasoning`/`reasoning_content` fields via
  `--reasoning-parser qwen3`.
- **New venvs at repo root**: `vllm-venv/` (vLLM 0.27.1, torch 2.13.0+cu130),
  `sglang-venv/` (SGLang 0.5.17, evaluation only) — plain pip wheels; the
  4090-era CUDA 12.8 / g++-14 pins are obsolete on this box. flashinfer JIT
  needs `ninja` on PATH and the `toolchain-fix` header shim (the launcher
  exports both).
- **Do not run the model at temperature 0** — greedy decoding produces
  unbounded thinking loops; use Qwen's recommended temp 0.6 / top_p 0.95.
- **Tool calling enabled on the production launcher**: added
  `--enable-auto-tool-choice --tool-call-parser qwen3_xml` (the model's chat
  template emits Qwen's XML `<function=...>` format). Without these, any
  client sending `tool_choice: "auto"` — Open WebUI does — got a 400.
- **Open WebUI added for browser chat** (vLLM serves no web UI, unlike
  llama.cpp): Docker container `open-webui` on the **host network**, UI on
  port 3000, pointed at `http://localhost:5000/v1`. Host networking is
  required because ufw blocks the Docker bridge subnet from reaching port
  5000. `--restart always`, data in the `open-webui` Docker volume. LAN
  access: `sudo ufw allow from 192.168.10.0/24 to any port 3000 proto tcp`.

## 2026-08-09 — Agent stack becomes the production default

- **systemd unit installed and enabled on the RTX PRO 6000 box.** The server
  had only ever been launched manually here; an unplanned reboot at 14:45 UTC
  silently took the API down, which is exactly what the unit prevents.
  Installed to `/etc/systemd/system/llama-server.service`, enabled at boot.
- **Production default switched from single Qwen3.6-27B to the agent stack**
  (`scripts/start-agent-stack.sh`): Qwen3.6-35B-A3B workers on port 5000
  (3 slots × 128K ctx), Qwen3.6-27B Q8_0 planner/judge on port 5001
  (2 slots × 128K ctx). ~77GB of 96GB VRAM.
- **`start-agent-stack.sh` hardened for systemd**: if either server dies, the
  script kills the other and exits non-zero so `Restart=on-failure` relaunches
  the pair together instead of leaving a half-alive stack.
- **Clean model names in the API**: all llama.cpp launchers now pass
  `--alias`, so `/v1/models` reports `Qwen3.6-35B-A3B` / `Qwen3.6-27B` etc.
  instead of the full GGUF file path.
- **Firewall**: only port 5000 was originally allowed from the LAN; the 27B
  web UI/API on 5001 needs its own rule:
  `sudo ufw allow from 192.168.10.0/24 to any port 5001 proto tcp`.

### Operational notes learned today

- Wait ~10s between stopping one server and starting another: the old process
  needs time to release VRAM, and the new one OOMs on its KV-cache allocation
  (32GB for the 27B at full context) if it races the teardown.
- Per-request context in the stack is **128K**, not the models' 262K native
  max — total `--ctx-size` is split across `--parallel` slots. The single-model
  `start-llama-27b.sh` gives the full 262K per request. Full 262K in the stack
  is possible by trading concurrency (35B: `--ctx-size 524288 --parallel 2`,
  27B: `--parallel 1`), costing only ~+3GB.

## Earlier

RTX PRO 6000 migration, MTP speculative-decoding defaults, and benchmark
results predate this changelog — see `git log` and `docs/BENCHMARKS.md`.
