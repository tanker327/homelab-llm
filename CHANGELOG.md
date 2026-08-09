# Changelog

System-level changes to the inference server (host config, systemd, firewall)
are recorded here — code changes are tracked by git history (`git log`).

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
