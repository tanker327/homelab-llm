# NInfer on the RTX 4090 box

Qwen3.8-27B served by [sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090)
(an `sm_89` fork of [Neroued/ninfer](https://github.com/Neroued/ninfer)) in
Docker, with the full native 262K context, MTP n=3 speculative decoding and
OpenAI + Anthropic compatible APIs on **port 8080**. This is the first
Qwen3.8-class engine that fits on 24GB with full context; the llama.cpp
launchers in `../scripts/` remain as the Qwen3.6 fallback on port 5000.

Installed 2026-09-16 from the external "NInfer on RTX 4090: Setup Guide".
The corrected, reproducible version of that guide is **[SETUP-GUIDE.md](./SETUP-GUIDE.md)**.
Everything lives under **`/opt/ninfer/`** on the host (not in this repo);
`compose.yaml` here is a copy of the live one.

## Host layout

```
/opt/ninfer/
├── ninfer-4090/          git clone of the fork
├── PINNED_COMMIT         commit the image was built from
├── models/qwen3_8_27b.ninfer      18.2 GB container-v2 artifact, HF revision 35269130 (read-only mount)
├── models/qwen3_8_27b.v3.ninfer   20.4 GB container-v3 artifact from HF `main` (unusable by this fork, kept)
├── MODEL_REVISION        HF revision the live model came from
├── run-build.sh / run-download-v2.sh   detached helpers used for the long build/download
├── slots/                saved sessions (--slot-save-path)
├── logs/requests.jsonl   per-request records (--request-log-jsonl)
├── compose.yaml          the service definition (copy kept in this folder)
├── build.log / download.log
└── SHA256SUMS            published checksum for the model
```

Docker image: `ninfer-4090:sm89`, also tagged with the first 8 chars of the
pinned commit for rollback.

## What was done on this host

Steps from the guide that were already satisfied and skipped:

- Driver 595.91.07 (guide needs 580+). No display manager runs and the 4090
  shows 1 MiB used, so the box is headless in practice even though the
  systemd default target is still `graphical.target`. Not rebooted.
- Docker, Compose v5 and `nvidia-container-toolkit` 1.20 were already
  installed and the `nvidia` runtime already registered (dcgm-exporter uses it).
- GPU power cap (optional in the guide) was **not** applied.

Steps performed:

1. `docker run --rm --gpus all nvidia/cuda:13.1.2-runtime-ubuntu24.04 nvidia-smi` — 4090 visible.
2. Cloned the fork to `/opt/ninfer/ninfer-4090`, pinned the commit in `/opt/ninfer/PINNED_COMMIT`.
3. `docker build --tag ninfer-4090:sm89 .` and a second tag `ninfer-4090:<commit8>`.
4. Downloaded the model. **The fork's `scripts/download-qwen38.sh` pulls HF `main`, which moved to
   container version 3 on 2026-09-15; the fork (v0.6.1-rtx3090, branch `rtx4090-port`, commit
   `1bd56c9a`) reads version 2 only and dies with `artifact magic is not NInfer v2`.** The fork's
   own model card pins the 2026-08-14 revision `35269130` (18,210,531,328 bytes, SHA-256
   `eec39564…`), whose required upstream commit `52320554` is in the fork's history. That is
   what `/opt/ninfer/run-download-v2.sh` fetches and verifies. The 2026-09-06 revision `dc370fb6`
   is also v2 but needs upstream `385b30ce`, which the fork lacks.
5. Wrote `/opt/ninfer/compose.yaml` (Profile A: `rk4v4-e8` KV, 262K, text only) and `docker compose up -d`.
   `ninfer-serve` rejects `--flag=value` (`unknown argument: --host=0.0.0.0`), so every value is
   its own list item in the compose command.
   The first `docker build` failed on a transient DNS error inside the build container
   (`Could not resolve 'archive.ubuntu.com'`); a plain retry succeeded.
6. ufw: `allow from 192.168.10.0/24 to any port 8080 proto tcp`.

## Profile in use

Profile A from the guide. To change profile edit `/opt/ninfer/compose.yaml`
and `docker compose up -d` again:

| Profile | `--kv-dtype` | `--max-context` / `--kv-capacity` | extra | slack |
|---|---|---:|---|---:|
| **A (live)** | `rk4v4-e8` | 262144 | – | ~1.4 GiB |
| B vision | `rk4v4-e8` | 262144 | `--vision` | ~780 MiB |
| C precision | `int8` | 172032 | – | ~136 MiB |
| D slack | `rk2v4-e8` | 262144 | – | ~2.4 GiB |

The server checks that the configured context fits before it listens and
exits with the shortfall if not; nothing else may hold VRAM at start.

## Chat client: Open WebUI on :3000

`open-webui` is a second service in the same `compose.yaml`
(image `ghcr.io/open-webui/open-webui:main`, data in `/opt/ninfer/open-webui/`,
`restart: unless-stopped`). It reaches NInfer over the compose network as
`http://ninfer:8080/v1`, Ollama probing is disabled, and ufw allows :3000 from
`192.168.10.0/24`.

- Open **http://192.168.10.7:3000**. The first visit creates the admin
  account (local to this box; nothing leaves the LAN).
- The model list is fetched from NInfer, so `qwen3.8-27b` appears
  automatically. Reasoning shows as a collapsible "thinking" block.
- NInfer serves no page at `/`; port 8080 is API only.

```bash
cd /opt/ninfer
docker compose logs -f open-webui
docker compose pull open-webui && docker compose up -d open-webui   # update
```

## Operating

```bash
cd /opt/ninfer
docker compose up -d            # start (restart: unless-stopped survives reboots)
docker compose logs -f ninfer   # logs
docker compose down             # stop and free the GPU
curl -s http://127.0.0.1:8080/v1/models | jq
curl -s http://127.0.0.1:8080/metrics | grep -E 'llamacpp:|ninfer:'
curl -s http://127.0.0.1:8080/slots | jq
```

Only one engine can own the GPU: stop this container before running any
launcher from `../scripts/` (they need ~23 GB) and vice versa.

### Client notes (differ from the llama.cpp engines)

- Base URL `http://192.168.10.7:8080/v1`; model id from `/v1/models`.
- Thinking on/off is the **top-level** `"enable_thinking": false`.
  `chat_template_kwargs` is rejected, so `clients/chat.py` (which sends it)
  does not work unmodified against this engine.
- `reasoning_effort` accepts only `low`, `medium`, `xhigh`, `none`. Map `high` to `xhigh`.
- `max_tokens` works. Default output cap is 8192 (`--default-max-tokens`).
- Streaming errors arrive as an in-band SSE error event after HTTP 200.
- Keep the start of prompts byte-stable to benefit from prefix caching.
- Anthropic Messages API is also served (`/v1/messages`).

### Updating / rollback

Upstream `Neroued/ninfer` added container-v3 reading on 2026-09-15; when the fork merges it,
`qwen3_8_27b.v3.ninfer` can replace the v2 file (and `download-qwen38.sh` will work again).

```bash
cd /opt/ninfer/ninfer-4090
git fetch && git log --oneline HEAD..origin/main
git pull && git rev-parse HEAD > ../PINNED_COMMIT
docker build --tag ninfer-4090:$(cut -c1-8 ../PINNED_COMMIT) .
# point compose.yaml `image:` at the new tag, docker compose up -d; revert the tag to roll back
```

Saved slot files may not load across engine versions; treat `slots/` as cache.

## Verification record (2026-09-16, first start)

Engine ready 6.0 s after container start (weights 16.7 GiB at 5.24 GiB/s,
host state 1.15 GiB + host KV 8 GiB pinned, CUDA graphs 744 ms). VRAM in use
22,592 MiB of 24,564. `/v1/models` reports `context_window: 262144`,
`vision: false`. Reachable from the LAN address on :8080.

| Request | Result |
|---|---|
| Code prompt, thinking off, 148 output tokens | 153 tok/s decode, prefill 191 tok/s on 23 tokens, draft acceptance 105/126 (83%) |
| Arithmetic, thinking on, `reasoning_effort: low` | correct answer, 52 reasoning tokens in `reasoning_content`, 163 tok/s, acceptance 42/45 |

Both within the guide's expected ~125–150 tok/s range for shallow code.
`/metrics` exposes `llamacpp:*` and `ninfer:*` counters; `/slots` shows the
retained session checkpoints. `logs/requests.jsonl` is written as root by the
container.

Not exercised yet: long-context prefill at depth, `--max-concurrency 2`,
Profile B vision, slot save/restore, and the benchmark comparison against the
llama.cpp launchers in `../scripts/`.
