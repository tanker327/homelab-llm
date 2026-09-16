# RTX 4090 Server (24GB)

Everything for the **second inference box**, the one with an RTX 4090. The
rest of this repo (root `scripts/`, `systemd/`, `docs/`, CLAUDE.md) targets
the RTX PRO 6000 (96GB) production server and **none of its launchers fit in
24GB** — see [What does not fit](#what-does-not-fit-in-24gb). This folder is
self-contained: its own setup script, launchers, systemd unit and tuning notes.

**What is running on the box (since 2026-09-16): [NInfer serving Qwen3.8-27B](./ninfer/README.md)**
in Docker on **port 8080** — full 262K context, MTP n=3, ~153–163 tok/s, 22.6 GB
VRAM, `restart: unless-stopped`. It owns the GPU; the llama.cpp launchers below
are the Qwen3.6 alternatives on port 5000 and need the NInfer container stopped
first (`cd /opt/ninfer && docker compose down`).

The shared git-ignored directories (`llama.cpp/`, `models/`, `venv/`,
`vllm-venv/`, `sglang-venv/`, `toolchain-fix/`) still live at the **repo
root** on this machine too; every script here resolves paths relative to the
repo root, so the folder can be used from any checkout location.

## Host

Recorded 2026-09-16 on the fresh box, before any inference setup.

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB, compute capability 8.9 (SM89, Ada) |
| Driver | 595.91.07 (CUDA 13.2 runtime). No CUDA toolkit installed yet |
| CPU / RAM | i9-13900K (32 threads) / 61 GB |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-31, glibc 2.43 |
| Disk | ~172 GB free on `/` (model zoo below needs ~30 GB for the default, ~75 GB for all GGUFs + GPTQ) |
| LAN | `192.168.10.7` (the 96GB box is `192.168.10.106`); Tailscale `100.91.237.44` |
| Already running | `docker`, `node_exporter` (:9100), `dcgm-exporter` container (:9400), `tailscaled`. Both exporters are ufw-allowed from `192.168.10.200` only |
| Not yet present | `cmake`, `nvcc`, `ninja`, `hf`, `llama.cpp/`, `models/`, `venv/`, the systemd unit, a ufw rule for :5000 |
| Added since | ufw rule for :8080 from `192.168.10.0/24`; `/opt/ninfer/` (NInfer image, model, compose) — see [ninfer/](./ninfer/README.md) |

`uv` 0.12 and Python 3.14 are installed; `setup.sh` creates a Python 3.12 venv
via `uv` regardless.

## What fits in 24GB

Numbers are from the original 4090 build of this project (git `main`,
2026-08), same GPU model. They have **not** been re-measured on this host yet.

| Launcher | Engine | Model | VRAM | Context | Single-stream | Notes |
|---|---|---|---|---|---|---|
| [`ninfer/`](./ninfer/README.md) **(running)** | NInfer (Docker, :8080) | Qwen3.8-27B groupwise-int (17 GB) | 22.6 GB | 262K per request | 153–163 tok/s measured here | MTP n=3, OpenAI + Anthropic APIs, thinking with `reasoning_effort`. See its README for the artifact-version pin |
| `start-llama-35b-moe.sh` **(default)** | llama.cpp | Qwen3.6-35B-A3B MXFP4_MOE (20.2 GB file) | ~23.2 GB | 96K total (`--ctx-size 98304`), split across `--parallel` slots | ~163 tok/s | MoE, 3B active. Fastest option on this card. ~300 tok/s aggregate ceiling |
| `start-llama-9b.sh` | llama.cpp | Qwen3.5-9B Q4_K_M | ~6 GB | 128K | ~125 tok/s | Dense. Leaves ~18 GB free — the pick if the GPU must be shared |
| `start-llama-27b.sh` | llama.cpp | Qwen3.6-27B Q4_K_M | ~23 GB | 96K | slower than the MoE | Dense, all params active per token. Quality pick, speed cost |
| `start-vllm-35b-mtp.sh` | vLLM | Qwen3.6-35B-A3B GPTQ-Int4 (22.7 GB) | tight; `--cpu-offload-gb 4` | 32K | — | MTP n=5 speculative decoding. Needs `vllm-venv/`, not built by `setup.sh` |
| `start-sglang-35b-mtp.sh` | SGLang | same GPTQ-Int4 | `--mem-fraction-static 0.92` | 32K | — | NEXTN n=5. Needs `sglang-venv/`. 4090-era toolchain pins, see script header |

The vLLM and SGLang launchers were verified on the *previous* 4090 host
(CUDA 12.8 era). On this Ubuntu 26.04 / driver 595 box they are untested; treat
them as starting points and expect to rebuild the venvs.

### Default config and the `--parallel` trade-off

`start-llama-35b-moe.sh` runs `--ctx-size 98304 --parallel 2`. llama.cpp's
`--ctx-size` is the **total** KV cache, divided evenly across slots:

| `--parallel` | Per-request context | Concurrent streams | Use case |
|---|---|---|---|
| 1 | 96K | 1 (others queue) | Solo, long documents |
| **2 (default)** | **48K** | **2** | You + one background agent |
| 4 | 24K | 4 | Small team / multi-agent |
| 8 | 12K | 8 | Max throughput, ~300 tok/s aggregate |

Measured throughput curves for 4 / 8 / 10 are in
[docs/CONCURRENCY.md](./docs/CONCURRENCY.md). Change the flag in the launcher
and `sudo systemctl restart llama-server`.

Going past 96K total needs KV quantization (`--cache-type-k q8_0
--cache-type-v q8_0`, roughly halves KV memory); the model alone is ~21.7 GB
resident. Untested here.

## What does NOT fit in 24GB

Every launcher in the root `scripts/` directory assumes 96GB:

- `start-production.sh` / `start-vllm-flashnext-nvfp4-mtp.sh` — ~90 GB, and the script refuses to start unless ~94 GB VRAM is free.
- `start-llama-flashnext.sh` — ~76 GB.
- `start-vllm-38-27b-*.sh`, `start-vllm-27b-fp8.sh` — 86–90 GB.
- `start-agent-stack.sh` — ~77 GB (two models resident).
- Root `start-llama-27b.sh` (Q8_0 + MTP, 524K ctx, ~64 GB) and root `start-llama-35b-moe.sh` (MTP GGUF, 524K ctx) — same model families as here but sized for 96GB.
- `start-llama-122b-moe.sh`, `start-llama-coder-next.sh` — experimental, far too large.

Do not point this box's systemd unit at anything outside `rtx4090/`.

## Setup

```bash
./rtx4090/scripts/setup.sh
```

Idempotent. It installs `cmake`/`build-essential` and (if no `nvcc`) the
`cuda-toolkit-13-1` package, builds `llama.cpp/build/bin/llama-server` for
the auto-detected compute capability (89 here) with the glibc-2.42+ `rsqrt`
header shim the root setup also uses, downloads the default GGUF into
`models/`, creates `venv/` with `openai` + `huggingface-hub`, and adds the
ufw rule for `192.168.10.0/24 → :5000`.

Extra GGUFs for the alternative launchers (download into `models/`):

```bash
./venv/bin/hf download unsloth/Qwen3.5-9B-GGUF  Qwen3.5-9B-Q4_K_M.gguf  --local-dir models
./venv/bin/hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir models
```

### Systemd

```bash
sudo cp rtx4090/systemd/llama-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server
sudo systemctl status llama-server
journalctl -u llama-server -f
```

The unit runs `rtx4090/scripts/start-llama-35b-moe.sh` with `Restart=on-failure`,
5 s delay. Cold start is ~5 s (warm page cache). To run another launcher,
`sudo systemctl stop llama-server` first — every engine binds port 5000, and
the MoE default already uses ~23 GB of the 24.

## Usage

The shared clients and benchmarks at the repo root work against this box
unchanged, from the repo root:

```bash
./venv/bin/python clients/chat.py
./venv/bin/python benchmarks/bench_concurrency.py      # the 4090 --parallel benchmark
./venv/bin/python benchmarks/bench_serving.py --label x --port 5000 --workload agent --levels 1,2,4 --runs 3
./venv/bin/python tools/probe_max_input.py             # max usable prompt length
```

API is OpenAI-compatible on `http://192.168.10.7:5000/v1`. Send
`"model": "local"` (all launchers here pass `--alias local`; llama.cpp ignores
the field anyway, but the vLLM/SGLang launchers validate it). Full endpoint
reference: [../docs/API.md](../docs/API.md) — it describes the 96GB box, but
the llama.cpp sections (`reasoning_content`, streaming format, `/tokenize`,
web UI at `/`) apply here too.

Things that differ from the 96GB box:

- **Model family is Qwen3.6, not Qwen3.8.** Recommended thinking-mode sampling is temp 0.6 / top_p 0.95. `clients/chat.py` sends the Qwen3.8 defaults (temp 1.0) and a `reasoning_effort` chat-template kwarg; whether the Qwen3.6 template honours that kwarg is unverified on this box.
- **Never use temperature 0** — same unbounded-thinking loop as on the big box.
- **`max_tokens` on `/v1/chat/completions` is broken on llama.cpp** (truncation mid-reasoning gives empty content or a 500). Omit it or use `stop`.
- No `reasoning_effort` pin server-side; no vision; no tool-call parser flags. The launchers are the plain, measured configs from `main`.

## Layout

```
rtx4090/
├── README.md                     this file
├── scripts/
│   ├── setup.sh                  build llama.cpp (SM89) + download default GGUF + venv + ufw
│   ├── start-llama-35b-moe.sh    DEFAULT: Qwen3.6-35B-A3B MXFP4_MOE, 96K ctx, --parallel 2
│   ├── start-llama-9b.sh         Qwen3.5-9B Q4_K_M, 128K ctx, ~6GB
│   ├── start-llama-27b.sh        Qwen3.6-27B Q4_K_M, 96K ctx, ~23GB
│   ├── start-vllm-35b-mtp.sh     vLLM + GPTQ-Int4 + MTP n=5 (untested on this host)
│   └── start-sglang-35b-mtp.sh   SGLang + GPTQ-Int4 + NEXTN n=5 (untested on this host)
├── systemd/
│   └── llama-server.service      ExecStart → rtx4090/scripts/start-llama-35b-moe.sh
├── docs/
│   └── CONCURRENCY.md            --parallel 4/8/10 throughput measurements on the 4090
└── ninfer/
    ├── README.md                 NInfer + Qwen3.8-27B on :8080 (what runs on the box), setup record, gotchas
    ├── SETUP-GUIDE.md            step-by-step install guide, corrected from the real install
    └── compose.yaml              copy of the live /opt/ninfer/compose.yaml (Profile A)
```
