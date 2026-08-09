# Benchmarks — RTX PRO 6000 Blackwell (96GB)

All measurements taken 2026-08-08/09 on this server. Kept as the reference
record for the RTX 4090 → RTX PRO 6000 migration and the MTP/quant tuning
that followed. Concurrency tuning advice lives in `CONCURRENCY.md`; this file
is the raw record.

## Environment

| Component | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell, 96GB GDDR7 (97,887 MiB), SM120, 600W cap |
| Driver / CUDA | 595.71.05 / CUDA 13.1 (`cuda-toolkit-13-1`, Ubuntu packages) |
| OS | Ubuntu 26.04 LTS (glibc 2.42 — needs the `toolchain-fix` header shim, see setup.sh) |
| llama.cpp | master @ 2026-08-08, `-DCMAKE_CUDA_ARCHITECTURES=120`, flash attention on |
| Method | `timings.predicted_per_second` from `/v1/chat/completions`; bench prompt: `/no_think List the integers from 1 to 200...` (~1K tokens out) |

## Single-stream generation speed

Server config: `--ctx-size 524288 --parallel 2` unless noted.

| Model / config | tok/s | Notes |
|---|---|---|
| Qwen3.6-35B-A3B MXFP4, no MTP (RTX 4090, historical) | ~163 | old server baseline |
| Qwen3.6-35B-A3B MXFP4, no MTP | 260 | same model, new GPU: 1.6x from hardware |
| Qwen3.6-35B-A3B MXFP4 **MTP n-max 3** | **418–430** | +60% from MTP; production speed pick |
| Qwen3.6-27B Q8_0 MTP n-max 3 | **139–145** | production quality pick |
| Qwen3.6-27B UD-Q6_K_XL MTP n-max 3 | 148 | only ~5% over Q8 — not worth the precision loss |

## MTP draft-window A/B (`--spec-draft-n-max`)

MTP self-speculation merged into llama.cpp May 2026; requires MTP GGUFs.
`n-max 3` won for both models on this GPU (community guidance suggested 2 for
dense models — measurement said otherwise):

| Model | n-max 2 | n-max 3 |
|---|---|---|
| 35B-A3B MXFP4 | 380 tok/s | **418 tok/s** |
| 27B Q8_0 | 115 tok/s | **139 tok/s** |

## Quantization A/B — Qwen3.6-27B

| Quant | File size | tok/s (MTP n-max 3) | Verdict |
|---|---|---|---|
| **Q8_0** | 29.0GB | ~140 | **chosen** — near-lossless, speed penalty tiny |
| UD-Q6_K_XL | 26.0GB | ~148 | +5% speed didn't justify quality risk |

Theory predicted 15–20% speedup from the smaller weights; MTP overhead
dominates decode, collapsing the real gap to ~5%.

## Concurrency matrix with GPU telemetry

`benchmarks/bench_matrix.py`, servers at `--parallel 8 --ctx-size 524288`,
N simultaneous requests, nvidia-smi sampled at 1s intervals.

### Qwen3.6-27B Q8_0 (dense)

| N | agg tok/s | per-req tok/s | avg lat (s) | power avg/max | temp max | util avg |
|---|---|---|---|---|---|---|
| 1 | 145 | 145 | 36.3 | 569/580W | 74C | 97% |
| 2 | 235 | 130 | 30.2 | 594/602W | 80C | 96% |
| 3 | 300 | 114 | 42.9 | 514/601W | 81C | 94% |
| 4 | 286 | 99 | 44.6 | 527/601W | 83C | 93% |
| 6 | 390* | 84 | 45.6 | 530/601W | 83C | 91% |
| 8 | 289 | 72 | 68.5 | 559/602W | 86C | 93% |

\* completion-length variance; treat as noise.

### Qwen3.6-35B-A3B MXFP4 (MoE, 3B active)

| N | agg tok/s | per-req tok/s | avg lat (s) | power avg/max | temp max | util avg |
|---|---|---|---|---|---|---|
| 1 | 430 | 432 | 12.3 | 414/418W | 76C | 89% |
| 2 | 627 | 342 | 11.5 | 439/453W | 77C | 87% |
| 3 | 636 | 261 | 20.9 | 394/453W | 75C | 88% |
| 4 | 715 | 215 | 16.6 | 376/411W | 74C | 86% |
| 6 | 790 | 176 | 21.7 | 378/420W | 72C | 85% |
| 8 | 818 | 136 | 29.6 | 375/406W | 72C | 81% |

### Key findings

- **Sweet spots:** dense 27B → N=2–3 (power-capped from the first request);
  MoE 35B → N=6 (790 agg tok/s; 6→8 adds +3.5% aggregate for −23% per-stream).
- **Power efficiency:** MoE ≈ 1.04 tok/J vs dense ≈ 0.25 tok/J at N=1 — the
  MoE is ~3x faster per stream and ~4x cheaper per token. Active-parameter
  count, not total size, sets the power bill.
- **Thermals:** dense 27B is the hard workload (sustained ~600W, 86C peak);
  the MoE never exceeded 453W and ran cooler with 8 streams than the dense
  model with 1.
- Both models retain their MTP speedup under concurrency.

## VRAM footprints (`--ctx-size 524288 --parallel 2`)

| Configuration | VRAM used |
|---|---|
| 35B-A3B MXFP4 alone | ~32GB |
| 27B Q8_0 alone | ~63GB |
| Agent stack (both resident: 35B x3 slots @393K total + 27B x2 slots @262K total) | ~77GB |

## Reproducing

```bash
# server must be running with enough slots for the levels tested
./venv/bin/python benchmarks/bench_matrix.py --label my-config --levels 1,2,4,8 --json out.jsonl
```
