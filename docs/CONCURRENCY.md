# Concurrency & `--parallel` Tuning

How many simultaneous requests llama.cpp can serve, and how `--parallel` affects throughput vs. per-request speed on this machine (RTX 4090, 24 GB).

Current setting: `scripts/start-llama-35b-moe.sh` passes `--parallel 8`.

## TL;DR

- GPU throughput ceiling for `Qwen3.6-35B-A3B-MXFP4_MOE` is **~290–305 tok/s aggregate**, regardless of `--parallel`.
- Single-request speed is **~163 tok/s** for any value of `--parallel` (only one slot active).
- More slots = more concurrent users, **but each user's stream gets slower** because they share the GPU.
- More slots also **shrink per-slot KV cache**: `--ctx-size / --parallel`. With `--ctx-size 98304`:
  - `--parallel 4` → 24 K per slot
  - `--parallel 8` → 12 K per slot (current)
  - `--parallel 10` → 9.8 K per slot

## How to bench

Server must be running. The script fires N requests in parallel and reports aggregate and per-request tok/s.

```bash
./venv/bin/python benchmarks/bench_concurrency.py
```

Concurrency levels tested: 1, 2, 4, 6, 8, 10. Prompt is `/no_think` + a fixed-length task ("list 1..200") so every request produces ~3.4 K tokens with low variance.

To bench a different `--parallel` value:

1. Edit `scripts/start-llama-35b-moe.sh` (`--parallel N`).
2. `sudo systemctl restart llama-server`.
3. Re-run `benchmarks/bench_concurrency.py`.

## Results (Qwen3.6-35B-A3B-MXFP4_MOE, RTX 4090)

### Aggregate throughput (tok/s) — total tokens / wall time

| N concurrent | parallel=4 | parallel=8 | parallel=10 |
|---|---|---|---|
| 1  | 157 | 163 | 163 |
| 2  | 200 | **226** | 207 |
| 4  | 272 | **304** | 293 |
| 6  | 214\* | 263 | **286** |
| 8  | 292 | 284 | 288 |
| 10 | 277 | **304** | **305** |

\* noisy run — completion-length variance.

### Per-request speed (tok/s each request actually sees)

| N | parallel=4 | parallel=8 | parallel=10 |
|---|---|---|---|
| 1  | 157 | 164 | 164 |
| 2  | 119 | 121 | 127 |
| 4  | 82  | 89  | 90  |
| 6  | 75  | 55  | 60  |
| 8  | 81  | 47  | 45  |
| 10 | 77  | 46  | 38  |

With `--parallel 4`, requests beyond N=4 queue (so the next batch runs at the fast 4-slot rate). With `--parallel 8` or `10`, all N stream simultaneously at a slower per-stream rate.

## Choosing `--parallel`

| Use case | Pick | Why |
|---|---|---|
| Mostly solo / occasional 2–4 concurrent | `--parallel 4` | Same single-user speed, 24 K per-slot context for long prompts, queues the rare 5+ burst |
| Multi-user up to ~8 concurrent (default here) | `--parallel 8` | Best aggregate at N=2/4/10, 12 K per-slot context still usable for chat |
| 10+ concurrent users acceptable at ~38 tok/s each | `--parallel 10` | Marginal aggregate gain; per-slot context drops to 9.8 K |

If you need long-context requests, lean toward fewer slots (or raise `--ctx-size` proportionally — costs VRAM).
