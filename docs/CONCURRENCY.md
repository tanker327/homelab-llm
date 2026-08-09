# Concurrency & `--parallel` Tuning

How many simultaneous requests llama.cpp can serve and where the GPU's sweet
spot sits. Measured with `benchmarks/bench_matrix.py` (adds GPU power/temp/util
sampling); the older `benchmarks/bench_concurrency.py` reports the same
throughput columns without telemetry.

## RTX PRO 6000 Blackwell (96GB) — current hardware

Both models with MTP (`--spec-type draft-mtp --spec-draft-n-max 3`), server
started with `--parallel 8 --ctx-size 524288`, N parallel requests fired at
once. Aggregate = total generated tokens / wall time; per-req = what each
stream sees. Power cap is 600W.

### Qwen3.6-27B Q8_0 (dense) — quality pick

| N | agg tok/s | per-req tok/s | power avg/max | temp max | util |
|---|---|---|---|---|---|
| 1 | 145 | 145 | 569/580W | 74C | 97% |
| 2 | 235 | 130 | 594/602W | 80C | 96% |
| **3** | **300** | **114** | 514/601W | 81C | 94% |
| 4 | 286 | 99 | 527/601W | 83C | 93% |
| 6 | 390 | 84 | 530/601W | 83C | 91% |
| 8 | 289 | 72 | 559/602W | 86C | 93% |

Dense saturates the chip immediately: near the 600W cap from N=1 and the
hottest workload measured (86C at N=8). Aggregate gains beyond N=3 are noisy
to negative while per-request speed keeps dropping. **Sweet spot: N=2–3.**
The launcher ships `--parallel 2` (keeps the full 262K native context per
slot); use 3 if you routinely have 3 concurrent users and can live with
174K ctx/slot.

### Qwen3.6-35B-A3B MXFP4 (MoE, 3B active) — throughput pick

| N | agg tok/s | per-req tok/s | power avg/max | temp max | util |
|---|---|---|---|---|---|
| 1 | 430 | 432 | 414/418W | 76C | 89% |
| 2 | 627 | 342 | 439/453W | 77C | 87% |
| 4 | 715 | 215 | 376/411W | 74C | 86% |
| **6** | **790** | **176** | 378/420W | 72C | 85% |
| 8 | 818 | 136 | 375/406W | 72C | 81% |

The MoE never comes close to the power cap (≤453W) and runs cooler under
8 streams than the dense model does under 1. Throughput scales all the way
to N=8, but 6→8 buys only +3.5% aggregate while per-request drops 23%.
**Sweet spot: N=6.** The launcher ships `--parallel 2` for max context;
switch to 6 for multi-user serving.

### Takeaways

- The MoE is ~3x faster per stream AND ~4x more power-efficient per token
  (1.04 tok/J vs 0.25 tok/J at N=1) — active-parameter count, not total
  size, dictates cost on this GPU.
- Dense 27B is thermally the harder workload: expect sustained ~600W.
- Both models hold their MTP speedup under concurrency.

## RTX 4090 (24GB) — historical results

Kept for reference from the previous server. Qwen3.6-35B-A3B MXFP4, no MTP,
`--ctx-size 98304`.

### Aggregate throughput (tok/s)

| N concurrent | parallel=4 | parallel=8 | parallel=10 |
|---|---|---|---|
| 1  | 157 | 163 | 163 |
| 2  | 200 | **226** | 207 |
| 4  | 272 | **304** | 293 |
| 6  | 214 | 263 | **286** |
| 8  | 292 | 284 | 288 |
| 10 | 277 | **304** | **305** |

### Per-request speed (tok/s)

| N | parallel=4 | parallel=8 | parallel=10 |
|---|---|---|---|
| 1  | 157 | 164 | 164 |
| 2  | 119 | 121 | 127 |
| 4  | 82  | 89  | 90  |
| 6  | 75  | 55  | 60  |
| 8  | 81  | 47  | 45  |
| 10 | 77  | 46  | 38  |

GPU ceiling was ~290–305 tok/s aggregate regardless of `--parallel`; the
RTX PRO 6000 MoE numbers are ~2.7x that.

## How to bench

Server must be running with enough slots for the levels you test:

```bash
./venv/bin/python benchmarks/bench_matrix.py --label my-config --levels 1,2,4,8
./venv/bin/python benchmarks/bench_concurrency.py   # legacy, no GPU telemetry
```

To test a different `--parallel`, edit the launcher, restart the server, re-run.
