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

---

# Addendum: Qwen3.6-27B engine bake-off (2026-08-11/12)

Full three-engine comparison for the dense 27B — llama.cpp (incumbent) vs
vLLM 0.27.1 vs SGLang 0.5.17 — optimizing multi-agent aggregate throughput.
Harness: `benchmarks/bench_serving.py` (streaming TTFT, token-exact prompts,
unique prefixes to defeat prefix caching, GPU telemetry, tok/J). Raw data:
`benchmarks/results/*.jsonl`; environment: `benchmarks/results/env.md`.

## Winner: vLLM + Qwen3.6-27B-FP8 + MTP n=3 + FP8 KV cache

`scripts/start-vllm-27b-fp8.sh` — 262K max-model-len, ~86GB VRAM.

| Metric | Incumbent (llama.cpp Q8_0+MTP) | Winner | Δ |
|---|---|---|---|
| Agent agg tok/s, best N (12K in/2K out) | 244 @ N=8 (p95 TTFT 35s) | **530 @ N=8 (p95 12s)** | **+117%** |
| Single-stream decode, 2K ctx | 130–141 | 118 | −12% |
| Decode @ 49K ctx | 120 | 120 | = |
| Judge TTFT (49K in) | 16–19s | 7.2s | −60% |
| Prefill tok/s (2K) | ~3,300 | ~6,000–7,300 | ~2x |
| tok/J at best N | 0.49 | 0.97–0.99 | ~2x |
| Power at best N | 495–570W (capped at N=1) | 505–548W (never capped) | — |
| Max NIAH-correct context | untested (262K config) | 254,976 tokens | — |
| Quality smoke (10 tasks, vs Q8_0 ref 10/10) | 10/10 | 10/10 | = |

## Key findings

1. **Speculative decoding is the entire ballgame.** Base dense decode is
   memory-bound at ~50 tok/s on ALL THREE engines (llama.cpp no-MTP 51,
   vLLM no-spec 50, SGLang no-spec 51). MTP/NEXTN with the checkpoint's own
   MTP head gives 2.4–2.6x with ~0.85–0.92 acceptance, and the gain does
   NOT invert under batching (still ~2x at N=8).
2. **FP8 KV cache is a speed feature here, not just capacity.** vLLM
   `--kv-cache-dtype fp8_e4m3`: agent N=8 434→530 agg (+22%), 49K-ctx decode
   67→120 tok/s (+79%), zero quality regression (10/10). Adopted.
3. **The incumbent's published ~300 agg peak was a short-prompt artifact.**
   With realistic 12K-token prompts, llama.cpp's serialized ~2.4K tok/s
   prefill caps it at 244 agg with 23–35s p95 TTFT. The FP8 engines prefill
   2–3x faster and batch prefill properly.
4. **Power cap explains the plateaus.** llama.cpp Q8_0 draws 570–600W from
   the first stream (0.2–0.25 tok/J); FP8 tensor-core kernels never reach
   the 600W cap and land at 0.8–1.0 tok/J at high N — >3x the incumbent's
   energy efficiency at 2x+ the throughput.
5. **SGLang: best single-stream, broken burst scheduler.** Fastest TTFT
   (251ms), fastest prefill (8.1–8.6K tok/s), and 120 tok/s decode at 49K —
   but at 8-wide bursts aggregate DROPS (425@N=6 → 350@N=8, p95 TTFT 24–44s)
   with or without spec decode. vLLM keeps scaling to N=8. With multi-agent
   throughput as the priority, vLLM wins.
6. **SM120 + CUDA 13.1 is a solved problem** (as of vLLM 0.27.1 / SGLang
   0.5.17): plain pip wheels work; the 4090-era CUDA 12.8 + g++-14 pins are
   obsolete. Only requirements: `ninja` on PATH and the `toolchain-fix`
   header shim exported for flashinfer's runtime JIT (glibc 2.43).
7. **Never run this model at temperature 0** — greedy decoding produces
   unbounded thinking loops (observed 54K+ reasoning tokens on a trivial
   task). Use Qwen's recommended temp 0.6 / top_p 0.95.

## Concurrency matrices (agent workload: 12,288 in / 2,048 out, 3-run medians)

| Config | N=1 | N=2 | N=4 | N=6 | N=8 | notes |
|---|---|---|---|---|---|---|
| llama.cpp Q8_0+MTP3 | 90 | 102* | 140 | 180* | **244** | p95 TTFT 35s @ N=8 |
| vLLM FP8+MTP3 | 98 | 188 | 310 | 407 | **434** | p95 11.8s; accept ~0.9 |
| vLLM FP8+MTP3+**KV-fp8** | — | — | 348 | — | **530** | p95 12.0s; 0.97 tok/J |
| vLLM FP8 no-spec | 47 | 85 | 153 | 214 | 259 | spec-off baseline |
| SGLang FP8+NEXTN3 | 121 | 218 | 340 | **425** | 350 | N=8 regresses; p95 24s |
| SGLang FP8 no-spec | 48 | 86 | 156 | 214 | 158 | N=8 p95 44s |

\* high run-to-run variance (llama.cpp slot scheduling).

## Single-stream (phase2.jsonl)

| Config | W1 decode (2K in/1K out) | W1 TTFT | prefill tok/s | judge decode (49K in) | judge TTFT |
|---|---|---|---|---|---|
| llama.cpp Q8_0+MTP3 | 120–141 | 750ms | 3,300 | 120 | 16–19s |
| llama.cpp Q8_0 no-MTP | 51 | 690ms | 3,700 | — | — |
| vLLM FP8+MTP3 | 118 | 342ms | 6,000 | 67 (120 with KV-fp8) | 7.0s |
| SGLang FP8+NEXTN3 | 128 | 251ms | 8,100–8,600 | 120 | 7.3s |

## Soak

2h sustained agent N=4 on the winner config: see `benchmarks/results/soak.jsonl`
(errors / VRAM creep / throughput drift summarized in RESUME-STATE.md).

## Production notes

- The winner needs ~86GB → **cannot coexist with the 35B MoE agent stack**.
  Switching the systemd default to it is a separate decision: either the 27B
  replaces the stack, or the stack keeps llama.cpp for the 27B half.
- Restore/keep production: `sudo systemctl start llama-server` runs the
  unchanged agent stack. The bake-off changed no production config.
- Reproduce any row:
  `venv/bin/python benchmarks/bench_serving.py --label X --port 5000 --workload agent --levels 1,2,4,6,8 --runs 3 --json out.jsonl`

# Addendum: Qwen3.8-27B-FP8 upgrade (2026-08-15)

Qwen3.8-27B (open-weights 2026-08-13, Apache 2.0) replaced Qwen3.6-27B-FP8 as
the production default two days after the bake-off above. Same engine (vLLM
0.27.1 — the `Qwen3_5ForConditionalGeneration` arch was already registered),
same launcher pattern; raw data in `benchmarks/results/qwen38_bench.jsonl`
and `qwen38_quality.jsonl`.

## Why upgrade

- Quality: SWE-bench Pro 61.7 (3.6: 53.5), Terminal Bench 73.0 (63.4),
  OSWorld-Verified 84.3 (63.9), MathVision 94.6 (85.1) — vendor-reported.
- Architecture: hybrid — 48 of 64 layers are Gated DeltaNet linear attention,
  full attention every 4th layer. KV cache only exists on the 16 full-attention
  layers → **1.55M KV tokens** fit at 0.90 util (dense 3.6 fit far fewer),
  enabling 16 concurrent 262K-capable sequences.
- Local gates: 10/10 quality smoke (thinking sampling now temp 1.0 / top_p
  0.95 per Qwen3.8 recommendation), NIAH-correct to **261,696 tokens**
  (3.6: ~255K) with TTFT ~88s at that length.

## Tuning matrix (agent workload: 12,288 in / 2,048 out, 3-run medians)

| config    | N=1 dec p50 | N=8 agg | N=16 agg | tok/J @max | notes |
|-----------|------------:|--------:|---------:|-----------:|-------|
| n=3, s=8  |        96.4 |     436 |        — |       0.81 | baseline (3.6 flags) |
| n=2, s=8  |        90.9 |     424 |        — |       0.80 | strictly worse |
| n=4, s=8  |       100.3 |     430 |        — |       0.75 | **erratic stalls** (runs collapsing to 13 and 3.6 tok/s) — rejected |
| **n=3, s=16** |    91.4 |     449 |  **592** |   **0.99** | **production** |

vs Qwen3.6 production record: ~530 agg @ N=8, ~118 tok/s single-stream.
Qwen3.8 wins aggregate (+12% at its higher ceiling) and efficiency but loses
single-stream: its MTP head is a *single* layer applied recursively for n>1
(vLLM warns about this), so per-position acceptance decays 0.83/0.71/0.61 and
overall acceptance is ~0.6 vs 3.6's 0.85–0.92. n=3 is the sweet spot; n=4's
stalls had no logged cause and are disqualifying regardless.

## Soak

40 iterations of agent N=4 (2026-08-15, `benchmarks/results/qwen38_soak.jsonl`;
16 iters against the pre-promotion manual server, 24 against the live systemd
service): **0 errors**, agg 272–288 tok/s throughout (one transient dip to 243
with a single 8.5s TTFT p95 blip, self-recovered), VRAM flat at 88.0GB (no
creep), thermals stabilized at 85–86C / ~510W.

## Production config

`scripts/start-vllm-38-27b-fp8.sh`: 262K max-model-len, `--max-num-seqs 16`,
FP8 KV (`fp8_e4m3`), MTP n=3, `--reasoning-parser qwen3`, `--tool-call-parser
qwen3_xml` (alias of `qwen3_coder` in vLLM 0.27.1). ~88GB VRAM. Rollback =
point systemd `ExecStart` back at `start-vllm-27b-fp8.sh`.

# Addendum: reasoning-effort sweep + serving-feature verification (2026-08-16)

All against the live production server (vLLM 0.27.1, Qwen3.8-27B-FP8, MTP n=3,
FP8 KV). Harness: `benchmarks/bench_effort.py`; raw data:
`benchmarks/results/qwen38_effort.jsonl`.

## reasoning_effort × thinking mode (6 assert-gated coding tasks × 3 runs)

Tasks run 6-way concurrent per config; thinking configs sampled 1.0/0.95/top_k
20, non-thinking 0.7/0.8/presence 1.5. One task (`logparse`) had a broken
assert in runs 1–2 (harness bug, models were right); those rows are excluded.

| effort  | pass  | avg completion tok/task | wall p50 | wall max |
|---------|------:|------------------------:|---------:|---------:|
| xhigh (template default) | 14/16 | 17,807 | 221s | 384s |
| medium  | 14/16 |  4,309 |  22s | 371s |
| low     | 15/16 |  1,856 |  17.5s | 30s |
| thinking off | 13/16 | 707 | 6s | 16s |

**Conclusion:** on self-contained coding tasks, xhigh buys no pass-rate gain
over low/medium while costing ~10× tokens and ~10–12× median wall time; low
also has the tightest tail latency. Thinking off is measurably weaker. Client
defaults changed to `reasoning_effort: medium` (chat.py adds an
`effort <low|medium|xhigh|off>` command; orchestrate.py takes `--effort`,
planner/judge stay medium). Pass-rate deltas among thinking levels are within
noise at this sample size; the cost deltas are not.

## Feature verification on the production config

- **Vision works as-is** (plan Phase 10): PNG UI screenshot via base64
  `image_url` → all fields/buttons/colors/error text correctly read, 3.5s
  wall, no serve-config change needed (`language_model_only: false` in the
  FP8 checkpoint; tower loaded).
- **Tool calling works**: `qwen3_xml` parser yields clean `tool_calls` with
  valid JSON args, sensible multi-turn chaining after tool results, in both
  thinking and non-thinking modes.
- **preserve_thinking verified** (template default on): re-injects prior-turn
  reasoning only if the client echoes it back as `reasoning_content` on
  assistant history messages (prompt_tokens 227→439 in the probe);
  `preserve_thinking: false` strips it. Plain OpenAI-SDK clients drop
  reasoning silently — chat.py now echoes it.
- **API field correction**: vLLM 0.27.1 returns reasoning in
  `message.reasoning` (non-stream) and `delta.reasoning` (stream) —
  `reasoning_content` is empty/absent. CLAUDE.md updated.

## NVFP4 candidate validation (plan Phases 0–1, serving bench pending)

Remote checkpoint audit of Qwen3.8-27B NVFP4 exports:

| repo | export | MTP head | vision | size |
|------|--------|----------|--------|-----:|
| Inferact/Qwen3.8-27B-NVFP4 | **modelopt (fast path)** | yes | yes | 24.6 GiB |
| unsloth/Qwen3.8-27B-NVFP4 | compressed-tensors (slow-path risk) + FP8-KV calib | yes | yes | 21.8 GiB |
| RadixArk/Qwen3.8-27B-NVFP4 | modelopt | **no — eliminated** | yes | 20.4 GiB |

Inferact (primary candidate) downloaded to
`models/Qwen3.8-27B-NVFP4-Inferact/` for a future bake-off. Not yet served:
requires stopping production (sudo). Decision rule: NVFP4 must match FP8
within ~2% task success and be ≥15% faster to displace it.

## Not yet run (needs a service window)

FP8-vs-default KV A/B on the 3.8 hybrid (expect small effect: only 16/64
layers hold KV), `--max-num-batched-tokens` tuning, 1M-context YaRN probe
(`--hf-overrides '{"text_config":{"max_position_embeddings":1010000}}'`),
NVFP4 serving bench, BF16 quality reference.

# Addendum: service-window tests — NVFP4 bake-off, KV A/B, 1M YaRN (2026-08-16)

Production stopped for a test window; all runs on identical launcher flags
unless noted. Raw data: `qwen38_nvfp4_bench.jsonl`, `qwen38_nvfp4_quality.jsonl`,
`qwen38_nvfp4_effort.jsonl`, `qwen38_kvab_bench.jsonl`.

## NVFP4 (Inferact modelopt) vs FP8 — NVFP4 wins on everything but vision

Agent workload, 3-run medians, same flags (MTP n=3, FP8 KV, 262K, s=16):

| config | N=1 dec p50 | N=8 agg | N=16 agg | accept | tok/J @max | VRAM |
|--------|------------:|--------:|---------:|-------:|-----------:|-----:|
| FP8 (production) | 91.4 | 449 | 592 | ~0.6 | 0.99 | 88.0G |
| **NVFP4-Inferact** | **128.9 (+41%)** | **563 (+25%)** | **676 (+14%)** | 0.89–0.97 | 1.19 | 87.8G |

Quality: 10/10 smoke, NIAH/synthesis included. Coding-task success (bench_effort
medium+low ×3): 32/36 (89%) vs FP8's 29/32 (91%) — parity within noise; every
failure on both is the same hard `logparse` task. Tool calling verified. KV
capacity 1.65M tokens. Much of the speed gain is the higher MTP acceptance.

**Trade-off: vLLM 0.27.1 serves this checkpoint text-only** (no registered
multimodal processor for the quantized tower) — vision requests are rejected.
FP8 keeps vision. Launcher: `scripts/start-vllm-38-27b-nvfp4.sh` (manual);
production switch is a policy call: text speed vs vision.

## KV cache A/B on Qwen3.8 (FP8 weights): fp8_e4m3 confirmed, emphatically

Plan v2 predicted a small effect (only 16/64 layers hold KV). Wrong — FP8 KV
is a large speed win here because MTP acceptance depends on it:

| KV dtype | N=1 dec p50 | N=8 agg | accept | KV tokens |
|----------|------------:|--------:|-------:|----------:|
| default (bf16) | 79.2 | 349 | 0.58 | 800K |
| fp8_e4m3 (production) | 91.4 (+15%) | 449 (+29%) | ~0.6→0.94* | 1.55M |

*acceptance as reported by the harness; the bf16-KV run's 0.58 vs fp8-KV runs
0.89–0.97 on NVFP4 and ~0.6 on FP8's original record — directionally, fp8 KV
never hurts and usually helps acceptance. Keep `--kv-cache-dtype fp8_e4m3`.

## --max-num-batched-tokens: default 8192 stays

NVFP4, s=16: mbt=16384 ties at N=16 (676.8 vs 676.0) and regresses N=8
(495 vs 563, TTFT p50 7.8s vs 6.3s). No change.

## 1M-context YaRN probe: works

NVFP4 + `--max-model-len 1010000 --max-num-seqs 4` + YaRN factor 4.0 via
`--hf-overrides '{"text_config":{"max_position_embeddings":1010000,"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}'`:
starts clean, KV capacity 1.74M tokens (1.72× concurrency at 1M). Needle at
25% depth: **PASS at 500K (245s wall)** and **PASS at 980K (851s wall,
prefill-dominated)**. Viable as a special-occasion config; not daily (TTFT).
