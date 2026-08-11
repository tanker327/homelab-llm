# Bake-off session checkpoint — FINAL (2026-08-12)

**OUTCOME: vLLM FP8+MTP3+KV-fp8 wins** — 530 agg tok/s @ N=8 (2.2x incumbent),
120 tok/s decode @ 49K ctx, NIAH-correct to 254,976 tokens, 10/10 quality.
Full results: `docs/BENCHMARKS.md` addendum. Launcher:
`scripts/start-vllm-27b-fp8.sh`. All phases complete except the 2h soak
(in flight, → soak.jsonl) and the production-switch decision (user's call:
winner needs ~86GB, cannot coexist with the 35B agent stack).

---

## Original checkpoint below (historical)


Resume file for the Qwen3.6-27B serving bake-off (plan:
`~/.claude/plans/cool-let-s-have-a-abstract-gadget.md`). Written before a
Claude Code restart. Everything measured so far is on disk in this directory;
the servers and any in-flight benchmark die with the session and must be
relaunched per "How to relaunch" below.

## Phase status

| Phase | Status |
|---|---|
| 0 prep (env, download, tooling) | DONE — env.md, `models/Qwen3.6-27B-FP8/` (29GB), bench_serving.py + quality_smoke.py |
| 1 engine gates | DONE — both PASS (see env.md table) |
| 2 single-stream | DONE for C3/C4/C5/C6 (phase2.jsonl). MISSING: C1/C2 (llama.cpp Q8_0 ± MTP) |
| 3 concurrency | DONE for all four challenger configs (phase3.jsonl incl. complete `P3-vllm-fp8-mtp3`). MISSING: llama.cpp incumbent matrix with the same 12K-in workload |
| 4 long-ctx/KV | NOT STARTED |
| 5 quality gate | quality_smoke.py ready; Q8_0 reference NOT yet recorded (first attempt invalidated: temp-0 thinking loop, then service stop killed rerun). Candidates not run. quality.jsonl deleted/empty |
| 6 decision/soak | NOT STARTED |

## Results so far (headline numbers, agent workload 12288in/2048out)

| Config | N=1 | N=4 | N=6 | N=8 | notes |
|---|---|---|---|---|---|
| SGLang FP8+NEXTN3 | 121 | 340 | **425** | 350 (p95 24s!) | sweet spot N=6, 0.80 tok/J |
| SGLang FP8 no-spec | 48 | 156 | 214 | 158 (p95 44s!) | N=8 collapse is scheduler, not spec |
| vLLM FP8 no-spec | 47 | 153 | 214 | **259** (p95 11s) | keeps scaling at N=8 |
| vLLM FP8+MTP3 | 98 | 313 | 408 | **434** (p95 11.8s) | still climbing at N=8; accept ~0.9; 0.81 tok/J — best config so far |
| llama.cpp Q8_0+MTP3 (incumbent, from BENCHMARKS.md, ~1K-tok prompts) | 145 | 286 | ~390* | 289 | power-capped 600W from N=1, ~0.25 tok/J |

Single-stream (phase2.jsonl): W1 decode — SGLang+NEXTN 128, vLLM+MTP 118,
either no-spec ~50, incumbent 139–145 (record; C1 re-run pending). Judge
(49K in) decode — SGLang+NEXTN ~120 (!), vLLM+MTP ~67, no-spec ~46.
Prefill — SGLang ~8100–8600, vLLM ~6000–7300, llama.cpp (prod stack) ~3300.
FP8 engines never power-cap (~470–530W); spec decode is +2–2.5× here and
does NOT invert under batching.

## Remaining work, in order

1. **llama.cpp block** (no sudo needed — manual launcher):
   - C1: `./scripts/start-llama-27b.sh` (Q8_0+MTP3, port 5000), then
     w1×5 + judge×3 → phase2.jsonl (label `C1-llamacpp-q8-mtp3`) and
     agent matrix ×3 → phase3.jsonl (label `P3-llamacpp-q8-mtp3`)
   - Q8_0 quality reference: `venv/bin/python benchmarks/quality_smoke.py --label q8-reference --port 5000 --json benchmarks/results/quality.jsonl`
   - C2 (no MTP): same launcher minus `--spec-type draft-mtp --spec-draft-n-max 3` flags (copy command from script, strip spec flags); w1×5 only + agent matrix if time
2. **Phase 4** on the two leaders (likely SGLang+NEXTN and vLLM+MTP):
   max ctx via `tools/probe_max_input.py --port 5000`; FP8 KV
   (vLLM: `--kv-cache-dtype fp8_e4m3` — needs flashinfer backend; SGLang: check
   `--kv-cache-dtype` options); llama.cpp counterpart `-ctk q8_0 -ctv q8_0`;
   judge×3 + agent N=4 + VRAM-per-ctx slope each
3. **Phase 5**: quality_smoke on winner configs (FP8, FP8+KV-fp8, FP8+MTP)
   vs Q8_0 reference. Gate: no task fails that Q8_0 passes (1 retry).
4. **Phase 6**: weighted score (0.40 agg @ sweet-N, 0.15 TTFT p95 load,
   0.15 single-stream, 0.10 long-ctx, 0.10 stability, 0.10 VRAM headroom);
   2–4h soak on winner; BENCHMARKS.md addendum; follow-ups list
   (launcher script, systemd/stack integration, docs, CHANGELOG).
   Restore production: `sudo systemctl start llama-server` (user password).

## How to relaunch engines (from repo root; ALWAYS: check `nvidia-smi` shows
<2GB used and port 5000 free first; wait ~10s after killing anything)

vLLM with MTP (add/remove `--speculative-config` for spec on/off):
```bash
export PATH="$PWD/vllm-venv/bin:$PATH" NVCC_PREPEND_FLAGS="-I$PWD/toolchain-fix"
vllm-venv/bin/vllm serve models/Qwen3.6-27B-FP8 --host 0.0.0.0 --port 5000 \
  --served-model-name local --max-model-len 131072 --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```
SGLang with NEXTN (drop the three `--speculative-*` flags for spec off):
```bash
export PATH="$PWD/sglang-venv/bin:$PATH" NVCC_PREPEND_FLAGS="-I$PWD/toolchain-fix"
sglang-venv/bin/python -m sglang.launch_server --model-path models/Qwen3.6-27B-FP8 \
  --host 0.0.0.0 --port 5000 --served-model-name local --mem-fraction-static 0.90 \
  --context-length 131072 --max-running-requests 8 \
  --mamba-scheduler-strategy extra_buffer --reasoning-parser qwen3 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```
Health: poll `curl -o /dev/null -w "%{http_code}" localhost:5000/health` for
**200** (SGLang serves 503 while warming). Boot ~1–3 min.

## Gotchas learned this session

- `ninja` must be on PATH for vLLM (flashinfer JIT) — venv bin has it; the
  PATH export above covers it. toolchain-fix shim likewise.
- Engines validate the `model` field → everything launches with
  `--served-model-name local`; harness sends "local".
- vLLM streams reasoning under `delta.reasoning` (not `reasoning_content`);
  bench_serving handles all variants.
- temp 0 → endless thinking loops (54K tokens observed). quality_smoke uses
  temp 0.6 / top_p 0.95 / max_tokens 16384, truncation counts as fail.
- Production llama-server systemd unit is STOPPED (user did it 17:15, state
  "failed" is cosmetic). GPU work needs no sudo until final restore.
- glibc is 2.43 (docs said 2.42). SGLANG_ENABLE_SPEC_V2 is obsolete.
- Server logs: benchmarks/results/*.serverlog.
