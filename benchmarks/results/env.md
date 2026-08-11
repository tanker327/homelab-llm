# Benchmark environment — Qwen3.6-27B serving bake-off

Captured 2026-08-11. All bake-off results in this directory were measured on
this configuration unless a row notes otherwise.

| Component | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB, compute cap 12.0 (SM120) |
| Power limit | 600.00 W |
| Driver | 595.71.05 |
| CUDA toolkit | 13.1, V13.1.115 (nvcc build cuda_13.1.r13.1/compiler.37061995_0) |
| OS / kernel | Ubuntu 26.04, 7.0.0-29-generic |
| glibc | 2.43 (Ubuntu GLIBC 2.43-2ubuntu2.3) — note: repo docs say 2.42; the rsqrt/nvcc header conflict and `toolchain-fix/` shim still apply |
| llama.cpp (incumbent) | master `687e778` (2026-08-08), built for SM120, GNU 15.2.0. NOT rebuilt for this bake-off — it is the production binary under test. |
| Python (bench venv) | 3.12.13 |

## Models under test

| Model | Source | Size |
|---|---|---|
| Qwen3.6-27B-MTP-Q8_0.gguf | on disk (production) | 28 GB |
| Qwen3.6-27B-FP8 (safetensors) | `Qwen/Qwen3.6-27B-FP8` | ~29 GB |

## Engine installs

(filled in during Phase 1)

| Engine | Version | Torch | Gate result |
|---|---|---|---|
| llama.cpp | 687e778 | — | incumbent, passes by construction |
| vLLM | 0.27.1 (pip wheel) | 2.13.0+cu130 | **PASS** 2026-08-11: FP8+MTP3 @128K, boot 110s, 5×N=4 smoke zero errors, ~420 agg tok/s, ~470W (not power-capped). Needs `ninja` on PATH (venv bin) + toolchain-fix shim for flashinfer JIT. Arch resolves as `Qwen3_5MTP`. Streams reasoning under `delta.reasoning`. |
| SGLang | 0.5.17 (pip wheel) | 2.11.0+cu130 | **PASS** 2026-08-11: FP8+NEXTN3 @128K, boot ~130s, 5×N=4 smoke zero errors, ~390 agg tok/s, ~465W. `SGLANG_ENABLE_SPEC_V2` obsolete (V2 always on). Drafter loads as `Qwen3_5ForCausalLMMTP`. No per-request accept-rate in API. |

Both engine venvs installed 2026-08-11 via `uv` with stock pip wheels — no
source builds, no CUDA 12.8 pin, no g++-14 workaround needed on this
driver/toolkit. Python headers come from uv's managed CPython; `ninja` is
venv-local. `NVCC_PREPEND_FLAGS="-I<repo>/toolchain-fix"` still required at
launch for any runtime JIT (flashinfer GDN kernels).

Checkpoint notes (`models/Qwen3.6-27B-FP8/`): ships `mtp.safetensors` (MTP
head present → spec decode available in vLLM/SGLang); architecture is
`Qwen3_5ForConditionalGeneration` — the model is **multimodal** (vision tower
kept in bf16, excluded from FP8 quant), so a text-only load mode saves VRAM
if the engine supports one.
