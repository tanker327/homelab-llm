#!/usr/bin/env python3
"""Serving benchmark for the Qwen3.6-27B engine bake-off (llama.cpp / vLLM / SGLang).

Merges bench_matrix.py's concurrency harness + GPU telemetry with
probe_max_input.py's streaming TTFT measurement and token-exact prompt
building. Works against any OpenAI-compatible server.

Per request: TTFT, decode tok/s (engine timings preferred), prefill tok/s.
Per level:   aggregate tok/s, TTFT p50/p95, power/temp/util/VRAM, tok/J,
             MTP acceptance (llama.cpp timings or vLLM /metrics delta).

Usage:
  bench_serving.py --label C1-llamacpp-q8-mtp3 --workload w1 --levels 1 --runs 5 \
      --json benchmarks/results/phase2.jsonl
  bench_serving.py --label C4-vllm-fp8-mtp3 --workload agent --levels 1,2,4,6,8 \
      --runs 3 --port 5000 --json benchmarks/results/phase3.jsonl
"""

import argparse
import json
import re
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Optional

WORKLOADS = {
    # name: (input_tokens, output_tokens)
    "w1": (2048, 1024),      # interactive coding request
    "agent": (12288, 2048),  # orchestrate.py worker-shaped request
    "judge": (49152, 2048),  # orchestrate.py judge-shaped request (prefill-heavy)
}

FILLER_BASE = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim "
    "veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
    "commodo consequat. Duis aute irure dolor in reprehenderit in voluptate. "
)
# Output length is controlled by max_tokens, so the task must never finish on
# its own. /no_think keeps all engines out of reasoning mode (same token
# budget everywhere) and sidesteps llama.cpp's truncate-mid-reasoning 500.
TASK = (
    "\n\n/no_think Ignore the filler text above. Count upward from 1, "
    "comma-separated, without stopping and without any other text."
)

_TOKENIZE_KEY: Optional[str] = None   # cached "content" or "prompt"
_TOKENIZE_BROKEN = False              # server has no usable /tokenize
_CHARS_PER_TOKEN: Optional[float] = None


def tokenize(text: str, base_url: str, timeout: int = 120) -> int:
    """Token count via the server's /tokenize; calibrated estimate as fallback."""
    global _TOKENIZE_KEY, _TOKENIZE_BROKEN, _CHARS_PER_TOKEN
    if not _TOKENIZE_BROKEN:
        keys = [_TOKENIZE_KEY] if _TOKENIZE_KEY else ["content", "prompt"]
        last_err = None
        for key in keys:
            body = {key: text}
            if key == "prompt":
                body["model"] = "local"  # vLLM/SGLang validate this; launchers set --served-model-name local
            try:
                req = urllib.request.Request(
                    f"{base_url}/tokenize",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    resp = json.loads(r.read())
                tokens = resp.get("tokens") or []
                count = resp.get("count")
                n = count if count is not None else len(tokens)
                _TOKENIZE_KEY = key
                return n
            except (urllib.error.HTTPError, urllib.error.URLError,
                    json.JSONDecodeError, socket.timeout) as e:
                last_err = e
                continue
        _TOKENIZE_BROKEN = True
        print(f"  [warn] /tokenize unusable ({last_err}); falling back to "
              "usage.prompt_tokens calibration")
    # Fallback: calibrate chars/token once with a max_tokens=1 request.
    if _CHARS_PER_TOKEN is None:
        sample = (FILLER_BASE * 40)[:8000]
        body = {
            "model": "local",
            "messages": [{"role": "user", "content": sample}],
            "max_tokens": 1, "stream": False, "temperature": 0,
        }
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            usage = json.loads(r.read()).get("usage") or {}
        pt = usage.get("prompt_tokens")
        if not pt:
            raise RuntimeError("calibration request returned no usage.prompt_tokens")
        _CHARS_PER_TOKEN = len(sample) / max(1, pt - 16)  # ~16 template tokens
        print(f"  [info] calibrated {_CHARS_PER_TOKEN:.2f} chars/token")
    return int(len(text) / _CHARS_PER_TOKEN)


def build_prompt_base(target_tokens: int, base_url: str) -> tuple[str, int]:
    """Filler+task prompt of ~target_tokens. Unique prefix is added per request."""
    fixed_n = tokenize(TASK, base_url)
    filler_target = max(0, target_tokens - fixed_n - 12)  # 12 = prefix budget
    if filler_target < 50:
        return TASK.strip(), fixed_n
    chars = int(filler_target * 3.5)
    filler = (FILLER_BASE * (chars // len(FILLER_BASE) + 2))[:chars]
    for _ in range(6):
        n = tokenize(filler, base_url)
        if abs(n - filler_target) <= max(8, target_tokens // 500):
            break
        ratio = len(filler) / max(1, n)
        target_chars = max(1, int(filler_target * ratio))
        filler = (FILLER_BASE * (target_chars // len(FILLER_BASE) + 2))[:target_chars]
    prompt = filler + TASK
    return prompt, tokenize(prompt, base_url)


class GpuSampler:
    """1 Hz nvidia-smi sampling: power, temp, util, VRAM."""

    QUERY = "power.draw,temperature.gpu,utilization.gpu,memory.used"

    def __init__(self):
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", f"--query-gpu={self.QUERY}",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=5,
                ).strip().split("\n")[0].split(",")
                self.samples.append(tuple(float(x) for x in out))
            except Exception:
                pass
            self._stop.wait(1.0)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=3)

    def stats(self) -> dict:
        if not self.samples:
            return {}
        power = [s[0] for s in self.samples]
        temp = [s[1] for s in self.samples]
        util = [s[2] for s in self.samples]
        vram = [s[3] for s in self.samples]
        return {
            "power_avg_w": sum(power) / len(power),
            "power_max_w": max(power),
            "temp_max_c": max(temp),
            "util_avg_pct": sum(util) / len(util),
            "vram_max_mb": max(vram),
        }


def stream_request(base_url: str, prompt: str, max_tokens: int,
                   timeout_s: int) -> dict:
    """One streaming chat completion; returns per-request metrics or error."""
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t_start = time.monotonic()
    first_t: Optional[float] = None
    usage: Optional[dict] = None
    timings: Optional[dict] = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw in resp:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices and first_t is None:
                    delta = choices[0].get("delta") or {}
                    # field name varies: llama.cpp reasoning_content, vLLM reasoning
                    if (delta.get("content") or delta.get("reasoning_content")
                            or delta.get("reasoning")):
                        first_t = time.monotonic()
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("timings"):
                    timings = chunk["timings"]
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode(errors="replace")[:200]
        except Exception:
            pass
        return {"error": f"HTTP {e.code}: {detail}"}
    except (socket.timeout, urllib.error.URLError, ConnectionError, OSError) as e:
        return {"error": str(e)[:200]}

    t_end = time.monotonic()
    completion_tokens = (usage or {}).get("completion_tokens") or 0
    prompt_tokens = (usage or {}).get("prompt_tokens") or 0
    ttft_s = (first_t - t_start) if first_t else None

    decode_tps: Optional[float] = None
    prefill_tps: Optional[float] = None
    accept_rate: Optional[float] = None
    if timings:
        decode_tps = timings.get("predicted_per_second")
        prefill_tps = timings.get("prompt_per_second")
        draft_n = timings.get("draft_n")
        draft_acc = timings.get("draft_n_accepted")
        if draft_n:
            accept_rate = (draft_acc or 0) / draft_n
    if decode_tps is None and first_t and completion_tokens >= 16:
        dt = t_end - first_t
        if dt > 0.1:
            decode_tps = completion_tokens / dt
    if prefill_tps is None and ttft_s and prompt_tokens:
        prefill_tps = prompt_tokens / ttft_s  # approximate: includes queueing

    return {
        "elapsed_s": t_end - t_start,
        "ttft_ms": ttft_s * 1000 if ttft_s else None,
        "decode_tps": decode_tps,
        "prefill_tps": prefill_tps,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "accept_rate": accept_rate,
    }


def scrape_spec_metrics(base_url: str) -> Optional[tuple[float, float]]:
    """(accepted, drafted) totals from a vLLM-style /metrics endpoint."""
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=5) as r:
            text = r.read().decode(errors="replace")
    except Exception:
        return None
    def total(name: str) -> Optional[float]:
        vals = re.findall(rf"^{re.escape(name)}(?:{{[^}}]*}})?\s+([0-9.eE+-]+)",
                          text, re.M)
        return sum(float(v) for v in vals) if vals else None
    acc = total("vllm:spec_decode_num_accepted_tokens_total")
    dra = total("vllm:spec_decode_num_draft_tokens_total")
    if acc is None or dra is None:
        return None
    return acc, dra


def pct(values: list[float], p: float) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def run_level(base_url: str, prompt_base: str, n: int, max_tokens: int,
              timeout_s: int) -> tuple[list[dict], int, float, dict]:
    results: list[Optional[dict]] = [None] * n

    def worker(i: int):
        # Unique prefix per request defeats prefix caching (vLLM APC / SGLang
        # radix / llama.cpp slot reuse) without touching engine defaults.
        prompt = f"[req {uuid.uuid4().hex[:8]}] {prompt_base}"
        results[i] = stream_request(base_url, prompt, max_tokens, timeout_s)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    with GpuSampler() as gpu:
        wall_start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.monotonic() - wall_start
    ok = [r for r in results if r and "error" not in r]
    errors = [r["error"] for r in results if r and "error" in r]
    if errors:
        print(f"  [err] {len(errors)} failed request(s); first: {errors[0]}")
    return ok, len(errors), wall, gpu.stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--workload", choices=sorted(WORKLOADS), default=None)
    ap.add_argument("--in-tokens", type=int, default=None)
    ap.add_argument("--out-tokens", type=int, default=None)
    ap.add_argument("--levels", default="1")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1200, help="seconds per request")
    ap.add_argument("--json", help="append one JSON line per (level, run)")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    base_url = args.base_url or f"http://localhost:{args.port}"
    if args.workload:
        in_tok, out_tok = WORKLOADS[args.workload]
    else:
        in_tok, out_tok = 2048, 1024
    in_tok = args.in_tokens or in_tok
    out_tok = args.out_tokens or out_tok
    workload_name = args.workload or f"{in_tok}in-{out_tok}out"
    levels = [int(x) for x in args.levels.split(",")]

    print(f"[{args.label}] {base_url}  workload={workload_name} "
          f"({in_tok} in / {out_tok} out)  levels={levels}  runs={args.runs}")
    prompt_base, actual_tok = build_prompt_base(in_tok, base_url)
    print(f"  prompt built: {actual_tok} tokens (target {in_tok})")

    if not args.no_warmup:
        w = stream_request(base_url, f"[warmup] {prompt_base}", min(64, out_tok),
                           args.timeout)
        if "error" in w:
            print(f"  [fatal] warmup failed: {w['error']}")
            return 1
        print(f"  warmup ok (ttft {w['ttft_ms']:.0f}ms)")

    header = (f"{'N':>3} {'run':>4} {'agg_tok/s':>10} {'dec_p50':>8} "
              f"{'ttft_p50':>9} {'ttft_p95':>9} {'prefill':>9} {'tok/J':>6} "
              f"{'pwr_avg':>8} {'temp':>5} {'vram_GB':>8} {'acc':>5} {'err':>4}")
    print(header)
    print("-" * len(header))

    for n in levels:
        for run_i in range(1, args.runs + 1):
            spec_before = scrape_spec_metrics(base_url)
            ok, err_count, wall, g = run_level(
                base_url, prompt_base, n, out_tok, args.timeout)
            spec_after = scrape_spec_metrics(base_url) if spec_before else None

            if not ok:
                print(f"{n:>3} {run_i:>4}  ALL FAILED ({err_count} errors)")
                continue

            total_completion = sum(r["completion_tokens"] for r in ok)
            agg = total_completion / wall if wall > 0 else 0.0
            dec_p50 = pct([r["decode_tps"] for r in ok], 0.50)
            ttft_p50 = pct([r["ttft_ms"] for r in ok], 0.50)
            ttft_p95 = pct([r["ttft_ms"] for r in ok], 0.95)
            prefill_vals = [r["prefill_tps"] for r in ok if r["prefill_tps"]]
            prefill_mean = (sum(prefill_vals) / len(prefill_vals)
                            if prefill_vals else None)
            accept_vals = [r["accept_rate"] for r in ok if r["accept_rate"] is not None]
            accept = sum(accept_vals) / len(accept_vals) if accept_vals else None
            if accept is None and spec_before and spec_after:
                d_acc = spec_after[0] - spec_before[0]
                d_dra = spec_after[1] - spec_before[1]
                if d_dra > 0:
                    accept = d_acc / d_dra
            power_avg = g.get("power_avg_w")
            tok_j = agg / power_avg if power_avg else None

            print(f"{n:>3} {run_i:>4} {agg:>10.1f} "
                  f"{dec_p50 or 0:>8.1f} "
                  f"{(ttft_p50 or 0):>8.0f}m {(ttft_p95 or 0):>8.0f}m "
                  f"{(prefill_mean or 0):>9.0f} "
                  f"{(tok_j or 0):>6.2f} {(power_avg or 0):>7.0f}W "
                  f"{g.get('temp_max_c', 0):>4.0f}C "
                  f"{g.get('vram_max_mb', 0) / 1024:>8.1f} "
                  f"{('%.2f' % accept) if accept is not None else '    -':>5} "
                  f"{err_count:>4}")

            if args.json:
                with open(args.json, "a") as f:
                    f.write(json.dumps({
                        "label": args.label,
                        "workload": workload_name,
                        "in_tokens": actual_tok,
                        "out_tokens": out_tok,
                        "n": n,
                        "run": run_i,
                        "wall_s": round(wall, 1),
                        "agg_tok_s": round(agg, 1),
                        "decode_tok_s_p50": round(dec_p50, 1) if dec_p50 else None,
                        "ttft_ms_p50": round(ttft_p50, 0) if ttft_p50 else None,
                        "ttft_ms_p95": round(ttft_p95, 0) if ttft_p95 else None,
                        "prefill_tok_s_mean": round(prefill_mean, 0) if prefill_mean else None,
                        "total_completion_tokens": total_completion,
                        "tok_per_j": round(tok_j, 3) if tok_j else None,
                        "mtp_accept_rate": round(accept, 3) if accept is not None else None,
                        "errors": err_count,
                        **{k: round(v, 1) for k, v in g.items()},
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
