#!/usr/bin/env python3
"""Concurrency + GPU telemetry benchmark for the running llama.cpp server.

Like bench_concurrency.py, but also samples nvidia-smi (power draw, GPU temp,
utilization) at 1s intervals during each concurrency level, and can emit JSON
for later analysis.

Usage:
  bench_matrix.py --label 27B-Q8            # table to stdout
  bench_matrix.py --label 35B --levels 1,2,4,8 --json results.jsonl
"""

import argparse
import json
import subprocess
import threading
import time
import urllib.request

BASE = "http://localhost:5000"
URL = f"{BASE}/v1/chat/completions"

PROMPT = (
    "/no_think List the integers from 1 to 200, comma-separated, on a single line. "
    "Output only the list and nothing else."
)


def query():
    body = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    elapsed = time.perf_counter() - start
    usage = data.get("usage", {})
    timings = data.get("timings", {})
    return {
        "elapsed": elapsed,
        "completion_tokens": usage.get("completion_tokens", 0),
        "gen_tok_s": timings.get("predicted_per_second") or 0.0,
    }


class GpuSampler:
    """Samples power.draw / temperature.gpu / utilization.gpu once per second."""

    def __init__(self):
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=power.draw,temperature.gpu,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=5,
                ).strip().split(",")
                self.samples.append(
                    (float(out[0]), float(out[1]), float(out[2]))
                )
            except Exception:
                pass
            self._stop.wait(1.0)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=3)

    def stats(self):
        if not self.samples:
            return {}
        power = [s[0] for s in self.samples]
        temp = [s[1] for s in self.samples]
        util = [s[2] for s in self.samples]
        return {
            "power_avg_w": sum(power) / len(power),
            "power_max_w": max(power),
            "temp_max_c": max(temp),
            "util_avg_pct": sum(util) / len(util),
        }


def run_concurrent(n):
    results = [None] * n
    errors = [None] * n

    def worker(i):
        try:
            results[i] = query()
        except Exception as e:
            errors[i] = e

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall_start
    ok = [r for r in results if r is not None]
    err_count = sum(1 for e in errors if e is not None)
    return ok, err_count, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="model")
    ap.add_argument("--levels", default="1,2,3,4,6,8")
    ap.add_argument("--json", help="append one JSON line per level to this file")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    with urllib.request.urlopen(f"{BASE}/props", timeout=5) as resp:
        props = json.loads(resp.read())
    slots = props.get("total_slots", "?")
    print(f"[{args.label}] server slots={slots}  warming up...")
    query()

    header = (
        f"{'N':>3}  {'agg_tok/s':>10}  {'per_req_tok/s':>14}  {'avg_lat(s)':>11}  "
        f"{'power_avg':>10}  {'power_max':>10}  {'temp_max':>9}  {'util_avg':>9}"
    )
    print(header)
    print("-" * len(header))

    for n in levels:
        with GpuSampler() as gpu:
            results, err_count, wall = run_concurrent(n)
        g = gpu.stats()
        if not results:
            print(f"{n:>3}  ALL FAILED ({err_count} errors)")
            continue
        total_tokens = sum(r["completion_tokens"] for r in results)
        agg = total_tokens / wall
        per_req = sum(r["gen_tok_s"] for r in results) / len(results)
        avg_lat = sum(r["elapsed"] for r in results) / len(results)
        suffix = f"  ({err_count} err)" if err_count else ""
        print(
            f"{n:>3}  {agg:>10.1f}  {per_req:>14.1f}  {avg_lat:>11.1f}  "
            f"{g.get('power_avg_w', 0):>9.0f}W  {g.get('power_max_w', 0):>9.0f}W  "
            f"{g.get('temp_max_c', 0):>8.0f}C  {g.get('util_avg_pct', 0):>8.0f}%{suffix}"
        )
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps({
                    "label": args.label, "n": n, "agg_tok_s": round(agg, 1),
                    "per_req_tok_s": round(per_req, 1), "avg_lat_s": round(avg_lat, 1),
                    "wall_s": round(wall, 1), "total_tokens": total_tokens,
                    "errors": err_count, **{k: round(v, 1) for k, v in g.items()},
                }) + "\n")


if __name__ == "__main__":
    main()
