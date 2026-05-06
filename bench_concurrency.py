#!/usr/bin/env python3
"""Concurrency benchmark for the running llama.cpp server on :5000."""
import json
import time
import threading
import urllib.request

BASE = "http://localhost:5000"
URL = f"{BASE}/v1/chat/completions"
MODEL = "Qwen3.6-35B-A3B-MXFP4_MOE.gguf"
CONCURRENCY_LEVELS = [1, 2, 4, 6, 8, 10]

# /no_think disables Qwen's reasoning so output length is bounded and consistent.
# Asking for an exact, self-terminating output further cuts variance across requests.
PROMPT = (
    "/no_think List the integers from 1 to 200, comma-separated, on a single line. "
    "Output only the list and nothing else."
)


def query():
    body = json.dumps({
        "model": MODEL,
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
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "gen_tok_s": timings.get("predicted_per_second") or 0.0,
        "prompt_tok_s": timings.get("prompt_per_second") or 0.0,
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
    wall_elapsed = time.perf_counter() - wall_start
    ok = [r for r in results if r is not None]
    return ok, errors, wall_elapsed


def server_slots():
    try:
        with urllib.request.urlopen(f"{BASE}/props", timeout=5) as resp:
            return json.loads(resp.read()).get("total_slots", "?")
    except Exception:
        return "?"


def main():
    slots = server_slots()
    print(f"Server total_slots = {slots}")
    print("Warming up (one untimed request)...")
    query()

    print()
    header = (
        f"{'N':>3}  {'wall(s)':>8}  {'compl_tok':>10}  "
        f"{'agg_tok/s':>10}  {'avg_per_req_tok/s':>18}  {'avg_lat(s)':>11}  "
        f"{'min_lat':>8}  {'max_lat':>8}"
    )
    print(header)
    print("-" * len(header))

    for n in CONCURRENCY_LEVELS:
        results, errors, wall = run_concurrent(n)
        err_count = sum(1 for e in errors if e is not None)
        if not results:
            print(f"{n:>3}  ALL FAILED  ({err_count} errors); first: {errors[0]}")
            continue
        total_tokens = sum(r["completion_tokens"] for r in results)
        avg_per_req_speed = sum(r["gen_tok_s"] for r in results) / len(results)
        latencies = [r["elapsed"] for r in results]
        avg_lat = sum(latencies) / len(latencies)
        agg = total_tokens / wall
        suffix = f"  ({err_count} errors)" if err_count else ""
        print(
            f"{n:>3}  {wall:>8.2f}  {total_tokens:>10}  "
            f"{agg:>10.2f}  {avg_per_req_speed:>18.2f}  {avg_lat:>11.2f}  "
            f"{min(latencies):>8.2f}  {max(latencies):>8.2f}{suffix}"
        )


if __name__ == "__main__":
    main()
