#!/usr/bin/env python3
"""Probe the running inference server on :5000 for max usable prompt length.

Binary-searches prompt size, validates retrieval via needle-in-a-haystack,
records prefill/decode throughput. Talks only to the running server — does
not start/stop engines.

Usage:
    ./venv/bin/python probe_max_input.py
    ./venv/bin/python probe_max_input.py --lo 1024 --hi 262144
    ./venv/bin/python probe_max_input.py --no-needle    # acceptance only
"""
import argparse
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

NEEDLE = "The secret passcode is BLUE-OWL-7142."
NEEDLE_KEY = "BLUE-OWL-7142"
QUESTION = (
    "\n\nWhat is the secret passcode hidden somewhere above? "
    "Reply with only the passcode and nothing else."
)
FILLER_BASE = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim "
    "veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
    "commodo consequat. Duis aute irure dolor in reprehenderit in voluptate. "
)


@dataclass
class Result:
    target: int
    prompt_tokens: int
    outcome: str
    ttft_ms: Optional[float]
    decode_tps: Optional[float]
    content_preview: str
    error: Optional[str]
    vram_mb: Optional[int]


_TOKENIZE_KEY: Optional[str] = None  # cached "content" or "prompt"


def fetch_engine(port: int) -> str:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5) as r:
            return json.loads(r.read())["data"][0]["id"]
    except Exception as e:
        return f"<unknown: {e}>"


def tokenize(text: str, port: int) -> int:
    """Return token count via the server's /tokenize endpoint."""
    global _TOKENIZE_KEY
    keys = [_TOKENIZE_KEY] if _TOKENIZE_KEY else ["content", "prompt"]
    last_err = None
    for key in keys:
        body = {key: text}
        if key == "prompt":
            body["model"] = "any"
        try:
            req = urllib.request.Request(
                f"http://localhost:{port}/tokenize",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.loads(r.read())
            tokens = resp.get("tokens") or []
            count = resp.get("count")
            n = count if count is not None else len(tokens)
            _TOKENIZE_KEY = key
            return n
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as e:
            last_err = e
            continue
    raise RuntimeError(f"/tokenize failed for both 'content' and 'prompt' shapes: {last_err}")


def build_prompt(target_tokens: int, depth_frac: float, port: int) -> tuple[str, int]:
    fixed = NEEDLE + QUESTION
    fixed_n = tokenize(fixed, port)
    filler_target = target_tokens - fixed_n
    if filler_target < 100:
        return fixed, fixed_n

    chars = int(filler_target * 3.5)
    filler = (FILLER_BASE * (chars // len(FILLER_BASE) + 2))[:chars]
    for _ in range(6):
        n = tokenize(filler, port)
        if abs(n - filler_target) <= 4:
            break
        ratio = len(filler) / max(1, n)
        target_chars = max(1, int(filler_target * ratio))
        filler = (FILLER_BASE * (target_chars // len(FILLER_BASE) + 2))[:target_chars]

    split = int(len(filler) * depth_frac)
    prompt = filler[:split] + " " + NEEDLE + " " + filler[split:] + QUESTION
    return prompt, tokenize(prompt, port)


def vram_used_mb() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip()
        return int(out.split("\n")[0])
    except Exception:
        return None


def probe(target: int, port: int, depth_frac: float, no_needle: bool, timeout_s: int) -> Result:
    try:
        prompt, actual = build_prompt(target, depth_frac, port)
    except Exception as e:
        return Result(target, 0, "BUILD_FAIL", None, None, "", str(e)[:300], None)

    body = {
        "model": "any",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )

    t_start = time.monotonic()
    first_t: Optional[float] = None
    content_text = ""
    usage: Optional[dict] = None
    timings: Optional[dict] = None  # llama.cpp emits {predicted_per_second, prompt_per_second, ...}

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
                if choices:
                    delta = choices[0].get("delta") or {}
                    text = delta.get("content")
                    if text:
                        if first_t is None:
                            first_t = time.monotonic()
                        content_text += text
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("timings"):
                    timings = chunk["timings"]
    except urllib.error.HTTPError as e:
        err = ""
        try:
            err = e.read().decode(errors="replace")
        except Exception:
            pass
        markers = ("context", "length", "too long", "exceeds", "maximum", "max_model_len")
        if e.code in (400, 413, 422) and any(m in err.lower() for m in markers):
            outcome = "REJECTED_LENGTH"
        else:
            outcome = "REJECTED_OTHER"
        return Result(target, actual, outcome, None, None, "", f"HTTP {e.code}: {err[:280]}", vram_used_mb())
    except (socket.timeout, urllib.error.URLError) as e:
        return Result(target, actual, "TIMEOUT", None, None, "", str(e)[:300], vram_used_mb())

    t_end = time.monotonic()
    completion_tokens = (usage or {}).get("completion_tokens") or 0
    prompt_tokens = (usage or {}).get("prompt_tokens") or actual
    ttft_ms = (first_t - t_start) * 1000 if first_t else None
    # Prefer engine-reported decode rate (llama.cpp emits this in the stream's
    # final chunk and it's accurate even for tiny completions). Fall back to
    # wall-clock only when the sample is large enough to be meaningful.
    decode_tps: Optional[float] = None
    if timings and timings.get("predicted_per_second"):
        decode_tps = timings["predicted_per_second"]
    else:
        decode_dt = (t_end - first_t) if first_t else None
        if decode_dt and decode_dt > 0.1 and completion_tokens >= 16:
            decode_tps = completion_tokens / decode_dt

    if no_needle:
        outcome = "ACCEPTED"
    elif NEEDLE_KEY in content_text:
        outcome = "ACCEPTED+CORRECT"
    else:
        outcome = "ACCEPTED+WRONG"

    preview = content_text.strip().replace("\n", " ")[:80]
    return Result(target, prompt_tokens, outcome, ttft_ms, decode_tps, preview, None, vram_used_mb())


def fmt_row(r: Result) -> str:
    ttft = f"{r.ttft_ms:>6.0f}" if r.ttft_ms is not None else "     -"
    tps = f"{r.decode_tps:>5.1f}" if r.decode_tps is not None else "    -"
    vram = f"{r.vram_mb:>5}" if r.vram_mb is not None else "    -"
    return f"  {r.target:>7}  pt={r.prompt_tokens:>7}  {r.outcome:<18}  ttft={ttft}ms  tps={tps}  vram={vram}MB  {r.content_preview}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--lo", type=int, default=1024)
    p.add_argument("--hi", type=int, default=262144)
    p.add_argument("--tolerance", type=int, default=512)
    p.add_argument("--depth-frac", type=float, default=0.5)
    p.add_argument("--no-needle", action="store_true")
    p.add_argument("--timeout", type=int, default=300, help="seconds per request")
    args = p.parse_args()

    engine = fetch_engine(args.port)
    print(f"Engine on :{args.port}: {engine}")
    print(f"Probing in [{args.lo}, {args.hi}] tokens, tolerance={args.tolerance}, depth={args.depth_frac}")
    print()
    print("   target   prompt_tk  outcome             ttft       tps    vram     preview")

    history: list[Result] = []
    lo, hi = args.lo, args.hi

    print("\n--- Phase 1: bisect acceptance boundary ---")
    aborted = False
    while hi - lo > args.tolerance:
        mid = (lo + hi) // 2
        r = probe(mid, args.port, args.depth_frac, args.no_needle, args.timeout)
        history.append(r)
        print(fmt_row(r))
        if r.outcome.startswith("ACCEPTED"):
            lo = mid
        elif r.outcome == "REJECTED_LENGTH":
            hi = mid
        elif r.outcome == "TIMEOUT":
            hi = mid
        else:
            print(f"  Aborting: {r.outcome} | {r.error}")
            aborted = True
            break

    accepted = [r for r in history if r.outcome.startswith("ACCEPTED")]
    rejected = [r for r in history if not r.outcome.startswith("ACCEPTED") and r.outcome != "BUILD_FAIL"]
    max_accepted = max((r.target for r in accepted), default=None)

    max_correct = None
    if not args.no_needle and accepted and not aborted:
        correct = [r for r in accepted if r.outcome == "ACCEPTED+CORRECT"]
        max_correct_so_far = max((r.target for r in correct), default=None)

        if max_correct_so_far == max_accepted:
            max_correct = max_correct_so_far
        else:
            lo2 = max_correct_so_far if max_correct_so_far is not None else args.lo
            hi2 = max_accepted
            print(f"\n--- Phase 2: bisect correctness in [{lo2}, {hi2}] ---")
            while hi2 - lo2 > args.tolerance:
                mid = (lo2 + hi2) // 2
                r = probe(mid, args.port, args.depth_frac, False, args.timeout)
                history.append(r)
                print(fmt_row(r))
                if r.outcome == "ACCEPTED+CORRECT":
                    lo2 = mid
                else:
                    hi2 = mid
            correct = [r for r in history if r.outcome == "ACCEPTED+CORRECT"]
            max_correct = max((r.target for r in correct), default=None)

    print("\n=== Summary ===")
    print(f"Engine:              {engine}")
    print(f"Max accepted prompt: {max_accepted} tokens" if max_accepted else "Max accepted prompt: (none — all probes failed)")
    if not args.no_needle:
        if max_correct:
            print(f"Max correct (NIAH):  {max_correct} tokens")
        else:
            print("Max correct (NIAH):  (none — model never returned the passcode)")
    if rejected:
        first_reject = min(rejected, key=lambda r: r.target)
        print(f"First failure at:    {first_reject.target} ({first_reject.outcome})")
    print(f"Probes run:          {len(history)}")


if __name__ == "__main__":
    main()
