#!/usr/bin/env python3
"""10-task quality smoke suite for the Qwen3.6-27B engine bake-off.

Pass/fail gate, not a leaderboard: run against the Q8_0 reference and each
candidate config; a candidate must not fail any task the reference passes.
Pinned sampling: temperature 0, thinking left on, generous output budget.

Tasks: 4 coding (scripted asserts) · 2 instruction-format · 2 JSON schema ·
2 long-context (NIAH + synthesis, ~32K tokens).

Usage:
  quality_smoke.py --label q8-reference --port 5001
  quality_smoke.py --label vllm-fp8-mtp3 --port 5000 --json benchmarks/results/quality.jsonl
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_serving import FILLER_BASE, tokenize  # noqa: E402


def chat(base_url: str, prompt: str, timeout_s: int = 900) -> str:
    """Non-streaming completion; returns final content (reasoning stripped).

    Sampling: Qwen's recommended thinking-mode settings (temp 0.6, top_p
    0.95). Greedy (temp 0) sends this model into endless-reasoning loops —
    observed 54K+ thinking tokens on a simple coding task. The 16K cap makes
    runaway reasoning terminate; a capped answer scores as a failure, which
    is the intended gate semantic (unusable in practice = fail).
    """
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 16384,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = json.loads(r.read())
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    # Engines without a reasoning parser may leave <think> blocks inline.
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    return content.strip()


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else text


def run_python(code: str, asserts: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + asserts + "\nprint('OK')\n")
        path = f.name
    try:
        out = subprocess.run([sys.executable, path], capture_output=True,
                             text=True, timeout=30)
        ok = out.returncode == 0 and "OK" in out.stdout
        return ok, (out.stderr or out.stdout)[-300:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        Path(path).unlink(missing_ok=True)


# --- coding tasks -----------------------------------------------------------

CODING = [
    ("code-rle",
     "Write a Python function `rle(s)` that run-length encodes a string: "
     "consecutive repeated characters become `<char><count>` (count always "
     "present, even when 1). Example: rle('aaabccd') == 'a3b1c2d1'. "
     "Empty string returns ''. Reply with a single python code block.",
     "assert rle('aaabccd') == 'a3b1c2d1'\n"
     "assert rle('') == ''\n"
     "assert rle('a') == 'a1'\n"
     "assert rle('aabbaa') == 'a2b2a2'"),
    ("code-merge-intervals",
     "Write a Python function `merge_intervals(intervals)` taking a list of "
     "[start, end] pairs and returning the merged, sorted list of "
     "non-overlapping intervals. Touching intervals ([1,2],[2,3]) merge. "
     "Reply with a single python code block.",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
     "assert merge_intervals([[1,2],[2,3]]) == [[1,3]]\n"
     "assert merge_intervals([]) == []\n"
     "assert merge_intervals([[5,7],[1,3]]) == [[1,3],[5,7]]"),
    ("code-semver",
     "Write a Python function `cmp_semver(a, b)` comparing two semantic "
     "version strings like '1.10.2' (major.minor.patch, no pre-release tags). "
     "Return -1 if a<b, 0 if equal, 1 if a>b. Numeric comparison, not "
     "lexicographic. Reply with a single python code block.",
     "assert cmp_semver('1.10.2', '1.9.9') == 1\n"
     "assert cmp_semver('2.0.0', '2.0.0') == 0\n"
     "assert cmp_semver('0.9.1', '0.10.0') == -1\n"
     "assert cmp_semver('10.0.0', '9.99.99') == 1"),
    ("code-flatten",
     "Write a Python function `flatten(x)` that flattens arbitrarily nested "
     "lists/tuples into a flat list, preserving order; non-sequence items "
     "(including strings, which must NOT be split) pass through as elements. "
     "Reply with a single python code block.",
     "assert flatten([1, [2, [3, (4, 5)]], 'ab']) == [1, 2, 3, 4, 5, 'ab']\n"
     "assert flatten([]) == []\n"
     "assert flatten([[[1]]]) == [1]\n"
     "assert flatten(['xy', ['z']]) == ['xy', 'z']"),
]

# --- format tasks -----------------------------------------------------------

def check_three_lines(text: str) -> bool:
    lines = [ln for ln in text.strip().split("\n") if ln.strip()]
    return len(lines) == 3 and all(re.match(r"^\d", ln.strip()) for ln in lines)


def check_csv_row(text: str) -> bool:
    t = text.strip().strip("`")
    return bool(re.fullmatch(r"Mercury,Venus,Earth,Mars", t))


FORMAT = [
    ("fmt-three-lines",
     "Name three primary colors. Reply with exactly 3 lines, each line "
     "starting with its number followed by a period (1. 2. 3.), and no other "
     "text before or after.",
     check_three_lines),
    ("fmt-csv",
     "List the first four planets from the Sun in order as a single CSV line "
     "with no spaces after commas and no trailing newline text. Output only "
     "the CSV line.",
     check_csv_row),
]

# --- JSON tasks -------------------------------------------------------------

def check_weather_call(text: str) -> bool:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return False
    return (obj.get("name") == "get_weather"
            and isinstance(obj.get("arguments"), dict)
            and obj["arguments"].get("city") == "Tokyo"
            and obj["arguments"].get("unit") in ("celsius", "c"))


def check_user_json(text: str) -> bool:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return False
    return (set(obj) == {"username", "age", "tags"}
            and obj["username"] == "eric_w"
            and obj["age"] == 34
            and obj["tags"] == ["admin", "ops"])


JSONTASKS = [
    ("json-tool-call",
     'You have a tool: {"name": "get_weather", "parameters": {"city": '
     '"string", "unit": "celsius|fahrenheit"}}. The user asks: "What\'s the '
     'weather in Tokyo, in celsius?" Reply with ONLY the JSON tool call in '
     'the form {"name": ..., "arguments": {...}} and nothing else.',
     check_weather_call),
    ("json-extract",
     "Extract into JSON with keys username (string), age (integer), tags "
     "(array of strings), from: 'User eric_w is 34 years old and has roles "
     "admin and ops.' Use tags [\"admin\", \"ops\"]. Reply with ONLY the "
     "JSON object.",
     check_user_json),
]

# --- long-context tasks -----------------------------------------------------

NEEDLE = "The secret passcode is BLUE-OWL-7142."
FACTS = [
    "Fact alpha: the server rack is in Freiburg.",
    "Fact beta: the rack holds exactly 14 machines.",
    "Fact gamma: each machine has 2 power feeds.",
]


def build_filler(target_tokens: int, base_url: str) -> str:
    chars = int(target_tokens * 3.5)
    filler = (FILLER_BASE * (chars // len(FILLER_BASE) + 2))[:chars]
    for _ in range(4):
        n = tokenize(filler, base_url)
        if abs(n - target_tokens) <= 64:
            break
        ratio = len(filler) / max(1, n)
        filler = (FILLER_BASE * (int(target_tokens * ratio) // len(FILLER_BASE) + 2)
                  )[:int(target_tokens * ratio)]
    return filler


def longctx_niah(base_url: str) -> str:
    filler = build_filler(32000, base_url)
    split = len(filler) // 2
    return (filler[:split] + " " + NEEDLE + " " + filler[split:]
            + "\n\nWhat is the secret passcode hidden above? Reply with only "
              "the passcode.")


def longctx_synth(base_url: str) -> str:
    filler = build_filler(32000, base_url)
    third = len(filler) // 3
    parts = (filler[:third] + " " + FACTS[0] + " " + filler[third:2 * third]
             + " " + FACTS[1] + " " + filler[2 * third:] + " " + FACTS[2])
    return (parts + "\n\nUsing the three facts hidden above: how many power "
                    "feeds are there in total in the Freiburg rack? Reply "
                    "with only the number.")


def check_niah(text: str) -> bool:
    return "BLUE-OWL-7142" in text


def check_synth(text: str) -> bool:
    nums = re.findall(r"\d+", text)
    return bool(nums) and nums[-1] == "28"


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--json", help="append one JSON line per config run")
    ap.add_argument("--retry-failed", action="store_true", default=True,
                    help="retry each failed task once (flake allowance)")
    args = ap.parse_args()
    base_url = args.base_url or f"http://localhost:{args.port}"

    results: dict[str, bool] = {}
    notes: dict[str, str] = {}

    def record(name: str, attempt):
        ok, note = attempt()
        if not ok and args.retry_failed:
            ok, note = attempt()
            note = f"(retried) {note}"
        results[name] = ok
        notes[name] = note
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  — {note[:120]}"))

    print(f"[{args.label}] quality smoke vs {base_url}")

    for name, prompt, asserts in CODING:
        record(name, lambda p=prompt, a=asserts: run_python(
            extract_code(chat(base_url, p)), a))

    for name, prompt, checker in FORMAT + JSONTASKS:
        record(name, lambda p=prompt, c=checker:
               ((lambda t: (c(t), t[:150]))(chat(base_url, p))))

    print("  building 32K long-context prompts...")
    for name, builder, checker in [("longctx-niah", longctx_niah, check_niah),
                                   ("longctx-synth", longctx_synth, check_synth)]:
        prompt = builder(base_url)
        record(name, lambda p=prompt, c=checker:
               ((lambda t: (c(t), t[:150]))(chat(base_url, p, timeout_s=1200))))

    passed = sum(results.values())
    print(f"\n[{args.label}] {passed}/10 passed: "
          + " ".join(f"{k}={'P' if v else 'F'}" for k, v in results.items()))

    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps({
                "label": args.label, "passed": passed,
                "results": results,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
