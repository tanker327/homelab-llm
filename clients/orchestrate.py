#!/usr/bin/env python3
"""Planner/worker/judge pipeline over the two-model agent stack.

  Qwen3.6-27B  (port 5001)  plans:  decompose the task into independent subtasks
  Qwen3.6-35B  (port 5000)  works:  run all subtasks in parallel
  Qwen3.6-27B  (port 5001)  judges: synthesize worker output into one answer

Start the servers first:  ./scripts/start-agent-stack.sh

Usage:
  orchestrate.py "Compare Rust and Go for a CLI tool"          # decompose mode
  orchestrate.py --mode bestof "Write a haiku about GPUs"      # N attempts, judge picks
  orchestrate.py --workers 3 --show-work "..."
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

PLANNER_URL = "http://localhost:5001/v1"
WORKERS_URL = "http://localhost:5000/v1"

planner = OpenAI(base_url=PLANNER_URL, api_key="none")
workers = OpenAI(base_url=WORKERS_URL, api_key="none")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def chat(client: OpenAI, prompt: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(model="local", messages=messages)
    return resp.choices[0].message.content or ""


def plan(task: str, n: int) -> list[str]:
    prompt = (
        f"Decompose the following task into exactly {n} independent subtasks that "
        "can each be worked on without seeing the others' results. Cover different "
        "aspects; do not overlap. Respond with ONLY a JSON array of subtask strings, "
        f"no other text.\n\nTask: {task}"
    )
    raw = chat(planner, prompt)
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        log("planner returned no JSON array; falling back to best-of mode")
        return [task] * n
    subtasks = [str(s) for s in json.loads(match.group(0))]
    return subtasks[:n] if len(subtasks) >= n else subtasks + [task] * (n - len(subtasks))


def work(subtasks: list[str]) -> list[str]:
    with ThreadPoolExecutor(max_workers=len(subtasks)) as pool:
        return list(pool.map(lambda s: chat(workers, s), subtasks))


def judge(task: str, subtasks: list[str], results: list[str], mode: str) -> str:
    blocks = "\n\n".join(
        f"### Worker {i + 1}\nSubtask: {s}\nResult:\n{r}"
        for i, (s, r) in enumerate(zip(subtasks, results))
    )
    if mode == "bestof":
        instruction = (
            "Each worker attempted the same task. Evaluate the attempts, pick the "
            "best one, and output an improved final version of it. Output only the "
            "final answer."
        )
    else:
        instruction = (
            "Each worker completed one subtask of the original task. Evaluate their "
            "output critically, fix any errors, and synthesize everything into one "
            "coherent final answer to the original task. Output only the final answer."
        )
    return chat(planner, f"Original task: {task}\n\n{blocks}\n\n{instruction}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task")
    ap.add_argument("--workers", type=int, default=3, help="number of parallel workers")
    ap.add_argument("--mode", choices=["decompose", "bestof"], default="decompose")
    ap.add_argument("--show-work", action="store_true", help="print plan and worker output")
    args = ap.parse_args()

    if args.mode == "bestof":
        subtasks = [args.task] * args.workers
        log(f"[bestof] {args.workers} workers attempting the task")
    else:
        log("[plan] 27B decomposing task...")
        subtasks = plan(args.task, args.workers)
        for i, s in enumerate(subtasks):
            log(f"  {i + 1}. {s}")

    log(f"[work] {len(subtasks)} x 35B workers running in parallel...")
    results = work(subtasks)
    if args.show_work:
        for i, r in enumerate(results):
            log(f"\n--- worker {i + 1} ---\n{r}\n")

    log("[judge] 27B evaluating and synthesizing...")
    print(judge(args.task, subtasks, results, args.mode))


if __name__ == "__main__":
    main()
