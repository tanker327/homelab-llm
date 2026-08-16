#!/usr/bin/env python3
"""Phase 4 (plan v2): reasoning_effort x thinking-mode sweep on the live server.

6 coding tasks with scripted asserts, run at reasoning_effort xhigh/medium/low
(thinking sampler 1.0/0.95/top_k20) and with thinking disabled (0.7/0.8/pp1.5).
Tasks within a config run concurrently (server allows 16 seqs); configs run
sequentially so wall times are comparable across configs.
"""
import concurrent.futures as cf
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

BASE = "http://localhost:5000"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("effort_results.jsonl")
RUN_ID = sys.argv[2] if len(sys.argv) > 2 else "r1"

CONFIGS = {
    "xhigh":   {"kw": {"reasoning_effort": "xhigh"},  "samp": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}},
    "medium":  {"kw": {"reasoning_effort": "medium"}, "samp": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}},
    "low":     {"kw": {"reasoning_effort": "low"},    "samp": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}},
    "nothink": {"kw": {"enable_thinking": False},     "samp": {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5}},
}

# Optional comma-separated subset, e.g. EFFORT_CONFIGS=medium,low
import os
_subset = os.environ.get("EFFORT_CONFIGS")
if _subset:
    CONFIGS = {k: v for k, v in CONFIGS.items() if k in _subset.split(",")}

PROMPT_SUFFIX = (
    "\n\nReturn the complete implementation in a single ```python code block. "
    "No usage examples needed, just the implementation."
)

TASKS = {
    "dedupe": {
        "prompt": "Write a Python function `dedupe(rows, key)` that takes a list of dicts and a key name, and returns a new list where only one dict per distinct key-value survives: the LAST occurrence's dict, but placed at the position where that key-value FIRST appeared. Rows missing the key are kept as-is in their original positions." + PROMPT_SUFFIX,
        "test": """
rows=[{'id':1,'v':'a'},{'id':2,'v':'b'},{'x':9},{'id':1,'v':'c'},{'id':3,'v':'d'},{'id':2,'v':'e'}]
out=dedupe(rows,'id')
assert out==[{'id':1,'v':'c'},{'id':2,'v':'e'},{'x':9},{'id':3,'v':'d'}], out
assert dedupe([],'id')==[]
assert dedupe([{'a':1}],'id')==[{'a':1}]
r2=[{'id':1},{'id':1},{'id':1}]
assert dedupe(r2,'id')==[{'id':1}]
""",
    },
    "lru_ttl": {
        "prompt": "Write a Python class `LRUTTLCache(capacity, ttl, clock)` where clock is a zero-arg callable returning current time in seconds. Methods: `get(key)` returns value or None if missing/expired (expired entries are removed on access; a get refreshes LRU recency but NOT the TTL). `put(key, value)` inserts/updates (updating resets both TTL and recency); when over capacity, evict the least-recently-used non-expired entry, but evict expired entries first if any exist." + PROMPT_SUFFIX,
        "test": """
t=[0.0]
clk=lambda: t[0]
c=LRUTTLCache(2, 10, clk)
c.put('a',1); t[0]=1; c.put('b',2)
assert c.get('a')==1
c.put('c',3)  # evicts b (LRU since a was refreshed)
assert c.get('b') is None and c.get('a')==1 and c.get('c')==3
t[0]=10.5  # a expired (born 0, ttl 10); c alive (born 1)
assert c.get('a') is None
c.put('d',4); c.put('e',5)  # capacity 2: c evicted as LRU
assert c.get('c') is None and c.get('d')==4 and c.get('e')==5
t[0]=15.0
c.put('d',44)  # update resets ttl -> d now expires at 25
t[0]=21.0     # e (born 10.5) expired at 20.5; d alive until 25
assert c.get('d')==44 and c.get('e') is None
""",
    },
    "intervals": {
        "prompt": "Write a Python function `max_weight_schedule(intervals)` where intervals is a list of (start, end, weight) tuples (end exclusive; weights positive). Return a tuple `(total_weight, chosen)` where chosen is a list of the selected non-overlapping intervals (touching endpoints allowed, i.e. one may start exactly when another ends) achieving the maximum total weight, sorted by start. Use an O(n log n) DP; among equal-weight optima any valid answer is fine." + PROMPT_SUFFIX,
        "test": """
w,ch=max_weight_schedule([(1,4,2),(2,6,4),(5,7,4),(4,5,1)])
assert w==7, w
assert sorted(ch)==sorted([(1,4,2),(4,5,1),(5,7,4)]) or w==sum(c[2] for c in ch)
s=set(ch)
ch2=sorted(ch)
for i in range(len(ch2)-1): assert ch2[i][1]<=ch2[i+1][0]
w2,_=max_weight_schedule([])
assert w2==0
w3,c3=max_weight_schedule([(0,10,5)])
assert w3==5 and c3==[(0,10,5)]
w4,c4=max_weight_schedule([(0,3,3),(3,6,3),(6,9,3),(0,9,8)])
assert w4==9
import random
random.seed(7)
iv=[(s,s+random.randint(1,5),random.randint(1,9)) for s in [random.randint(0,50) for _ in range(200)]]
w5,c5=max_weight_schedule(iv)
c5=sorted(c5)
for i in range(len(c5)-1): assert c5[i][1]<=c5[i+1][0]
assert w5==sum(c[2] for c in c5)
""",
    },
    "logparse": {
        "prompt": r'''Write a Python function `parse_log(line)` parsing one nginx combined-format access log line: `ip - user [time] "request" status bytes "referer" "user_agent"`. Return a dict with keys ip, user, time, method, path, protocol, status (int), bytes (int, 0 if "-"), referer, user_agent. user is None if "-", referer None if "-". Quoted fields may contain escaped quotes (\") and escaped backslashes; unescape them. If the request line is malformed (not 3 space-separated parts), set method/path/protocol to None. Raise ValueError on lines that don't match the overall format.''' + PROMPT_SUFFIX,
        "test": r'''
d=parse_log('1.2.3.4 - alice [10/Aug/2026:12:00:00 +0000] "GET /x?a=1 HTTP/1.1" 200 512 "-" "Mozilla/5.0"')
assert d['ip']=='1.2.3.4' and d['user']=='alice' and d['method']=='GET' and d['path']=='/x?a=1'
assert d['protocol']=='HTTP/1.1' and d['status']==200 and d['bytes']==512 and d['referer'] is None
assert d['user_agent']=='Mozilla/5.0' and d['time']=='10/Aug/2026:12:00:00 +0000'
d2=parse_log('5.6.7.8 - - [01/Jan/2026:00:00:00 +0000] "POST /a b HTTP/2" 404 - "http://r.com" "UA \\" quote"')
assert d2['user'] is None and d2['bytes']==0 and d2['method'] is None and d2['path'] is None
assert d2['user_agent']=='UA " quote' and d2['referer']=='http://r.com'
d3=parse_log('9.9.9.9 - - [01/Jan/2026:00:00:00 +0000] "x" 200 1 "a\\\\b" "u"')
assert d3['referer']=='a\\b' and d3['method'] is None
try:
    parse_log('garbage'); assert False
except ValueError: pass
''',
    },
    "toposort": {
        "prompt": "Write a Python function `topo_sort(edges, nodes)` where nodes is an iterable of node names and edges a list of (u, v) pairs meaning u must come before v. Return a topological order as a list; when several nodes are available, always pick the lexicographically smallest (i.e. the unique deterministic order). Raise ValueError('cycle') if the graph has a cycle. Edges may mention nodes not in `nodes`; include them too." + PROMPT_SUFFIX,
        "test": """
assert topo_sort([('b','c'),('a','c')], ['a','b','c'])==['a','b','c']
assert topo_sort([], ['z','y'])==['y','z']
assert topo_sort([('c','a')], ['a','b','c'])==['b','c','a']
assert topo_sort([('x','q')], ['q'])==['x','q']
try:
    topo_sort([('a','b'),('b','a')], ['a','b']); assert False
except ValueError as e: assert str(e)=='cycle'
out=topo_sort([('a','d'),('b','d'),('d','e'),('c','e')], ['a','b','c','d','e'])
assert out==['a','b','c','d','e'], out
""",
    },
    "calc": {
        "prompt": "Write a Python function `evaluate(expr)` that evaluates an arithmetic expression string with integers, + - * / % operators, parentheses, and unary minus. Division and modulo must truncate toward zero (C semantics, NOT Python floor semantics): evaluate('7/-2')==-3, evaluate('-7%2')==-1. Normal precedence and left associativity. Whitespace anywhere. Raise ValueError on invalid syntax or division by zero. Return an int." + PROMPT_SUFFIX,
        "test": """
assert evaluate('1+2*3')==7
assert evaluate('(1+2)*3')==9
assert evaluate('7/-2')==-3
assert evaluate('-7%2')==-1
assert evaluate('7%-2')==1
assert evaluate('- (3 + 4) * 2')==-14
assert evaluate('2-3-4')==-5
assert evaluate('100/3/3')==11
assert evaluate('--5')==5
for bad in ['','1+','(1','2**3x','1/0','5%0']:
    try:
        evaluate(bad); assert False, bad
    except ValueError: pass
""",
    },
}


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text or "", re.S)
    return blocks[-1] if blocks else (text or "")


def run_asserts(code, test):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + test)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=30)
        return p.returncode == 0, (p.stderr or "")[-400:]
    except subprocess.TimeoutExpired:
        return False, "test timeout"
    finally:
        Path(path).unlink(missing_ok=True)


def one(config_name, cfg, task_name, task):
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": task["prompt"]}],
        "stream": False,
        "max_tokens": 32768,
        "chat_template_kwargs": cfg["kw"],
        **cfg["samp"],
    }
    t0 = time.time()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            data = json.loads(r.read())
    except Exception as e:
        return {"run": RUN_ID, "config": config_name, "task": task_name, "error": str(e)[:200]}
    wall = time.time() - t0
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    passed, err = run_asserts(extract_code(content), task["test"])
    return {
        "run": RUN_ID, "config": config_name, "task": task_name,
        "passed": passed, "wall_s": round(wall, 1),
        "completion_tokens": data["usage"]["completion_tokens"],
        "reasoning_chars": len(reasoning), "content_chars": len(content),
        "finish": data["choices"][0].get("finish_reason"),
        "err": "" if passed else err,
    }


def main():
    for config_name, cfg in CONFIGS.items():
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=6) as ex:
            futs = [ex.submit(one, config_name, cfg, tn, t) for tn, t in TASKS.items()]
            results = [f.result() for f in futs]
        batch_wall = round(time.time() - t0, 1)
        with OUT.open("a") as f:
            for r in results:
                r["batch_wall_s"] = batch_wall
                f.write(json.dumps(r) + "\n")
        npass = sum(1 for r in results if r.get("passed"))
        toks = sum(r.get("completion_tokens", 0) for r in results)
        print(f"[{RUN_ID}] {config_name}: {npass}/{len(results)} pass, "
              f"batch {batch_wall}s, {toks} completion tokens", flush=True)


if __name__ == "__main__":
    main()
