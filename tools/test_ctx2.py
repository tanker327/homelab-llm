#!/usr/bin/env python3
"""Test maximum context window - high range."""
import subprocess
import time
import urllib.request
import json

SERVER = "/home/tanker/projects/temp/llama.cpp/build/bin/llama-server"
MODEL = "/home/tanker/projects/temp/models/Qwen3.5-35B-A3B-Q4_K_M.gguf"

CTX_SIZES = [65536, 81920, 98304, 114688, 131072]


def test_ctx(ctx_size):
    print(f"\n--- Testing ctx_size={ctx_size} ({ctx_size//1024}K) ---")
    proc = subprocess.Popen(
        [
            SERVER,
            "--model", MODEL,
            "--host", "127.0.0.1", "--port", "5050",
            "--n-gpu-layers", "99",
            "--ctx-size", str(ctx_size),
            "--flash-attn", "on",
            "--reasoning-format", "deepseek",
            "-np", "1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    started = False
    for i in range(60):
        time.sleep(2)
        if proc.poll() is not None:
            output = proc.stdout.read().decode()
            if "out of memory" in output.lower() or "CUDA" in output:
                print(f"  OOM at {ctx_size} ({ctx_size//1024}K)")
            else:
                print(f"  FAILED (code {proc.returncode})")
                for line in output.strip().split("\n")[-3:]:
                    print(f"    {line}")
            return False
        try:
            req = urllib.request.Request("http://127.0.0.1:5050/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if b"ok" in resp.read():
                    started = True
                    break
        except Exception:
            continue

    if not started:
        proc.kill()
        proc.wait()
        print(f"  Timeout")
        return False

    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    used, total = [int(x.strip()) for x in smi.split(",")]
    print(f"  OK: VRAM={used}/{total} MiB (free={total-used} MiB)")

    try:
        req_data = json.dumps({"model": "test", "messages": [{"role": "user", "content": "hi"}]}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:5050/v1/chat/completions",
            data=req_data, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            speed = data["timings"]["predicted_per_second"]
            print(f"  Inference: {speed:.1f} tok/s")
    except Exception as e:
        print(f"  Inference failed: {e}")

    proc.terminate()
    proc.wait()
    time.sleep(3)
    return True


last_ok = 49152
for ctx in CTX_SIZES:
    if test_ctx(ctx):
        last_ok = ctx
    else:
        break

print(f"\n=== Maximum working context: {last_ok} ({last_ok//1024}K tokens) ===")
