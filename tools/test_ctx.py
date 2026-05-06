#!/usr/bin/env python3
"""Test maximum context window for llama-server on RTX 4090."""
import subprocess
import time
import urllib.request
import json
import signal

SERVER = "/home/tanker/projects/temp/llama.cpp/build/bin/llama-server"
MODEL = "/home/tanker/projects/temp/models/Qwen3.5-35B-A3B-Q4_K_M.gguf"

# Test these context sizes
CTX_SIZES = [8192, 16384, 24576, 32768, 40960, 49152]


def test_ctx(ctx_size):
    print(f"\n--- Testing ctx_size={ctx_size} ---")
    proc = subprocess.Popen(
        [
            SERVER,
            "--model", MODEL,
            "--host", "127.0.0.1", "--port", "5050",
            "--n-gpu-layers", "99",
            "--ctx-size", str(ctx_size),
            "--flash-attn", "on",
            "--reasoning-format", "deepseek",
            "-np", "1",  # single slot to minimize VRAM
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for server to be ready or fail
    started = False
    for i in range(60):
        time.sleep(2)
        if proc.poll() is not None:
            output = proc.stdout.read().decode()
            if "CUDA out of memory" in output or "out of memory" in output.lower():
                print(f"  FAILED: OOM at ctx_size={ctx_size}")
            else:
                print(f"  FAILED: Server exited (code {proc.returncode})")
                # Print last few lines for debugging
                for line in output.strip().split("\n")[-5:]:
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
        print(f"  FAILED: Server didn't start in time")
        proc.kill()
        proc.wait()
        return False

    # Get VRAM usage
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        used, total = [int(x.strip()) for x in smi.split(",")]
        free = total - used
        print(f"  OK: ctx_size={ctx_size}, VRAM={used}/{total} MiB (free={free} MiB)")
    except Exception:
        print(f"  OK: ctx_size={ctx_size}")

    # Quick test to make sure it actually works
    try:
        req_data = json.dumps({"model": "test", "messages": [{"role": "user", "content": "hi"}]}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:5050/v1/chat/completions",
            data=req_data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            toks = data["usage"]["completion_tokens"]
            speed = data["timings"]["predicted_per_second"]
            print(f"  Inference OK: {toks} tokens at {speed:.1f} tok/s")
    except Exception as e:
        print(f"  Inference failed: {e}")

    proc.terminate()
    proc.wait()
    time.sleep(3)
    return True


def main():
    print("=== Context Window Test ===")
    print(f"Model: Qwen3.5-35B-A3B-Q4_K_M.gguf")
    print(f"GPU: RTX 4090 (24GB)")
    print(f"Testing: {CTX_SIZES}")

    last_ok = 0
    for ctx in CTX_SIZES:
        ok = test_ctx(ctx)
        if ok:
            last_ok = ctx
        else:
            break

    print(f"\n=== Result ===")
    print(f"Maximum working context: {last_ok}")


if __name__ == "__main__":
    main()
