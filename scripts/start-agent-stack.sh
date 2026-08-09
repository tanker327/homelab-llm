#!/bin/bash
# Agent stack: both Qwen3.6 models resident at once (needs ~80GB VRAM).
#
#   port 5000  Qwen3.6-35B-A3B MTP  — 3 worker slots x 131072 ctx
#   port 5001  Qwen3.6-27B Q8_0 MTP — planner/judge, 2 slots x 131072 ctx
#
# Drive it with:  ./venv/bin/python clients/orchestrate.py "your task"
# Port 5000 keeps the same API as the plain server, so chat.py and the
# web UI still work while the stack is up.
#
# Stop the single-model server first (it also binds 5000):
#   sudo systemctl stop llama-server   # or: pkill -x llama-server

DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$DIR/llama.cpp/build/bin/llama-server"

"$BIN" \
  --model "$DIR/models/Qwen3.6-35B-A3B-MTP-MXFP4_MOE.gguf" \
  --alias Qwen3.6-35B-A3B \
  --host 0.0.0.0 \
  --port 5000 \
  --n-gpu-layers 99 \
  --ctx-size 393216 \
  --parallel 3 \
  --flash-attn on \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --reasoning-format deepseek &
WORKERS_PID=$!

"$BIN" \
  --model "$DIR/models/Qwen3.6-27B-MTP-Q8_0.gguf" \
  --alias Qwen3.6-27B \
  --host 0.0.0.0 \
  --port 5001 \
  --n-gpu-layers 99 \
  --ctx-size 262144 \
  --parallel 2 \
  --flash-attn on \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --reasoning-format deepseek &
PLANNER_PID=$!

trap 'kill $WORKERS_PID $PLANNER_PID 2>/dev/null' INT TERM
# If either server dies, take the other down too so systemd restarts the pair.
wait -n
STATUS=$?
kill $WORKERS_PID $PLANNER_PID 2>/dev/null
wait
exit $STATUS
