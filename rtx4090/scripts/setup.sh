#!/bin/bash
# RTX 4090 (24GB) server setup: build llama.cpp for this GPU, download the
# default GGUF, create the client venv, open the firewall. Idempotent.
# Shared assets (llama.cpp/, models/, venv/, toolchain-fix/) live at the
# REPO ROOT, two levels up from this script.
set -e

DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$DIR"

echo "=== RTX 4090: llama.cpp + Qwen3.6-35B-A3B MXFP4_MOE setup ==="
echo "repo root: $DIR"
echo ""

# 1. System dependencies
echo "[1/5] Installing system dependencies..."
sudo apt install -y cmake build-essential

# CUDA toolkit: nvcc is needed to build llama.cpp with GGML_CUDA (SM89).
if ! command -v nvcc &>/dev/null && ! ls /usr/local/cuda*/bin/nvcc &>/dev/null; then
    echo "Installing CUDA toolkit 13.1..."
    sudo apt install -y cuda-toolkit-13-1
fi
NVCC="$(command -v nvcc || ls -1 /usr/local/cuda*/bin/nvcc 2>/dev/null | sort -V | tail -1)"

# 2. uv
if ! command -v uv &>/dev/null; then
    echo "[2/5] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[2/5] uv already installed"
fi

# 3. Build llama.cpp
if [ ! -f "$DIR/llama.cpp/build/bin/llama-server" ]; then
    echo "[3/5] Building llama.cpp..."
    if [ ! -d "$DIR/llama.cpp" ]; then
        git clone https://github.com/ggml-org/llama.cpp
    fi
    cd "$DIR/llama.cpp"
    # Ubuntu 26.04's glibc >= 2.42 declares C23 rsqrt/rsqrtf with noexcept,
    # which clashes with CUDA <= 13.1 math headers. Shadow the header with a
    # copy that skips rsqrt under __CUDACC__ and feed it to nvcc only.
    MATHCALLS=/usr/include/x86_64-linux-gnu/bits/mathcalls.h
    if grep -q '^__MATHCALL_VEC (rsqrt' "$MATHCALLS" 2>/dev/null; then
        mkdir -p "$DIR/toolchain-fix/bits"
        sed 's|^__MATHCALL_VEC (rsqrt,, (_Mdouble_ __x));|#ifndef __CUDACC__\n&\n#endif|' \
            "$MATHCALLS" > "$DIR/toolchain-fix/bits/mathcalls.h"
        export NVCC_PREPEND_FLAGS="-I$DIR/toolchain-fix ${NVCC_PREPEND_FLAGS:-}"
    fi
    # RTX 4090 -> 89. Auto-detect so the same script works elsewhere.
    CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
    cmake -B build -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-89}" \
        -DCMAKE_CUDA_COMPILER="$NVCC"
    cmake --build build --config Release -j"$(nproc)" -- llama-server
    cd "$DIR"
else
    echo "[3/5] llama-server already built"
fi

# 4. Default model (what rtx4090/systemd/llama-server.service runs)
MODEL_FILE="Qwen3.6-35B-A3B-MXFP4_MOE.gguf"
if [ ! -f "$DIR/models/$MODEL_FILE" ]; then
    echo "[4/5] Downloading $MODEL_FILE (~20.2GB)..."
    mkdir -p "$DIR/models"
    if [ ! -d "$DIR/venv" ]; then
        uv venv --python 3.12 venv
    fi
    uv pip install --python venv/bin/python "huggingface-hub[hf_xet]"
    venv/bin/hf download unsloth/Qwen3.6-35B-A3B-GGUF \
        "$MODEL_FILE" \
        --local-dir "$DIR/models"
else
    echo "[4/5] Model already downloaded"
fi

# 5. Client venv
if [ ! -d "$DIR/venv" ]; then
    echo "[5/5] Setting up Python venv..."
    uv venv --python 3.12 venv
fi
uv pip install --python venv/bin/python openai "huggingface-hub[hf_xet]"

# Firewall: LAN only
if sudo ufw status 2>/dev/null | grep -q "active"; then
    sudo ufw allow from 192.168.10.0/24 to any port 5000 proto tcp comment "llama.cpp server" 2>/dev/null || true
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Start server:  ./rtx4090/scripts/start-llama-35b-moe.sh"
echo "Systemd:       sudo cp rtx4090/systemd/llama-server.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now llama-server"
echo "Chat:          ./venv/bin/python clients/chat.py"
echo "Web UI:        http://localhost:5000"
echo "API:           http://localhost:5000/v1/chat/completions"
