#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=== llama.cpp + Qwen3.5-35B-A3B Setup ==="
echo ""

# 1. Install dependencies
echo "[1/5] Installing system dependencies..."
sudo apt install -y cmake build-essential

# 2. Install uv (if needed)
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
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
    cmake --build build --config Release -j$(nproc) -- llama-server
    cd "$DIR"
else
    echo "[3/5] llama-server already built"
fi

# 4. Download model
if [ ! -f "$DIR/models/Qwen3.5-35B-A3B-Q4_K_M.gguf" ]; then
    echo "[4/5] Downloading model (~20.5GB)..."
    mkdir -p "$DIR/models"
    if [ ! -d "$DIR/venv" ]; then
        uv venv --python 3.12 venv
    fi
    uv pip install --python venv/bin/python huggingface-hub[hf_xet]
    venv/bin/hf download unsloth/Qwen3.5-35B-A3B-GGUF \
        Qwen3.5-35B-A3B-Q4_K_M.gguf \
        --local-dir "$DIR/models"
else
    echo "[4/5] Model already downloaded"
fi

# 5. Set up chat client
if [ ! -d "$DIR/venv" ]; then
    echo "[5/5] Setting up Python venv..."
    uv venv --python 3.12 venv
fi
uv pip install --python venv/bin/python openai

# Open firewall if ufw is active
if sudo ufw status 2>/dev/null | grep -q "active"; then
    sudo ufw allow from 192.168.10.0/24 to any port 5000 proto tcp comment "llama.cpp server" 2>/dev/null || true
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Start server:  ./start-llama.sh"
echo "Chat:          ./venv/bin/python chat.py"
echo "Web UI:        http://localhost:5000"
echo "API:           http://localhost:5000/v1/chat/completions"
