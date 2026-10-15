#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== l3ds Environment Setup ==="
cd "$ROOT_DIR"

if ! command -v uv &>/dev/null; then
    echo "Error: 'uv' is required but not installed." >&2
    exit 1
fi

if [ ! -d ".venv-decisions" ]; then
    echo "Creating virtual environment with Python 3.12..."
    uv venv --python 3.12 .venv-decisions
fi

echo "Installing PyTorch with CUDA..."
uv pip install --python .venv-decisions torch --index-url https://download.pytorch.org/whl/cu124

echo "Installing Nagi, Hugging Face, and dependencies..."
uv pip install --python .venv-decisions \
    transformers \
    peft \
    huggingface_hub \
    pyyaml \
    numpy \
    accelerate \
    "nagi-decisions @ git+https://github.com/nagisanzenin/nagi.git"

echo "Building release binary for l3ds CLI..."
cargo build --release

echo "=== l3ds is ready! ==="
l3ms --list stack
