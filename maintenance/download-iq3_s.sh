#!/usr/bin/env bash
# maintenance/download-iq3_s.sh
# Downloads ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF (IQ3_S: 51.05 GB weights + 26.82 GB PLE)
set -euo pipefail

DEST_DIR="/home/kchauhan/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_S"
mkdir -p "${DEST_DIR}"
cd "${DEST_DIR}"

BASE_URL="https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/IQ3_S"

echo "=== Staging ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO IQ3_S ==="
echo "Target directory: ${DEST_DIR}"

# Shard 1: Weights & MoE Backbone (51.05 GB)
if [[ ! -f "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" || -f "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf.part" ]]; then
    echo "[1/2] Downloading Shard 1 (51.05 GB weights)..."
    wget -c --progress=dot:giga "${BASE_URL}/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
else
    echo "[1/2] Shard 1 already present."
fi

# Shard 2: PLE 51B N-Gram Table on SSD (26.82 GB)
if [[ ! -f "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" || -f "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf.part" ]]; then
    echo "[2/2] Downloading Shard 2 (26.82 GB PLE table)..."
    wget -c --progress=dot:giga "${BASE_URL}/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"
else
    echo "[2/2] Shard 2 already present."
fi

echo "=== IQ3_S download complete. Verifying file sizes ==="
ls -lh "${DEST_DIR}"
