#!/usr/bin/env bash
# maintenance/download-coder.sh
# Downloads ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF (IQ1_M: 27.58 GB weights)
# Reuses the identical 26.82 GB PLE table from IQ3_S via hardlink to save 27 GB of disk and network.
set -euo pipefail

DEST_DIR="/home/kchauhan/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/IQ1_M"
IQ3_S_DIR="/home/kchauhan/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_S"
mkdir -p "${DEST_DIR}"
cd "${DEST_DIR}"

BASE_URL="https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/resolve/main/IQ1_M"

echo "=== Staging ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO Coder (IQ1_M) ==="
echo "Target directory: ${DEST_DIR}"

# Shard 1: Coder Weights & MoE Backbone (27.58 GB)
if [[ ! -f "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf" || -f "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf.part" ]]; then
    echo "[1/2] Downloading Coder Shard 1 (27.58 GB weights)..."
    wget -c --progress=dot:giga "${BASE_URL}/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
else
    echo "[1/2] Coder Shard 1 already present."
fi

# Shard 2: PLE Table (26.82 GB) - Check if IQ3_S table can be hardlinked
TARGET_SHARD2_BYTES=28800138432
if [[ ! -f "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf" ]]; then
    if [[ -f "${IQ3_S_DIR}/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" ]]; then
        iq3_s_size=$(stat -c %s "${IQ3_S_DIR}/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" 2>/dev/null || echo 0)
        if [[ "${iq3_s_size}" -ge "${TARGET_SHARD2_BYTES}" ]]; then
            echo "[2/2] Linking identical 26.82 GB PLE table from IQ3_S (zero disk duplication)..."
            ln "${IQ3_S_DIR}/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf"
        else
            echo "[2/2] Waiting for IQ3_S Shard 2 download to complete (${iq3_s_size} / ${TARGET_SHARD2_BYTES} bytes)..."
            while [[ "${iq3_s_size}" -lt "${TARGET_SHARD2_BYTES}" ]]; do
                sleep 10
                iq3_s_size=$(stat -c %s "${IQ3_S_DIR}/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" 2>/dev/null || echo 0)
                printf "  Waiting for PLE Shard 2: %'d / %'d bytes\r" "${iq3_s_size}" "${TARGET_SHARD2_BYTES}"
            done
            echo ""
            echo "[2/2] Linking completed 26.82 GB PLE table from IQ3_S (zero disk duplication)..."
            ln "${IQ3_S_DIR}/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf"
        fi
    else
        echo "[2/2] Downloading Shard 2 PLE table..."
        wget -c --progress=dot:giga "${BASE_URL}/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf"
    fi
else
    echo "[2/2] Shard 2 already present."
fi


echo "=== Coder IQ1_M staging complete ==="
ls -lh "${DEST_DIR}"
