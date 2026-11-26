#!/usr/bin/env bash
# maintenance/download-nemotron-3.5-bolt.sh
# Downloads nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF (20.51 GiB + mmproj 1.61 GiB)
# and nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF (0.93 GiB DFlash drafter)
# to /mnt/dev/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF. Resumable; preserves existing files.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname -- "$0")/.." && pwd)"
HF_BIN="${HF_BIN:-${REPO_ROOT}/.venv/bin/hf}"
DEST_DIR="${DEST_DIR:-/mnt/dev/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF}"

if ! mountpoint -q /mnt/media; then
    echo "ERROR: /mnt/media is not mounted. Mount it before downloading." >&2
    exit 1
fi

if [[ ! -x "${HF_BIN}" ]]; then
    echo "ERROR: hf CLI not found at ${HF_BIN}" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"
echo "=== Downloading NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB + mmproj ==="
echo "Target: ${DEST_DIR}"

"${HF_BIN}" download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF \
    NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf \
    mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf \
    --local-dir "${DEST_DIR}"

echo "=== Downloading LiLiCorr DFlash Drafter ==="
"${HF_BIN}" download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF \
    NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf \
    --local-dir "${DEST_DIR}"

echo "=== Nemotron 3.5 Bolt 74B-A9B download complete ==="
ls -lh "${DEST_DIR}"
