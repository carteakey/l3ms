#!/usr/bin/env bash
# maintenance/download-inkling-2-mini.sh
# Downloads unsloth/Inkling-2-Mini-GGUF (UD-Q4_K_XL 16.42 GiB + mmproj-F16 0.60 GiB + DSpark Q8_0 0.37 GiB)
# to /mnt/dev/models/unsloth/Inkling-2-Mini-GGUF. Resumable; preserves existing files.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname -- "$0")/.." && pwd)"
HF_BIN="${HF_BIN:-${REPO_ROOT}/.venv/bin/hf}"
DEST_DIR="${DEST_DIR:-/mnt/dev/models/unsloth/Inkling-2-Mini-GGUF}"

if ! mountpoint -q /mnt/media; then
    echo "ERROR: /mnt/media is not mounted. Mount it before downloading." >&2
    exit 1
fi

if [[ ! -x "${HF_BIN}" ]]; then
    echo "ERROR: hf CLI not found at ${HF_BIN}" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"
echo "=== Downloading Inkling 2 Mini (UD-Q4_K_XL + mmproj-F16 + DSpark Q8_0) ==="
echo "Target: ${DEST_DIR}"

"${HF_BIN}" download unsloth/Inkling-2-Mini-GGUF \
    Inkling-2-Mini-UD-Q4_K_XL.gguf \
    mmproj-F16.gguf \
    dspark-Inkling-2-Mini-Q8_0.gguf \
    --local-dir "${DEST_DIR}"

echo "=== Inkling 2 Mini download complete ==="
ls -lh "${DEST_DIR}"
