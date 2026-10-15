#!/usr/bin/env bash
# Restore an archived media model from NAS back to local fast NVMe.
# Usage: ./maintenance/restore-media-model.sh <model_name>
set -euo pipefail

NAS_ROOT="/mnt/storage/models/media"
LOCAL_ROOT="$HOME/models/media"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model_name>"
    echo "Available archived models on NAS:"
    ls -1 "$NAS_ROOT" 2>/dev/null || echo "No models found at $NAS_ROOT"
    exit 1
fi

MODEL="$1"
SRC="$NAS_ROOT/$MODEL"
DST="$LOCAL_ROOT/$MODEL"

if [[ ! -d "$SRC" ]]; then
    echo "Error: Model not found on NAS: $SRC" >&2
    exit 1
fi

mkdir -p "$LOCAL_ROOT"

echo "=== Restoring $MODEL from NAS ==="
echo "From: $SRC ($(du -sh "$SRC" | cut -f1))"
echo "To:   $DST"

rsync -rltD --info=progress2 "$SRC" "$LOCAL_ROOT/"

echo "Verification:"
du -sh "$DST"
echo "[SUCCESS] Restored $MODEL to $LOCAL_ROOT"
