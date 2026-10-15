#!/usr/bin/env bash
# Archive selected media models to NAS (/mnt/storage/archive/media-models)
# and replace local directories with symlinks to free NVMe disk space.
set -euo pipefail

DEST_ROOT="/mnt/storage/models/media"
SRC_ROOT="$HOME/models/media"

# Migrate any initial test transfers from /mnt/storage/archive/media-models
if [[ -d "/mnt/storage/archive/media-models" && ! -d "$DEST_ROOT" ]]; then
    mkdir -p "/mnt/storage/models"
    mv "/mnt/storage/archive/media-models" "$DEST_ROOT"
fi
mkdir -p "$DEST_ROOT"

MODELS=(
    "HeartMuLa-GGUF"
    "ltx-2.5"
    "Krea-2"
    "MiniMax-H3-Q4-GGUF"
    "Z-Image-Turbo"
    "MiniMax-H3-ComfyUI"
)

echo "=== Media Models NAS Archiving Started: $(date) ==="
df -h "$SRC_ROOT"

for model in "${MODELS[@]}"; do
    src="$SRC_ROOT/$model"
    dst="$DEST_ROOT/$model"

    if [[ ! -e "$src" ]]; then
        echo "[SKIP] Source does not exist: $src"
        continue
    fi

    if [[ -L "$src" ]]; then
        echo "[SKIP] Already a symlink: $src -> $(readlink "$src")"
        continue
    fi

    echo "--- Archiving: $model ---"
    echo "Source: $src ($(du -sh "$src" | cut -f1))"
    echo "Destination: $dst"

    rsync -rltD --info=progress2 "$src" "$DEST_ROOT/"

    echo "Verifying transfer for $model..."
    src_size=$(du -s "$src" | cut -f1)
    dst_size=$(du -s "$dst" | cut -f1)

    echo "Source size: $src_size KB | Dest size: $dst_size KB"

    # Allow minor filesystem metadata delta (< 2% or 100MB)
    size_diff=$(( src_size > dst_size ? src_size - dst_size : dst_size - src_size ))
    max_allowed_diff=$(( src_size / 50 + 102400 ))

    if [[ "$size_diff" -le "$max_allowed_diff" ]]; then
        echo "Verification passed. Removing local source $src..."
        rm -rf "$src"
        echo "[SUCCESS] Archived to NAS and removed local: $model"
        df -h "$SRC_ROOT"
    else
        echo "[ERROR] Size mismatch between $src ($src_size) and $dst ($dst_size). Retaining source!" >&2
        exit 1
    fi
done

echo "=== Media Models NAS Archiving Complete: $(date) ==="
df -h "$SRC_ROOT"
