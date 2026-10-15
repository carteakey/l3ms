#!/usr/bin/env bash
# Archive selected media models only after checksum verification on a mounted NAS.
set -euo pipefail
DEST_ROOT="${L2M2_MEDIA_NAS_ROOT:-/mnt/storage/models/media}"
SRC_ROOT="${L2M2_MEDIA_LOCAL_ROOT:-$HOME/models/media}"
NAS_MOUNT="${L2M2_MEDIA_NAS_MOUNT:-/mnt/storage}"
check_nas() {
    # Consult the kernel mount table without resolving/statting the path: probing
    # an autofs placeholder can hang while the NAS is offline.
    if ! findmnt -rnC -M "$NAS_MOUNT" -t noautofs -o FSTYPE >/dev/null; then
        echo "[ERROR] NAS is unavailable or only an autofs placeholder; retaining local models." >&2
        exit 1
    fi
    [[ ! -L "$DEST_ROOT" && ! -L "$NAS_MOUNT" ]] &&
        mountpoint -q -- "$NAS_MOUNT" &&
        [[ "$(realpath -e -- "$DEST_ROOT")" == "$(realpath -e -- "$NAS_MOUNT")/"* ]] || {
        echo "[ERROR] NAS must be mounted with a real destination below it; retaining local models." >&2
        exit 1
    }
}
check_source_mounts() {
    local mounts mounted
    mounts="$(findmnt -rn -o TARGET)"
    while IFS= read -r mounted; do
        # findmnt --raw hex-escapes whitespace and backslashes in mount paths.
        printf -v mounted '%b' "$mounted"
        if [[ "$mounted" == "$src" || "$mounted" == "$src/"* ]]; then
            echo "Refusing to remove a source containing a mount: $mounted" >&2
            exit 1
        fi
    done <<< "$mounts"
}
# Do not create directories on an unmounted NAS or migrate old archives implicitly.
check_nas
[[ -d "$SRC_ROOT" && ! -L "$SRC_ROOT" ]] || { echo "Invalid local model root" >&2; exit 1; }
SRC_ROOT="$(realpath -e -- "$SRC_ROOT")"
DEST_ROOT="$(realpath -e -- "$DEST_ROOT")"
[[ "$SRC_ROOT" != "$DEST_ROOT" && "$SRC_ROOT/" != "$DEST_ROOT/"* && "$DEST_ROOT/" != "$SRC_ROOT/"* ]] || exit 1
MODELS=(HeartMuLa-GGUF ltx-2.5 Krea-2 MiniMax-H3-Q4-GGUF Z-Image-Turbo MiniMax-H3-ComfyUI)
for model in "${MODELS[@]}"; do
    src="$SRC_ROOT/$model"
    dst="$DEST_ROOT/$model"
    if [[ -L "$src" || ! -e "$src" ]]; then
        echo "[SKIP] Missing or already symlinked: $src"
        continue
    fi
    [[ -d "$src" && ! -L "$dst" ]] || { echo "Unsafe model path; retaining $src" >&2; exit 1; }
    check_source_mounts
    check_nas
    rsync -rltD --ignore-existing --info=progress2 -- "$src" "$DEST_ROOT/"
    check_nas
    [[ -d "$dst" && ! -L "$dst" ]] || exit 1
    # Checksums detect changed bytes; --delete also reports unexpected archive entries.
    changes="$(rsync -rclnD --delete --itemize-changes -- "$src/" "$dst/")"
    if [[ -n "$changes" ]]; then
        printf '[ERROR] Archive differs; retaining %s:\n%s\n' "$src" "$changes" >&2
        exit 1
    fi
    check_nas
    [[ -d "$src" && ! -L "$src" && "$(realpath -e -- "$src")" == "$SRC_ROOT/$model" ]] || exit 1
    check_source_mounts
    rm -rf --one-file-system --preserve-root -- "$src"
    echo "[SUCCESS] Checksum-verified archive: $dst"
done
