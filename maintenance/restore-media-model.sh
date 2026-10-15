#!/usr/bin/env bash
# Restore an archived model without replacing any existing local path.
set -euo pipefail
NAS_ROOT="${L2M2_MEDIA_NAS_ROOT:-/mnt/storage/models/media}"
LOCAL_ROOT="${L2M2_MEDIA_LOCAL_ROOT:-$HOME/models/media}"
NAS_MOUNT="${L2M2_MEDIA_NAS_MOUNT:-/mnt/storage}"
[[ $# == 1 && "$1" != .* && "$1" != */* ]] || {
    echo "Usage: $0 <model_name> (one directory name)" >&2; exit 1;
}
MODEL="$1"
findmnt -rnC -M "$NAS_MOUNT" -t noautofs -o FSTYPE >/dev/null || {
    echo "NAS is unavailable or only an autofs placeholder." >&2; exit 1;
}
mountpoint -q -- "$NAS_MOUNT" &&
    [[ ! -L "$NAS_ROOT" && ! -L "$NAS_MOUNT" && "$(realpath -e -- "$NAS_ROOT")" == "$(realpath -e -- "$NAS_MOUNT")/"* ]] || {
    echo "NAS is not mounted with a real archive root." >&2; exit 1;
}
SRC="$NAS_ROOT/$MODEL"
DST="$LOCAL_ROOT/$MODEL"
[[ -d "$SRC" && ! -L "$SRC" ]] || { echo "Invalid archived model: $SRC" >&2; exit 1; }
[[ ! -e "$DST" && ! -L "$DST" ]] || { echo "Preserving existing local path: $DST" >&2; exit 1; }
mkdir -p -- "$LOCAL_ROOT"
[[ ! -L "$LOCAL_ROOT" ]] || { echo "Local root must be a real directory" >&2; exit 1; }
# Keep a failed/incomplete copy for inspection; never overwrite a local model.
stage="$(mktemp -d -- "$LOCAL_ROOT/.restore-$MODEL.XXXXXXXX")"
echo "Restoring into $stage"
rsync -rltD --info=progress2 -- "$SRC/" "$stage/"
changes="$(rsync -rclnD --delete --itemize-changes -- "$SRC/" "$stage/")"
[[ -z "$changes" ]] || { echo "Restore verification failed; staged copy retained: $stage" >&2; exit 1; }
mv -T -n -- "$stage" "$DST"
[[ ! -d "$stage" ]] || { echo "Destination appeared; staged copy retained: $stage" >&2; exit 1; }
echo "[SUCCESS] Restored $MODEL to $DST"
