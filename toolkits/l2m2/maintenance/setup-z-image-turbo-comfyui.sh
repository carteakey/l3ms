#!/usr/bin/env bash
# Download pinned official Z-Image-Turbo ComfyUI weights and setup workflows.
set -euo pipefail

root="$(cd -- "$(dirname -- "$BASH_SOURCE")/.." && pwd)"
workspace="$root/vendor/ComfyUI"
manifest="$root/models/z-image-turbo-comfyui-files.txt"
comfy_revision="20b46065ef7f34fa21e8abebe8d0cca90a904d6d"
workflow_dir="$root/workflows/z-image-turbo"
py="$root/.venv-comfy/bin/python"

model_root="$(printenv L2M2_Z_IMAGE_TURBO_ROOT || true)"
if [[ -z "$model_root" ]]; then
    if [[ -d "/run/media/kchauhan/Games/models/media" ]]; then
        model_root="/run/media/kchauhan/Games/models/media/Z-Image-Turbo"
    else
        model_root="$HOME/models/media/Z-Image-Turbo"
    fi
fi
case "$model_root" in
    /*) ;;
    *) model_root="$PWD/$model_root" ;;
esac
mkdir -p "$model_root" "$root/.toolkit/model_config_versions" "$workflow_dir"
model_root="$(cd -- "$model_root" && pwd -P)"

[[ -d "$workspace/.git" ]] || {
    echo "ComfyUI checkout missing; run maintenance/setup-comfyui.sh first." >&2
    exit 1
}
installed_revision="$(git -C "$workspace" rev-parse HEAD)"
[[ "$installed_revision" == "$comfy_revision" ]] || {
    echo "ComfyUI must be pinned to $comfy_revision; run maintenance/setup-comfyui.sh." >&2
    exit 1
}
[[ -x "$py" ]] || {
    echo "ComfyUI Python environment missing; run maintenance/setup-comfyui.sh first." >&2
    exit 1
}
[[ -f "$manifest" ]] || { echo "Missing model manifest: $manifest" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "sha256sum is required." >&2; exit 1; }

stamp="$(date +%Y%m%d-%H%M%S)"
cp "$manifest" "$root/.toolkit/model_config_versions/z-image-turbo-comfyui-files-$stamp.txt"

missing_bytes=0
while read -r repo revision repo_file relative expected_size expected_sha; do
    [[ -n "$repo" && "$repo" != \#* ]] || continue
    path="$model_root/$relative"
    if [[ -f "$path" ]]; then
        actual_size="$(stat -c %s "$path")"
        actual_sha="$(sha256sum "$path" | awk '{print $1}')"
        if [[ "$actual_size" != "$expected_size" || "$actual_sha" != "$expected_sha" ]]; then
            echo "Existing Z-Image-Turbo file differs from the pinned artifact; preserving it: $path" >&2
            exit 1
        fi
    elif [[ -e "$path" || -L "$path" ]]; then
        echo "Existing Z-Image-Turbo path is not a regular file; preserving it: $path" >&2
        exit 1
    else
        missing_bytes=$((missing_bytes + expected_size))
    fi
done < "$manifest"

available_kib="$(df -Pk "$model_root" | awk 'END {print $4}')"
required_bytes=$((missing_bytes + 1073741824))
if (( available_kib * 1024 < required_bytes )); then
    printf 'Insufficient space: need %s bytes including a 1 GiB margin; have %s bytes.\n' \
        "$required_bytes" "$((available_kib * 1024))" >&2
    printf 'Set L2M2_Z_IMAGE_TURBO_ROOT to specify a filesystem with sufficient space.\n' >&2
    exit 1
fi

while read -r repo revision repo_file relative expected_size expected_sha; do
    [[ -n "$repo" && "$repo" != \#* ]] || continue
    path="$model_root/$relative"
    target_dir="$(dirname -- "$path")"
    mkdir -p "$target_dir"
    if [[ ! -f "$path" ]]; then
        echo "Downloading $repo_file from $repo at $revision -> $path"
        "$py" -c "
import os, shutil
from huggingface_hub import hf_hub_download

repo = '$repo'
revision = '$revision'
repo_file = '$repo_file'
final_path = '$path'
model_root = '$model_root'
cache_dir = os.path.join(model_root, '.cache', 'huggingface')

downloaded = hf_hub_download(
    repo_id=repo,
    filename=repo_file,
    revision=revision,
    local_dir=model_root,
    cache_dir=cache_dir
)
if downloaded != final_path and os.path.exists(downloaded):
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    shutil.move(downloaded, final_path)
"
    fi
    actual_size="$(stat -c %s "$path")"
    actual_sha="$(sha256sum "$path" | awk '{print $1}')"
    if [[ "$actual_size" != "$expected_size" || "$actual_sha" != "$expected_sha" ]]; then
        echo "Pinned Z-Image-Turbo file failed size/SHA-256 verification: $path" >&2
        exit 1
    fi

    target="$workspace/models/$relative"
    mkdir -p "$(dirname -- "$target")"
    if [[ -L "$target" && "$(readlink -f "$target")" == "$(readlink -f "$path")" ]]; then
        continue
    fi
    if [[ -e "$target" || -L "$target" ]]; then
        echo "Refusing to replace an existing ComfyUI model path: $target" >&2
        exit 1
    fi
    ln -s -- "$path" "$target"
done < "$manifest"

sum_file="$model_root/Z-IMAGE-TURBO-SHA256SUMS"
sum_temp="$(mktemp "$model_root/.z-image-turbo-sums.XXXXXX")"
(
    cd -- "$model_root"
    while read -r repo revision repo_file relative expected_size expected_sha; do
        [[ -n "$repo" && "$repo" != \#* ]] || continue
        sha256sum "$relative"
    done < "$manifest"
) > "$sum_temp"
if [[ -e "$sum_file" ]]; then
    if ! cmp -s "$sum_temp" "$sum_file"; then
        rm -- "$sum_temp"
        echo "Existing checksum file differs; preserving it: $sum_file" >&2
        exit 1
    fi
    rm -- "$sum_temp"
else
    mv -- "$sum_temp" "$sum_file"
fi

echo "Z-Image-Turbo ComfyUI assets installed at $model_root"
