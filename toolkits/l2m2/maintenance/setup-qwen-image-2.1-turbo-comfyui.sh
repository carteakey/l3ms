#!/usr/bin/env bash
# Download pinned Qwen Image 2.1 Turbo ComfyUI INT8 ConvRot weights.
set -euo pipefail

root="$(cd -- "$(dirname -- "$BASH_SOURCE")/.." && pwd)"
downloader="$root/model_downloader/download_hf_model.py"
if [[ ! -f "$downloader" ]]; then
    downloader="$root/../../model_downloader/download_hf_model.py"
fi
[[ -f "$downloader" ]] || { echo "Model downloader missing: $downloader" >&2; exit 1; }
workspace="$root/vendor/ComfyUI"
manifest="$root/models/qwen-image-2.1-turbo-comfyui-files.txt"
comfy_revision="6bfaacc67c2103481e5f0c84d75257cd0581d86a"
hf="$root/.venv-comfy/bin/hf"

model_root="$(printenv L2M2_QWEN_IMAGE_21_TURBO_ROOT || printenv L2M2_QWEN_IMAGE_21_ROOT || true)"
if [[ -z "$model_root" ]]; then
    model_root="$HOME/models/media/Qwen-Image-2.1"
fi
case "$model_root" in
    /*) ;;
    *) model_root="$PWD/$model_root" ;;
esac
mkdir -p "$model_root" "$root/.toolkit/model_config_versions"
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
[[ -x "$hf" ]] || {
    echo "ComfyUI Python environment missing; run maintenance/setup-comfyui.sh first." >&2
    exit 1
}
[[ -f "$manifest" ]] || { echo "Missing model manifest: $manifest" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "sha256sum is required." >&2; exit 1; }

stamp="$(date +%Y%m%d-%H%M%S)"
cp "$manifest" "$root/.toolkit/model_config_versions/qwen-image-2.1-turbo-comfyui-files-$stamp.txt"

missing_bytes=0
while read -r repo revision repo_file relative expected_size expected_sha; do
    [[ -n "$repo" && "$repo" != \#* ]] || continue
    path="$model_root/$relative"
    if [[ -f "$path" ]]; then
        actual_size="$(stat -c %s "$path")"
        actual_sha="$(sha256sum "$path" | awk '{print $1}')"
        if [[ "$actual_size" != "$expected_size" || "$actual_sha" != "$expected_sha" ]]; then
            echo "Existing Qwen Turbo file differs from the pinned artifact; preserving it: $path" >&2
            exit 1
        fi
    elif [[ -e "$path" || -L "$path" ]]; then
        echo "Existing Qwen Turbo path is not a regular file; preserving it: $path" >&2
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
    exit 1
fi

while read -r repo revision repo_file relative expected_size expected_sha; do
    [[ -n "$repo" && "$repo" != \#* ]] || continue
    path="$model_root/$relative"
    target_dir="$(dirname -- "$path")"
    mkdir -p "$target_dir"
    if [[ ! -f "$path" ]]; then
        echo "Downloading $repo_file from $repo at $revision"
        HF_HUB_DISABLE_XET=1 "$root/.venv-comfy/bin/python" "$downloader" \
            --repo-id "$repo" --revision "$revision" --allow-patterns "$repo_file" --local-dir "$target_dir"
    fi
    actual_size="$(stat -c %s "$path")"
    actual_sha="$(sha256sum "$path" | awk '{print $1}')"
    if [[ "$actual_size" != "$expected_size" || "$actual_sha" != "$expected_sha" ]]; then
        echo "Pinned Qwen Turbo file failed size/SHA-256 verification: $path" >&2
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

echo "Qwen Image 2.1 Turbo assets installed at $model_root"
