#!/usr/bin/env bash
# Install the official MiniMax-H3 Ref2VA quality baseline for ComfyUI.
# Downloads are resumable and existing files are preserved by huggingface_hub.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="$root/vendor/ComfyUI"
model_root="${L2M2_H3_COMFY_ROOT:-$HOME/models/media/MiniMax-H3-ComfyUI}"
python="$root/.venv-comfy/bin/python"
repo="Comfy-Org/MiniMax-H3"

[[ -x "$python" ]] || {
    echo "ComfyUI environment missing; run maintenance/setup-comfyui.sh first." >&2
    exit 1
}

mkdir -p "$model_root" "$root/.toolkit/model_config_versions"
stamp="$(date +%Y%m%d-%H%M%S)"
cp "$root/models/minimax-h3-comfyui-files.txt" \
    "$root/.toolkit/model_config_versions/minimax-h3-comfyui-files-$stamp.txt"

mapfile -t files < <(grep -Ev '^[[:space:]]*(#|$)' "$root/models/minimax-h3-comfyui-files.txt")
"$root/.venv-comfy/bin/hf" download "$repo" \
    "${files[@]}" --local-dir "$model_root"

while IFS= read -r relative; do
    [[ -n "$relative" ]] || continue
    source_path="$model_root/$relative"
    target_path="$workspace/models/$relative"
    [[ -f "$source_path" ]] || { echo "Missing downloaded file: $source_path" >&2; exit 1; }
    mkdir -p "$(dirname "$target_path")"
    if [[ -e "$target_path" && ! -L "$target_path" ]]; then
        echo "Refusing to replace existing model: $target_path" >&2
        exit 1
    fi
    ln -sfn "$source_path" "$target_path"
done < <(printf '%s\n' "${files[@]}")

: > "$model_root/SHA256SUMS"
for relative in "${files[@]}"; do
    sha256sum "$model_root/$relative" >> "$model_root/SHA256SUMS"
done
echo "MiniMax-H3 Ref2VA baseline installed at $model_root"
