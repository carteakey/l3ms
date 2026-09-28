#!/usr/bin/env bash
# Download pinned official Qwen Image 2.1 ComfyUI weights and workflows.
set -euo pipefail

root="$(cd -- "$(dirname -- "$BASH_SOURCE")/.." && pwd)"
workspace="$root/vendor/ComfyUI"
manifest="$root/models/qwen-image-2.1-comfyui-files.txt"
model_repo="Comfy-Org/Qwen-Image-2.1"
model_revision="ace0edeb3791a594ddfa36ed5f41a178a394e921"
comfy_revision="6bfaacc67c2103481e5f0c84d75257cd0581d86a"
workflow_revision="371a7b7171bbd11e9cc92ef615ba5ad223d7e5b4"
workflow_dir="$root/workflows/qwen-image-2.1"
hf="$root/.venv-comfy/bin/hf"

model_root="$(printenv L2M2_QWEN_IMAGE_21_ROOT || true)"
if [[ -z "$model_root" ]]; then
    model_root="$HOME/models/media/Qwen-Image-2.1"
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
[[ -x "$hf" ]] || {
    echo "ComfyUI Python environment missing; run maintenance/setup-comfyui.sh first." >&2
    exit 1
}
[[ -f "$manifest" ]] || { echo "Missing model manifest: $manifest" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "sha256sum is required." >&2; exit 1; }

stamp="$(date +%Y%m%d-%H%M%S)"
cp "$manifest" "$root/.toolkit/model_config_versions/qwen-image-2.1-comfyui-files-$stamp.txt"

missing_bytes=0
while read -r relative expected_size expected_sha; do
    [[ -n "$relative" && "$relative" != \#* ]] || continue
    path="$model_root/$relative"
    if [[ -f "$path" ]]; then
        actual_size="$(stat -c %s "$path")"
        actual_sha="$(sha256sum "$path" | awk '{print $1}')"
        if [[ "$actual_size" != "$expected_size" || "$actual_sha" != "$expected_sha" ]]; then
            echo "Existing Qwen file differs from the pinned artifact; preserving it: $path" >&2
            exit 1
        fi
    elif [[ -e "$path" || -L "$path" ]]; then
        echo "Existing Qwen path is not a regular file; preserving it: $path" >&2
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

while read -r relative expected_size expected_sha; do
    [[ -n "$relative" && "$relative" != \#* ]] || continue
    path="$model_root/$relative"
    if [[ ! -f "$path" ]]; then
        echo "Downloading $relative from $model_repo at $model_revision"
        "$hf" download "$model_repo" "$relative" \
            --revision "$model_revision" --local-dir "$model_root"
    fi
    actual_size="$(stat -c %s "$path")"
    actual_sha="$(sha256sum "$path" | awk '{print $1}')"
    if [[ "$actual_size" != "$expected_size" || "$actual_sha" != "$expected_sha" ]]; then
        echo "Pinned Qwen file failed size/SHA-256 verification: $path" >&2
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

fetch_workflow() {
    name="$1"
    expected_blob="$2"
    url="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/$workflow_revision/templates/$name"
    destination="$workflow_dir/$name"
    if [[ -e "$destination" || -L "$destination" ]]; then
        if [[ -f "$destination" && "$(git -C "$root" hash-object "$destination")" == "$expected_blob" ]]; then
            return
        fi
        echo "Refusing to replace an existing workflow: $destination" >&2
        return 1
    fi
    temporary="$(mktemp "$workflow_dir/.workflow.XXXXXX")"
    if ! curl --fail --location --retry 3 --silent --show-error \
        --output "$temporary" "$url"; then
        rm -- "$temporary"
        return 1
    fi
    actual_blob="$(git -C "$root" hash-object "$temporary")"
    if [[ "$actual_blob" != "$expected_blob" ]]; then
        echo "Workflow checksum mismatch for $name; keeping the temporary download at $temporary" >&2
        return 1
    fi
    mv -- "$temporary" "$destination"
}

fetch_workflow image_qwen_image_2_1_t2i.json 8491e178f6918c6c1d7c37ebce552889fb7ec372
fetch_workflow image_qwen_image_2_1_image_edit.json cfc020c8b04d349643c01fe3ab51968256680400

sum_file="$model_root/QWEN-IMAGE-2.1-SHA256SUMS"
sum_temp="$(mktemp "$model_root/.qwen-image-2.1-sums.XXXXXX")"
(
    cd -- "$model_root"
    while read -r relative expected_size expected_sha; do
        [[ -n "$relative" && "$relative" != \#* ]] || continue
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

echo "Qwen Image 2.1 ComfyUI assets installed at $model_root"
echo "Official workflows are in $workflow_dir"
