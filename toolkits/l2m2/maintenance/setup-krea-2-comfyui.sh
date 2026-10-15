#!/usr/bin/env bash
# Download pinned official Krea 2 ComfyUI weights, Turbo distill LoRA, and workflows.
set -euo pipefail

root="$(cd -- "$(dirname -- "$BASH_SOURCE")/.." && pwd)"
workspace="$root/vendor/ComfyUI"
manifest="$root/models/krea-2-comfyui-files.txt"
comfy_revision="20b46065ef7f34fa21e8abebe8d0cca90a904d6d"
workflow_revision="a7acaf8cee9ccccdaf4b26ce04a9fced4c4d13de"
workflow_dir="$root/workflows/krea-2"
hf="$root/.venv-comfy/bin/hf"

model_root="$(printenv L2M2_KREA_2_ROOT || printenv L2M2_KREA2_ROOT || true)"
if [[ -z "$model_root" ]]; then
    model_root="$HOME/models/media/Krea-2"
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
cp "$manifest" "$root/.toolkit/model_config_versions/krea-2-comfyui-files-$stamp.txt"

missing_bytes=0
while read -r repo revision repo_file relative expected_size expected_sha; do
    [[ -n "$repo" && "$repo" != \#* ]] || continue
    path="$model_root/$relative"
    if [[ -f "$path" ]]; then
        actual_size="$(stat -c %s "$path")"
        actual_sha="$(sha256sum "$path" | awk '{print $1}')"
        if [[ "$actual_size" != "$expected_size" || "$actual_sha" != "$expected_sha" ]]; then
            echo "Existing Krea file differs from the pinned artifact; preserving it: $path" >&2
            exit 1
        fi
    elif [[ -e "$path" || -L "$path" ]]; then
        echo "Existing Krea path is not a regular file; preserving it: $path" >&2
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
    printf 'Set L2M2_KREA_2_ROOT to specify a filesystem with sufficient space.\n' >&2
    exit 1
fi

while read -r repo revision repo_file relative expected_size expected_sha; do
    [[ -n "$repo" && "$repo" != \#* ]] || continue
    path="$model_root/$relative"
    target_dir="$(dirname -- "$path")"
    mkdir -p "$target_dir"
    if [[ ! -f "$path" ]]; then
        echo "Downloading $repo_file from $repo at $revision"
        if [[ "$repo_file" == "$relative" ]]; then
            "$hf" download "$repo" "$repo_file" \
                --revision "$revision" --local-dir "$model_root"
        else
            "$hf" download "$repo" "$repo_file" \
                --revision "$revision" --local-dir "$target_dir"
        fi
    fi
    actual_size="$(stat -c %s "$path")"
    actual_sha="$(sha256sum "$path" | awk '{print $1}')"
    if [[ "$actual_size" != "$expected_size" || "$actual_sha" != "$expected_sha" ]]; then
        echo "Pinned Krea file failed size/SHA-256 verification: $path" >&2
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

fetch_workflow image_krea2_turbo_t2i.json 74f3d6e07214fc3ded1a31170a5ca5198761339c
fetch_workflow image_krea2_turbo_t2i_int8.json 59a58337dfa8d5f43b036102b8888bc74ff43531
fetch_workflow image_krea2_turbo_int8_image_style_reference.json d7bdffd78723c594acce1435f1bd5aa86ff45ee8

sum_file="$model_root/KREA-2-SHA256SUMS"
sum_temp="$(mktemp "$model_root/.krea-2-sums.XXXXXX")"
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

echo "Krea 2 ComfyUI assets installed at $model_root"
echo "Official workflows are in $workflow_dir"
