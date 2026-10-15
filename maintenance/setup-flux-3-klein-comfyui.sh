#!/usr/bin/env bash
# Download pinned official FLUX.3 Image Klein ComfyUI weights and setup workflows.
set -euo pipefail

root="$(cd -- "$(dirname -- "$BASH_SOURCE")/.." && pwd)"
workspace="$root/vendor/ComfyUI"
manifest="$root/models/flux-3-klein-comfyui-files.txt"
comfy_revision="20b46065ef7f34fa21e8abebe8d0cca90a904d6d"
workflow_dir="$root/workflows/flux-3-klein"
py="$root/.venv-comfy/bin/python"

model_root="$(printenv L2M2_FLUX_3_KLEIN_ROOT || true)"
if [[ -z "$model_root" ]]; then
    if [[ -d "$HOME/models/media" ]]; then
        model_root="$HOME/models/media/FLUX.3-Klein"
    elif [[ -d "/run/media/kchauhan/Games/models/media" ]]; then
        model_root="/run/media/kchauhan/Games/models/media/FLUX.3-Klein"
    else
        model_root="$HOME/models/media/FLUX.3-Klein"
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
cp "$manifest" "$root/.toolkit/model_config_versions/flux-3-klein-comfyui-files-$stamp.txt"

# Reuse existing flux2-vae.safetensors if already available on disk
existing_vae=""
for candidate in \
    "$HOME/models/media/FLUX.2-Klein-9B/vae/flux2-vae.safetensors" \
    "$workspace/models/vae/flux2-vae.safetensors" \
    "/run/media/kchauhan/Games/models/media/FLUX.2-Klein-9B/vae/flux2-vae.safetensors"; do
    if [[ -f "$candidate" ]]; then
        existing_vae="$candidate"
        break
    fi
done

if [[ -n "$existing_vae" && ! -e "$model_root/vae/flux2-vae.safetensors" && ! -L "$model_root/vae/flux2-vae.safetensors" ]]; then
    mkdir -p "$model_root/vae"
    echo "Reusing existing verified VAE from $existing_vae"
    ln -s -- "$existing_vae" "$model_root/vae/flux2-vae.safetensors"
fi

missing_bytes=0
while read -r repo revision repo_file relative expected_size expected_sha; do
    [[ -n "$repo" && "$repo" != \#* ]] || continue
    path="$model_root/$relative"
    if [[ -f "$path" ]]; then
        actual_size="$(stat -L -c %s "$path")"
        actual_sha="$(sha256sum "$path" | awk '{print $1}')"
        if [[ "$actual_size" != "$expected_size" || "$actual_sha" != "$expected_sha" ]]; then
            echo "Existing FLUX.3 Klein file differs from the pinned artifact; preserving it: $path" >&2
            exit 1
        fi
    elif [[ -e "$path" || -L "$path" ]]; then
        if [[ -L "$path" && -f "$path" ]]; then
            actual_size="$(stat -L -c %s "$path")"
            actual_sha="$(sha256sum "$path" | awk '{print $1}')"
            if [[ "$actual_size" == "$expected_size" && "$actual_sha" == "$expected_sha" ]]; then
                continue
            fi
        fi
        echo "Existing FLUX.3 Klein path is not a valid regular file/link; preserving it: $path" >&2
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
    printf 'Set L2M2_FLUX_3_KLEIN_ROOT to specify a filesystem with sufficient space.\n' >&2
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
import os, sys, shutil, urllib.request

repo = '$repo'
revision = '$revision'
repo_file = '$repo_file'
final_path = '$path'
expected_size = int('$expected_size')
model_root = '$model_root'
cache_dir = os.path.join(model_root, '.cache', 'huggingface')

# Check token
token = None
token_path = os.path.expanduser('~/.cache/huggingface/token')
if os.path.exists(token_path):
    with open(token_path) as tf:
        token = tf.read().strip() or None

downloaded = None
# Try huggingface_hub first
try:
    from huggingface_hub import hf_hub_download
    downloaded = hf_hub_download(
        repo_id=repo,
        filename=repo_file,
        revision=revision,
        local_dir=model_root,
        cache_dir=cache_dir,
        token=token
    )
except Exception as e:
    print(f'Huggingface_hub direct download failed ({e}); trying authenticated curl fallback...', file=sys.stderr)

if not downloaded or not os.path.exists(downloaded) or os.stat(downloaded).st_size != expected_size:
    # Direct curl fallback with stream
    urls = [
        f'https://huggingface.co/{repo}/resolve/{revision}/{repo_file}',
        f'https://huggingface.co/ml-resources/flux-3-image-klein-comfy/resolve/main/{repo_file}'
    ]
    part_path = final_path + '.part'
    success = False
    for url in urls:
        print(f'Attempting download from {url}...')
        req = urllib.request.Request(url)
        if token and 'black-forest-labs' in url:
            req.add_header('Authorization', f'Bearer {token}')
        try:
            with urllib.request.urlopen(req) as resp, open(part_path, 'wb') as out_f:
                shutil.copyfileobj(resp, out_f, length=16*1024*1024)
            if os.path.exists(part_path) and os.stat(part_path).st_size == expected_size:
                shutil.move(part_path, final_path)
                success = True
                break
            else:
                if os.path.exists(part_path):
                    os.unlink(part_path)
        except Exception as dl_err:
            print(f'URL {url} failed: {dl_err}', file=sys.stderr)
            if os.path.exists(part_path):
                os.unlink(part_path)
    if not success:
        sys.exit(1)
elif downloaded != final_path and os.path.exists(downloaded):
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    shutil.move(downloaded, final_path)
"
    fi
    actual_size="$(stat -L -c %s "$path")"
    actual_sha="$(sha256sum "$path" | awk '{print $1}')"
    if [[ "$actual_size" != "$expected_size" || "$actual_sha" != "$expected_sha" ]]; then
        echo "Pinned FLUX.3 Klein file failed size/SHA-256 verification: $path" >&2
        exit 1
    fi

    target="$workspace/models/$relative"
    mkdir -p "$(dirname -- "$target")"
    if [[ -L "$target" && ! -e "$target" ]]; then
        rm -f "$target"
    elif [[ -L "$target" && "$(readlink -f "$target")" == "$(readlink -f "$path")" ]]; then
        continue
    fi
    if [[ -f "$target" ]] && [[ "$(stat -L -c %s "$target")" == "$expected_size" ]]; then
        target_sha="$(sha256sum "$target" | awk '{print $1}')"
        if [[ "$target_sha" == "$expected_sha" ]]; then
            echo "ComfyUI already has verified file at $target; keeping it"
            continue
        fi
    fi
    if [[ -e "$target" || -L "$target" ]]; then
        echo "Refusing to replace an existing ComfyUI model path: $target" >&2
        exit 1
    fi
    ln -s -- "$path" "$target"
done < "$manifest"

sum_file="$model_root/FLUX-3-KLEIN-SHA256SUMS"
sum_temp="$(mktemp "$model_root/.flux-3-klein-sums.XXXXXX")"
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

echo "FLUX.3 Image Klein ComfyUI assets successfully installed at $model_root"
