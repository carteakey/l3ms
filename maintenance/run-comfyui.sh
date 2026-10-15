#!/usr/bin/env bash
# Default CPU mode keeps the GPU available during setup/benchmarking.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-cpu}"
case "$mode" in
 cpu) device=(--cpu) ;;
 gpu)
   if systemctl --user is-active --quiet llama-swap.service; then
     echo 'Stop llama-swap before starting GPU ComfyUI.' >&2; exit 2
   fi
   used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)"
   (( used < 1500 )) || { echo "GPU is busy (${used} MiB)." >&2; exit 2; }
   device=(--lowvram --reserve-vram 1.0) ;;
 *) echo 'Usage: run-comfyui.sh [cpu|gpu]' >&2; exit 2 ;;
esac
mkdir -p "$root/.toolkit/comfyui/user" "$HOME/media-output/comfyui"
cd "$root/vendor/ComfyUI"
exec "$root/.venv-comfy/bin/python" main.py --listen 127.0.0.1 --port 8188 \
 --disable-auto-launch --disable-api-nodes --user-directory "$root/.toolkit/comfyui/user" \
 --output-directory "$HOME/media-output/comfyui" "${device[@]}"
