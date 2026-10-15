#!/usr/bin/env bash
# Generate Qwen Image 2.1 Turbo text-to-image through the local ComfyUI API.
set -euo pipefail
root="$(cd -- "$(dirname -- "$BASH_SOURCE")/.." && pwd)"
python="$(printenv L2M2_PYTHON || true)"
if [[ -z "$python" ]]; then
    python="python3"
fi
exec "$python" "$root/media-models/qwen_image_21_turbo.py" "$@"
