#!/usr/bin/env bash
# Generate text-to-image with FLUX.3 Image Klein through the local ComfyUI API.
set -euo pipefail
root="$(cd -- "$(dirname -- "$BASH_SOURCE")/.." && pwd)"
python="$(printenv L2M2_PYTHON || true)"
if [[ -z "$python" ]]; then
    python="python3"
fi
exec "$python" "$root/media-models/flux_3_klein.py" "$@"
