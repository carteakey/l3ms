#!/usr/bin/env bash
# Give the MCP server an explicit workspace and the correct runtime Python.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export VIRTUAL_ENV="$root/.venv-comfy"
export COMFY_LOCAL_URL=http://127.0.0.1:8188
exec "$root/.venv-comfy-mcp/bin/comfy" --workspace "$root/vendor/ComfyUI" "$@"
