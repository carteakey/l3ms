#!/usr/bin/env bash
# STDIO server; stdout is reserved for MCP messages.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export COMFY_BIN="$root/maintenance/comfy-cli.sh"
export COMFY_LOCAL_URL=http://127.0.0.1:8188
cd "$root/vendor/ComfyUI"
exec "$root/.venv-comfy-mcp/bin/comfy-mcp"
