#!/usr/bin/env bash
# Register the official local MCP server with Codex, preserving config first.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
config="${CODEX_HOME:-$HOME/.codex}/config.toml"
if [[ -f "$config" ]]; then
    cp -p "$config" "$config.bak-comfy-$(date +%Y%m%d-%H%M%S)-$$"
fi
codex mcp add comfy-local -- "$root/maintenance/run-comfy-mcp.sh"
codex mcp get comfy-local
