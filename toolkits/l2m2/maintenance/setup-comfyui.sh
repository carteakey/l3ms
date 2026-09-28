#!/usr/bin/env bash
# Install the local ComfyUI + official MCP toolchain, without model downloads.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
uv="${L2M2_UV:-$HOME/.local/bin/uv}"
comfy_ref=6bfaacc67c2103481e5f0c84d75257cd0581d86a
mcp_ref=eb8ea47397fb13cdcf0a818b83bfba48bd4ff236
checkout() {
    local url="$1" dir="$2" ref="$3"
    if [[ ! -d "$dir/.git" ]]; then git clone --depth 1 "$url" "$dir"; fi
    [[ -z "$(git -C "$dir" status --porcelain)" ]] || { echo "Dirty vendor checkout: $dir" >&2; exit 1; }
    if [[ "$(git -C "$dir" rev-parse HEAD)" != "$ref" ]]; then
        git -C "$dir" fetch --depth 1 origin "$ref"
        git -C "$dir" checkout --detach "$ref"
    fi
}
mkdir -p "$root/vendor" "$root/.toolkit/comfyui"
checkout https://github.com/Comfy-Org/ComfyUI.git "$root/vendor/ComfyUI" "$comfy_ref"
checkout https://github.com/Comfy-Org/comfy-mcp.git "$root/vendor/comfy-mcp" "$mcp_ref"
[[ -x "$root/.venv-comfy/bin/python" ]] || "$uv" venv --python 3.13.15 "$root/.venv-comfy"
[[ -x "$root/.venv-comfy-mcp/bin/python" ]] || "$uv" venv --python 3.13.15 "$root/.venv-comfy-mcp"
"$uv" pip install --python "$root/.venv-comfy/bin/python" -r "$root/runtime-locks/comfyui-py313-linux.txt" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
"$uv" pip install --python "$root/.venv-comfy-mcp/bin/python" -r "$root/runtime-locks/comfy-mcp-py313-linux.txt"
"$uv" pip install --python "$root/.venv-comfy-mcp/bin/python" --no-deps "$root/vendor/comfy-mcp"
"$uv" pip freeze --python "$root/.venv-comfy/bin/python" > "$root/.toolkit/comfyui/requirements-comfy-resolved.txt"
"$uv" pip freeze --python "$root/.venv-comfy-mcp/bin/python" > "$root/.toolkit/comfyui/requirements-mcp-resolved.txt"
echo 'Installed. No models downloaded; no server or generation started.'
