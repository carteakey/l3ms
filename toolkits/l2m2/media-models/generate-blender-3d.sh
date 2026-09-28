#!/usr/bin/env bash
# Run Blender 3D operations, headless Python modeling/rendering, or manage the MCP bridge.

set -euo pipefail

usage() {
    cat <<'USAGE_EOF'
Usage:
  l2m2 --media blender-3d --extra '--code "import bpy; bpy.ops.mesh.primitive_monkey_add()"'
  l2m2 --media blender-3d --extra '--launch'
  l2m2 --media blender-3d --extra '--check'

Options:
  --prompt TEXT             3D prompt / instruction for Blender MCP
  --code PYTHON_CODE        Execute Python code directly in Blender background
  --script FILE             Execute a Python script file in Blender background
  --render FILE             Render the current or loaded scene to an image file (PNG)
  --output PATH             Output directory or file path (default: $HOME/media-output)
  --launch                  Launch interactive Blender GUI with MCP add-on loaded
  --check                   Verify Blender executable, addon, and MCP listener status
  --port PORT               MCP server socket port (default: 9876)
  --help                    Show this help message

Environment:
  L3MS_BLENDER_BIN          Blender executable path (default: ~/.local/bin/blender or blender)
  L3MS_BLENDER_MCP_PORT     MCP socket port (default: 9876)
  L3MS_MEDIA_OUTPUT_DIR     Output directory (default: $HOME/media-output)
USAGE_EOF
}

find_blender() {
    if [[ -n "${L3MS_BLENDER_BIN:-}" && -x "${L3MS_BLENDER_BIN}" ]]; then
        printf '%s\n' "$L3MS_BLENDER_BIN"
        return 0
    fi
    if command -v blender >/dev/null 2>&1; then
        command -v blender
        return 0
    fi
    if [[ -x "$HOME/.local/bin/blender" ]]; then
        printf '%s\n' "$HOME/.local/bin/blender"
        return 0
    fi
    return 1
}

prompt=""
code=""
script=""
render_target=""
launch=0
check=0
port="${L3MS_BLENDER_MCP_PORT:-9876}"
output_dir="${L3MS_MEDIA_OUTPUT_DIR:-${HOME}/media-output}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt)
            [[ $# -ge 2 ]] || { echo "ERROR: --prompt requires an argument" >&2; exit 1; }
            prompt="$2"
            shift 2
            ;;
        --code)
            [[ $# -ge 2 ]] || { echo "ERROR: --code requires an argument" >&2; exit 1; }
            code="$2"
            shift 2
            ;;
        --script)
            [[ $# -ge 2 ]] || { echo "ERROR: --script requires an argument" >&2; exit 1; }
            script="$2"
            shift 2
            ;;
        --render)
            [[ $# -ge 2 ]] || { echo "ERROR: --render requires an argument" >&2; exit 1; }
            render_target="$2"
            shift 2
            ;;
        --output)
            [[ $# -ge 2 ]] || { echo "ERROR: --output requires an argument" >&2; exit 1; }
            output_dir="$2"
            shift 2
            ;;
        --launch)
            launch=1
            shift
            ;;
        --check)
            check=1
            shift
            ;;
        --port)
            [[ $# -ge 2 ]] || { echo "ERROR: --port requires an argument" >&2; exit 1; }
            port="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

blender_bin="$(find_blender || true)"
if [[ -z "$blender_bin" ]]; then
    echo "ERROR: Blender executable not found. Set L3MS_BLENDER_BIN or ensure blender is on PATH." >&2
    exit 1
fi

if [[ $check -eq 1 ]]; then
    printf 'Blender executable: %s\n' "$blender_bin"
    "$blender_bin" --version | head -n 2
    addon_path="$HOME/.config/blender/5.2/scripts/addons/blender_mcp.py"
    if [[ -f "$addon_path" ]]; then
        printf '[OK] blender_mcp addon: %s\n' "$addon_path"
    else
        printf '[WARN] blender_mcp addon not found at %s\n' "$addon_path"
    fi
    if ss -tlpn 2>/dev/null | grep -q ":${port} "; then
        printf '[OK] Blender MCP listener: active on port %s\n' "$port"
    else
        printf '[INFO] Blender MCP listener: port %s idle (start via Blender sidebar -> MCP for Blender -> Start MCP Server)\n' "$port"
    fi
    exit 0
fi

if [[ $launch -eq 1 ]]; then
    printf 'Launching Blender GUI (%s)...\n' "$blender_bin"
    exec "$blender_bin"
fi

if [[ -n "$code" ]]; then
    printf 'Executing Python code in Blender background...\n'
    "$blender_bin" --background --python-expr "$code"
    exit 0
fi

if [[ -n "$script" ]]; then
    [[ -f "$script" ]] || { echo "ERROR: Script file not found: $script" >&2; exit 1; }
    printf 'Executing Python script (%s) in Blender background...\n' "$script"
    "$blender_bin" --background --python "$script"
    exit 0
fi

if [[ -n "$render_target" ]]; then
    mkdir -p "$output_dir"
    output_file="$output_dir/render_$(date +%Y%m%d_%H%M%S).png"
    printf 'Rendering Blender scene to %s...\n' "$output_file"
    "$blender_bin" --background "$render_target" --render-output "$output_file" --render-frame 1
    printf 'Render completed: %s\n' "$output_file"
    exit 0
fi

if [[ -n "$prompt" ]]; then
    if ss -tlpn 2>/dev/null | grep -q ":${port} "; then
        printf 'Blender MCP listener is active on port %s.\n' "$port"
        printf 'Prompt ready for MCP agent: "%s"\n' "$prompt"
        exit 0
    else
        echo "Blender MCP listener on port $port is not running."
        echo "Launch Blender with: l2m2 --media blender-3d --extra '--launch'"
        echo "Then press N -> MCP for Blender -> Start MCP Server."
        exit 1
    fi
fi

usage
exit 0
