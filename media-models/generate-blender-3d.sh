#!/usr/bin/env bash
# Run Blender 3D operations, headless Python modeling/rendering, or manage the MCP bridge.

set -euo pipefail

usage() {
    cat <<'USAGE_EOF'
Usage:
  ./media-models/generate-blender-3d.sh --code "import bpy; bpy.ops.mesh.primitive_monkey_add()"
  ./media-models/generate-blender-3d.sh --launch
  ./media-models/generate-blender-3d.sh --check

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
  L3MS_BLENDER_HEADLESS_ONLY
                            Set to 1 for one code/script/render action only
                            (no GUI/MCP; render requires a fresh output target)
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
headless_only=0
[[ "${L3MS_BLENDER_HEADLESS_ONLY:-}" == "1" ]] && headless_only=1
code_count=0
script_count=0
render_count=0
prompt_seen=0
launch_seen=0
check_seen=0
port_seen=0
output_seen=0
help_seen=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt)
            [[ $# -ge 2 ]] || { echo "ERROR: --prompt requires an argument" >&2; exit 1; }
            prompt="$2"
            prompt_seen=$((prompt_seen + 1))
            shift 2
            ;;
        --code)
            [[ $# -ge 2 ]] || { echo "ERROR: --code requires an argument" >&2; exit 1; }
            code="$2"
            code_count=$((code_count + 1))
            shift 2
            ;;
        --script)
            [[ $# -ge 2 ]] || { echo "ERROR: --script requires an argument" >&2; exit 1; }
            script="$2"
            script_count=$((script_count + 1))
            shift 2
            ;;
        --render)
            [[ $# -ge 2 ]] || { echo "ERROR: --render requires an argument" >&2; exit 1; }
            render_target="$2"
            render_count=$((render_count + 1))
            shift 2
            ;;
        --output)
            [[ $# -ge 2 ]] || { echo "ERROR: --output requires an argument" >&2; exit 1; }
            output_dir="$2"
            output_seen=$((output_seen + 1))
            shift 2
            ;;
        --launch)
            launch=1
            launch_seen=$((launch_seen + 1))
            shift
            ;;
        --check)
            check=1
            check_seen=$((check_seen + 1))
            shift
            ;;
        --port)
            [[ $# -ge 2 ]] || { echo "ERROR: --port requires an argument" >&2; exit 1; }
            port="$2"
            port_seen=$((port_seen + 1))
            shift 2
            ;;
        --help|-h)
            if [[ $headless_only -eq 1 ]]; then
                help_seen=$((help_seen + 1))
                shift
            else
                usage
                exit 0
            fi
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ $headless_only -eq 1 ]]; then
    action_count=$((code_count + script_count + render_count))
    if [[ $action_count -ne 1 || $prompt_seen -ne 0 || $launch_seen -ne 0 || \
          $check_seen -ne 0 || $port_seen -ne 0 || $help_seen -ne 0 || \
          $output_seen -gt 1 ]]; then
        echo "ERROR: Headless Blender mode requires exactly one of --code, --script, or --render; other standalone options are unavailable." >&2
        exit 1
    fi
    if [[ $render_count -eq 1 ]]; then
        if [[ $output_seen -ne 1 || -z "$render_target" || ! -f "$render_target" || \
              "$output_dir" != /* || ! -d "$output_dir" || -L "$output_dir" ]]; then
            echo "ERROR: Headless render requires an existing scene and an existing absolute output directory." >&2
            exit 1
        fi
        render_output_prefix="${output_dir%/}/result_####"
        render_output_file="${output_dir%/}/result_0001.png"
        if [[ -e "$render_output_file" || -L "$render_output_file" ]]; then
            echo "ERROR: Headless render output already exists." >&2
            exit 1
        fi
    elif [[ $output_seen -ne 0 ]]; then
        echo "ERROR: --output is only valid with --render in headless Blender mode." >&2
        exit 1
    elif [[ $code_count -eq 1 && -z "$code" ]]; then
        echo "ERROR: Headless Blender code is empty." >&2
        exit 1
    elif [[ $script_count -eq 1 && ! -f "$script" ]]; then
        echo "ERROR: Headless Blender script file is missing." >&2
        exit 1
    fi
fi

blender_bin="$(find_blender || true)"
if [[ -z "$blender_bin" ]]; then
    echo "ERROR: Blender executable not found. Set L3MS_BLENDER_BIN or ensure blender is on PATH." >&2
    exit 1
fi

if [[ $headless_only -eq 1 ]]; then
    if [[ $code_count -eq 1 ]]; then
        # --code is trusted local Python, not a sandboxed input format.
        "$blender_bin" --background --python-exit-code 1 --python-expr "$code" >/dev/null 2>&1
        exit 0
    fi
    if [[ $script_count -eq 1 ]]; then
        "$blender_bin" --background --python-exit-code 1 --python "$script" >/dev/null 2>&1
        exit 0
    fi
    "$blender_bin" --background "$render_target" --render-output "$render_output_prefix" \
        --render-format PNG --use-extension 1 --render-frame 1 >/dev/null 2>&1
    if [[ ! -f "$render_output_file" || -L "$render_output_file" || ! -s "$render_output_file" ]]; then
        echo "ERROR: Blender did not produce the expected render output." >&2
        exit 1
    fi
    exit 0
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
        echo "Launch Blender with: ./media-models/generate-blender-3d.sh --launch"
        echo "Then press N -> MCP for Blender -> Start MCP Server."
        exit 1
    fi
fi

usage
exit 0
