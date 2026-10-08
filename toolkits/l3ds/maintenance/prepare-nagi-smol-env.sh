#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCK_FILE="$ROOT_DIR/requirements/nagi-smol.lock.txt"
VENV_DIR="$ROOT_DIR/.venv"

usage() {
    cat <<'EOF'
Usage: maintenance/prepare-nagi-smol-env.sh [--dry-run | --install]

  --dry-run  (default) Print the pinned Nagi-Smol setup plan without changing
             files, creating a venv, accessing package indexes, or fetching
             model weights.
  --install  Create a fresh child .venv and install the pinned Nagi-Smol-only
             Python/CUDA environment. This may download multi-gigabyte wheels.
             It does not download model weights.
EOF
}

MODE="${1:---dry-run}"
if [ "$#" -gt 1 ]; then
    usage >&2
    exit 2
fi

case "$MODE" in
    --dry-run|--install) ;;
    --help|-h)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

if [ ! -f "$LOCK_FILE" ]; then
    echo "Error: missing Nagi-Smol lock file: $LOCK_FILE" >&2
    exit 1
fi

if [ "$MODE" = "--dry-run" ]; then
    echo "=== L3DS Nagi-Smol environment plan (no changes) ==="
    echo "Python: CPython 3.12.13"
    echo "Target: $VENV_DIR (a fresh .venv is required; existing .venv will not be changed)"
    echo "Lock:   $LOCK_FILE"
    echo "Scope:  Nagi only; Torch 2.6.0+cu124; no SemIf/Laya/hosted SDK"
    echo "Model weights: not part of this setup and will not be downloaded"
    echo "Install is opt-in and may fetch multi-gigabyte CUDA wheels: --install"
    exit 0
fi

if [ -e "$VENV_DIR" ] || [ -L "$VENV_DIR" ]; then
    echo "Error: refusing to modify existing environment: $VENV_DIR" >&2
    echo "Move or preserve that environment yourself, then retry only if this fresh setup is intended." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: 'uv' is required for the pinned Nagi-Smol setup." >&2
    exit 1
fi

stage_dir=""
cleanup() {
    if [ -n "$stage_dir" ]; then
        stage_parent="${stage_dir%/*}"
        stage_name="${stage_dir##*/}"
        if [[ "$stage_parent" == "$ROOT_DIR" \
            && "$stage_name" == .nagi-smol-env.* \
            && -d "$stage_dir" \
            && ! -L "$stage_dir" ]]; then
            rm -rf -- "$stage_dir"
        else
            echo "Warning: leaving unverified staging path untouched: $stage_dir" >&2
        fi
    fi
}
trap cleanup EXIT

stage_dir="$(mktemp -d "$ROOT_DIR/.nagi-smol-env.XXXXXX")"
stage_parent="${stage_dir%/*}"
stage_name="${stage_dir##*/}"
if [[ "$stage_parent" != "$ROOT_DIR" \
    || "$stage_name" != .nagi-smol-env.* \
    || ! -d "$stage_dir" \
    || -L "$stage_dir" ]]; then
    echo "Error: mktemp returned an unexpected staging path; leaving it untouched." >&2
    exit 1
fi
echo "Creating a staged CPython 3.12.13 environment."
echo "Installing the pinned Nagi-Smol dependency closure (model weights excluded)."
UV_PYTHON_DOWNLOADS=never uv venv --python 3.12.13 --relocatable "$stage_dir"
uv pip sync \
    --strict \
    --torch-backend cu124 \
    --python "$stage_dir/bin/python" \
    "$LOCK_FILE"

if [ -e "$VENV_DIR" ] || [ -L "$VENV_DIR" ]; then
    echo "Error: environment appeared during setup; preserving it and discarding staging." >&2
    exit 1
fi

mv -T -n -- "$stage_dir" "$VENV_DIR"
if [ -e "$stage_dir" ] || [ -L "$stage_dir" ]; then
    echo "Error: environment appeared during setup; refusing to replace it." >&2
    exit 1
fi
stage_dir=""
echo "Pinned Nagi-Smol environment installed at $VENV_DIR."
echo "Model weights remain a separate, user-authorized acquisition step."
