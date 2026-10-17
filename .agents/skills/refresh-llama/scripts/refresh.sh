#!/usr/bin/env bash
# refresh.sh
# ----------
# Helper script for refresh-llama skill.
# Locates the L3MS repository and executes maintenance/refresh-llama.sh.

set -euo pipefail

L3MS_ROOT="${L3MS_ROOT:-/home/kchauhan/repos/l3ms}"

if [ ! -d "${L3MS_ROOT}" ]; then
    printf 'ERROR: L3MS repository root not found at %s\n' "${L3MS_ROOT}" >&2
    exit 1
fi

exec "${L3MS_ROOT}/maintenance/refresh-llama.sh" "$@"
