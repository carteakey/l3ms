#!/usr/bin/env bash
# Invoke via systemd-run --user for survival across terminal disconnections.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
restore=false
cleanup() { if "$restore"; then systemctl --user start llama-swap.service; fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if systemctl --user is-active --quiet llama-swap.service; then
    restore=true
    systemctl --user stop llama-swap.service
fi
"$root/maintenance/test-h3-batman.sh"
