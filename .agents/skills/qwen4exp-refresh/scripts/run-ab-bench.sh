#!/usr/bin/env bash
# run-ab-bench.sh
# ---------------
# Stops llama-swap.service, executes the 7-way A/B benchmark harness for Qwen3.8-Flash-Next,
# and safely restarts llama-swap.service.

set -euo pipefail

L3MS_ROOT="${L3MS_ROOT:-/home/kchauhan/repos/l3ms}"

log() { printf '\033[1;34m->\033[0m %s\n' "$*"; }
ok()  { printf '\033[1;32m✓\033[0m %s\n' "$*"; }

log "Pausing llama-swap router service for full-VRAM benchmark..."
systemctl --user stop llama-swap.service || true
sleep 3

cleanup() {
    log "Restoring llama-swap service..."
    systemctl --user start llama-swap.service || true
    systemctl --user is-active llama-swap.service && ok "llama-swap active."
}
trap cleanup EXIT

log "Launching Qwen3.8-Flash-Next refresh benchmark suite..."
(cd "${L3MS_ROOT}" && bash bench-models/bench-qwen38-refresh-ab.sh "$@")

ok "Benchmark complete. Results stored in ${L3MS_ROOT}/logs/results/qwen38-refresh-ab.jsonl"
