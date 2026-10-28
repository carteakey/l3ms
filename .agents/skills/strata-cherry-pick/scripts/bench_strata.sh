#!/usr/bin/env bash
# Benchmark Strata 4070-main-lite with safe GPU handoff.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

echo "=== Freeing GPU: Stopping llama-swap.service ==="
systemctl --user stop llama-swap.service || true
pkill -f "llama-server.*refresh-ab" 2>/dev/null || true
pkill -f "strata.*--pack" 2>/dev/null || true
sleep 2

cleanup() {
  echo "=== Restoring GPU: Starting llama-swap.service ==="
  systemctl --user start llama-swap.service || true
}
trap cleanup EXIT

echo "=== Running Strata 4070-main-lite A/B Benchmark ==="
cd "$REPO_ROOT"
ARMS="strata_lean" bash bench-models/bench-qwen38-refresh-ab.sh

echo ""
echo "=== Latest Benchmark Results (qwen38-refresh-ab.jsonl) ==="
tail -n 1 "$REPO_ROOT/logs/results/qwen38-refresh-ab.jsonl"
