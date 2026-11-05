#!/usr/bin/env bash
# Benchmark Strata 4070-main-lite with safe GPU handoff (gpu-handoff skill rules).
#
# MODE=code (default)  the rebuild-and-compare path: bench-qwen38-refresh-ab.sh arm strata_lean.
# MODE=tune           the switch path: bench-strata-tuning-ab.sh arm ladder, flags and env only,
#                     no rebuild. Override the ladder with TUNE_ARMS="base kv_grow base".
# Both take the plat tier off the router for the run and put it back on exit.
#
# Handoff rules followed here (docs in .agents/skills/gpu-handoff/SKILL.md): never `pkill -f`,
# find things by pid via nvidia-smi/ps; only kill model servers this stack spawned; verify the
# card is actually idle (~44 MiB is the signal) before spending 40 minutes on arms that would all
# OOM; and verify the router came back healthy instead of assuming it.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MODE="${MODE:-code}"
IDLE_MIB="${IDLE_MIB:-44}"          # idle VRAM on this card
MAX_MIB="${MAX_MIB:-200}"           # above this, something else has the card

# Model servers owned by this stack: llama-swap's children, the refresh bench's servers, and
# anything launched through serve/server.py (the tuning harness). Matched on cmdline, killed by pid.
stack_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read -r p; do
    [ -n "${p:-}" ] || continue
    cmd="$(ps -o args= -p "$p" 2>/dev/null || true)"
    [ -n "$cmd" ] || continue
    case "$cmd" in
      *llama-swap*|*vendor/llama.cpp*build/bin/llama-server*|*vendor/strata*build/strata*|*serve/server.py*|*refresh-ab*|*strata-tuning-ab*)
        echo "$p" ;;
    esac
  done
}

show_card() {
  echo "   VRAM used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) (idle signal is ~${IDLE_MIB} MiB)"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | while read -r line; do
    pid="$(printf '%s' "$line" | cut -d, -f1 | tr -d ' ')"
    printf '   holder: %s | %s\n' "$line" "$(ps -o args= -p "$pid" 2>/dev/null | cut -c1-110)"
  done
}

cleanup() {
  echo "=== Restoring GPU: starting llama-swap.service ==="
  systemctl --user start llama-swap.service || true
  sleep 8
  echo "   unit: $(systemctl --user is-active llama-swap.service 2>/dev/null || echo unknown)"
  journalctl --user -u llama-swap.service -n 20 --no-pager 2>/dev/null \
    | grep -E "listening|Health check passed" || echo "   no listening/Health line in the last 20 log lines yet (preload may still be loading)"
  show_card
}
trap cleanup EXIT

echo "=== Freeing the card: stopping llama-swap.service ==="
systemctl --user stop llama-swap.service || true
sleep 4

for round in 1 2; do
  pids="$(stack_pids || true)"
  [ -n "$pids" ] || break
  for p in $pids; do
    echo "   orphan model server (round $round): pid $p | $(ps -o args= -p "$p" 2>/dev/null | cut -c1-110)"
    kill "$p" 2>/dev/null || true
  done
  sleep 5
  [ "$round" = 1 ] || break
done
pids="$(stack_pids || true)"
for p in ${pids:-}; do
  echo "   still alive after SIGTERM, SIGKILL: pid $p"
  kill -9 "$p" 2>/dev/null || true
done
sleep 3

show_card
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1)"
if [ -n "$used" ] && [ "$used" -gt "$MAX_MIB" ]; then
  echo "Card is not free (${used} MiB used). If the holder above is not one of this stack's model"
  echo "servers, another session owns the GPU: leave it alone and ask before benching."
  echo "If it is yours and it will not die, re-run with MAX_MIB set above that value."
  exit 1
fi

echo "=== Running Strata 4070-main-lite A/B (MODE=$MODE) ==="
cd "$REPO_ROOT"
if [ "$MODE" = "tune" ]; then
  # The sweep refuses to run while the router is active; we stopped it above, so FORCE is the
  # honest way to say "yes, this is the handoff run".
  FORCE=1 ARMS="${TUNE_ARMS:-base reserve832 kv_grow ctx64k routeprior15 draft_en grnormfast base}" \
    bash bench-models/bench-strata-tuning-ab.sh
else
  ARMS="strata_lean" bash bench-models/bench-qwen38-refresh-ab.sh
fi

echo ""
echo "=== Latest benchmark rows ==="
if [ "$MODE" = "tune" ]; then
  tail -n 1 "$REPO_ROOT/logs/results/strata-tuning-ab.jsonl" 2>/dev/null || echo "no probe rows yet"
  tail -n 1 "$REPO_ROOT/logs/results/strata-tuning-diag.jsonl" 2>/dev/null || true
else
  tail -n 1 "$REPO_ROOT/logs/results/qwen38-refresh-ab.jsonl"
fi
