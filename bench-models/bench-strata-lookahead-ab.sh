#!/usr/bin/env bash
# Strata Qwen3.8-Flash-Next router-lookahead A/B.
#
# Requires exclusive GPU access: stop llama-swap before running and restore it
# afterward. The script only owns/kills process groups that it starts.
# Default order is interleaved to expose startup/environment drift.
#
#   ARMS="lookahead_on lookahead_off lookahead_off lookahead_on" \
#     bash bench-models/bench-strata-lookahead-ab.sh

set -u

REPO="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/bench-models/bench-env.sh"

PORT="${PORT:-8023}"
PYTHON="$REPO/vendor/strata/.venv/bin/python"
SERVER="$REPO/vendor/strata-latest/serve/server.py"
CONFIG="$REPO/vendor/strata-latest/strata-iq3_xxs.json"
PROBE="$REPO/bench-models/bench-qwen38-refresh-probe.py"
OUT="$REPO/logs/results/strata-lookahead-ab.jsonl"
TMP="/tmp/strata-lookahead-ab"
RUN=0
SRV=""

mkdir -p "$TMP" "$(dirname "$OUT")"

cleanup() {
  if [[ -n "$SRV" ]] && kill -0 "$SRV" 2>/dev/null; then
    kill -- "-$SRV" 2>/dev/null || true
    wait "$SRV" 2>/dev/null || true
  fi
  SRV=""
}
trap cleanup EXIT INT TERM

wait_ready() {
  local i
  for i in $(seq 1 450); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -qi '"status": *"ok"\|ok'; then
      return 0
    fi
    kill -0 "$SRV" 2>/dev/null || return 1
    sleep 2
  done
  return 1
}

run_arm() {
  local label="$1" setting log
  RUN=$((RUN + 1))
  case "$label" in
    lookahead_on)  setting=1 ;;
    lookahead_off) setting=0 ;;
    *) echo "unknown arm: $label" >&2; return 1 ;;
  esac
  log="$TMP/${RUN}-${label}.log"

  echo "== run $RUN: $label (STRATA_LOOKAHEAD=$setting)"
  setsid env STRATA_LOOKAHEAD="$setting" \
    taskset -c 0-11 "$PYTHON" "$SERVER" --engine strata --config "$CONFIG" \
    --fit-max-tokens --port "$PORT" >"$log" 2>&1 &
  SRV=$!

  if ! wait_ready; then
    echo "   FAILED to load"
    tail -n 20 "$log"
    cleanup
    return 1
  fi

  python3 "$PROBE" \
    --url "http://127.0.0.1:$PORT" \
    --label "${label}-run${RUN}" \
    --out "$OUT" \
    --warmups 3 \
    --tg-probes 4 \
    --tg-tokens 256 \
    --pp-sizes 2048,8192

  echo "   VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  echo "   RAM available: $(free -g | awk '/^Mem:/{print $7}') GiB"
  cleanup
  sleep 5
}

bench_preflight_warn
bench_emit_frontmatter | tee "$TMP/frontmatter.yaml"

for arm in ${ARMS:-lookahead_on lookahead_off lookahead_off lookahead_on}; do
  run_arm "$arm" || true
done

echo "Results: $OUT"
