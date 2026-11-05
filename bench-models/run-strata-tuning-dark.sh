#!/usr/bin/env bash
# Dark run of the Strata tuning ladder.
#
# The ladder measures one loaded model at a time, and the model it measures is the model this
# agent answers from, so the run has to happen while the agent is asleep: launch this detached,
# end the turn, and read the rows on the next turn. The grace delay is the window for the
# agent's own turn to finish before the tier is evicted; nothing is stopped before it expires.
#
#   setsid nohup bench-models/run-strata-tuning-dark.sh > logs/strata-dark-$(date +%H%M).log 2>&1 &
#
# What it does: preflight state to the log, stop llama-swap (unit + its engine children), wait
# for the card to read idle, run bench-models/bench-strata-tuning-ab.sh, restart the unit,
# confirm the served tier answers /health, and write a one-line status file next to the log.
# Serving configs are NOT edited here: promotion happens after the rows have been read.

set -uo pipefail

REPO="/home/kchauhan/repos/l3ms"
GRACE="${GRACE:-75}"                      # seconds before the router is stopped
ARMS="${ARMS:-}"                          # empty -> the harness's default (Exact-switch) ladder
TG_PROBES="${TG_PROBES:-4}"
TG_WARMUP="${TG_WARMUP:-6}"
DEEP_SIZES="${DEEP_SIZES:-}"
PP_SIZES="${PP_SIZES:-8192,32768}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$REPO/logs/strata-dark-$STAMP.log"
STATUS="$REPO/logs/strata-dark-$STAMP.status"
CPU_GATE="$REPO/maintenance/cpu-power-state.sh"

say() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*"; }

preflight() {
  say "grace: ${GRACE}s, then llama-swap stops"
  say "unit: $(systemctl --user is-active llama-swap.service)"
  say "VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  say "RAM available: $(free -g | awk '/^Mem:/{print $7}') GiB"
  say "engine pids: $(pgrep -f 'serve/server.py --engine strata' | tr '\n' ' ')"
}

free_card() {
  systemctl --user stop llama-swap.service
  sleep 6
  pkill -f 'serve/server.py --engine strata' 2>/dev/null
  pkill -f 'build/strata --serve' 2>/dev/null
  for _ in $(seq 1 30); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    say "VRAM after stop: ${used} MiB, engine pids: $(pgrep -f 'build/strata --serve' | tr '\n' ' ')"
    [ "${used:-99999}" -lt 300 ] && return 0
    sleep 3
  done
  say "VRAM never dropped below 300 MiB, continuing anyway (the harness preflight will warn)"
  return 0
}

verify_serving() {
  for _ in $(seq 1 40); do
    pid="$(pgrep -f 'serve/server.py --engine strata' | head -1)"
    if [ -n "$pid" ]; then
      port="$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | grep -A1 -- '--port' | tail -1)"
      body="$(curl -s -m 5 "http://127.0.0.1:${port:-0}/health" 2>/dev/null)"
      if [ -n "$body" ]; then
        say "engine pid $pid port ${port} health: ${body:0:110}"
        return 0
      fi
    fi
    sleep 5
  done
  say "no engine health answer within 200 s (check: journalctl --user -u llama-swap.service -n 40)"
  return 1
}

main() {
  say "dark run $STAMP"
  # A ladder measures the CPU power state it runs in, and every row this repo has recorded was
  # measured with governor=powersave (intel_pstate's active-mode default) and min_perf_pct=17.
  # Ask before spending 20 dark minutes on a row that cannot be compared with anything.
  # FORCE_CPU=1 bypasses this gate only; FORCE=1 belongs to the harness (run with the router up)
  # and is deliberately a different variable, so bypassing the power gate cannot also mute the
  # preflight that refuses to bench onto a busy card.
  if [ ! -x "$CPU_GATE" ]; then
    say "cpu gate: $CPU_GATE missing or not executable, continuing unmeasured"
  else
    "$CPU_GATE" check 2>&1 | while IFS= read -r l; do say "cpu gate:${l#   }"; done
    gate="${PIPESTATUS[0]}"
    if [ "$gate" -ne 0 ] && [ "${FORCE_CPU:-0}" != "1" ]; then
      say "refusing to run: the CPU power state is not the one a t/s row needs."
      say "fix it (sudo $CPU_GATE apply-cpupower --yes, or sudo /usr/local/bin/l3ms-cpu-power apply if that grant is installed) or bypass with FORCE_CPU=1, in which case"
      say "the rows are comparable only to other FORCE_CPU=1 runs in the same state."
      echo "$STAMP rc=3 cpu-gate=failed log=$LOG" > "$STATUS"
      return 3
    fi
    [ "$gate" -ne 0 ] && say "cpu gate bypassed (FORCE_CPU=1)"
  fi
  preflight
  sleep "$GRACE"
  preflight
  free_card

  if [ -n "$ARMS" ]; then export ARMS; fi
  export ARMS TG_PROBES TG_WARMUP DEEP_SIZES PP_SIZES
  say "ladder: ${ARMS:-default (Exact-switch ladder on the kv-grow baseline)}"
  say "arms=${ARMS:-harness default}  warmups=$TG_WARMUP  tg-probes=$TG_PROBES  deep-sizes=${DEEP_SIZES:-off}  pp-sizes=$PP_SIZES"

  "$REPO/bench-models/bench-strata-tuning-ab.sh"
  rc=$?
  say "ladder exit $rc"

  systemctl --user start llama-swap.service
  say "unit after start: $(systemctl --user is-active llama-swap.service)"
  verify_serving || rc=$((rc + 10))

  journalctl --user -u llama-swap.service --since "-2 min" --no-pager 2>/dev/null |
    grep -iE "health check passed|error|panic" | tail -6
  say "done (rc=$rc)"
  echo "$STAMP rc=$rc arms=${ARMS:-default} tg_probes=$TG_PROBES log=$LOG" > "$STATUS"
}

main "$@" 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
