#!/usr/bin/env bash
# Qwen3.8-Flash-Next — prefill (pp) matrix: placement x ctx x THP
#
# Arms (override with ARMS="fit64 ncmoe64 ncmoe32"):
#   fit64   : --fit on --fit-target 512, 64k   (current production base config)
#   ncmoe64 : -ngl 99 -ncmoe 46 -fit off, 64k  (pre-2026-08-28 config)
#   ncmoe32 : -ngl 99 -ncmoe 46 -fit off, 32k  (doc's original pp-ladder conditions)
#
# Each arm: fresh load, 1 warmup probe (expert-page fault-in), then 2 measured
# probes of the same ~617-token prompt. THP state is printed, not toggled.
# Requires llama-swap stopped (full-VRAM experiment).
set -euo pipefail

REPO="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO/vendor/llama.cpp-pr-test-27742/build/bin/llama-server"
MODEL="$HOME/models/qwen38-flash-next/AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf"
PORT=8019
THP="$(cat /sys/kernel/mm/transparent_hugepage/enabled)"
FILLER="$(python3 -c "print('The quick brown fox jumps over the lazy dog while engineers profile memory bandwidth on hybrid MoE inference. ' * 30)")"

probe() {
  # unique prefix per call forces a full re-prefill (otherwise KV prefix reuse
  # makes prompt_n collapse and the pp number meaningless)
  curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"[$(date +%s%N)] Summarize in one sentence: $FILLER\"}],\"max_tokens\":8,\"temperature\":0}" \
  | python3 -c "import json,sys; t=json.load(sys.stdin).get('timings',{}); n=t.get('prompt_n',0); print(f'{n/max(t.get(\"prompt_ms\",1),1)*1000:.0f} t/s over n={n}', end=' ')"
}

run_arm() { # run_arm <label> <ctx> <placement-args...>
  local label="$1" ctx="$2"; shift 2
  local LOG="/tmp/opencode/pp-$label.log"
  local EXTRA_ENV="${EXTRA_ENV:-}"
  echo "== arm: $label (ctx $ctx)  [THP: $THP]"
  env $EXTRA_ENV nohup taskset -c 0-11 "$BIN" -m "$MODEL" --alias "pp-$label" "$@" \
    -c "$ctx" --parallel 1 -b 4096 -ub 1024 \
    -fa on --jinja -ctk q8_0 -ctv q8_0 \
    -t 10 --threads-batch 12 --prio 2 \
    --tensor-read-lazy on --no-warmup \
    --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
  local SRV=$!
  trap 'kill $SRV 2>/dev/null || true' EXIT
  local ok=""
  for i in $(seq 1 150); do
    curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }
    grep -qiE 'out of memory|cudaMalloc failed' "$LOG" && { echo "   LOAD ERROR"; break; }
    sleep 2
  done
  [ -n "$ok" ] || { echo "   server failed:"; tail -3 "$LOG"; kill $SRV 2>/dev/null || true; trap - EXIT; return 1; }
  echo -n "   warmup: " ; probe; echo
  echo -n "   probes: " ; probe; probe; echo
  echo "   VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  kill $SRV 2>/dev/null || true
  wait $SRV 2>/dev/null || true
  trap - EXIT
  sleep 3
}

for arm in ${ARMS:-fit64 ncmoe64 ncmoe32}; do
  case "$arm" in
    fit64)   run_arm fit64   65536 --fit on --fit-target 512 ;;
    ncmoe64) run_arm ncmoe64 65536 -ngl 99 -ncmoe 46 -fit off ;;
    ncmoe32) run_arm ncmoe32 32768 -ngl 99 -ncmoe 46 -fit off ;;
  esac
done
echo "done."
