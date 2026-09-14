#!/usr/bin/env bash
# Qwen3.8-Flash-Next — 2x2 attribution: GGML_CUDA_GRAPH_OPT=1 x placement
# (fit-on 512 vs manual -ncmoe 46), at production 64k ctx.
#
# Motivation: live-router tg (~29 t/s) far exceeded direct-launch bench arms
# (~19 t/s); the bench omitted the router's GGML_CUDA_GRAPH_OPT=1 env. This
# isolates that env from the fit-on placement switch.
#
# Usage: stop llama-swap first (full-VRAM experiment), then run this script.
set -euo pipefail

REPO="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO/vendor/llama.cpp-pr-test-27742/build/bin/llama-server"
MODEL="$HOME/models/qwen38-flash-next/AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf"
PORT=8018
PROMPT='Refactor this Python function to use pathlib and add error handling:\ndef load(p):\n    f = open(p)\n    d = f.read()\n    f.close()\n    return json.loads(d)'

probe() {
  curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"'"$PROMPT"'"}],"max_tokens":320,"temperature":0.0}' \
  | python3 -c "import json,sys; t=json.load(sys.stdin).get('timings',{}); print(f'{t.get(\"predicted_n\",0)/max(t.get(\"predicted_ms\",1),1)*1000:.2f}')"
}

run_arm() { # run_arm <label> <env-string> <placement-args...>
  local label="$1" envstr="$2"; shift 2
  local LOG="/tmp/opencode/go2x2-$label.log"
  echo "== arm: $label (env: ${envstr:-none})"
  # shellcheck disable=SC2086
  env $envstr nohup taskset -c 0-11 "$BIN" \
    -m "$MODEL" --alias "qwen38fn-$label" \
    "$@" \
    -c 65536 --parallel 1 -b 4096 -ub 1024 \
    -fa on --jinja -ctk q8_0 -ctv q8_0 \
    -t 10 --threads-batch 12 --prio 2 \
    --tensor-read-lazy on \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --no-warmup --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
  local SRV=$!
  trap 'kill $SRV 2>/dev/null || true' EXIT
  local ok=""
  for i in $(seq 1 150); do
    curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }
    grep -qiE 'out of memory|cudaMalloc failed' "$LOG" && { echo "   LOAD ERROR"; break; }
    sleep 2
  done
  [ -n "$ok" ] || { echo "   server failed:"; tail -5 "$LOG"; kill $SRV 2>/dev/null || true; trap - EXIT; return 1; }
  probe >/dev/null  # warmup (table faults, first-touch)
  local vals=()
  for r in 1 2 3 4 5; do vals+=("$(probe)"); done
  echo "   tg: ${vals[*]}"
  echo "   median: $(printf '%s\n' "${vals[@]}" | sort -n | sed -n 3p)"
  echo "   VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  echo "   graph counters: $(grep -aic 'graph' "$LOG") graph-mention lines; last: $(grep -ai 'graph' "$LOG" | tail -1 | cut -c1-120)"
  kill $SRV 2>/dev/null || true
  wait $SRV 2>/dev/null || true
  trap - EXIT
  sleep 3
}

run_arm ncmoe-noenv  ""  -ngl 99 -ncmoe 46
run_arm ncmoe-graph  "GGML_CUDA_GRAPH_OPT=1"  -ngl 99 -ncmoe 46
run_arm fit-noenv    ""  --fit on --fit-target 512
run_arm fit-graph    "GGML_CUDA_GRAPH_OPT=1"  --fit on --fit-target 512

echo "done."
