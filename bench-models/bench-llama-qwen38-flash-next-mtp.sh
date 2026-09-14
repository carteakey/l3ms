#!/usr/bin/env bash
# Qwen3.8-Flash-Next — MTP (draft-mtp sidecar head) A/B on RTX 4070 12 GB
#
# Runs against the PR #27836 build (+ crusaderky detached-head patch) with the
# AtomicChat AD-4.27bpw split. 32k ctx so a Q4_K_M-quantized draft head fits VRAM
# (Q8_0 head = 3.5 GiB does not; head on CPU is impossible — RAM is at the
# 45.6 GiB design edge already).
#
# Usage:
#   bench-llama-qwen38-flash-next-mtp.sh            # baseline + MTP arms
#   ONLY=mtp bench-llama-qwen38-flash-next-mtp.sh   # single arm
#
# Requires: llama-swap stopped (full-VRAM experiment), performance governor.
set -euo pipefail

REPO="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$REPO/vendor/llama.cpp-pr-test-27836/build/bin/llama-server}"
MODEL="$HOME/models/qwen38-flash-next/AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf"
DRAFT="${DRAFT:-$HOME/models/qwen38-flash-next/mtp-heads/agentionai-mtp-Q4_K_M.gguf}"
PORT=8017
LOGDIR="$REPO/bench-models/logs"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOGDIR"

[ -x "$BIN" ] || { echo "llama-server not at $BIN"; exit 1; }
[ -f "$MODEL" ] || { echo "model missing: $MODEL"; exit 1; }

COMMON=(
  -m "$MODEL" --alias qwen38fn-bench
  -ngl 99 -ncmoe 46
  -c 32768 --parallel 1
  -b 4096 -ub 1024
  -fa on --jinja -fit off
  -ctk q8_0 -ctv q8_0
  -t 10 --threads-batch 12 --prio 2
  --tensor-read-lazy on
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
  --chat-template-kwargs '{"enable_thinking": false}'
  --no-warmup
  --host 127.0.0.1 --port "$PORT"
)

ARMS=()
if [ "${ONLY:-both}" != "mtp" ]; then ARMS+=("baseline"); fi
if [ "${ONLY:-both}" != "baseline" ]; then
  [ -f "$DRAFT" ] || { echo "draft head missing: $DRAFT"; exit 1; }
  ARMS+=("mtp")
fi

# 3 runs x 2 prompts x 2 temps per arm; server-reported timings only
PROMPTS=(
  "Explain how gated delta networks differ from standard attention in hybrid LLM architectures. Cover state compression, the delta rule, and why hybrids interleave both layer types."
  "Refactor this Python function to use pathlib and add error handling:\ndef load(p):\n    f = open(p)\n    d = f.read()\n    f.close()\n    return json.loads(d)"
)

probe() { # probe label temp prompt
  local label="$1" temp="$2" prompt="$3"
  python3 - "$PORT" "$temp" "$prompt" <<'EOF'
import json, sys, time, urllib.request
port, temp, prompt = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 320, "temperature": temp, "top_p": 0.95, "top_k": 20,
                   "stream": False}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
t0 = time.time()
r = json.loads(urllib.request.urlopen(req, timeout=600).read())
t = r.get("timings", {})
pp = t.get("prompt_n", 0) / max(t.get("prompt_ms", 1), 1) * 1000
tg = t.get("predicted_n", 0) / max(t.get("predicted_ms", 1), 1) * 1000
print(f"RESULT pp={pp:.1f} tg={tg:.2f} wall={time.time()-t0:.1f}s n={t.get('predicted_n',0)}")
EOF
}

for arm in "${ARMS[@]}"; do
  LOG="$LOGDIR/qwen38fn-mtp-$arm-$STAMP.log"
  echo "== arm: $arm (log: $LOG)"
  if [ "$arm" = "mtp" ]; then
    EXTRA_SPEC=()
    [ -n "${SPEC_PMIN:-}" ] && EXTRA_SPEC+=(--spec-draft-p-min "$SPEC_PMIN")
    nohup taskset -c "${cpu_range:-0-11}" "$BIN" "${COMMON[@]}" \
      --spec-draft-model "$DRAFT" --spec-type draft-mtp --spec-draft-n-max "${N_MAX:-2}" \
      --spec-draft-ngl 99 "${EXTRA_SPEC[@]}" > "$LOG" 2>&1 &
  else
    nohup taskset -c "${cpu_range:-0-11}" "$BIN" "${COMMON[@]}" > "$LOG" 2>&1 &
  fi
  SRV=$!
  trap 'kill $SRV 2>/dev/null || true' EXIT
  for i in $(seq 1 180); do curl -s "http://127.0.0.1:$PORT/health" | grep -q '"status":"ok"' && break; sleep 2; done
  curl -s "http://127.0.0.1:$PORT/health" | grep -q '"status":"ok"' || { echo "server failed to come up; tail:"; tail -20 "$LOG"; exit 1; }
  echo "   server up; warmup run"
  probe warmup 0.0 "Count from one to fifty, writing each number as a word." || true
  for temp in 0.0 1.0; do
    for pi in 0 1; do
      for run in 1 2 3; do
        out=$(probe "r$run" "$temp" "${PROMPTS[$pi]}")
        echo "   t=$temp p$pi r$run: $out"
        echo "$out" >> "$LOG.sum"
      done
    done
  done
  echo "   acceptance/spec counters:"
  grep -aiE 'accept|spec' "$LOG" | tail -6 || true
  kill $SRV; wait $SRV 2>/dev/null || true
  trap - EXIT
  sleep 3
done
echo "done. summaries in $LOGDIR/*.sum"
