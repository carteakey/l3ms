#!/usr/bin/env bash
# Qwen3.8-Flash-Next — MTP param sweep, warm-pool protocol
#
# Arms:
#   baseline : proven 0b7d6d57d build, prod combo params (ncmoe 46, agentionai Q4_K_M)
#   144-*    : unsloth PR #144 build (586b15ef8), shared-Q8_0 head, param variants
#
# Warm protocol per arm (fresh load):
#   2 throwaway generations (spec pool + expert page-in warm-up, discarded)
#   3x counting probe  (1->40, 64 tok, temp 0)  -> steady state = mean of last 2
#   2x code probe      (pathlib refactor, 320 tok, temp 0)
#   acceptance counters from server log
#
# Requires: llama-swap stopped (full-VRAM experiment):
#   systemctl --user stop llama-swap.service
set -uo pipefail

REPO="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$HOME/models/qwen38-flash-next/AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf"
HEAD_Q4="$HOME/models/qwen38-flash-next/mtp-heads/agentionai-mtp-Q4_K_M.gguf"
HEAD_SHARED="$HOME/models/unsloth/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
BIN_BASE="$REPO/vendor/llama.cpp-pr-test-28023-28068-27941/build/bin/llama-server-mtp0b7d"
BIN_144="$REPO/vendor/llama.cpp-mtp144-wt/build/bin/llama-server"
PORT=8023
LOGDIR="$REPO/bench-models/logs/results"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/mtp144-sweep-$STAMP.txt"

systemctl --user is-active llama-swap.service >/dev/null 2>&1 && {
  echo "ERROR: llama-swap is active. Stop it first:"; echo "  systemctl --user stop llama-swap.service"; exit 1; }

COMBO="--spec-type draft-mtp,ngram-mod --spec-draft-n-max 2 --spec-draft-p-min 0.7 --spec-ngram-mod-n-match 60 --spec-draft-ngl 99"

# label|binary|ncmoe|head|extra-args
ARMS=(
  ${SWEEP_ARMS:+"$@"}
)
if [ -z "${SWEEP_ARMS:-}" ]; then ARMS+=(
  "baseline|$BIN_BASE|46|$HEAD_Q4|$COMBO"
  "144-combo|$BIN_144|47|$HEAD_SHARED|$COMBO"
  "144-plain|$BIN_144|47|$HEAD_SHARED|--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.7 --spec-draft-ngl 99"
  "144-p075|$BIN_144|47|$HEAD_SHARED|--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75 --spec-draft-ngl 99"
  "144-p050|$BIN_144|47|$HEAD_SHARED|--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.5 --spec-draft-ngl 99"
  "144-nmax3|$BIN_144|47|$HEAD_SHARED|--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.7 --spec-draft-ngl 99"
  "144-psplit|$BIN_144|47|$HEAD_SHARED|--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.7 --spec-draft-p-split 0.10 --spec-draft-ngl 99"
  "144-ncmoe46|$BIN_144|46|$HEAD_SHARED|--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.7 --spec-draft-ngl 99 -ub 512"
) ; fi

CODE_PROMPT='Refactor this Python function to use pathlib and add error handling:\ndef load(p):\n    f = open(p)\n    d = f.read()\n    f.close()\n    return json.loads(d)'

probe_tg() { # max_tokens prompt -> "tg x.x n=N"
  curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}],\"max_tokens\":$1,\"temperature\":0,\"top_p\":0.95,\"top_k\":20}" \
  | python3 -c "
import json,sys
try: t=json.load(sys.stdin).get('timings',{})
except Exception: print('tg PARSE-FAIL n=0'); sys.exit()
n=t.get('predicted_n',0); ms=max(t.get('predicted_ms',1),1)
print(f'tg {n/ms*1000:.2f} n={n}')"
}

run_arm() {
  local spec="$1"
  IFS='|' read -r label bin ncmoe head extra <<< "$spec"
  local LOG="$LOGDIR/mtp144-$label-$STAMP.log"
  echo "== arm: $label ($(basename "$bin") @ $(git -C "$(dirname "$bin")" log --oneline -1 2>/dev/null | awk '{print $1}' || echo '?'), ncmoe $ncmoe, head $(basename "$head"))"
  echo "== arm: $label" >> "$SUMMARY"
  GGML_CUDA_GRAPH_OPT=1 nohup taskset -c 0-11 "$bin" -m "$MODEL" \
    --spec-draft-model "$head" \
    ${ncmoe:+-ngl 99 -ncmoe "$ncmoe"} \
    -c 32768 --parallel 1 \
    -b 4096 -ub 1024 \
    -fa on --jinja ${FITARGS:--fit off} \
    -ctk q8_0 -ctv q8_0 \
    -t 10 --threads-batch 12 --prio 2 \
    --lazy-mode on \
    $extra \
    --no-warmup \
    --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
  local SRV=$!
  trap 'kill $SRV 2>/dev/null || true' EXIT
  local ok=""
  for _ in $(seq 1 240); do
    curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }
    grep -qiE 'out of memory|cudaMalloc failed|std::bad_alloc|failed to allocate|error: [a-z ]*(argument|tensor)|not found' "$LOG" && break
    kill -0 $SRV 2>/dev/null || break
    sleep 2
  done
  if [ -z "$ok" ]; then
    echo "   LOAD FAILED: $(grep -aiE 'out of memory|cudaMalloc|error|not found' "$LOG" | tail -2 | tr '\n' ' ')"
    echo "   LOAD FAILED" >> "$SUMMARY"
    kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true; trap - EXIT; sleep 3; return
  fi
  echo "   loaded. VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)  RAM avail: $(free -g | awk '/^Mem:/{print $7}') GiB"
  echo -n "   warmup1: "; probe_tg 48 "Count from one to thirty, writing each number as a word."; echo
  echo -n "   warmup2: "; probe_tg 48 "Count from one to thirty, writing each number as a word."; echo
  local t1 t2 t3 c1 c2
  t1=$(probe_tg 64 "Count from 1 to 40, one number per line.")
  t2=$(probe_tg 64 "Count from 1 to 40, one number per line.")
  t3=$(probe_tg 64 "Count from 1 to 40, one number per line.")
  echo "   counting (steady=last2): $t1 | $t2 | $t3"
  echo "counting: $t1 | $t2 | $t3" >> "$SUMMARY"
  c1=$(probe_tg 320 "$CODE_PROMPT")
  c2=$(probe_tg 320 "$CODE_PROMPT")
  echo "   code: $c1 | $c2"
  echo "code: $c1 | $c2" >> "$SUMMARY"
  echo "   acceptance: $(grep -aoE 'acceptance[^,]*|draft_accept[^,]*' "$LOG" | tail -2 | tr '\n' ' ')"
  grep -aE 'accept' "$LOG" | tail -2 >> "$SUMMARY" 2>/dev/null || true
  kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true; trap - EXIT
  sleep 3
}

for arm in "${ARMS[@]}"; do run_arm "$arm"; done
echo; echo "==== SUMMARY ($SUMMARY) ===="; cat "$SUMMARY"
