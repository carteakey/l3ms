#!/usr/bin/env bash
# Qwen3.8-Flash-Next — Quantization A/B Benchmark:
# AtomicChat AD-4.27bpw vs ISTA-DASLab GSQ-RCO IQ3_XXS
#
# Compares:
#   atomicchat : AtomicChat AD-4.27bpw (~54.5 GB weights, 33 shards)
#   daslab-iq3 : ISTA-DASLab GSQ-RCO IQ3_XXS (47.0 GB weights, 2 shards)
#
# Evaluates:
#   - Memory footprint: VRAM allocated + resident host RAM cushion
#   - Layer fitting: number of MoE layers offloaded to GPU via --fit
#   - Prompt processing: prefill throughput (pp) with unique prefix nonces
#   - Decode throughput: steady-state generation (tg) across repeat probes
#   - Sanity check: chain-of-thought and output formatting
#
# Prerequisite: pause llama-swap first
#   systemctl --user stop llama-swap.service
set -euo pipefail

REPO="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO/vendor/llama.cpp-master/build/bin/llama-server"
PORT=8022

declare -A MODELS=(
  [atomicchat]="$HOME/models/qwen38-flash-next/AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf"
  [daslab-iq3]="$HOME/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
)

FILLER="$(python3 -c "print('The quick brown fox jumps over the lazy dog while engineers profile memory bandwidth on hybrid MoE inference. ' * 30)")"

# Capture standardized system frontmatter
if [[ -x "$REPO/bench-models/bench-env.sh" ]]; then
  "$REPO/bench-models/bench-env.sh" || true
fi

probe_pp() {
  curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"[$(date +%s%N)] Summarize in one sentence: $FILLER\"}],\"max_tokens\":8,\"temperature\":0}" \
  | python3 -c "import json,sys; t=json.load(sys.stdin).get('timings',{}); n=t.get('prompt_n',0); print(f'{n/max(t.get(\"prompt_ms\",1),1)*1000:.1f} t/s over n={n}', end=' ')"
}

probe_tg() {
  curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"[$(date +%s%N)] Count from 1 to 40, one number per line.\"}],\"max_tokens\":64,\"temperature\":0}" \
  | python3 -c "import json,sys; t=json.load(sys.stdin).get('timings',{}); n=t.get('predicted_n',0); print(f'{n/max(t.get(\"predicted_ms\",1),1)*1000:.2f} t/s over n={n}', end=' ')"
}

probe_quality() {
  curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Name the four inner planets of the solar system in order, one per line, no commentary."}],"max_tokens":96,"temperature":0}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'].strip()[:220].replace(chr(10),' | '))"
}

run_arm() {
  local arm="$1"
  local model_path="${MODELS[$arm]}"
  local LOG="/tmp/opencode/quant-ab-$arm.log"
  mkdir -p /tmp/opencode

  if [[ ! -f "$model_path" ]]; then
    echo "== arm: $arm — model file missing ($model_path)"
    return 1
  fi

  echo "=================================================================="
  echo "== Running arm: $arm"
  echo "   Model: $model_path"
  echo "   Size:  $(du -sh "$model_path" | cut -f1)"
  echo "=================================================================="

  nohup taskset -c 0-11 "$BIN" -m "$model_path" --alias "quant-ab-$arm" \
    --fit on --fit-target 512 \
    -c 65536 --parallel 1 -b 4096 -ub 1024 \
    -fa on --jinja -ctk q8_0 -ctv q8_0 \
    -t 10 --threads-batch 12 --prio 2 \
    --lazy-mode on \
    --spec-type ngram-mod \
    --spec-ngram-mod-n-match 60 --spec-ngram-mod-n-min 12 --spec-ngram-mod-n-max 24 \
    --no-warmup \
    --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
  local SRV=$!
  trap 'kill $SRV 2>/dev/null || true' EXIT

  echo "Waiting for server to load and allocate buffers..."
  local ok=""
  for i in $(seq 1 300); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
      ok=1
      break
    fi
    if grep -qiE 'out of memory|cudaMalloc failed|std::bad_alloc|failed to allocate' "$LOG"; then
      echo "   [ERROR] Server hit memory allocation failure:"
      grep -iE 'out of memory|cudaMalloc failed|std::bad_alloc|failed to allocate' "$LOG" | head -3
      break
    fi
    sleep 2
  done

  if [[ -z "$ok" ]]; then
    echo "   Server failed to start. Last log lines:"
    tail -10 "$LOG"
    kill $SRV 2>/dev/null || true
    trap - EXIT
    return 1
  fi

  # Extract GPU layer offload from log
  local offload_info
  offload_info=$(grep -E 'llm_load_tensors: offloaded|llm_load_print_meta: model type|llm_load_tensors: [0-9]+/[0-9]+ layers offloaded' "$LOG" | tail -2 | tr '\n' ' | ')
  echo "   Offload meta: $offload_info"

  echo -n "   Warmup probe: " ; probe_tg; echo
  echo -n "   Prefill pp:   " ; probe_pp; probe_pp; echo
  echo -n "   Decode tg:    " ; probe_tg; probe_tg; echo
  echo "   Sanity test:  $(probe_quality)"
  echo "   VRAM Used:    $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader)"
  echo "   RAM Avail:    $(free -h | awk '/^Mem:/{print $7}') (Total: $(free -h | awk '/^Mem:/{print $2}'))"

  kill $SRV 2>/dev/null || true
  wait $SRV 2>/dev/null || true
  trap - EXIT
  sleep 3
}

for arm in ${ARMS:-atomicchat daslab-iq3}; do
  run_arm "$arm" || true
done

echo "=== Quantization A/B Benchmark Complete ==="
