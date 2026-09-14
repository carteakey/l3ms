#!/usr/bin/env bash
# Fresh-prompt comparison: baseline, MTP-2, and ngram-mod chained before MTP-2.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/vendor/ik_llama.cpp/build/bin/llama-spec-bench}"
MODEL="${MODEL:-/home/kchauhan/models/qwen38-flash-next/AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf}"
HEAD="${HEAD:-/home/kchauhan/models/qwen38-flash-next/mtp-heads/agentionai-mtp-Q4_K_M.gguf}"
PROMPTS="${PROMPTS:-$ROOT/bench-models/prompts/qwen38-flash-next-unique.jsonl}"
CTX="${CTX:-16384}"
NCMOE="${NCMOE:-46}"
OUT="${OUT:-$ROOT/bench-models/logs/results/ik-mtp-unique-16k-20260909}"
RUN_ARMS="${RUN_ARMS:-base,mtp1,mtp2,ngram4-mtp1,ngram4-mtp2}"
mkdir -p "$OUT"

for path in "$BIN" "$MODEL" "$HEAD" "$PROMPTS"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }
done

router_was_active=false
if systemctl --user is-active --quiet llama-swap.service; then
  router_was_active=true
  systemctl --user stop llama-swap.service
fi
restore_router() {
  if [[ "$router_was_active" == true ]]; then
    systemctl --user start llama-swap.service
  fi
}
trap restore_router EXIT

common=(
  -m "$MODEL" --defer-ple --prefetch-experts --prefetch-experts-threads 8
  -ngl 99 -ncmoe "$NCMOE"
  -c "$CTX" -b 2048 -ub 512 -t 10 -tb 12
  -fa on -ctk q8_0 -ctv q8_0 -np 1 --jinja
  --temp 0 --seed 123 --prompts "$PROMPTS" --repeat 1 --output-format jsonl
)
mtp=( -md "$HEAD" -ngld 99 --spec-ckpt-mode gpu-fallback )

run_arm() {
  local label="$1"
  shift
  if [[ ",$RUN_ARMS," != *",$label,"* ]]; then
    return
  fi
  if [[ -s "$OUT/$label.jsonl" ]] && tail -n 1 "$OUT/$label.jsonl" | grep -q '"row_type":"summary"'; then
    echo "Skipping completed arm: $label"
    return
  fi
  echo "Running $label"
  set +e
  GGML_CUDA_NO_PINNED=1 "$BIN" "${common[@]}" "$@" \
    >"$OUT/$label.jsonl" 2>"$OUT/$label.log"
  local rc=$?
  set -e
  printf '%s\n' "$rc" >"$OUT/$label.exit"
  if (( rc == 0 )); then
    tail -n 1 "$OUT/$label.jsonl"
  else
    echo "Arm $label failed (exit $rc): $OUT/$label.log" >&2
  fi
}

run_arm base
run_arm mtp1 "${mtp[@]}" --spec-type mtp:n_max=1,p_min=0.7
run_arm mtp2 "${mtp[@]}" --spec-type mtp:n_max=2,p_min=0.7
run_arm ngram4-mtp1 "${mtp[@]}" \
  --spec-type ngram-mod:n_min=4 --spec-type mtp:n_max=1,p_min=0.7
run_arm ngram4-mtp2 "${mtp[@]}" \
  --spec-type ngram-mod:n_min=4 --spec-type mtp:n_max=2,p_min=0.7

echo "Unique-prompt results: $OUT"
