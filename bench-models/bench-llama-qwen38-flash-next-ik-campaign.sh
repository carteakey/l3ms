#!/usr/bin/env bash
# Resumable ik_llama Qwen3.8-Flash-Next speculative-decoding campaign.
# Each arm writes its own JSONL/log; completed arms are skipped on rerun.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/vendor/ik_llama.cpp/build/bin/llama-spec-bench}"
MODEL="${MODEL:-/home/kchauhan/models/qwen38-flash-next/AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf}"
HEAD="${HEAD:-/home/kchauhan/models/qwen38-flash-next/mtp-heads/agentionai-mtp-Q4_K_M.gguf}"
CTX="${CTX:-16384}"
NCMOE="${NCMOE:-46}"
REPEAT="${REPEAT:-3}"
PREDICT="${PREDICT:-128}"
CAMPAIGN="${CAMPAIGN:-ik-mtp-16k-params-20260909}"
OUT="${OUT:-$ROOT/bench-models/logs/results/$CAMPAIGN}"
mkdir -p "$OUT"

for path in "$BIN" "$MODEL" "$HEAD"; do
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
  -m "$MODEL" --defer-ple -ngl 99 -ncmoe "$NCMOE"
  -c "$CTX" -b 2048 -ub 512 -t 10 -tb 12
  -fa on -ctk q8_0 -ctv q8_0 -np 1 --jinja
  --temp 0 --seed 123 --predict "$PREDICT"
  --task code,story --repeat "$REPEAT" --output-format jsonl
)

run_arm() {
  local label="$1"
  shift
  if [[ -s "$OUT/$label.jsonl" ]] && tail -n 1 "$OUT/$label.jsonl" | grep -q '"row_type":"summary"'; then
    echo "Skipping completed arm: $label"
    return
  fi
  printf '%q ' "$BIN" "${common[@]}" "$@" >"$OUT/$label.command"
  printf '\n' >>"$OUT/$label.command"
  echo "Running $label: ctx=$CTX ncmoe=$NCMOE"
  set +e
  GGML_CUDA_NO_PINNED=1 "$BIN" "${common[@]}" "$@" \
    >"$OUT/$label.jsonl" 2>"$OUT/$label.log"
  local rc=$?
  set -e
  printf '%s\n' "$rc" >"$OUT/$label.exit"
  if (( rc != 0 )); then
    echo "Arm $label failed (exit $rc); see $OUT/$label.log" >&2
  else
    tail -n 1 "$OUT/$label.jsonl"
  fi
}

mtp=( -md "$HEAD" -ngld 99 --spec-ckpt-mode gpu-fallback )

run_arm base
run_arm mtp1-p070 "${mtp[@]}" --spec-type mtp:n_max=1,p_min=0.7
run_arm mtp1-p000 "${mtp[@]}" --spec-type mtp:n_max=1,p_min=0.0
run_arm mtp1-p050 "${mtp[@]}" --spec-type mtp:n_max=1,p_min=0.5
run_arm mtp2-p070 "${mtp[@]}" --spec-type mtp:n_max=2,p_min=0.7
run_arm ngram4-mtp1-p070 "${mtp[@]}" \
  --spec-type ngram-mod:n_min=4 --spec-type mtp:n_max=1,p_min=0.7
run_arm mtp1-p070-muge "${mtp[@]}" -muge --spec-type mtp:n_max=1,p_min=0.7

echo "Campaign results: $OUT"
