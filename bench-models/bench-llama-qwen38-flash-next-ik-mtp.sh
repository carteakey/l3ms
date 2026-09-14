#!/usr/bin/env bash
# Compare ik_llama.cpp baseline vs MTP on Qwen3.8-Flash-Next.
# The router is restored on exit. Override NCMOE, CTX, REPEAT, or PREDICT as needed.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/vendor/ik_llama.cpp/build/bin/llama-spec-bench}"
MODEL="${MODEL:-/home/kchauhan/models/qwen38-flash-next/AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf}"
HEAD="${HEAD:-/home/kchauhan/models/qwen38-flash-next/mtp-heads/agentionai-mtp-Q4_K_M.gguf}"
NCMOE="${NCMOE:-46}"
CTX="${CTX:-16384}"
REPEAT="${REPEAT:-3}"
PREDICT="${PREDICT:-128}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${OUT:-$ROOT/bench-models/logs/results/ik-mtp-$STAMP}"
mkdir -p "$OUT"

for path in "$BIN" "$MODEL" "$HEAD"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing $path" >&2
    exit 1
  fi
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

echo "Running baseline: ncmoe=$NCMOE ctx=$CTX"
GGML_CUDA_NO_PINNED=1 "$BIN" "${common[@]}" \
  >"$OUT/base.jsonl" 2>"$OUT/base.log"

echo "Running MTP n_max=1"
GGML_CUDA_NO_PINNED=1 "$BIN" "${common[@]}" \
  -md "$HEAD" -ngld 99 \
  --spec-type mtp:n_max=1,p_min=0.7 --spec-ckpt-mode gpu-fallback \
  >"$OUT/mtp1.jsonl" 2>"$OUT/mtp1.log"

echo "Results: $OUT"
tail -n 1 "$OUT/base.jsonl"
tail -n 1 "$OUT/mtp1.jsonl"
