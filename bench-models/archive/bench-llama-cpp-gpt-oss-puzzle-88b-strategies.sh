#!/usr/bin/env bash
# gpt-oss-puzzle-88B (MXFP4_MOE) — manual MoE offload strategy experiments
#
# Strategy overview:
#   all-cpu-moe   All MoE expert tensors on CPU
#   partial-cpu   Fit-shaped split: blk4 down-proj + blk5+ expert mats on CPU (default)
#   up-down-cpu   Up+down experts on CPU; gate remains on GPU
#   up-cpu        Only up experts on CPU
#   none          No tensor override, use integer N_CPU_MOE path
#
# Note:
#   Puzzle uses heterogeneous per-layer expert counts and mixed attention windows.
#   Keep strategy selection empirical; use fit script for first-pass placement.
#   Comma-separated -ot lists fail in llama-bench (interpreted as multi-run lists);
#   use semicolons in OVERRIDE_TENSOR for multi-pattern overrides.
#
# Usage:
#   ./bench-llama-cpp-gpt-oss-puzzle-88b-strategies.sh
#   STRATEGY=all-cpu-moe ./bench-llama-cpp-gpt-oss-puzzle-88b-strategies.sh
#   STRATEGY=partial-cpu TASKS=1024,256 ./bench-llama-cpp-gpt-oss-puzzle-88b-strategies.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

LLAMA_BENCH="${LLAMA_BENCH:-${REPO_DIR}/vendor/llama.cpp-pr-test-21032/build/bin/llama-bench}"
MODEL="${MODEL:-/home/kchauhan/models/SamPurkis/gpt-oss-puzzle-88B-GGUF/gpt-oss-puzzle-88B.MXFP4_MOE.gguf}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-37}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"

STRATEGY="${STRATEGY:-partial-cpu}"

if [ -z "${OVERRIDE_TENSOR:-}" ]; then
  case "${STRATEGY}" in
    all-cpu-moe)
      OVERRIDE_TENSOR=".ffn_(up|down|gate_up|gate)_(ch|)exps=CPU"
      ;;
    partial-cpu)
      OVERRIDE_TENSOR="blk\.4\.ffn_down.*=CPU;blk\.(5|[6-9]|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU"
      ;;
    up-down-cpu)
      OVERRIDE_TENSOR=".ffn_(up|down|gate_up)_(ch|)exps=CPU"
      ;;
    up-cpu)
      OVERRIDE_TENSOR=".ffn_(up|gate_up)_(ch|)exps=CPU"
      ;;
    none)
      OVERRIDE_TENSOR=""
      ;;
    *)
      echo "Unknown STRATEGY '${STRATEGY}'." >&2
      echo "Valid: all-cpu-moe | partial-cpu | up-down-cpu | up-cpu | none" >&2
      exit 1
      ;;
  esac
fi

export LLAMA_BENCH MODEL TASKS N_GPU_LAYERS THREADS CPU_RANGE FA MMP BATCH_SIZE UBATCH_SIZE OVERRIDE_TENSOR

echo "# model    : ${MODEL}"
echo "# strategy : ${STRATEGY}"
echo "# override : ${OVERRIDE_TENSOR:-<none>}"
echo "# tasks    : ${TASKS}"
echo "# ngl      : ${N_GPU_LAYERS}"
echo "# threads  : ${THREADS} (pinned ${CPU_RANGE})"
echo "# fa       : ${FA}  mmp: ${MMP}"
echo "# batch    : ${BATCH_SIZE}  ubatch: ${UBATCH_SIZE}"
echo

exec "${SCRIPT_DIR}/run-llama-bench.sh"
