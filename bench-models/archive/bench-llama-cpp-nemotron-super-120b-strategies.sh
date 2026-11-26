#!/usr/bin/env bash
# NVIDIA Nemotron 3 Super 120B A12B (UD-Q3_K_XL) — llama.cpp MoE strategy experiments
#
# Fit probe baseline on this host (32k ctx, 512 MiB margin) landed:
#   -ngl 89
#   blk 2+ experts on CPU (+ blk1 ffn_down)
#
# Usage:
#   ./bench-llama-cpp-nemotron-super-120b-strategies.sh
#   STRATEGY=partial-cpu ./bench-llama-cpp-nemotron-super-120b-strategies.sh
#   STRATEGY=up-down-cpu TASKS=1024,256 ./bench-llama-cpp-nemotron-super-120b-strategies.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

LLAMA_BENCH="${LLAMA_BENCH:-${REPO_DIR}/vendor-forks/llama.cpp-copilot/build/bin/llama-bench}"
MODEL="${MODEL:-/mnt/lab/models/unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF/UD-Q3_K_XL/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q3_K_XL-00001-of-00003.gguf}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"

STRATEGY="${STRATEGY:-all-cpu-moe}"

if [ -z "${OVERRIDE_TENSOR:-}" ]; then
  case "${STRATEGY}" in
    all-cpu-moe)
      OVERRIDE_TENSOR=".ffn_.*_(ch|)exps.=CPU"
      ;;
    up-down-cpu)
      OVERRIDE_TENSOR=".ffn_(up|down)_(ch|)exps.=CPU"
      ;;
    up-cpu)
      OVERRIDE_TENSOR=".ffn_(up)_(ch|)exps.=CPU"
      ;;
    partial-cpu)
      OVERRIDE_TENSOR="blk\\.([2-9]|[1-9][0-9])\\.ffn_(gate|up|down)_(ch|)exps=CPU"
      ;;
    none)
      OVERRIDE_TENSOR=""
      ;;
    *)
      echo "Unknown STRATEGY '${STRATEGY}'." >&2
      echo "Valid: all-cpu-moe | up-down-cpu | up-cpu | partial-cpu | none" >&2
      exit 1
      ;;
  esac
fi

export LLAMA_BENCH MODEL TASKS N_GPU_LAYERS THREADS CPU_RANGE FA MMP OVERRIDE_TENSOR

echo "# strategy : ${STRATEGY}"
echo "# override : ${OVERRIDE_TENSOR:-<none>}"
echo "# tasks    : ${TASKS}"
echo "# ngl      : ${N_GPU_LAYERS}"
echo "# threads  : ${THREADS} (pinned ${CPU_RANGE})"
echo "# fa       : ${FA}  mmp: ${MMP}"
echo

exec "${SCRIPT_DIR}/run-llama-bench.sh"
