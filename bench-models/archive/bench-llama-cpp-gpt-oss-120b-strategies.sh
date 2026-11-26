#!/usr/bin/env bash
# gpt-oss-120b (mxfp4) — manual MoE offload strategy experiments
#
# Model: 36 layers, 128 experts (4 active/tok), ~60 GB, RTX 4070 12 GB
# Architecture: gpt-oss — pure _exps tensors, no shared experts (_shexp).
# Simple regex patterns work correctly here (no (ch|) needed).
#
# Strategy overview (least → most VRAM):
#
#   all-cpu-moe   All MoE expert tensors on CPU; only attn + norms on GPU
#                 -ot ".ffn_.*_exps.=CPU"
#
#   partial-cpu   Experts on CPU from layer 4 onwards; blk 0-3 fully on GPU
#                 -ot "blk\.([4-9]|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(gate|up|down)_exps=CPU"
#
#   up-down-cpu   Gate proj stays on GPU; up+down experts on CPU
#                 -ot ".ffn_(up|down)_exps=CPU"
#
#   up-cpu        Only up projection experts on CPU; gate+down on GPU
#                 -ot ".ffn_up_exps=CPU"
#                 128 experts × fewer projections — more likely to fit than on 512-expert models
#
#   none          No override; use N_CPU_MOE integer path if set
#
# Set OVERRIDE_TENSOR directly in env to bypass strategy selection entirely:
#   OVERRIDE_TENSOR=".ffn_gate_exps=CPU" ./bench-llama-cpp-gpt-oss-120b-strategies.sh
#
# Usage:
#   ./bench-llama-cpp-gpt-oss-120b-strategies.sh
#   STRATEGY=all-cpu-moe  ./bench-llama-cpp-gpt-oss-120b-strategies.sh
#   STRATEGY=partial-cpu  ./bench-llama-cpp-gpt-oss-120b-strategies.sh
#   STRATEGY=up-down-cpu  ./bench-llama-cpp-gpt-oss-120b-strategies.sh
#   STRATEGY=up-cpu       ./bench-llama-cpp-gpt-oss-120b-strategies.sh
#   STRATEGY=partial-cpu TASKS=1024,256 THREADS=12 ./bench-llama-cpp-gpt-oss-120b-strategies.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

export GGML_CUDA_GRAPH_OPT="${GGML_CUDA_GRAPH_OPT:-1}"

MODEL="${MODEL:-/mnt/lab/models/ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"

BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"

STRATEGY="${STRATEGY:-all-cpu-moe}"

if [ -z "${OVERRIDE_TENSOR:-}" ]; then
  case "${STRATEGY}" in
    all-cpu-moe)
      # Smallest VRAM footprint: every expert tensor to CPU.
      # Attention, norms, embeddings stay on GPU.
      OVERRIDE_TENSOR=".ffn_.*_exps=CPU"
      ;;
    partial-cpu)
      # Blk 0-3 fully on GPU for fast early-layer prefill.
      # From blk 4 onward, all three expert projections go to CPU.
      OVERRIDE_TENSOR="blk\.([4-9]|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(gate|up|down)_exps=CPU"
      ;;
    up-down-cpu)
      # Gate projection experts stay on GPU; up+down go to CPU.
      OVERRIDE_TENSOR=".ffn_(up|down)_exps=CPU"
      ;;
    up-cpu)
      # Only up projection experts spill to CPU; gate+down on GPU.
      # 128 experts makes this more viable than on 512-expert models.
      OVERRIDE_TENSOR=".ffn_up_exps=CPU"
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

export MODEL TASKS N_GPU_LAYERS THREADS CPU_RANGE FA MMP BATCH_SIZE UBATCH_SIZE OVERRIDE_TENSOR

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
