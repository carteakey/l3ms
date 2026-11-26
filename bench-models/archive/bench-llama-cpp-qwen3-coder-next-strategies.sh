#!/usr/bin/env bash
# Qwen3-Coder-Next (UD-Q4_K_XL) — manual MoE offload strategy experiments
#
# Model: 48 layers, 512 experts (10 active/tok), ~47 GB, RTX 4070 12 GB
#
# Strategy overview (least → most VRAM):
#
#   all-cpu-moe   All MoE expert tensors on CPU; only attn + shared weights on GPU
#                 -ot ".ffn_.*_exps.=CPU"
#
#   up-down-cpu   Gate proj stays on GPU; up+down experts on CPU
#                 -ot ".ffn_(up|down)_exps.=CPU"
#
#   up-cpu        Only up projection experts on CPU; gate+down on GPU
#                 -ot ".ffn_(up)_exps.=CPU"
#
#   partial-cpu   Gate/up/down experts on CPU from layer 6 onwards only;
#                 early layers fully on GPU for better prefill
#                 -ot "\.(6|7|8|9|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(gate|up|down)_exps.=CPU"
#
#   none          No override; use N_CPU_MOE integer path if set
#
# Set OVERRIDE_TENSOR directly in env to bypass strategy selection entirely
# and supply a fully custom regex, e.g.:
#   OVERRIDE_TENSOR=".ffn_gate_exps.=CPU" ./bench-llama-cpp-qwen3-coder-next-strategies.sh
#
# Usage:
#   ./bench-llama-cpp-qwen3-coder-next-strategies.sh
#   STRATEGY=up-down-cpu ./bench-llama-cpp-qwen3-coder-next-strategies.sh
#   STRATEGY=partial-cpu TASKS=1024,256 THREADS=12 ./bench-llama-cpp-qwen3-coder-next-strategies.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

MODEL="${MODEL:-/mnt/lab/models/unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-UD-Q4_K_XL.gguf}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"

# STRATEGY selects the OVERRIDE_TENSOR preset below.
# Ignored when OVERRIDE_TENSOR is set directly in env.
STRATEGY="${STRATEGY:-all-cpu-moe}"

if [ -z "${OVERRIDE_TENSOR:-}" ]; then
  case "${STRATEGY}" in
    all-cpu-moe)
      # Smallest VRAM footprint: every MoE expert tensor goes to CPU.
      # Attention, norms, embeddings, and shared FFN stay on GPU.
      OVERRIDE_TENSOR=".ffn_.*_exps.=CPU"
      ;;
    up-down-cpu)
      # Mid tier: gate projection experts stay on GPU; up+down go to CPU.
      # Keeps the hot routing path on GPU while freeing most expert VRAM.
      OVERRIDE_TENSOR=".ffn_(up|down)_exps.=CPU"
      ;;
    up-cpu)
      # More VRAM available: only up projection experts spill to CPU.
      # Gate and down experts remain on GPU.
      OVERRIDE_TENSOR=".ffn_(up)_exps.=CPU"
      ;;
    partial-cpu)
      # Layers 0-5 fully on GPU for fast early-layer prefill.
      # From layer 6 onward, all three expert projections go to CPU.
      OVERRIDE_TENSOR="\.(6|7|8|9|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(gate|up|down)_exps.=CPU"
      ;;
    none)
      # No tensor override. Set N_CPU_MOE in env to use integer MoE offload.
      OVERRIDE_TENSOR=""
      ;;
    *)
      echo "Unknown STRATEGY '${STRATEGY}'." >&2
      echo "Valid: all-cpu-moe | up-down-cpu | up-cpu | partial-cpu | none" >&2
      exit 1
      ;;
  esac
fi

export MODEL TASKS N_GPU_LAYERS THREADS CPU_RANGE FA MMP OVERRIDE_TENSOR

echo "# strategy : ${STRATEGY}"
echo "# override : ${OVERRIDE_TENSOR:-<none>}"
echo "# tasks    : ${TASKS}"
echo "# ngl      : ${N_GPU_LAYERS}"
echo "# threads  : ${THREADS} (pinned ${CPU_RANGE})"
echo "# fa       : ${FA}  mmp: ${MMP}"
echo

exec "${SCRIPT_DIR}/run-llama-bench.sh"
