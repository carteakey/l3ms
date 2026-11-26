#!/usr/bin/env bash
# Qwen3.5-122B-A10B (UD-IQ4_XS) — manual MoE offload strategy experiments
#
# Model: 48 layers, 256 experts (8 active/tok), ~70 GB, RTX 4070 12 GB
#
# Strategy overview (least → most VRAM):
#
#   all-cpu-moe   All MoE expert tensors on CPU; only attn + shared weights on GPU
#                 -ot ".ffn_.*_exps.=CPU"
#
#   partial-cpu   Gate/up/down experts on CPU from layer 3 onwards (fit-params recommended);
#                 early layers 0-2 fully on GPU for better prefill.
#                 Matches both routed experts (_exps) and shared experts (_shexp).
#                 -ot "blk\.(3|[4-9]|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(up|down|gate)_(ch|)exps=CPU"
#
#   up-down-cpu   Gate proj stays on GPU; up+down experts (routed + shared) on CPU
#                 -ot ".ffn_(up|down)_(ch|)exps=CPU"
#
#   up-cpu        Only up projection experts (routed + shared) on CPU; gate+down on GPU
#                 -ot ".ffn_up_(ch|)exps=CPU"
#                 WARNING: likely OOM on 12 GB VRAM — 256 experts × gate+down × 48 layers
#
#   none          No override; use N_CPU_MOE integer path if set
#
# Note: Qwen3.5-122B-A10B has 256 experts (vs 512 in Qwen3-Coder-Next), so
# up-down-cpu and even up-cpu are more likely to fit in 12 GB than on larger
# models. Test carefully and fall back to all-cpu-moe if you hit CUDA OOM.
#
# IMPORTANT: This model uses tensor names with BOTH routed experts (_exps) and
# shared experts (_shexp). Patterns must match "_(ch|)exps" to catch both.
# Simple patterns like ".ffn_.*_exps.=CPU" only match routed experts and will
# cause CUDA OOM because the shared experts remain on GPU unexpectedly.
# The fit-params output confirmed the correct pattern form: ffn_(up|down|gate)_(ch|)exps=CPU
#
# Set OVERRIDE_TENSOR directly in env to bypass strategy selection entirely
# and supply a fully custom regex, e.g.:
#   OVERRIDE_TENSOR=".ffn_gate_exps.=CPU" ./bench-llama-cpp-qwen3-5-122b-a10b-strategies.sh
#
# Usage:
#   ./bench-llama-cpp-qwen3-5-122b-a10b-strategies.sh
#   STRATEGY=all-cpu-moe  ./bench-llama-cpp-qwen3-5-122b-a10b-strategies.sh
#   STRATEGY=partial-cpu  ./bench-llama-cpp-qwen3-5-122b-a10b-strategies.sh
#   STRATEGY=up-down-cpu  ./bench-llama-cpp-qwen3-5-122b-a10b-strategies.sh
#   STRATEGY=up-cpu       ./bench-llama-cpp-qwen3-5-122b-a10b-strategies.sh
#   STRATEGY=partial-cpu TASKS=1024,256 THREADS=12 ./bench-llama-cpp-qwen3-5-122b-a10b-strategies.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

MODEL="${MODEL:-/mnt/lab/models/unsloth/Qwen3.5-122B-A10B-GGUF/UD-IQ4_XS/Qwen3.5-122B-A10B-UD-IQ4_XS-00001-of-00003.gguf}"
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
      # Matches both routed experts (_exps) and shared experts (_shexp via _(ch|)exps).
      # Attention, norms, embeddings, and shared non-expert FFN stay on GPU.
      OVERRIDE_TENSOR=".ffn_(up|down|gate)_(ch|)exps=CPU"
      ;;
    partial-cpu)
      # Layers 0-2 fully on GPU for fast early-layer prefill.
      # From layer 3 onward, all three expert projections (routed + shared) go to CPU.
      # Pattern derived from llama-fit-params output for this model.
      OVERRIDE_TENSOR="blk\.(3|[4-9]|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(up|down|gate)_(ch|)exps=CPU"
      ;;
    up-down-cpu)
      # Mid tier: gate projection experts stay on GPU; up+down (routed + shared) go to CPU.
      # Keeps the hot routing path on GPU while freeing most expert VRAM.
      OVERRIDE_TENSOR=".ffn_(up|down)_(ch|)exps=CPU"
      ;;
    up-cpu)
      # More VRAM available: only up projection experts (routed + shared) spill to CPU.
      # Gate and down experts remain on GPU.
      # WARNING: may OOM on 12 GB — monitor nvidia-smi before running.
      OVERRIDE_TENSOR=".ffn_up_(ch|)exps=CPU"
      ;;
    none)
      # No tensor override. Set N_CPU_MOE in env to use integer MoE offload.
      OVERRIDE_TENSOR=""
      ;;
    *)
      echo "Unknown STRATEGY '${STRATEGY}'." >&2
      echo "Valid: all-cpu-moe | partial-cpu | up-down-cpu | up-cpu | none" >&2
      exit 1
      ;;
  esac
fi

export MODEL TASKS N_GPU_LAYERS THREADS CPU_RANGE FA MMP OVERRIDE_TENSOR

echo "# model    : ${MODEL}"
echo "# strategy : ${STRATEGY}"
echo "# override : ${OVERRIDE_TENSOR:-<none>}"
echo "# tasks    : ${TASKS}"
echo "# ngl      : ${N_GPU_LAYERS}"
echo "# threads  : ${THREADS} (pinned ${CPU_RANGE})"
echo "# fa       : ${FA}  mmp: ${MMP}"
echo

exec "${SCRIPT_DIR}/run-llama-bench.sh"
