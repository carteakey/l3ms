#!/usr/bin/env bash
# NVIDIA Nemotron 3 Super 120B A12B (UD-Q3_K_XL) — ik_llama.cpp strategy experiments
#
# STRATEGY presets:
#   fused          fused-moe only (default)
#   fused-ger      fused-moe + grouped-expert-routing
#   fused-mqkv     fused-moe + merge-qkv
#   fused-muge     fused-moe + merge-up-gate
#   all            all flags on
#   none           no ik-specific extras

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

IK_BENCH="${IK_BENCH:-${REPO_DIR}/vendor-forks/ik_llama.cpp-copilot/build/bin/llama-bench}"
MODEL="${MODEL:-/mnt/lab/models/unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF/UD-Q3_K_XL/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q3_K_XL-00001-of-00003.gguf}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
N_CPU_MOE="${N_CPU_MOE:-88}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"

STRATEGY="${STRATEGY:-fused}"

FUSED_MOE=0
MERGE_UP_GATE=0
MERGE_QKV=0
GROUPED_ROUTING=0
ROPE_CACHE=0

case "${STRATEGY}" in
  fused)         FUSED_MOE=1 ;;
  fused-ger)     FUSED_MOE=1; GROUPED_ROUTING=1 ;;
  fused-mqkv)    FUSED_MOE=1; MERGE_QKV=1 ;;
  fused-muge)    FUSED_MOE=1; MERGE_UP_GATE=1 ;;
  all)           FUSED_MOE=1; MERGE_UP_GATE=1; MERGE_QKV=1; GROUPED_ROUTING=1 ;;
  none)          ;;
  *)
    echo "Unknown STRATEGY '${STRATEGY}'." >&2
    echo "Valid: fused | fused-ger | fused-mqkv | fused-muge | all | none" >&2
    exit 1
    ;;
esac

export IK_BENCH MODEL TASKS N_GPU_LAYERS N_CPU_MOE THREADS CPU_RANGE FA MMP
export FUSED_MOE MERGE_UP_GATE MERGE_QKV GROUPED_ROUTING ROPE_CACHE

echo "# strategy       : ${STRATEGY}"
echo "# fused-moe      : ${FUSED_MOE}"
echo "# merge-up-gate  : ${MERGE_UP_GATE}"
echo "# merge-qkv      : ${MERGE_QKV}"
echo "# grouped-routing: ${GROUPED_ROUTING}"
echo "# tasks          : ${TASKS}"
echo "# threads        : ${THREADS} (pinned ${CPU_RANGE})"
echo "# fa             : ${FA}  mmp: ${MMP}"
echo

exec "${SCRIPT_DIR}/run-ik-llama-bench.sh"
