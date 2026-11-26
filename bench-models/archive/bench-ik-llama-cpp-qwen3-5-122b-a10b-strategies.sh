#!/usr/bin/env bash
# Qwen3.5-122B-A10B (UD-IQ4_XS) — ik_llama.cpp MoE strategy experiments
#
# Model: 48 layers, 256 experts (8 active/tok), ~70 GB, RTX 4070 12 GB
#
# ik_llama-specific flags (all combinable):
#   FUSED_MOE=1          Fuse expert computation kernel (default on, major tg win)
#   MERGE_UP_GATE=1      Repack up+gate expert weights into single matrix.
#                        Costs load time + extra RAM. +tg, -pp.
#   MERGE_QKV=1          Merge Q/K/V projection weights. Small tg improvement.
#   GROUPED_ROUTING=1    Group expert routing for better memory access patterns.
#   ROPE_CACHE=1         Cache RoPE computations. May help on long contexts.
#
# STRATEGY presets:
#   fused          fused-moe only (best all-round, default)
#   fused-ger      fused-moe + grouped-expert-routing
#   fused-mqkv     fused-moe + merge-qkv
#   fused-muge     fused-moe + merge-up-gate (+tg, -pp, extra RAM required)
#   all            all flags on (best tg, worst pp)
#   none           no ik_llama flags (compare against plain llama.cpp)
#
# Note: Qwen3.5-122B-A10B has 256 experts (vs 512 in Qwen3-Coder-Next).
# fused-muge RAM overhead is proportionally lower — check available RAM before
# enabling. fused-ger and fused-mqkv are low-risk and worth trying first.
#
# Usage:
#   ./bench-ik-llama-cpp-qwen3-5-122b-a10b-strategies.sh
#   STRATEGY=fused     ./bench-ik-llama-cpp-qwen3-5-122b-a10b-strategies.sh
#   STRATEGY=fused-ger ./bench-ik-llama-cpp-qwen3-5-122b-a10b-strategies.sh
#   STRATEGY=fused-mqkv ./bench-ik-llama-cpp-qwen3-5-122b-a10b-strategies.sh
#   STRATEGY=fused-muge ./bench-ik-llama-cpp-qwen3-5-122b-a10b-strategies.sh
#
# Sweep all safe strategies in one go:
#   for s in fused fused-ger fused-mqkv; do
#     STRATEGY=$s ./bench-ik-llama-cpp-qwen3-5-122b-a10b-strategies.sh
#   done

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

MODEL="${MODEL:-/mnt/lab/models/unsloth/Qwen3.5-122B-A10B-GGUF/UD-IQ4_XS/Qwen3.5-122B-A10B-UD-IQ4_XS-00001-of-00003.gguf}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
N_CPU_MOE="${N_CPU_MOE:-48}"
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

export MODEL TASKS N_GPU_LAYERS N_CPU_MOE THREADS CPU_RANGE FA MMP
export FUSED_MOE MERGE_UP_GATE MERGE_QKV GROUPED_ROUTING ROPE_CACHE

echo "# model          : ${MODEL}"
echo "# strategy       : ${STRATEGY}"
echo "# fused-moe      : ${FUSED_MOE}"
echo "# merge-up-gate  : ${MERGE_UP_GATE}"
echo "# merge-qkv      : ${MERGE_QKV}"
echo "# grouped-routing: ${GROUPED_ROUTING}"
echo "# rope-cache     : ${ROPE_CACHE}"
echo "# tasks          : ${TASKS}"
echo "# n_cpu_moe      : ${N_CPU_MOE}"
echo "# threads        : ${THREADS} (pinned ${CPU_RANGE})"
echo "# fa             : ${FA}  mmp: ${MMP}"
echo

exec "${SCRIPT_DIR}/run-ik-llama-bench.sh"
