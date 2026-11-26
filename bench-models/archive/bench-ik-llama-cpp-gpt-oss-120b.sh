#!/usr/bin/env bash
# gpt-oss-120b (mxfp4) — ik_llama.cpp default bench
#
# Architecture:
#   36 transformer blocks, 128 experts per MoE layer, 4 active/token
#   ~60 GB on disk (split GGUF), RTX 4070 12 GB
#
# RAM CONSTRAINT: do NOT use N_CPU_MOE=36 (all experts on CPU) — that loads
# ~60 GB into RAM and will hard-crash a 64 GB system. The partial-cpu -ot
# below keeps ~50 GB in RAM and ~10.5 GB in VRAM (safe).
#
# FUSED_MOE default is 0 — on hybrid CPU+GPU inference with this model,
# fused-moe hurts pp badly without meaningful tg gain (learned from 122B bench).
# Set FUSED_MOE=1 explicitly to test.
#
# Usage:
#   ./bench-ik-llama-cpp-gpt-oss-120b.sh
#   FUSED_MOE=1  ./bench-ik-llama-cpp-gpt-oss-120b.sh
#   TASKS=1024,256 ./bench-ik-llama-cpp-gpt-oss-120b.sh
#   N_GPU_LAYERS=37 FUSED_MOE=1 GROUPED_ROUTING=1 ./bench-ik-llama-cpp-gpt-oss-120b.sh

export GGML_CUDA_GRAPH_OPT="${GGML_CUDA_GRAPH_OPT:-1}"

MODEL="${MODEL:-/mnt/lab/models/ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-37}"
# N_CPU_MOE intentionally unset — OVERRIDE_TENSOR controls expert placement.
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"

# fit-params recommended: blk 0-4 on GPU, blk 5-36 experts on CPU
OVERRIDE_TENSOR="${OVERRIDE_TENSOR:-blk\.(5|[6-9]|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(up|down|gate)_(ch|)exps=CPU}"

# ik_llama-specific — fused-moe off by default for hybrid CPU+GPU
# (fused kernel optimised for all-GPU inference; hurts pp on CPU-spill configs)
FUSED_MOE="${FUSED_MOE:-0}"
MERGE_UP_GATE="${MERGE_UP_GATE:-0}"
MERGE_QKV="${MERGE_QKV:-0}"
GROUPED_ROUTING="${GROUPED_ROUTING:-0}"
ROPE_CACHE="${ROPE_CACHE:-0}"

export MODEL TASKS N_GPU_LAYERS THREADS CPU_RANGE FA MMP BATCH_SIZE UBATCH_SIZE OVERRIDE_TENSOR
export FUSED_MOE MERGE_UP_GATE MERGE_QKV GROUPED_ROUTING ROPE_CACHE

echo "# model          : ${MODEL}"
echo "# tasks          : ${TASKS}"
echo "# ngl            : ${N_GPU_LAYERS}"
echo "# threads        : ${THREADS} (pinned ${CPU_RANGE})"
echo "# fa             : ${FA}  mmp: ${MMP}"
echo "# batch          : ${BATCH_SIZE}  ubatch: ${UBATCH_SIZE}"
echo "# override       : ${OVERRIDE_TENSOR}"
echo "# fused-moe      : ${FUSED_MOE}"
echo "# merge-up-gate  : ${MERGE_UP_GATE}"
echo "# merge-qkv      : ${MERGE_QKV}"
echo "# grouped-routing: ${GROUPED_ROUTING}"
echo

exec "$(dirname -- "$0")/run-ik-llama-bench.sh"
