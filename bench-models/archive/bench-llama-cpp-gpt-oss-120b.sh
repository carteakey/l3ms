#!/usr/bin/env bash
# gpt-oss-120b (mxfp4) — default llama.cpp bench
#
# Architecture:
#   36 transformer blocks, 128 experts per MoE layer, 4 active/token
#   ~60 GB on disk (split GGUF), RTX 4070 12 GB
#
# No shared experts (_shexp) — pure _exps tensors only.
# Simple patterns like ".ffn_.*_exps.=CPU" work correctly for this model.
#
# Env vars carried from run scripts:
#   GGML_CUDA_GRAPH_OPT=1   — CUDA graph optimisation
#   BATCH_SIZE=2048         — logical batch
#   UBATCH_SIZE=512         — physical batch
#
# N_CPU_MOE=36 puts all MoE expert layers on CPU (safe default for 12 GB VRAM).
# Lower N_CPU_MOE to push more expert layers onto GPU if VRAM allows.
#
# SAFE DEFAULT: uses fit-derived partial-cpu -ot (-ngl 37, blk 0-4 on GPU,
# blk 5-36 experts on CPU). This keeps RAM at ~50 GB — safe for 64 GB systems.
# N_CPU_MOE is intentionally unset; OVERRIDE_TENSOR controls placement instead.
# DO NOT use N_CPU_MOE=36 (all experts on CPU) — that pushes ~60 GB into RAM
# and will OOM/crash a 64 GB system.
#
# Usage:
#   ./bench-llama-cpp-gpt-oss-120b.sh
#   TASKS=1024,256 ./bench-llama-cpp-gpt-oss-120b.sh
#   N_GPU_LAYERS=37 ./bench-llama-cpp-gpt-oss-120b.sh

export GGML_CUDA_GRAPH_OPT="${GGML_CUDA_GRAPH_OPT:-1}"

MODEL="${MODEL:-/mnt/lab/models/ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-37}"
# N_CPU_MOE intentionally unset — OVERRIDE_TENSOR controls expert placement.
# Setting N_CPU_MOE=36 would load all experts into RAM and OOM a 64 GB system.
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
# fit-params recommended: blk 0-4 on GPU, blk 5-36 experts on CPU (~50 GB RAM, ~10.5 GB VRAM)
OVERRIDE_TENSOR="${OVERRIDE_TENSOR:-blk\.(5|[6-9]|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(up|down|gate)_(ch|)exps=CPU}"

export MODEL TASKS N_GPU_LAYERS THREADS CPU_RANGE FA MMP BATCH_SIZE UBATCH_SIZE OVERRIDE_TENSOR

exec "$(dirname -- "$0")/run-llama-bench.sh"
