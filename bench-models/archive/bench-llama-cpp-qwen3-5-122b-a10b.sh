#!/usr/bin/env bash
# Qwen3.5-122B-A10B (UD-IQ4_XS) — default llama.cpp bench
#
# Architecture:
#   48 transformer blocks, 256 experts per MoE layer, 8 active/token
#   ~70 GB on disk (split GGUF), RTX 4070 12 GB
#
# N_CPU_MOE=48 puts all MoE expert layers on CPU (safe default for 12 GB VRAM).
# Lower N_CPU_MOE to push more expert layers onto GPU if VRAM allows.
#
# Usage:
#   ./bench-llama-cpp-qwen3-5-122b-a10b.sh
#   N_CPU_MOE=36 ./bench-llama-cpp-qwen3-5-122b-a10b.sh
#   TASKS=1024,256 ./bench-llama-cpp-qwen3-5-122b-a10b.sh

MODEL="${MODEL:-/mnt/lab/models/unsloth/Qwen3.5-122B-A10B-GGUF/UD-IQ4_XS/Qwen3.5-122B-A10B-UD-IQ4_XS-00001-of-00003.gguf}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
N_CPU_MOE="${N_CPU_MOE:-48}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"

export MODEL TASKS N_GPU_LAYERS N_CPU_MOE THREADS CPU_RANGE FA MMP

exec "$(dirname -- "$0")/run-llama-bench.sh"
