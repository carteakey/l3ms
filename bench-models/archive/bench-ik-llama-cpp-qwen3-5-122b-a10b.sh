#!/usr/bin/env bash
# Qwen3.5-122B-A10B (UD-IQ4_XS) — ik_llama.cpp default bench
#
# Architecture:
#   48 transformer blocks, 256 experts per MoE layer, 8 active/token
#   ~70 GB on disk (split GGUF), RTX 4070 12 GB
#
# ik_llama enables FUSED_MOE by default — the single largest tg improvement
# over stock llama.cpp. Compare this result against the llama.cpp baseline
# (bench-llama-cpp-qwen3-5-122b-a10b.sh) to quantify the uplift.
#
# N_CPU_MOE=48 puts all 48 MoE expert layers on CPU (safe default for 12 GB).
# Lower to push more layers onto GPU if VRAM allows.
#
# Usage:
#   ./bench-ik-llama-cpp-qwen3-5-122b-a10b.sh
#   N_CPU_MOE=36 ./bench-ik-llama-cpp-qwen3-5-122b-a10b.sh
#   FUSED_MOE=0  ./bench-ik-llama-cpp-qwen3-5-122b-a10b.sh   # compare without fused-moe
#   TASKS=1024,256 ./bench-ik-llama-cpp-qwen3-5-122b-a10b.sh

MODEL="${MODEL:-/mnt/lab/models/unsloth/Qwen3.5-122B-A10B-GGUF/UD-IQ4_XS/Qwen3.5-122B-A10B-UD-IQ4_XS-00001-of-00003.gguf}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
N_CPU_MOE="${N_CPU_MOE:-48}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"

# ik_llama-specific — fused-moe on by default (major tg win)
FUSED_MOE="${FUSED_MOE:-1}"

export MODEL TASKS N_GPU_LAYERS N_CPU_MOE THREADS CPU_RANGE FA MMP FUSED_MOE

exec "$(dirname -- "$0")/run-ik-llama-bench.sh"
