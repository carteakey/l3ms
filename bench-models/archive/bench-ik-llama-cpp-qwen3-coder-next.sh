#!/usr/bin/env bash
# Qwen3-Coder-Next (UD-Q4_K_XL) — ik_llama.cpp default bench
# ik_llama uses --n-cpu-moe (not -ncmoe) and has fused-moe on by default.
MODEL="${MODEL:-/mnt/lab/models/unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-UD-Q4_K_XL.gguf}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
N_CPU_MOE="${N_CPU_MOE:-40}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"
export MODEL TASKS N_GPU_LAYERS N_CPU_MOE THREADS CPU_RANGE FA MMP
exec "$(dirname -- "$0")/run-ik-llama-bench.sh"
