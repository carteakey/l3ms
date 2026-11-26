#!/usr/bin/env bash
# NVIDIA Nemotron 3 Super 120B A12B (UD-Q3_K_XL) — default llama.cpp bench

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

LLAMA_BENCH="${LLAMA_BENCH:-${REPO_DIR}/vendor-forks/llama.cpp-copilot/build/bin/llama-bench}"
MODEL="${MODEL:-/mnt/lab/models/unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF/UD-Q3_K_XL/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q3_K_XL-00001-of-00003.gguf}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
N_CPU_MOE="${N_CPU_MOE:-88}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"

export LLAMA_BENCH MODEL TASKS N_GPU_LAYERS N_CPU_MOE THREADS CPU_RANGE FA MMP
exec "${SCRIPT_DIR}/run-llama-bench.sh"
