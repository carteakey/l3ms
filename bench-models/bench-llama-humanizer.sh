#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-/home/kchauhan/models/jialinyyzz/humanizer/humanizer-12b-Q6_K.gguf}"
LLAMA_BENCH="${LLAMA_BENCH:-${REPO_DIR}/vendor/llama.cpp-master/build/bin/llama-bench}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
FA="${FA:-1}"
TASKS="${TASKS:-512,128}"
REPETITIONS="${REPETITIONS:-3}"
OUTPUT_FMT="${OUTPUT_FMT:-md}"
MODEL_KEY="${MODEL_KEY:-humanizer-12b}"
STRATEGY="${STRATEGY:-full-gpu-q6k}"
CPU_RANGE="${CPU_RANGE:-0-11}"

export MODEL LLAMA_BENCH N_GPU_LAYERS FA TASKS REPETITIONS OUTPUT_FMT MODEL_KEY STRATEGY CPU_RANGE
exec "${SCRIPT_DIR}/run-llama-bench.sh"
