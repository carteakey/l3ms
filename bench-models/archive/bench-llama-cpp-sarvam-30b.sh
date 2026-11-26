#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SARVAM_PR_NUMBER="${SARVAM_PR_NUMBER:-20275}"
MODEL="${MODEL:-/mnt/lab/models/Sumitc13/sarvam-30b-GGUF/sarvam-30B-Q6_K.gguf}"
LLAMA_BENCH="${LLAMA_BENCH:-${REPO_DIR}/vendor/llama.cpp-pr-test-${SARVAM_PR_NUMBER}/build/bin/llama-bench}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
FA="${FA:-1}"
TASKS="${TASKS:-512,128}"
REPETITIONS="${REPETITIONS:-3}"
OUTPUT_FMT="${OUTPUT_FMT:-md}"

export MODEL LLAMA_BENCH N_GPU_LAYERS FA TASKS REPETITIONS OUTPUT_FMT
exec "${SCRIPT_DIR}/run-llama-bench.sh"
