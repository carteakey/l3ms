#!/usr/bin/env bash
# NVIDIA Nemotron 3.5 Bolt 74B-A9B (24GB IQ2 QAT) — llama.cpp master bench
#
# Uses vendor/llama.cpp-master (b12621+) and q8_0 KV cache.

REPO_DIR="$(cd "$(dirname -- "$0")/.." && pwd)"
LLAMA_BENCH="${LLAMA_BENCH:-${REPO_DIR}/vendor/llama.cpp-master/build/bin/llama-bench}"
MODEL="${MODEL:-/mnt/dev/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-1}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"

export LLAMA_BENCH MODEL TASKS N_GPU_LAYERS THREADS CPU_RANGE FA MMP BATCH_SIZE UBATCH_SIZE CACHE_TYPE_K CACHE_TYPE_V

exec "$(dirname -- "$0")/run-llama-bench.sh"
