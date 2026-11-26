#!/usr/bin/env bash
# gpt-oss-puzzle-88B (MXFP4_MOE) — default llama.cpp bench
#
# Architecture notes (from issue/PR context):
#   - heterogeneous MoE (per-layer experts: 128 or 64)
#   - per-layer sliding window patterns
#   - requires puzzle-support changes merged via PR #21032
#
# Usage:
#   ./bench-llama-cpp-gpt-oss-puzzle-88b.sh
#   TASKS=1024,256 ./bench-llama-cpp-gpt-oss-puzzle-88b.sh
#   N_CPU_MOE=32 ./bench-llama-cpp-gpt-oss-puzzle-88b.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

LLAMA_BENCH="${LLAMA_BENCH:-${REPO_DIR}/vendor/llama.cpp-pr-test-21032/build/bin/llama-bench}"
MODEL="${MODEL:-/home/kchauhan/models/SamPurkis/gpt-oss-puzzle-88B-GGUF/gpt-oss-puzzle-88B.MXFP4_MOE.gguf}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
N_CPU_MOE="${N_CPU_MOE:-48}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"

export LLAMA_BENCH MODEL TASKS N_GPU_LAYERS N_CPU_MOE THREADS CPU_RANGE FA MMP BATCH_SIZE UBATCH_SIZE

exec "${SCRIPT_DIR}/run-llama-bench.sh"
