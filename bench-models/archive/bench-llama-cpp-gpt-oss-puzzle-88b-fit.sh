#!/usr/bin/env bash
# gpt-oss-puzzle-88B (MXFP4_MOE) — automatic fit-based bench
#
# Delegates to run-llama-fit-bench.sh which computes optimal placement with
# llama-fit-params and reuses those args for llama-bench.
#
# Usage:
#   ./bench-llama-cpp-gpt-oss-puzzle-88b-fit.sh
#   FIT_TARGET=2048 ./bench-llama-cpp-gpt-oss-puzzle-88b-fit.sh
#   FIT_CTX=32768 TASKS=4096,512 ./bench-llama-cpp-gpt-oss-puzzle-88b-fit.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

LLAMA_BENCH="${LLAMA_BENCH:-${REPO_DIR}/vendor/llama.cpp-pr-test-21032/build/bin/llama-bench}"
LLAMA_FIT="${LLAMA_FIT:-${REPO_DIR}/vendor/llama.cpp-pr-test-21032/build/bin/llama-fit-params}"
MODEL="${MODEL:-/home/kchauhan/models/SamPurkis/gpt-oss-puzzle-88B-GGUF/gpt-oss-puzzle-88B.MXFP4_MOE.gguf}"
TASKS="${TASKS:-512,128}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
FIT_TARGET="${FIT_TARGET:-512}"
FIT_CTX="${FIT_CTX:-32768}"

export LLAMA_BENCH LLAMA_FIT MODEL TASKS THREADS CPU_RANGE FA MMP BATCH_SIZE UBATCH_SIZE FIT_TARGET FIT_CTX

exec "${SCRIPT_DIR}/run-llama-fit-bench.sh"
