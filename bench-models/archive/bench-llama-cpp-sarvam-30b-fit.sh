#!/usr/bin/env bash
# Sarvam-30B (Q6_K) — automatic fit-based bench
# Uses llama-fit-params to compute placement, then runs llama-bench.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-/mnt/lab/models/Sumitc13/sarvam-30b-GGUF/sarvam-30B-Q6_K.gguf}"

if [ -z "${LLAMA_FIT:-}" ]; then
  for candidate in \
    "${REPO_DIR}/vendor/llama.cpp-pr-test-20275/build/bin/llama-fit-params" \
    "${REPO_DIR}/vendor/llama.cpp-pr-test/build/bin/llama-fit-params" \
    "${REPO_DIR}/vendor/llama.cpp-sarvam/build/bin/llama-fit-params"
  do
    if [ -x "${candidate}" ]; then
      LLAMA_FIT="${candidate}"
      break
    fi
  done
  LLAMA_FIT="${LLAMA_FIT:-${REPO_DIR}/vendor/llama.cpp-pr-test-20275/build/bin/llama-fit-params}"
fi

if [ -z "${LLAMA_BENCH:-}" ]; then
  for candidate in \
    "${REPO_DIR}/vendor/llama.cpp-pr-test-20275/build/bin/llama-bench" \
    "${REPO_DIR}/vendor/llama.cpp-pr-test/build/bin/llama-bench" \
    "${REPO_DIR}/vendor/llama.cpp-sarvam/build/bin/llama-bench"
  do
    if [ -x "${candidate}" ]; then
      LLAMA_BENCH="${candidate}"
      break
    fi
  done
  LLAMA_BENCH="${LLAMA_BENCH:-${REPO_DIR}/vendor/llama.cpp-pr-test-20275/build/bin/llama-bench}"
fi

TASKS="${TASKS:-512,128}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
FIT_TARGET="${FIT_TARGET:-512}"
FIT_CTX="${FIT_CTX:-4096}"
REPETITIONS="${REPETITIONS:-3}"
OUTPUT_FMT="${OUTPUT_FMT:-md}"

export MODEL LLAMA_FIT LLAMA_BENCH TASKS THREADS CPU_RANGE FA MMP BATCH_SIZE UBATCH_SIZE FIT_TARGET FIT_CTX REPETITIONS OUTPUT_FMT
exec "${SCRIPT_DIR}/run-llama-fit-bench.sh"
