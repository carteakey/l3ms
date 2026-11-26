#!/usr/bin/env bash
# Sarvam-30B (Q6_K) — print fit-params output only
# Useful to inspect -ngl/-ts/-ot before updating run scripts.

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

FIT_TARGET="${FIT_TARGET:-512}"
FIT_CTX="${FIT_CTX:-4096}"
THREADS="${THREADS:-10}"
FA="${FA:-1}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"

export MODEL LLAMA_FIT FIT_TARGET FIT_CTX THREADS FA BATCH_SIZE UBATCH_SIZE
exec "${SCRIPT_DIR}/run-llama-fit-params.sh"
