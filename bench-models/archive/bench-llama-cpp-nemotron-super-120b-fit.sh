#!/usr/bin/env bash
# NVIDIA Nemotron 3 Super 120B A12B (UD-Q3_K_XL) — automatic fit-based bench
# Delegates to run-llama-fit-bench.sh; set env vars to override defaults.

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

LLAMA_BENCH="${LLAMA_BENCH:-${REPO_DIR}/vendor-forks/llama.cpp-copilot/build/bin/llama-bench}"
LLAMA_FIT="${LLAMA_FIT:-${REPO_DIR}/vendor-forks/llama.cpp-copilot/build/bin/llama-fit-params}"
MODEL="${MODEL:-/mnt/lab/models/unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF/UD-Q3_K_XL/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q3_K_XL-00001-of-00003.gguf}"
TASKS="${TASKS:-512,128}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"
FIT_TARGET="${FIT_TARGET:-512}"
FIT_CTX="${FIT_CTX:-32768}"

export LLAMA_BENCH LLAMA_FIT MODEL TASKS THREADS CPU_RANGE FA MMP FIT_TARGET FIT_CTX
exec "${SCRIPT_DIR}/run-llama-fit-bench.sh"
