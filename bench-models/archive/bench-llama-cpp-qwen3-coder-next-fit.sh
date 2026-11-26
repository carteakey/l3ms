#!/usr/bin/env bash
# Qwen3-Coder-Next (UD-Q4_K_XL) — automatic fit-based bench
# Delegates to run-llama-fit-bench.sh; set env vars to override any default.
MODEL="${MODEL:-/mnt/lab/models/unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-UD-Q4_K_XL.gguf}"
TASKS="${TASKS:-512,128}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"
FIT_TARGET="${FIT_TARGET:-512}"
FIT_CTX="${FIT_CTX:-65536}"
export MODEL TASKS THREADS CPU_RANGE FA MMP FIT_TARGET FIT_CTX
exec "$(dirname -- "$0")/run-llama-fit-bench.sh"
