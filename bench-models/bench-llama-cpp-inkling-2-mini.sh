#!/usr/bin/env bash
# Inkling 2 Mini 26.4B-A5.4B (UD-Q4_K_XL) - custom Ada sm_89 llama.cpp-inkling2-lean bench
#
# Matches served llama-swap.yaml flags: elastic MoE cache + visit-weighted budget +
# -cmoe 7800 MiB + --moe-cache-cpu-share 0.70 + -b 2048 -ub 2048 + token_embd on CPU + q8_0 KV cache.
# See docs/inkling-2-mini-optimization.md for full tuning sweep history.

REPO_DIR="$(cd "$(dirname -- "$0")/.." && pwd)"
LLAMA_BENCH="${LLAMA_BENCH:-${REPO_DIR}/vendor/llama.cpp-inkling2-lean/build/bin/llama-bench}"
MODEL="${MODEL:-/mnt/dev/models/unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
THREADS="${THREADS:-12}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-2048}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
LLAMA_MOE_CACHE_ELASTIC="${LLAMA_MOE_CACHE_ELASTIC:-1}"
LLAMA_MOE_CACHE_VISIT_BUDGET="${LLAMA_MOE_CACHE_VISIT_BUDGET:-1}"

export LLAMA_BENCH MODEL TASKS N_GPU_LAYERS THREADS CPU_RANGE FA MMP BATCH_SIZE UBATCH_SIZE CACHE_TYPE_K CACHE_TYPE_V LLAMA_MOE_CACHE_ELASTIC LLAMA_MOE_CACHE_VISIT_BUDGET

exec "$(dirname -- "$0")/run-llama-bench.sh"
