#!/usr/bin/env bash
# Qwen3.8-27B-UD-Q2_K_XL — llama.cpp benchmark, BIG-CONTEXT variant (RTX 4070 12 GB)
#
# Small-quant companion to the Ridge baseline. Frees ~2.5 GiB vs Ridge, which
# the fit solver spends on a large KV cache. Same hybrid arch (only 16/64
# layers are attention), so KV stays cheap even at huge context.
#
# Measured (2026-08-19, fit-target 128, KV q5_0/q4_1):
#   tg128: ~39.6 t/s  pp128: ~411 t/s  pp512: ~873 t/s
#   Fits ctx 262144 (native window) fully in VRAM.
#   (ctx 1000000 "fits" per the solver estimate but is NOT verified as a real
#    KV allocation — llama-bench --fit-ctx is a planning hint only.)
set -euo pipefail
LLAMA_BIN="/home/kchauhan/runtime-builds/llama.cpp-571d0d5/build-cublas/bin/llama-bench"
MODEL="$HOME/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf"
CTX="${CTX:-262144}"
[ -f "$MODEL" ] || { echo "Model not found at $MODEL"; exit 1; }
[ -f "$LLAMA_BIN" ] || { echo "llama-bench not found at $LLAMA_BIN"; exit 1; }
taskset -c 0-11 $LLAMA_BIN \
  -m $MODEL \
  -ngl 0 \
  -p 128 -n 128 -r 1 \
  --fit-target 128 --fit-ctx $CTX \
  -ctk q5_0 -ctv q4_1 \
  -fa on 2>/dev/null | grep -aE "pp128|tg128"