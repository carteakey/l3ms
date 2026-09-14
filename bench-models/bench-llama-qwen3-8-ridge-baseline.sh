#!/usr/bin/env bash
# Qwen3.8-27B-Ridge-3.7bpw — LOCKED-IN llama.cpp benchmark (RTX 4070 12 GB)
#
# Uses llama-bench: exits cleanly and prints stable tg/pp t/s.
# (llama-cli --fit in build 571d0d5 hangs after generation — do NOT use it here.)
#
# Winning config (measured 2026-08-19):
#   --fit-target 128           (solver auto-sheds whole layers as ctx grows; never OOMs)
#   KV q5_0/q4_1               (post-suggested quants; buys context for free)
#   NO MTP                     (a wash on this card; MTP head steals VRAM)
#
# Measured (pp128 / tg128, fit-target 128 @ ctx 16384):
#   pp128: ~409 t/s   tg128: ~34.6 t/s
#
# NOTE: LLAMA_BIN is the stable CUDA build. vendor/... build dirs get pruned
#       by cleanup jobs; runtime-builds/ persists.
set -euo pipefail
LLAMA_BIN="/home/kchauhan/runtime-builds/llama.cpp-571d0d5/build-cublas/bin/llama-bench"
MODEL="$HOME/models/empero-ai/Qwen3.8-27B-Ridge-GGUF/Qwen3.8-27B-Ridge-3.7bpw.gguf"
[ -f "$MODEL" ] || { echo "Model not found at $MODEL"; exit 1; }
[ -f "$LLAMA_BIN" ] || { echo "llama-bench not found at $LLAMA_BIN"; exit 1; }
taskset -c 0-11 $LLAMA_BIN \
  -m $MODEL \
  -ngl 0 \
  -p 128 -n 128 -r 1 \
  --fit-target 128 --fit-ctx 16384 \
  -ctk q5_0 -ctv q4_1 \
  -fa on 2>/dev/null | grep -aE "pp128|tg128"

# ---------------------------------------------------------------------------
# ALTERNATIVES — not finalized, left as notes for later tuning
# ---------------------------------------------------------------------------
#
# 1) MTP (native embedded head). ~0 benefit when GPU-resident (13.5 vs 13.3 t/s).
#    llama-bench has NO draft-mtp support; use llama-cli instead but it hangs
#    after generation in 571d0d5 (needs manual kill). Add when memory-bound:
#      --spec-type draft-mtp --spec-draft-n-max 2
#      --cache-type-k-draft q5_0 --cache-type-v-draft q4_1
#
# 2) Manual FFN-only offload to CPU (reddit post's -ot approach).
#    WORSE than --fit here: 12/16/24 FFN blocks -> 13.5/11.3/9.3 t/s @ 16k,
#    and OOM'd at ctx=100000 (KV stays on GPU). Pattern:
#      -ot 'blk\.(0|1|...|N)\.ffn_(gate|up|down).*=CPU'
#
# 3) Bigger context via smaller quant. Ridge maxes ~131k @ 7 t/s. For native
#    262k, a smaller quant frees VRAM for KV. Candidates (NOT benchmarked yet):
#      - IQ2_XXS bartowski  8.75 GiB  (~3 GiB freed for KV)
#      - UD-IQ2_XXS unsloth 8.39 GiB  (~3.3 GiB freed)
#      - UD-IQ2_M  unsloth  9.61 GiB
#    NOTE: Q3_K_M (14.6 GiB) is LARGER than Ridge and does NOT fit 12 GB.
#    IQ2 is a quality tradeoff (see post comments).
#
# 4) For llama-swap serving, mirror this into llama-swap.yaml once finalized.