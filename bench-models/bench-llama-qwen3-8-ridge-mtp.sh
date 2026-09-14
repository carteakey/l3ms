#!/usr/bin/env bash
# Qwen3.8-27B-Ridge-3.7bpw — llama.cpp MTP comparison (NOT the default)
#
# Kept for regression: MTP showed ~0 benefit on the RTX 4070 12 GB when the
# model is GPU-resident (see baseline script §1). llama-bench has no
# draft-mtp support, so this uses llama-cli.
#
# CAVEAT: llama-cli --fit in build 571d0d5 hangs after printing the stats
# line (never exits, holds VRAM). We cap it with timeout and grep the line
# before killing. A stale process may linger — check/kill llama-cli if the
# next run OOMs.
set -euo pipefail
LLAMA_BIN="/home/kchauhan/runtime-builds/llama.cpp-571d0d5/build-cublas/bin/llama-cli"
MODEL="$HOME/models/empero-ai/Qwen3.8-27B-Ridge-GGUF/Qwen3.8-27B-Ridge-3.7bpw.gguf"
[ -f "$MODEL" ] || { echo "Model not found at $MODEL"; exit 1; }
[ -f "$LLAMA_BIN" ] || { echo "llama-cli not found at $LLAMA_BIN"; exit 1; }
timeout 300 taskset -c 0-11 $LLAMA_BIN \
  -m $MODEL \
  --fit on \
  --fit-target 128 \
  --ctx-size 16384 \
  --cache-type-k q5_0 \
  --cache-type-v q4_1 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --cache-type-k-draft q5_0 \
  --cache-type-v-draft q4_1 \
  -fa 1 \
  --threads 10 \
  --temp 0.0 \
  -n 128 \
  -p "Explain how speculative decoding works in large language model inference, in three short paragraphs." \
  --no-display-prompt > /tmp/qwen38-ridge-mtp.out 2>&1; rc=$?; \
  tr '\r' '\n' < /tmp/qwen38-ridge-mtp.out | grep -aE "Generation:" | tail -1; \
  rm -f /tmp/qwen38-ridge-mtp.out; \
  if [ $rc -eq 124 ]; then echo "(note: llama-cli hung after gen; killed by timeout. kill any lingering llama-cli before next run)"; fi