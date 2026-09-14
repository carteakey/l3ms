#!/usr/bin/env bash
# Baseline script (adapted to UD-Q6_K): llama.cpp spec draft-mtp
# Flags: --fit on --fit-target 512 --ctx-size 131072 --spec-type draft-mtp --spec-draft-p-min 0.75 --spec-draft-n-max 2
set -euo pipefail
LLAMA_BIN="$HOME/runtime-builds/llama.cpp-dc72703/build/bin/llama-cli"
MODEL="$HOME/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q6_K.gguf"
[ -f "$MODEL" ] || { echo "Model not found at $MODEL"; exit 1; }
taskset -c 0-11 $LLAMA_BIN \
  -m $MODEL \
  --fit on \
  --fit-target 512 \
  --ctx-size 131072 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --cache-type-k-draft q8_0 \
  --cache-type-v-draft q8_0 \
  --spec-type draft-mtp \
  --spec-draft-p-min 0.75 \
  --spec-draft-n-max 2 \
  -st \
  --no-mmap \
  --mlock \
  --threads 8 \
  --temp 0.0 \
  -n 128 \
  -p "Explain how speculative decoding works in large language model inference, in three short paragraphs." \
  --no-display-prompt 2>&1 | tee ~/repos/l3ms/bench-models/logs/qwen3-6-Q6K-MTP-reddit-baseline.log
