#!/usr/bin/env bash
# Test script based on Reddit post: Qwen3.6-35B-A3B-UD-Q4_K_XL @ 110 tok/s
# Flags: --fit --fit-margin 1664 --ctx-size 131072 --multi-token-prediction --draft-p-min 0.75 --draft-max 2

IK_BIN="vendor/ik_llama.cpp/build/bin/llama-cli"
MODEL="/mnt/lab/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"

# Verify model exists
if [ ! -f "$MODEL" ]; then
    echo "Model not found at $MODEL"
    exit 1
fi

# Run benchmark directly with llama-cli
taskset -c 0-11 $IK_BIN \
  -m $MODEL \
  --fit \
  --fit-margin 1664 \
  --ctx-size 131072 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --cache-type-k-draft q8_0 \
  --cache-type-v-draft q8_0 \
  --multi-token-prediction \
  --draft-p-min 0.75 \
  --draft-max 2 \
  -st \
  --no-mmap \
  --mlock \
  --threads 8 \
  --temp 0.0 \
  -n 128 \
  -p "Explain how speculative decoding works in large language model inference, in three short paragraphs." \
  --no-display-prompt 2>&1 | tee ik_cli_bench.log
