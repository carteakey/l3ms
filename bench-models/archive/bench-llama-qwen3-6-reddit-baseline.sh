#!/usr/bin/env bash
# Baseline script: llama.cpp @ 89 tok/s
# Flags: --fit on --fit-target 512 --ctx-size 131072 --spec-type draft-mtp --spec-draft-p-min 0.75 --spec-draft-n-max 2

# Fallback to llama-cli if llama-server is missing
LLAMA_BIN="vendor/llama.cpp/build/bin/llama-cli"
MODEL="/mnt/lab/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"

# Verify model exists
if [ ! -f "$MODEL" ]; then
    echo "Model not found at $MODEL"
    exit 1
fi

# Run benchmark directly with llama-cli in non-interactive mode
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
  --no-display-prompt 2>&1 | tee llama_cli_bench.log
