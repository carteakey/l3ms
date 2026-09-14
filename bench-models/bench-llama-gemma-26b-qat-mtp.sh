#!/usr/bin/env bash
# Gemma 4 26B QAT (UD-Q4_K_XL) + MTP drafter — fixed fit placement from llama-fit-params
# (--fit with an external draft model loops in build 571d0d5; placement derived standalone)
set -euo pipefail
LLAMA_BIN="$HOME/runtime-builds/llama.cpp-dc72703/build/bin/llama-cli"
MODEL="$HOME/models/unsloth/gemma-4-26B-A4B-it-qat-GGUF/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
DRAFT="$HOME/models/unsloth/gemma-4-26B-A4B-it-qat-GGUF/mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
[ -f "$MODEL" ] || { echo "Model not found at $MODEL"; exit 1; }
[ -f "$DRAFT" ] || { echo "Draft not found at $DRAFT"; exit 1; }
OT="blk\.13\.ffn_(gate|gate_up|down).*=CPU,blk\.(14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30)\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU"
taskset -c 0-11 $LLAMA_BIN \
  -m $MODEL \
  --spec-draft-model $DRAFT \
  -c 131072 -ngl 31 -ot "$OT" \
  -ctk f16 -ctv f16 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -fa 1 \
  --no-mmap \
  --mlock \
  --threads 8 \
  -st \
  --temp 0.0 \
  -n 128 \
  -p "Explain how speculative decoding works in large language model inference, in three short paragraphs." \
  --no-display-prompt 2>&1 | tee ~/repos/l3ms/bench-models/logs/gemma-26b-qat-mtp.log
