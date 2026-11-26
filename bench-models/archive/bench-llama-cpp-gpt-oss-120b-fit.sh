#!/usr/bin/env bash
# gpt-oss-120b (mxfp4) — automatic fit-based bench
#
# Delegates to run-llama-fit-bench.sh which runs llama-fit-params to compute
# the optimal -ngl/-ot placement for available VRAM, then passes those args
# directly into llama-bench.
#
# Architecture:
#   36 transformer blocks, 128 experts per MoE layer, 4 active/token
#   ~60 GB on disk (split GGUF), RTX 4070 12 GB
#
# Useful knobs:
#   FIT_TARGET=2048  — leave 2 GB VRAM headroom instead of default
#   FIT_CTX=32768    — ensure fit allocates for 32k context minimum
#   TASKS=4096,512   — bench at a more realistic context size
#
# Usage:
#   ./bench-llama-cpp-gpt-oss-120b-fit.sh
#   FIT_TARGET=2048 ./bench-llama-cpp-gpt-oss-120b-fit.sh
#   FIT_CTX=32768 TASKS=4096,512 ./bench-llama-cpp-gpt-oss-120b-fit.sh
#
# To inspect what fit-params would choose without running bench:
#   MODEL=/mnt/lab/models/ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
#     ./run-llama-fit-params.sh

export GGML_CUDA_GRAPH_OPT="${GGML_CUDA_GRAPH_OPT:-1}"

MODEL="${MODEL:-/mnt/lab/models/ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf}"
TASKS="${TASKS:-512,128}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
FIT_TARGET="${FIT_TARGET:-512}"
FIT_CTX="${FIT_CTX:-32768}"

export MODEL TASKS THREADS CPU_RANGE FA MMP BATCH_SIZE UBATCH_SIZE FIT_TARGET FIT_CTX

exec "$(dirname -- "$0")/run-llama-fit-bench.sh"
