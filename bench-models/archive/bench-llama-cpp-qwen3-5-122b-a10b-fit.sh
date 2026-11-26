#!/usr/bin/env bash
# Qwen3.5-122B-A10B (UD-IQ4_XS) — automatic fit-based bench
#
# Delegates to run-llama-fit-bench.sh which runs llama-fit-params to compute
# the optimal -ngl/-ot placement for available VRAM, then passes those args
# directly into llama-bench.
#
# Architecture:
#   48 transformer blocks, 256 experts per MoE layer, 8 active/token
#   ~70 GB on disk (split GGUF), RTX 4070 12 GB
#
# Useful knobs:
#   FIT_TARGET=2048  — leave 2 GB VRAM headroom instead of 1 GB (default)
#   FIT_CTX=65536    — ensure fit allocates for 64k context minimum
#   TASKS=4096,512   — bench at a more realistic coding-session context size
#
# Usage:
#   ./bench-llama-cpp-qwen3-5-122b-a10b-fit.sh
#   FIT_TARGET=2048 ./bench-llama-cpp-qwen3-5-122b-a10b-fit.sh
#   FIT_CTX=65536 TASKS=4096,512 ./bench-llama-cpp-qwen3-5-122b-a10b-fit.sh
#
# To inspect what fit-params would choose without running bench:
#   MODEL=/mnt/lab/models/unsloth/Qwen3.5-122B-A10B-GGUF/UD-IQ4_XS/Qwen3.5-122B-A10B-UD-IQ4_XS-00001-of-00003.gguf \
#     ./run-llama-fit-params.sh

MODEL="${MODEL:-/mnt/lab/models/unsloth/Qwen3.5-122B-A10B-GGUF/UD-IQ4_XS/Qwen3.5-122B-A10B-UD-IQ4_XS-00001-of-00003.gguf}"
TASKS="${TASKS:-512,128}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"
FIT_TARGET="${FIT_TARGET:-512}"
FIT_CTX="${FIT_CTX:-65536}"

export MODEL TASKS THREADS CPU_RANGE FA MMP FIT_TARGET FIT_CTX

exec "$(dirname -- "$0")/run-llama-fit-bench.sh"
