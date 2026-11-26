#!/usr/bin/env bash
# CPU-only bench: no GPU offload, no flash attention
MODEL="${MODEL:-/mnt/lab/models/qwen/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf}"
N_GPU_LAYERS=0
FA=0
export MODEL N_GPU_LAYERS FA
exec "$(dirname -- "$0")/run-llama-bench.sh"
