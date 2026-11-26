#!/usr/bin/env bash
MODEL="${MODEL:-/mnt/lab/models/qwen/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf}"
export MODEL
exec "$(dirname "$0")/run-llama-bench.sh"
