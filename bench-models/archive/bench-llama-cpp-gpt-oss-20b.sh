#!/usr/bin/env bash
MODEL="${MODEL:-/mnt/lab/models/ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-mxfp4.gguf}"
N_CPU_MOE="${N_CPU_MOE:-4}"
export MODEL N_CPU_MOE
exec "$(dirname "$0")/run-llama-bench.sh"
