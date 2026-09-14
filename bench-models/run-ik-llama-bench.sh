#!/usr/bin/env bash
# Shared ik_llama.cpp bench runner. Set env vars before exec'ing this script.
#
# Required (no default):
#   MODEL             - path to .gguf model file
#
# Result logging (bench-models/logs/results/<MODEL_KEY>.jsonl):
#   MODEL_KEY         - short slug for the results file (default: model filename stem)
#   STRATEGY          - descriptive label for this run (e.g. "fused", "fused-mqkv", "fused-muge")
#   NOTES             - free-text annotation appended to the JSONL record
#
# Optional (ik_llama defaults shown):
#   IK_BENCH          - path to ik_llama llama-bench binary (default: ../vendor/ik_llama.cpp/build/bin/llama-bench)
#
#   --- runner ---
#   CPU_RANGE         - taskset CPU affinity range, e.g. "0-11" (default: unset)
#
#   --- test parameters ---
#   TASKS             - pp,tg pair via -pg, e.g. "512,128" (default: unset)
#   N_PROMPT          - prompt token count (default: unset)
#   N_GEN             - generation token count (default: unset)
#   BATCH_SIZE        - logical batch size (default: unset)
#   UBATCH_SIZE       - physical batch size (default: unset)
#   THREADS           - CPU threads for generation (default: unset)
#   N_GPU_LAYERS      - layers offloaded to GPU (default: unset)
#   N_CPU_MOE         - MoE layers kept on CPU (default: unset)
#   FA                - flash attention 0|1 (default: unset)
#   MMP               - mmap 0|1 (default: unset)
#   CACHE_TYPE_K      - KV cache type for K: f16|q8_0|q4_0|... (default: unset)
#   CACHE_TYPE_V      - KV cache type for V: f16|q8_0|q4_0|... (default: unset)
#   OVERRIDE_TENSOR   - tensor override pattern (default: unset)
#   NUMA              - numa mode: distribute|isolate|numactl (default: unset)
#   REPETITIONS       - repetitions per test (default: unset)
#   OUTPUT_FMT        - output format: csv|json|jsonl|md|sql (default: unset)
#
#   --- ik_llama-specific ---
#   FUSED_MOE         - fuse MoE expert kernel 0|1 (default: 1 in ik_llama)
#   MERGE_UP_GATE     - repack up+gate expert weights 0|1 (default: 0)
#                       WARNING: nearly doubles RAM usage (~73 GB for Q3CN)
#   MERGE_QKV         - merge Q/K/V projection weights 0|1 (default: 0)
#   GROUPED_ROUTING   - group expert routing 0|1 (default: 0)
#   ROPE_CACHE        - cache RoPE computations 0|1 (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

IK_BENCH="${IK_BENCH:-${REPO_DIR}/vendor/ik_llama.cpp/build/bin/llama-bench}"

# --- validation ---

if [ -z "${MODEL:-}" ]; then
  echo "MODEL is not set" >&2
  exit 1
fi

if [ ! -x "${IK_BENCH}" ]; then
  echo "ik_llama llama-bench not found/executable: ${IK_BENCH}" >&2
  exit 1
fi

if [ ! -f "${MODEL}" ]; then
  echo "model file not found: ${MODEL}" >&2
  exit 1
fi

# --- build command ---

cmd=("${IK_BENCH}" -m "${MODEL}")

[ -n "${REPETITIONS:-}"   ] && cmd+=(-r "${REPETITIONS}")
[ -n "${OUTPUT_FMT:-}"    ] && cmd+=(-o "${OUTPUT_FMT}")
[ -n "${NUMA:-}"          ] && cmd+=(--numa "${NUMA}")

if [ -n "${TASKS:-}" ]; then
  cmd+=(-pg "${TASKS}")
else
  [ -n "${N_PROMPT:-}" ] && cmd+=(-p "${N_PROMPT}")
  [ -n "${N_GEN:-}"    ] && cmd+=(-n "${N_GEN}")
fi

[ -n "${BATCH_SIZE:-}"    ] && cmd+=(-b    "${BATCH_SIZE}")
[ -n "${UBATCH_SIZE:-}"   ] && cmd+=(-ub   "${UBATCH_SIZE}")
[ -n "${THREADS:-}"       ] && cmd+=(-t    "${THREADS}")
[ -n "${N_GPU_LAYERS:-}"  ] && cmd+=(-ngl  "${N_GPU_LAYERS}")
[ -n "${N_CPU_MOE:-}"     ] && cmd+=(--n-cpu-moe "${N_CPU_MOE}")
[ -n "${FA:-}"            ] && cmd+=(-fa   "${FA}")
[ -n "${MMP:-}"           ] && cmd+=(-mmp  "${MMP}")
[ -n "${CACHE_TYPE_K:-}"  ] && cmd+=(-ctk  "${CACHE_TYPE_K}")
[ -n "${CACHE_TYPE_V:-}"  ] && cmd+=(-ctv  "${CACHE_TYPE_V}")
[ -n "${OVERRIDE_TENSOR:-}" ] && cmd+=(-ot "${OVERRIDE_TENSOR}")

# ik_llama-specific flags
[ -n "${FUSED_MOE:-}"       ] && cmd+=(-fmoe  "${FUSED_MOE}")
[ -n "${MERGE_UP_GATE:-}"   ] && cmd+=(-muge  "${MERGE_UP_GATE}")
[ -n "${MERGE_QKV:-}"       ] && cmd+=(-mqkv  "${MERGE_QKV}")
[ -n "${GROUPED_ROUTING:-}" ] && cmd+=(-ger   "${GROUPED_ROUTING}")
[ -n "${ROPE_CACHE:-}"      ] && cmd+=(-rcache "${ROPE_CACHE}")

# --- logging & environment capture ---

source "${SCRIPT_DIR}/bench-env.sh"
bench_preflight_warn || true

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
_model_slug="$(basename "${MODEL}" .gguf)"
LOG_FILE="${LOG_DIR}/$(date +%Y-%m-%d_%H-%M-%S)_${_model_slug}.log"

# Emit system frontmatter to log file
bench_emit_frontmatter > "${LOG_FILE}"

# --- launch ---

if command -v taskset >/dev/null 2>&1 && [ -n "${CPU_RANGE:-}" ]; then
  taskset -c "${CPU_RANGE}" "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
else
  "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
fi

# --- log structured result ---
BACKEND="${BACKEND:-ik_llama.cpp}" LOG_FILE="${LOG_FILE}" \
  "${SCRIPT_DIR}/log-result.sh" || true

