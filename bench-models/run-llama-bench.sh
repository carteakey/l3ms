#!/usr/bin/env bash
# Shared llama-bench runner. Set env vars before exec'ing this script.
#
# Required (no default):
#   MODEL             - path to .gguf model file
#
# Result logging (bench-models/logs/results/<MODEL_KEY>.jsonl):
#   MODEL_KEY         - short slug for the results file (default: model filename stem)
#   STRATEGY          - descriptive label for this run (e.g. "all-cpu-moe", "fit", "partial-cpu")
#   NOTES             - free-text annotation appended to the JSONL record
#
# Optional (llama-bench defaults shown):
#   LLAMA_BENCH       - path to llama-bench binary (default: ../vendor/llama.cpp/build/bin/llama-bench)
#
#   --- runner behaviour ---
#   CPU_RANGE         - taskset CPU affinity range, e.g. "0-11" (default: unset, no taskset)
#   REPETITIONS       - number of times to repeat each test (default: 5)
#   PRIO              - process/thread priority -1..3 (default: 0)
#   DELAY             - seconds between tests (default: 0)
#   OUTPUT_FMT        - stdout format: csv|json|jsonl|md|sql (default: md)
#   OUTPUT_ERR_FMT    - stderr format: csv|json|jsonl|md|sql (default: unset)
#   NUMA              - numa mode: distribute|isolate|numactl (default: unset)
#   VERBOSE           - set to 1 to pass --verbose (default: unset)
#   PROGRESS          - set to 1 to pass --progress (default: unset)
#   NO_WARMUP         - set to 1 to skip warmup runs (default: unset)
#
#   --- test parameters ---
#   TASKS             - shorthand pp,tg pair passed via -pg, e.g. "512,128" (default: unset)
#   N_PROMPT          - prompt token count (default: 512)
#   N_GEN             - generation token count (default: 128)
#   N_DEPTH           - speculative decoding depth (default: 0)
#   BATCH_SIZE        - logical batch size (default: 2048)
#   UBATCH_SIZE       - physical batch size (default: 512)
#   CACHE_TYPE_K      - KV cache type for K: f16|q8_0|q4_0|... (default: f16)
#   CACHE_TYPE_V      - KV cache type for V: f16|q8_0|q4_0|... (default: f16)
#   THREADS           - CPU threads (default: 6)
#   CPU_MASK          - hex CPU affinity mask (default: unset)
#   CPU_STRICT        - strict CPU affinity 0|1 (default: 0)
#   POLL              - polling level 0..100 (default: 50)
#   N_GPU_LAYERS      - layers offloaded to GPU (default: 99)
#   N_CPU_MOE         - MoE expert layers kept on CPU (default: unset, flag omitted)
#   SPLIT_MODE        - tensor split mode: none|layer|row (default: layer)
#   MAIN_GPU          - index of primary GPU (default: 0)
#   NO_KV_OFFLOAD    - disable KV cache offload 0|1 (default: 0)
#   FA                - flash attention 0|1 (default: 0)
#   DEVICE            - device list e.g. "0/1" (default: unset, auto)
#   MMP               - mmap 0|1 (default: 1)
#   DIRECT_IO         - direct IO 0|1 (default: 0)
#   EMBEDDINGS        - embeddings mode 0|1 (default: 0)
#   TENSOR_SPLIT      - tensor split ratios e.g. "3/1" (default: unset)
#   OVERRIDE_TENSOR   - tensor override pattern (default: unset)
#   NO_OP_OFFLOAD     - disable op offload 0|1 (default: 0)
#   NO_HOST           - no host mode 0|1 (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

LLAMA_BENCH="${LLAMA_BENCH:-${REPO_DIR}/vendor/llama.cpp/build/bin/llama-bench}"

# --- validation ---

if [ -z "${MODEL:-}" ]; then
  echo "MODEL is not set" >&2
  exit 1
fi

if [ ! -x "${LLAMA_BENCH}" ]; then
  echo "llama-bench not found/executable: ${LLAMA_BENCH}" >&2
  exit 1
fi

if [ ! -f "${MODEL}" ]; then
  echo "model file not found: ${MODEL}" >&2
  exit 1
fi

# --- build command ---

cmd=("${LLAMA_BENCH}" -m "${MODEL}")

# runner behaviour
[ -n "${NUMA:-}"           ] && cmd+=(--numa "${NUMA}")
[ -n "${REPETITIONS:-}"    ] && cmd+=(-r "${REPETITIONS}")
[ -n "${PRIO:-}"           ] && cmd+=(--prio "${PRIO}")
[ -n "${DELAY:-}"          ] && cmd+=(--delay "${DELAY}")
[ -n "${OUTPUT_FMT:-}"     ] && cmd+=(-o "${OUTPUT_FMT}")
[ -n "${OUTPUT_ERR_FMT:-}" ] && cmd+=(-oe "${OUTPUT_ERR_FMT}")
[ "${VERBOSE:-0}"   = "1"  ] && cmd+=(--verbose)
[ "${PROGRESS:-0}"  = "1"  ] && cmd+=(--progress)
[ "${NO_WARMUP:-0}" = "1"  ] && cmd+=(--no-warmup)

# test parameters — -pg and -p/-n are mutually exclusive; prefer -pg when TASKS is set
if [ -n "${TASKS:-}" ]; then
  cmd+=(-pg "${TASKS}")
else
  [ -n "${N_PROMPT:-}" ] && cmd+=(-p "${N_PROMPT}")
  [ -n "${N_GEN:-}"    ] && cmd+=(-n "${N_GEN}")
fi

[ -n "${N_DEPTH:-}"         ] && cmd+=(-d "${N_DEPTH}")
[ -n "${BATCH_SIZE:-}"      ] && cmd+=(-b "${BATCH_SIZE}")
[ -n "${UBATCH_SIZE:-}"     ] && cmd+=(-ub "${UBATCH_SIZE}")
[ -n "${CACHE_TYPE_K:-}"    ] && cmd+=(-ctk "${CACHE_TYPE_K}")
[ -n "${CACHE_TYPE_V:-}"    ] && cmd+=(-ctv "${CACHE_TYPE_V}")
[ -n "${THREADS:-}"         ] && cmd+=(-t "${THREADS}")
[ -n "${CPU_MASK:-}"        ] && cmd+=(-C "${CPU_MASK}")
[ -n "${CPU_STRICT:-}"      ] && cmd+=(--cpu-strict "${CPU_STRICT}")
[ -n "${POLL:-}"            ] && cmd+=(--poll "${POLL}")
[ -n "${N_GPU_LAYERS:-}"    ] && cmd+=(-ngl "${N_GPU_LAYERS}")
[ -n "${N_CPU_MOE:-}"       ] && cmd+=(-ncmoe "${N_CPU_MOE}")
[ -n "${SPLIT_MODE:-}"      ] && cmd+=(-sm "${SPLIT_MODE}")
[ -n "${MAIN_GPU:-}"        ] && cmd+=(-mg "${MAIN_GPU}")
[ -n "${NO_KV_OFFLOAD:-}"   ] && cmd+=(-nkvo "${NO_KV_OFFLOAD}")
[ -n "${FA:-}"              ] && cmd+=(-fa "${FA}")
[ -n "${DEVICE:-}"          ] && cmd+=(-dev "${DEVICE}")
[ -n "${MMP:-}"             ] && cmd+=(-mmp "${MMP}")
[ -n "${DIRECT_IO:-}"       ] && cmd+=(-dio "${DIRECT_IO}")
[ -n "${EMBEDDINGS:-}"      ] && cmd+=(-embd "${EMBEDDINGS}")
[ -n "${TENSOR_SPLIT:-}"    ] && cmd+=(-ts "${TENSOR_SPLIT}")
[ -n "${OVERRIDE_TENSOR:-}" ] && cmd+=(-ot "${OVERRIDE_TENSOR}")
[ -n "${NO_OP_OFFLOAD:-}"   ] && cmd+=(-nopo "${NO_OP_OFFLOAD}")
[ -n "${NO_HOST:-}"         ] && cmd+=(--no-host "${NO_HOST}")

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
BACKEND="${BACKEND:-llama.cpp}" LOG_FILE="${LOG_FILE}" \
  "${SCRIPT_DIR}/log-result.sh" || true

