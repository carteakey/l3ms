#!/usr/bin/env bash
# Shared two-stage fit→bench runner.
#
# Stage 1: runs llama-fit-params to compute optimal -ngl/-ts/-ot for free VRAM.
# Stage 2: feeds those args directly into llama-bench.
#
# Required (no default):
#   MODEL             - path to .gguf model file
#
# Result logging (bench-models/logs/results/<MODEL_KEY>.jsonl):
#   MODEL_KEY         - short slug for the results file (default: model filename stem)
#   STRATEGY          - defaults to "fit" if not set
#   NOTES             - free-text annotation appended to the JSONL record
#
# Optional — fit stage:
#   LLAMA_FIT         - path to llama-fit-params binary (default: ../vendor/llama.cpp/build/bin/llama-fit-params)
#   FIT_TARGET        - MiB of free VRAM margin to leave on each GPU (default: 1024)
#   FIT_CTX           - minimum context size fit is allowed to reduce to (default: 4096)
#
# Optional — bench stage (passed to llama-bench):
#   LLAMA_BENCH       - path to llama-bench binary (default: ../vendor/llama.cpp/build/bin/llama-bench)
#   CPU_RANGE         - taskset CPU affinity range, e.g. "0-11" (default: unset, no taskset)
#   TASKS             - shorthand pp,tg pair passed via -pg, e.g. "512,128" (default: unset)
#   N_PROMPT          - prompt token count (default: 512)
#   N_GEN             - generation token count (default: 128)
#   BATCH_SIZE        - logical batch size (default: 2048)
#   UBATCH_SIZE       - physical batch size (default: 512)
#   CACHE_TYPE_K      - KV cache type for K: f16|q8_0|q4_0|... (default: f16) — passed to both fit-params and bench
#   CACHE_TYPE_V      - KV cache type for V: f16|q8_0|q4_0|... (default: f16) — passed to both fit-params and bench
#   THREADS           - CPU threads (default: 6)
#   FA                - flash attention 0|1 (default: 1)
#   MMP               - mmap 0|1 (default: 0) — bench only, not passed to fit-params
#   REPETITIONS       - number of times to repeat each test (default: 5)
#   OUTPUT_FMT        - stdout format: csv|json|jsonl|md|sql (default: md)
#
# Compatibility notes (handled automatically):
#   - llama-fit-params does not accept -mmp; MMP is passed to llama-bench only.
#   - llama-bench does not accept -c; the fitted context size is stripped.
#   - llama-bench uses semicolons inside -ot/-ts values; commas are rewritten.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

LLAMA_FIT="${LLAMA_FIT:-${REPO_DIR}/vendor/llama.cpp/build/bin/llama-fit-params}"
LLAMA_BENCH="${LLAMA_BENCH:-${REPO_DIR}/vendor/llama.cpp/build/bin/llama-bench}"

FIT_TARGET="${FIT_TARGET:-1024}"
FIT_CTX="${FIT_CTX:-4096}"

# --- environment capture & preflight warnings ---
source "${SCRIPT_DIR}/bench-env.sh"
bench_preflight_warn || true

# --- logging (set up early so fit stage output is also captured) ---
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
THREADS="${THREADS:-6}"
FA="${FA:-1}"
MMP="${MMP:-0}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [ -z "${MODEL:-}" ]; then
  echo "MODEL is not set" >&2
  exit 1
fi

if [ ! -f "${MODEL}" ]; then
  echo "model file not found: ${MODEL}" >&2
  exit 1
fi

if [ ! -x "${LLAMA_FIT}" ]; then
  echo "llama-fit-params not found/executable: ${LLAMA_FIT}" >&2
  echo "Rebuild with: cmake --build . --target llama-fit-params" >&2
  exit 1
fi

if [ ! -x "${LLAMA_BENCH}" ]; then
  echo "llama-bench not found/executable: ${LLAMA_BENCH}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Print config
# ---------------------------------------------------------------------------

echo "# model      : ${MODEL}"
echo "# tasks      : ${TASKS:-${N_PROMPT:-512}pp + ${N_GEN:-128}tg}"
echo "# batch      : ${BATCH_SIZE} / ubatch: ${UBATCH_SIZE}"
echo "# threads    : ${THREADS}${CPU_RANGE:+ (pinned ${CPU_RANGE})}"
echo "# fa         : ${FA}  mmp: ${MMP}"
echo "# ctk        : ${CACHE_TYPE_K:-f16}  ctv: ${CACHE_TYPE_V:-f16}"
echo "# fit-target : ${FIT_TARGET} MiB margin per GPU"
echo "# fit-ctx    : ${FIT_CTX} tokens minimum context"
echo

# ---------------------------------------------------------------------------
# Stage 1: llama-fit-params
#
# Output format (single stdout line):
#   -c 4096 -ngl 49 -ot "blk\.8\.ffn_...=CPU,blk\.9\...."
#
# Stderr contains the human-readable fit diagnostics; we let that through.
# ---------------------------------------------------------------------------

echo "# [1/2] running llama-fit-params..."

fit_cmd=(
  "${LLAMA_FIT}"
  -m    "${MODEL}"
  -b    "${BATCH_SIZE}"
  -ub   "${UBATCH_SIZE}"
  -t    "${THREADS}"
  -fa   "${FA}"
  -fitt "${FIT_TARGET}"
  -fitc "${FIT_CTX}"
)
[ -n "${CACHE_TYPE_K:-}" ] && fit_cmd+=(-ctk "${CACHE_TYPE_K}")
[ -n "${CACHE_TYPE_V:-}" ] && fit_cmd+=(-ctv "${CACHE_TYPE_V}")

fit_line=$(
  "${fit_cmd[@]}" 2>/dev/null
) || true

if [ -z "${fit_line}" ]; then
  echo "# llama-fit-params produced no output — model fits as-is or fit failed." >&2
  echo "# Proceeding without fitted placement args." >&2
  fit_line=""
fi

echo "# fitted : ${fit_line:-<none, model fits as-is>}"
echo

# ---------------------------------------------------------------------------
# Post-process fitted args for llama-bench compatibility:
#
#   1. Strip -c <n>  — llama-bench has no -c flag; context is implied by -pg.
#   2. Commas → semicolons inside -ot "..." and -ts "..." values only.
#      llama-bench treats commas as multi-run separators, so tensor pattern
#      lists must use semicolons instead.
# ---------------------------------------------------------------------------

if command -v python3 >/dev/null 2>&1; then
  bench_line=$(printf '%s' "${fit_line}" | python3 -c "
import re, sys
line = sys.stdin.read().strip()
# Strip -c <value>
line = re.sub(r'-c\s+\S+', '', line)
# Commas -> semicolons inside -ot \"...\" and -ts \"...\" values
def repl(m): return m.group(0).replace(',', ';')
line = re.sub(r'-(ot|ts) \"[^\"]*\"', repl, line)
# Clean up extra whitespace
line = re.sub(r'  +', ' ', line).strip()
print(line, end='')
")
else
  # POSIX sed/sh fallback
  bench_line="${fit_line}"
  bench_line=$(printf '%s' "${bench_line}" | sed 's/-c [^ ]*  *//g')
  while printf '%s' "${bench_line}" | grep -q '"\([^"]*\),\([^"]*\)"'; do
    bench_line=$(printf '%s' "${bench_line}" | sed 's/"\([^"]*\),/"\1;/g')
  done
fi

# ---------------------------------------------------------------------------
# Stage 2: llama-bench
# ---------------------------------------------------------------------------

echo "# [2/2] running llama-bench..."

bench_cmd=(
  "${LLAMA_BENCH}"
  -m   "${MODEL}"
  -b   "${BATCH_SIZE}"
  -ub  "${UBATCH_SIZE}"
  -t   "${THREADS}"
  -fa  "${FA}"
  -mmp "${MMP}"
)

[ -n "${CACHE_TYPE_K:-}" ] && bench_cmd+=(-ctk "${CACHE_TYPE_K}")
[ -n "${CACHE_TYPE_V:-}" ] && bench_cmd+=(-ctv "${CACHE_TYPE_V}")
[ -n "${REPETITIONS:-}"  ] && bench_cmd+=(-r "${REPETITIONS}")
[ -n "${OUTPUT_FMT:-}"   ] && bench_cmd+=(-o "${OUTPUT_FMT}")

# -pg takes priority over separate -p/-n
if [ -n "${TASKS:-}" ]; then
  bench_cmd+=(-pg "${TASKS}")
else
  [ -n "${N_PROMPT:-}" ] && bench_cmd+=(-p "${N_PROMPT}")
  [ -n "${N_GEN:-}"    ] && bench_cmd+=(-n "${N_GEN}")
fi

# Append fitted placement args (eval needed to correctly expand the quoted -ot value)
if [ -n "${bench_line:-}" ]; then
  eval "fitted_arr=(${bench_line})"
  bench_cmd+=("${fitted_arr[@]}")
fi

_model_slug="$(basename "${MODEL}" .gguf)"
LOG_FILE="${LOG_DIR}/$(date +%Y-%m-%d_%H-%M-%S)_${_model_slug}_fit.log"

# Replay the printed config into the log, then run bench with tee
{
  bench_emit_frontmatter
  echo "# model      : ${MODEL}"
  echo "# tasks      : ${TASKS:-${N_PROMPT:-512}pp + ${N_GEN:-128}tg}"
  echo "# batch      : ${BATCH_SIZE} / ubatch: ${UBATCH_SIZE}"
  echo "# threads    : ${THREADS}${CPU_RANGE:+ (pinned ${CPU_RANGE})}"
  echo "# fa         : ${FA}  mmp: ${MMP}"
  echo "# ctk        : ${CACHE_TYPE_K:-f16}  ctv: ${CACHE_TYPE_V:-f16}"
  echo "# fit-target : ${FIT_TARGET} MiB margin per GPU"
  echo "# fit-ctx    : ${FIT_CTX} tokens minimum context"
  echo "# fitted     : ${fit_line:-<none>}"
  echo
  if command -v taskset >/dev/null 2>&1 && [ -n "${CPU_RANGE:-}" ]; then
    taskset -c "${CPU_RANGE}" "${bench_cmd[@]}" 2>&1
  else
    "${bench_cmd[@]}" 2>&1
  fi
} | tee "${LOG_FILE}"

# --- log structured result ---
BACKEND="${BACKEND:-llama.cpp}" \
STRATEGY="${STRATEGY:-fit}" \
OVERRIDE_TENSOR="${bench_line:-}" \
LOG_FILE="${LOG_FILE}" \
  "${SCRIPT_DIR}/log-result.sh" || true
