#!/usr/bin/env bash
# Mistral-Small-4-119B (UD-IQ4_XS) — MoE strategy sweep for best tg (tokens/s).
#
# By default this script sweeps a small set of -ot strategy presets and reports
# the best tg result. Set SWEEP=0 to run one strategy only.
#
# Usage:
#   ./bench-llama-cpp-mistral-small-4-119b-strategies.sh
#   SWEEP=0 STRATEGY=fit-128 ./bench-llama-cpp-mistral-small-4-119b-strategies.sh
#   TASKS=1024,256 N_GPU_LAYERS=37 ./bench-llama-cpp-mistral-small-4-119b-strategies.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [ -z "${LLAMA_BENCH:-}" ]; then
  for candidate in \
    "${REPO_DIR}/vendor/llama.cpp/build/bin/llama-bench" \
    "${REPO_DIR}/vendor/llama.cpp/llama-bench" \
    "${REPO_DIR}/vendor-forks/llama.cpp-copilot/build/bin/llama-bench" \
    "${REPO_DIR}/vendor-forks/llama.cpp-copilot/llama-bench" \
    "${REPO_DIR}/vendor/llama.cpp/build/bin/llama-bench" \
    "${REPO_DIR}/vendor-forks/llama.cpp-copilot/build/bin/llama-bench"
  do
    if [ -x "${candidate}" ]; then
      LLAMA_BENCH="${candidate}"
      break
    fi
  done
  LLAMA_BENCH="${LLAMA_BENCH:-${REPO_DIR}/vendor/llama.cpp/build/bin/llama-bench}"
fi

MODEL="${MODEL:-/mnt/lab/models/unsloth/Mistral-Small-4-119B-2603-GGUF/UD-IQ4_XS/Mistral-Small-4-119B-2603-UD-IQ4_XS-00001-of-00003.gguf}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-37}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
REPETITIONS="${REPETITIONS:-3}"
OUTPUT_FMT="${OUTPUT_FMT:-md}"
SWEEP="${SWEEP:-1}"
STRATEGY="${STRATEGY:-fit-128}"

if [ ! -x "${LLAMA_BENCH}" ]; then
  echo "llama-bench not found/executable: ${LLAMA_BENCH}" >&2
  exit 1
fi

if [ ! -f "${MODEL}" ]; then
  echo "model file not found: ${MODEL}" >&2
  exit 1
fi

if [ "${SWEEP}" = "1" ] && [ "${OUTPUT_FMT}" != "md" ]; then
  echo "SWEEP=1 requires OUTPUT_FMT=md (parser expects markdown output)." >&2
  exit 1
fi

strategy_override() {
  case "$1" in
    all-cpu-moe)
      printf '%s\n' '.ffn_(up|down|gate)_(ch|)exps=CPU'
      ;;
    partial-cpu)
      printf '%s\n' 'blk\.([5-9]|[1-9][0-9]|[1-9][0-9][0-9])\.ffn_(up|down|gate)_(ch|)exps=CPU'
      ;;
    up-down-cpu)
      printf '%s\n' '.ffn_(up|down)_(ch|)exps=CPU'
      ;;
    fit-512)
      # fit-params derived at FIT_TARGET=512, FIT_CTX=32768 on this host
      printf '%s\n' 'blk\.(30|3[1-6])\.ffn_(up|down|gate)_(ch|)exps=CPU'
      ;;
    fit-128)
      # fit-params derived at FIT_TARGET=128, FIT_CTX=32768 on this host
      printf '%s\n' 'blk\.(29|3[0-6])\.ffn_(up|down|gate)_(ch|)exps=CPU'
      ;;
    up-cpu)
      printf '%s\n' '.ffn_up_(ch|)exps=CPU'
      ;;
    none)
      printf '%s\n' ''
      ;;
    *)
      return 1
      ;;
  esac
}

extract_tg() {
  awk -F'|' '/\|[[:space:]]*tg[0-9]+[[:space:]]*\|/ { v=$(NF-1); gsub(/^[[:space:]]+|[[:space:]]+$/, "", v); n=split(v, a, /[[:space:]]+/); for (i=1; i<=n; ++i) if (a[i] != "") { print a[i]; exit } }' "$1"
}

float_gt() {
  awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > b) }'
}

run_one() {
  local strategy="$1"
  local override
  local tmp
  local tg
  local rc

  TG_RESULT=""

  if ! override="$(strategy_override "${strategy}")"; then
    echo "Unknown STRATEGY '${strategy}'." >&2
    echo "Valid: all-cpu-moe | partial-cpu | up-down-cpu | fit-512 | fit-128 | up-cpu | none" >&2
    return 3
  fi

  export MODEL LLAMA_BENCH TASKS N_GPU_LAYERS THREADS CPU_RANGE FA MMP BATCH_SIZE UBATCH_SIZE CACHE_TYPE_K CACHE_TYPE_V REPETITIONS OUTPUT_FMT OVERRIDE_TENSOR
  OVERRIDE_TENSOR="${override}"

  echo "# strategy : ${strategy}"
  echo "# override : ${OVERRIDE_TENSOR:-<none>}"
  echo "# tasks    : ${TASKS}"
  echo "# ngl      : ${N_GPU_LAYERS}"
  echo "# threads  : ${THREADS} (pinned ${CPU_RANGE})"
  echo "# kv       : ${CACHE_TYPE_K}/${CACHE_TYPE_V}"
  echo

  tmp="$(mktemp)"
  if bash "${SCRIPT_DIR}/run-llama-bench.sh" 2>&1 | tee "${tmp}"; then
    if grep -qi "failed to create context" "${tmp}"; then
      echo "Context creation failed for strategy '${strategy}' (likely OOM with this placement)." >&2
      rm -f "${tmp}"
      return 1
    fi
    if grep -qi "failed to load model" "${tmp}"; then
      if grep -qi "unknown model architecture" "${tmp}"; then
        echo "Model failed to load in llama-bench." >&2
        echo "Hint: Mistral-Small-4 needs a llama.cpp build with 'mistral4' support (PR #20649+)." >&2
        rm -f "${tmp}"
        return 2
      fi
      echo "Model failed to load for strategy '${strategy}' (likely OOM with this placement)." >&2
      rm -f "${tmp}"
      return 1
    fi

    tg="$(extract_tg "${tmp}")"
    if [ -n "${tg}" ]; then
      TG_RESULT="${tg}"
      rm -f "${tmp}"
      return 0
    fi
    echo "Could not parse tg from benchmark output." >&2
    rm -f "${tmp}"
    return 1
  fi

  rc=$?
  if grep -qi "failed to create context" "${tmp}"; then
    echo "Context creation failed for strategy '${strategy}' (likely OOM with this placement)." >&2
    rm -f "${tmp}"
    return 1
  fi
  if grep -qi "failed to load model" "${tmp}"; then
    if grep -qi "unknown model architecture" "${tmp}"; then
      echo "Model failed to load in llama-bench." >&2
      echo "Hint: Mistral-Small-4 needs a llama.cpp build with 'mistral4' support (PR #20649+)." >&2
      rm -f "${tmp}"
      return 2
    fi
    echo "Model failed to load for strategy '${strategy}' (likely OOM with this placement)." >&2
    rm -f "${tmp}"
    return 1
  fi
  rm -f "${tmp}"
  return "${rc}"
}

if [ "${SWEEP}" != "1" ]; then
  run_one "${STRATEGY}"
  rc=$?
  if [ "${rc}" -eq 0 ]; then
    echo
    echo "# tg result : ${TG_RESULT} t/s"
    exit 0
  fi
  exit "${rc}"
fi

strategies=(all-cpu-moe partial-cpu up-down-cpu fit-512 fit-128 up-cpu none)
best_strategy=""
best_tg=""
summary_lines=()
TG_RESULT=""

for s in "${strategies[@]}"; do
  if run_one "${s}"; then
    summary_lines+=("$(printf '%-12s  %8s t/s  ok' "${s}" "${TG_RESULT}")")
    if [ -z "${best_tg}" ] || float_gt "${TG_RESULT}" "${best_tg}"; then
      best_tg="${TG_RESULT}"
      best_strategy="${s}"
    fi
  else
    rc=$?
    if [ "${rc}" -eq 2 ]; then
      exit 2
    fi
    summary_lines+=("$(printf '%-12s  %8s      fail(rc=%s)' "${s}" "-" "${rc}")")
  fi
  echo
done

echo "# sweep summary (tg tokens/s)"
for row in "${summary_lines[@]}"; do
  echo "# ${row}"
done

if [ -n "${best_strategy}" ]; then
  echo
  echo "# best strategy : ${best_strategy}"
  echo "# best tg       : ${best_tg} t/s"
  echo "# rerun exactly : SWEEP=0 STRATEGY=${best_strategy} ./bench-llama-cpp-mistral-small-4-119b-strategies.sh"
  exit 0
fi

echo
echo "No successful tg result found." >&2
exit 1
