#!/usr/bin/env bash
# log-result.sh — parse a llama-bench log and append a JSONL record.
#
# Called automatically by run-llama-bench.sh, run-ik-llama-bench.sh, and
# run-llama-fit-bench.sh after each run. Can also be called manually:
#
#   MODEL_KEY=qwen3-coder-next LOG_FILE=bench-models/logs/2026-04-19_….log \
#     bench-models/log-result.sh
#
# Required:
#   LOG_FILE   - path to the llama-bench log file to parse
#   MODEL_KEY  - short slug for the results file (e.g. qwen3-coder-next)
#                Falls back to the model filename stem if unset.
#
# Context env vars (captured from the calling runner — all optional):
#   MODEL, BACKEND, STRATEGY, N_GPU_LAYERS, N_CPU_MOE, OVERRIDE_TENSOR,
#   FIT_CTX, FIT_TARGET, CACHE_TYPE_K, CACHE_TYPE_V, THREADS, REPETITIONS,
#   TASKS, NOTES
#
# Output:
#   bench-models/logs/results/<MODEL_KEY>.jsonl  — one JSON object per run appended

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/logs/results"
mkdir -p "${RESULTS_DIR}"

LOG_FILE="${LOG_FILE:-}"
if [ -z "${LOG_FILE}" ] || [ ! -f "${LOG_FILE}" ]; then
  echo "log-result: LOG_FILE not set or not found (${LOG_FILE:-<unset>})" >&2
  exit 1
fi

# Determine model key: explicit > MODEL env basename > log file basename
_model_stem="$(basename "${LOG_FILE}" .log | sed 's/^[0-9_-]*_//')"
MODEL_KEY="${MODEL_KEY:-${_model_stem}}"
RESULTS_FILE="${RESULTS_FILE:-${RESULTS_DIR}/${MODEL_KEY}.jsonl}"

# ---------------------------------------------------------------------------
# Parse llama-bench markdown table output.
# The table rows look like:
#   | ... | CUDA | 49 | 10 | ... | pp 512 | 502.34 ± 3.10 |
#   | ... | CUDA | 49 | 10 | ... | tg 128 |  39.62 ± 0.20 |
# We extract pp_ts, pp_std, tg_ts, tg_std from the last pp/tg pair found.
# ---------------------------------------------------------------------------

_parsed=$(python3 - "${LOG_FILE}" <<'PYEOF'
import re, sys, json

log = open(sys.argv[1]).read()

results = {"pp_tokens": None, "pp_ts": None, "pp_std": None,
           "tg_tokens": None, "tg_ts": None, "tg_std": None}

test_re = re.compile(r'^(pp|tg)\s*(\d+)$')
speed_re = re.compile(r'([\d.]+)\s*±\s*([\d.]+)')

for line in log.splitlines():
    if not line.startswith("|") or "---" in line:
        continue
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    for idx, cell in enumerate(cells):
        m = test_re.match(cell)
        if not m or idx + 1 >= len(cells):
            continue
        s = speed_re.search(cells[idx + 1])
        if not s:
            continue
        kind, tokens = m.group(1), int(m.group(2))
        results[f"{kind}_tokens"] = tokens
        results[f"{kind}_ts"]     = float(s.group(1))
        results[f"{kind}_std"]    = float(s.group(2))

# Also try the simpler jsonl output format (-o jsonl) if md parse found nothing
if results["pp_ts"] is None:
    jl_re = re.compile(r'^\{.*"test".*\}', re.MULTILINE)
    for line in jl_re.findall(log):
        try:
            obj = json.loads(line)
            test = obj.get("test", "")
            ts   = obj.get("avg_ts", obj.get("t_avg", None))
            std  = obj.get("std_ts", obj.get("t_std", None))
            if test.startswith("pp") and ts:
                results["pp_tokens"] = int(test.split()[1]) if len(test.split()) > 1 else None
                results["pp_ts"]  = ts
                results["pp_std"] = std
            elif test.startswith("tg") and ts:
                results["tg_tokens"] = int(test.split()[1]) if len(test.split()) > 1 else None
                results["tg_ts"]  = ts
                results["tg_std"] = std
        except (IndexError, KeyError, TypeError, ValueError):
            pass

print(json.dumps(results))
PYEOF
)

# ---------------------------------------------------------------------------
# Build the JSONL record
# ---------------------------------------------------------------------------

_ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
_git_sha="$(git -C "${SCRIPT_DIR}/.." rev-parse --short HEAD 2>/dev/null || echo '')"
_llama_bin="${LLAMA_BENCH:-${SCRIPT_DIR}/../vendor/llama.cpp/build/bin/llama-bench}"
_llama_ver="$(grep -E '^build:' "${LOG_FILE}" | tail -1 || true)"
if [ -z "${_llama_ver}" ]; then
  _llama_ver="$("${_llama_bin}" --help 2>&1 | grep -E '^build:' | head -1 || true)"
fi

PARSED_JSON="${_parsed}" \
RESULTS_FILE="${RESULTS_FILE}" \
RESULT_TS="${_ts}" \
RESULT_MODEL_KEY="${MODEL_KEY}" \
RESULT_MODEL="${MODEL:-}" \
RESULT_BACKEND="${BACKEND:-llama.cpp}" \
RESULT_STRATEGY="${STRATEGY:-}" \
RESULT_N_GPU_LAYERS="${N_GPU_LAYERS:-99}" \
RESULT_N_CPU_MOE="${N_CPU_MOE:-}" \
RESULT_OVERRIDE_TENSOR="${OVERRIDE_TENSOR:-}" \
RESULT_FIT_CTX="${FIT_CTX:-}" \
RESULT_FIT_TARGET="${FIT_TARGET:-}" \
RESULT_CTX="${TASKS:-${N_PROMPT:-512},${N_GEN:-128}}" \
RESULT_CACHE_TYPE_K="${CACHE_TYPE_K:-f16}" \
RESULT_CACHE_TYPE_V="${CACHE_TYPE_V:-f16}" \
RESULT_THREADS="${THREADS:-6}" \
RESULT_REPETITIONS="${REPETITIONS:-}" \
RESULT_GIT_SHA="${_git_sha}" \
RESULT_LLAMA_VERSION="${_llama_ver}" \
RESULT_LOG_FILE="${LOG_FILE}" \
RESULT_NOTES="${NOTES:-}" \
python3 - <<'PYEOF'
import json, os, sys

def maybe_int(value):
    return int(value) if value else None

parsed = json.loads(os.environ["PARSED_JSON"])

sys_info = None
log_file = os.environ.get("RESULT_LOG_FILE", "")
if log_file and os.path.isfile(log_file):
    try:
        with open(log_file, "r") as f:
            for _ in range(50):
                line = f.readline()
                if not line:
                    break
                if line.startswith("# bench_sys_json:"):
                    raw = line.split(":", 1)[1].strip()
                    sys_info = json.loads(raw)
                    break
    except Exception:
        pass

if not sys_info and os.environ.get("RESULT_SYS_JSON"):
    try:
        sys_info = json.loads(os.environ["RESULT_SYS_JSON"])
    except Exception:
        pass

record = {
    "ts":            os.environ["RESULT_TS"],
    "model_key":     os.environ["RESULT_MODEL_KEY"],
    "model":         os.environ["RESULT_MODEL"],
    "backend":       os.environ["RESULT_BACKEND"],
    "strategy":      os.environ["RESULT_STRATEGY"],
    "ngl":           maybe_int(os.environ["RESULT_N_GPU_LAYERS"]),
    "n_cpu_moe":     maybe_int(os.environ["RESULT_N_CPU_MOE"]),
    "override_tensor": os.environ["RESULT_OVERRIDE_TENSOR"],
    "fit_ctx":       maybe_int(os.environ["RESULT_FIT_CTX"]),
    "fit_target":    maybe_int(os.environ["RESULT_FIT_TARGET"]),
    "ctx":           os.environ["RESULT_CTX"],
    "ctk":           os.environ["RESULT_CACHE_TYPE_K"],
    "ctv":           os.environ["RESULT_CACHE_TYPE_V"],
    "threads":       maybe_int(os.environ["RESULT_THREADS"]),
    "repetitions":   maybe_int(os.environ["RESULT_REPETITIONS"]),
    "pp_tokens":     parsed.get("pp_tokens"),
    "pp_ts":         parsed.get("pp_ts"),
    "pp_std":        parsed.get("pp_std"),
    "tg_tokens":     parsed.get("tg_tokens"),
    "tg_ts":         parsed.get("tg_ts"),
    "tg_std":        parsed.get("tg_std"),
    "git_sha":       os.environ["RESULT_GIT_SHA"],
    "llama_version": os.environ["RESULT_LLAMA_VERSION"],
    "log_file":      os.environ["RESULT_LOG_FILE"],
    "notes":         os.environ["RESULT_NOTES"],
    "sys":           sys_info,
}

# Drop None values to keep records compact
record = {k: v for k, v in record.items() if v not in (None, "")}

results_file = os.environ["RESULTS_FILE"]
with open(results_file, "a") as f:
    f.write(json.dumps(record) + "\n")

pp = record.get("pp_ts"); tg = record.get("tg_ts")
print(f"log-result: wrote to {results_file}")
if pp or tg:
    print(f"  pp={pp} ± {record.get('pp_std','?')} t/s   tg={tg} ± {record.get('tg_std','?')} t/s")
else:
    print("  warning: no pp/tg values parsed from log — check log format")
if sys_info:
    gov = sys_info.get("cpu_governor", "?")
    ram = sys_info.get("ram_speed_mts", "?")
    print(f"  sys: governor={gov}, RAM={ram} MT/s")
PYEOF

