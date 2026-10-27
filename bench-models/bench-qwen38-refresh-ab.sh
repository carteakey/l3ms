#!/usr/bin/env bash
# Qwen3.8-Flash-Next refresh A/B — llama.cpp gold/exp old vs latest, Strata v0.1.30 vs latest.
#
# All arms serve the SAME quant (ISTA-DASLab GSQ-RCO IQ3_XXS) so llama.cpp and
# Strata are directly comparable (the AD-4.27bpw quant lives on the NAS).
# Requires llama-swap stopped (full-VRAM experiment):
#   systemctl --user stop llama-swap.service
# Arms (override with ARMS="gold_old gold_new ..."):
#   gold_old   master b11241 (1c4729414), production gold flags
#   gold_new   master b11475 (7e8324f5f), same flags
#   exp_old    vendor/llama.cpp-exp-stack-v2 llama-server-exp, production exp flags
#   exp_new    master b11475 + #28671 + #29887 (vendor/llama.cpp-exp-latest), same flags
#   exp_cache  exp_new + --moe-cache-mib (MOE_CACHE_MIB, default 2048)
#   strata_old vendor/strata v0.1.30 (+local THP patch)
#   strata_new vendor/strata-latest v0.1.40.2
# Results: logs/results/qwen38-refresh-ab.jsonl ; server logs: /tmp/refresh-ab/<arm>.log

REPO="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/bench-models/bench-env.sh"
M="$HOME/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
PORT=8023
OUT="$REPO/logs/results/qwen38-refresh-ab.jsonl"
TMP=/tmp/refresh-ab; mkdir -p "$TMP" "$(dirname "$OUT")"
V="$REPO/vendor"
MOE_CACHE_MIB="${MOE_CACHE_MIB:-2048}"

declare -A BIN=(
  [gold_old]="$V/llama.cpp-master/build/bin/llama-server-b11619-18-g683bbc0dc"
  [gold_new]="$V/llama.cpp-master/build/bin/llama-server-b11913"
  [exp_old]="$V/llama.cpp-exp-stack-v2/build/bin/llama-server"
  [exp_new]="$V/llama.cpp-exp-latest/build/bin/llama-server"
  [exp_cache]="$V/llama.cpp-exp-latest/build/bin/llama-server"
)

llama_arm() { # label ctx ub extra...
  local label="$1" ctx="$2" ub="$3"; shift 3
  GGML_CUDA_GRAPH_OPT=1 taskset -c 0-11 "${BIN[$label]}" -m "$M" --alias refresh-ab \
    --fit on --fit-target 512 -c "$ctx" --parallel 1 -b 4096 -ub "$ub" \
    -fa on --jinja -ctk q8_0 -ctv q8_0 -t 10 --threads-batch 12 --prio 2 \
    --lazy-mode on --spec-type ngram-mod \
    --spec-ngram-mod-n-match 60 --spec-ngram-mod-n-min 12 --spec-ngram-mod-n-max 24 \
    --temp 0 --no-warmup "$@" --host 127.0.0.1 --port "$PORT" > "$TMP/$label.log" 2>&1 &
  SRV=$!
}

strata_arm() { # label dir
  local label="$1" dir="$2" cfg="$TMP/$1.json"
  python3 - "$dir" "$cfg" <<'PY'
import json, os, sys
d, out = sys.argv[1], sys.argv[2]
c = json.load(open("/home/kchauhan/repos/l3ms/vendor/strata/strata-iq3_xxs.json"))
exe = os.path.join(d, "build/strata") if os.path.exists(os.path.join(d, "build/strata")) else os.path.join(d, "engine/strata")
c["exe"] = exe
c["cwd"] = d
c["log"] = out.replace(".json", ".engine.log")
c["port"] = int(sys.argv[3]) if len(sys.argv) > 3 else 8023
json.dump(c, open(out, "w"), indent=1)
PY
  taskset -c 0-11 "$V/strata/.venv/bin/python" "$dir/serve/server.py" --engine strata --config "$cfg" \
    --port "$PORT" > "$TMP/$label.log" 2>&1 &
  SRV=$!
}

strata_arm_cfg() { # label dir cfg_src
  local label="$1" dir="$2" src="$3" cfg="$TMP/$1.json"
  python3 - "$dir" "$src" "$cfg" <<'PY'
import json, os, sys
d, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(src))
exe = os.path.join(d, "build/strata") if os.path.exists(os.path.join(d, "build/strata")) else os.path.join(d, "engine/strata")
c["exe"] = exe
c["cwd"] = d
c["log"] = out.replace(".json", ".engine.log")
c["port"] = 8023
json.dump(c, open(out, "w"), indent=1)
PY
  taskset -c 0-11 "$V/strata/.venv/bin/python" "$dir/serve/server.py" --engine strata --config "$cfg" \
    --port "$PORT" > "$TMP/$label.log" 2>&1 &
  SRV=$!
}

run_arm() {
  local label="$1"
  case "$label" in
    gold_old|gold_new) llama_arm "$label" 98304 2048 ;;
    exp_old|exp_new)   llama_arm "$label" 65536 1024 ;;
    exp_cache)         llama_arm "$label" 65536 1024 --moe-cache-mib "$MOE_CACHE_MIB" ;;
    strata_old)        strata_arm "$label" "$V/strata-v0.1.44.1" ;;
    strata_new|strata_v145) strata_arm "$label" "$V/strata-latest" ;;
    strata_v026)       strata_arm "$label" "$V/strata-v0.2.6" ;;
    strata_ple_q8)     strata_arm_cfg "$label" "$V/strata-latest" "$V/strata-latest/strata-iq3_xxs-ple_q8.json" ;;
    strata_lean)       strata_arm_cfg "$label" "$V/strata-lean" "$V/strata-lean/strata-iq3_xxs-ple_q8.json" ;;
    *) echo "unknown arm $label"; return 1 ;;
  esac
  [[ -n "${BIN[$label]:-}" && ! -x "${BIN[$label]}" ]] && { echo "== $label: binary missing"; kill $SRV 2>/dev/null; return 1; }
  echo "== arm: $label"
  local ok=""
  for i in $(seq 1 450); do   # up to 15 min (Strata cold arena fill)
    curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -qi '"status": *"ok"\|ok' && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || { echo "   server exited"; break; }
    sleep 2
  done
  if [ -z "$ok" ]; then echo "   FAILED to load:"; tail -5 "$TMP/$label.log"; kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; return 1; fi
  python3 "$REPO/bench-models/bench-qwen38-refresh-probe.py" --url "http://127.0.0.1:$PORT" --label "$label" --out "$OUT"
  echo "   VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)  RAM avail: $(free -g | awk '/^Mem:/{print $7}') GiB"
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  pkill -f "strata.*--pack" 2>/dev/null
  sleep 5
}

bench_preflight_warn
bench_emit_frontmatter | tee "$TMP/frontmatter.yaml"
for arm in ${ARMS:-gold_old gold_new exp_old exp_new exp_cache strata_old strata_new}; do
  run_arm "$arm" || true
done
echo done.
