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
# Persist the raw artifacts (engine logs, server logs, generated configs, frontmatter) with the
# rows. /tmp used to hold them and a cleanup destroyed an earlier sweep's acceptance evidence.
TMP="${RUN_DIR:-$REPO/logs/results/refresh-ab-$(date +%Y%m%d-%H%M%S)}"; mkdir -p "$TMP" "$(dirname "$OUT")"
V="$REPO/vendor"
MOE_CACHE_MIB="${MOE_CACHE_MIB:-2048}"

declare -A BIN=(
  [gold_old]="$V/llama.cpp-master/build/bin/llama-server-b11913"
  [gold_new]="$V/llama.cpp-master/build/bin/llama-server-b12073-15-gd1044bc82"
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

strata_arm_cfg() { # label dir cfg_src [env_json]
  local label="$1" dir="$2" src="$3" envjson="${4:-}" cfg="$TMP/$1.json"
  python3 - "$dir" "$src" "$cfg" "$envjson" <<'PY'
import json, os, sys
d, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
envjson = sys.argv[4] if len(sys.argv) > 4 else ""
c = json.load(open(src))
exe = os.path.join(d, "build/strata") if os.path.exists(os.path.join(d, "build/strata")) else os.path.join(d, "engine/strata")
c["exe"] = exe
c["cwd"] = d
c["log"] = out.replace(".json", ".engine.log")
c["port"] = 8023
if envjson:                                   # engine settings the config carries (a switch arm)
    c["env"] = {**(c.get("env") or {}), **json.loads(envjson)}
json.dump(c, open(out, "w"), indent=1)
PY
  taskset -c 0-11 "$V/strata/.venv/bin/python" "$dir/serve/server.py" --engine strata --config "$cfg" \
    --port "$PORT" > "$TMP/$label.log" 2>&1 &
  SRV=$!
}

arm_meta() { # label -> JSON: engine, args, env, git commit, binary identity, frontmatter path
  local label="$1" cfg="$TMP/$1.json"
  python3 - "$label" "$cfg" "${BIN[$1]:-}" "$TMP/frontmatter.yaml" <<'PY'
import json, os, subprocess, sys
label, cfg_path, bin_path, front = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
def git(d, *a):
    try:
        return subprocess.check_output(["git", "-C", d, *a], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""
def ver(exe):
    try:
        return subprocess.check_output([exe, "--version"], stderr=subprocess.STDOUT).decode().strip().splitlines()[0]
    except Exception:
        return ""
meta = {"label": label, "frontmatter": front}
if os.path.exists(cfg_path):
    c = json.load(open(cfg_path))
    exe, d = c.get("exe", ""), c.get("cwd", "")
    meta.update({
        "engine": "strata", "exe": exe, "config_path": cfg_path,
        "engine_version": ver(exe) if exe else "",
        "git_commit": git(d, "rev-parse", "HEAD") if d else "",
        "git_describe": git(d, "describe", "--tags", "--always") if d else "",
        "git_branch": git(d, "rev-parse", "--abbrev-ref", "HEAD") if d else "",
        "args": c.get("args"), "env": c.get("env"), "model_name": c.get("model_name"),
        "tokenizer": c.get("tokenizer"), "sampling": c.get("sampling"),
        "exe_size": os.path.getsize(exe) if exe and os.path.exists(exe) else None,
        "exe_mtime": os.path.getmtime(exe) if exe and os.path.exists(exe) else None,
    })
else:
    meta.update({"engine": "llama.cpp", "exe": bin_path,
                 "engine_version": ver(bin_path) if bin_path else "",
                 "exe_size": os.path.getsize(bin_path) if bin_path and os.path.exists(bin_path) else None,
                 "exe_mtime": os.path.getmtime(bin_path) if bin_path and os.path.exists(bin_path) else None})
print(json.dumps(meta))
PY
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
    strata_master|strata_ple_q8) strata_arm_cfg "$label" "$V/strata-latest" "$V/strata-latest/strata-iq3_xxs-ple_q8.json" ;;
    strata_lean)       strata_arm_cfg "$label" "$V/strata-lean" "$V/strata-lean/strata-iq3_xxs-ple_q8.json" ;;
    strata_lean_ud)    strata_arm_cfg "$label" "$V/strata-lean" "$V/strata-lean/strata-ud-iq2_xs-ple_q8.json" '{"STRATA_ROUTE_TAIL_SKIP":"0"}' ;;
    strata_v0212)      strata_arm_cfg "$label" "$V/strata-v0.2.12" "$V/strata-lean/strata-ud-iq2_xs-ple_q8.json" '{"STRATA_ROUTE_TAIL_SKIP":"0"}' ;;
    strata_lean_share3072) strata_arm_cfg "$label" "$V/strata-lean" "$V/strata-lean/strata-ud-iq2_xs-ple_q8.json" '{"STRATA_ROUTE_TAIL_SKIP":"0","STRATA_PREFILL_CPU_SHARE_MAX":"3072"}' ;;
    strata_v0212_pf0)  strata_arm_cfg "$label" "$V/strata-v0.2.12" "$V/strata-lean/strata-ud-iq2_xs-ple_q8.json" '{"STRATA_ROUTE_TAIL_SKIP":"0","STRATA_PF_SMALL":"0"}' ;;
    strata_v0212_af2)  strata_arm_cfg "$label" "$V/strata-v0.2.12" "$V/strata-lean/strata-ud-iq2_xs-ple_q8.json" '{"STRATA_ROUTE_TAIL_SKIP":"0","STRATA_ADAPT_FETCH":"2"}' ;;
    strata_v0212_af1)  strata_arm_cfg "$label" "$V/strata-v0.2.12" "$V/strata-lean/strata-ud-iq2_xs-ple_q8.json" '{"STRATA_ROUTE_TAIL_SKIP":"0","STRATA_ADAPT_FETCH":"1"}' ;;
    strata_v0212_twopass) strata_arm_cfg "$label" "$V/strata-v0.2.12" "$V/strata-lean/strata-ud-iq2_xs-ple_q8.json" '{"STRATA_ROUTE_TAIL_SKIP":"0","STRATA_SELECT_TWO_PASS":"1"}' ;;
    strata_lean_af0)   strata_arm_cfg "$label" "$V/strata-adapt-fetch" "$V/strata-lean/strata-ud-iq2_xs-ple_q8.json" '{"STRATA_ROUTE_TAIL_SKIP":"0"}' ;;
    strata_lean_af1)   strata_arm_cfg "$label" "$V/strata-adapt-fetch" "$V/strata-lean/strata-ud-iq2_xs-ple_q8.json" '{"STRATA_ROUTE_TAIL_SKIP":"0","STRATA_ADAPT_FETCH":"1"}' ;;
    strata_lean_af2)   strata_arm_cfg "$label" "$V/strata-adapt-fetch" "$V/strata-lean/strata-ud-iq2_xs-ple_q8.json" '{"STRATA_ROUTE_TAIL_SKIP":"0","STRATA_ADAPT_FETCH":"2"}' ;;
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
  local meta; meta="$(arm_meta "$label")"
  printf '%s\n' "$meta" > "$TMP/$label.meta.json"
  python3 "$REPO/bench-models/bench-qwen38-refresh-probe.py" --url "http://127.0.0.1:$PORT" --label "$label" --out "$OUT" \
    --pp-sizes "${PP_SIZES:-2048,8192}" --meta "$meta"
  echo "   VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)  RAM avail: $(free -g | awk '/^Mem:/{print $7}') GiB"
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  pkill -f "strata.*--pack" 2>/dev/null
  fuser -k ${PORT}/tcp 2>/dev/null || true
  sleep 5
}

bench_preflight_warn
bench_emit_frontmatter | tee "$TMP/frontmatter.yaml"
for arm in ${ARMS:-gold_old gold_new exp_old exp_new exp_cache strata_old strata_new}; do
  run_arm "$arm" || true
done
echo done.
