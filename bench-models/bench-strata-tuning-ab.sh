#!/usr/bin/env bash
# bench-strata-tuning-ab.sh - Strata plat-tier tuning sweep: flags and env only, no rebuild.
#
# Same binary (vendor/strata-lean/build/strata), same quant (ISTA-DASLab GSQ-RCO IQ3_XXS),
# same probe as bench-qwen38-refresh-ab.sh. Every arm is an overlay on the PRODUCTION run
# config (vendor/strata-lean/strata-iq3_xxs-ple_q8.json): extra args, exported STRATA_* env,
# or a wider taskset mask. That is deliberate: a winner becomes one config line in the
# served tier instead of a cherry-pick.
#
# Full-VRAM experiment, the router must be stopped first (this script refuses otherwise):
#   systemctl --user stop llama-swap.service   # then run this
#   systemctl --user start llama-swap.service  # when it finishes
# FORCE=1 runs anyway (do not, a mid-sweep model swap invalidates every row).
#
# Arms (override with ARMS="base kv_grow ..."; run base first and last so deltas are paired):
#   base          production config exactly as served today (control)
#   reserve832    --vram-reserve-mib 832, what the engine itself asks for (#2065)
#   kv_grow       --kv-grow: the K/V maps only the cells a request reaches, the freed
#                 VRAM becomes expert slots (off by default in lean and in v0.2.7)
#   ctx64k        --max-context 65536: half the K/V reservation, no elastic growth
#   routeprior15  STRATA_ROUTE_PRIOR=1.5: bias routing to resident experts (changes output)
#   routeprior_kv STRATA_ROUTE_PRIOR=1.5 plus --kv-grow (both push the same bottleneck)
#   routeprior10  STRATA_ROUTE_PRIOR=1.0: the weaker point of the same switch (opt-in arm, not in
#                 the default ladder)
#   route_resident STRATA_ROUTE_RESIDENT=4: the other residency law, a margin swap inside
#                 STRATA_ROUTE_RESIDENT_RANKS (default 6-9) rather than a logit bonus (opt-in arm)
#   draft_en      --mtp-draft-vocab data/draft_vocab_en.bin (40,525 tokens instead of 106,299)
#   spec5         --spec 5
#   specprob      --spec 5 with STRATA_SPEC_PROB=1 (gated proposal law, Strata PR #3416)
#   no_suffix     --suffix-draft 0: MTP only, isolates how much speed is prompt-lookup drafting
#   lookup        --lookup-chain 5: longer suffix repeats become draft windows
#   prefill16k    --prefill auto:16384: bigger prompt chunks than the 8192 default cap
#   prefill_alias --prefill-head-alias: unload the output head while the prompt is read
#   imma          STRATA_PROMPT_ATTN_IMMA=1: int8 tensor-core prompt attention (sm_80+)
#   hcq8          STRATA_HC_Q8=1: Q8 hyper-connection GEMV (upstream default only on gfx12)
#   ple_mmap      --ple-io mmap: the n-gram table through the file cache instead of unbuffered reads
#   aux_ecores    --aux-cpus auto with taskset 0-15 (the 4 idle Alder Lake E-cores)
#   grnormfast    STRATA_GR_NORM_FAST=1: the register-split hyper-connection norm. It is
#                 opt-in on CUDA (fused_gr.cu:1179) and the served run config has no env block,
#                 so AGENTS.md listing it as a plat feature is wrong until this arm proves it.
#
# Identity gate, with its noise floor measured. Each arm runs bench-models/strata-identity-probe.py
# twice, before any speed probe: seven fixed greedy prompts (temperature 0, top_k 1, seed 42,
# cache_prompt off) over code, strict JSON, an explanation, and 4K and 20K document replays.
#
# The 2026-10-28 ladder showed why a plain identical-or-not gate is useless here: two boots of the
# same --kv-grow config differed on 4 of 7 prompts, and two passes inside one boot of the served tier
# differed on 3 of 7 (code_small, code_long, pp20k), every time at the same token count. With 3,277
# of 24,576 expert pairs cached, which experts are resident changes between passes, and the cached
# and streamed paths are not bitwise the same, so greedy output is not reproducible on this box. The
# gate is therefore comparative:
#   intra_diff_n     second pass against first pass, same boot       -> this arm's noise floor
#   identity_diff_n  first pass against the ladder's reference arm   -> the switch's divergence
# A switch counts as quality-neutral only when identity_diff_n is no larger than the control pair's
# own cross-boot divergence. Upstream calls the strict version the identity gate for an Exact switch
# (v0.2.7 docs/DETAILS.md line 1031); on a partially cached MoE the control cannot meet the strict
# version either, and that is the finding rather than a bug in the probe.
#
# Results: logs/results/strata-tuning-ab.jsonl    probe rows, comparable to the refresh A/B rows
#          logs/results/strata-tuning-diag.jsonl  per arm: expert slots, decode hit rate, PCIe
#          share, free VRAM at start, the #2065 warning, pool worker placement, arena page
#          backing, draft head token count, and read_bytes per generated token
# Server logs: /tmp/strata-tune/<arm>.log, engine logs: /tmp/strata-tune/<arm>.engine.log
#
# Why a diag file: AGENTS.md treats /proc/<pid>/io read_bytes as the ground truth for
# whether a "faster" run is actually spilling to SSD. A probe row alone cannot tell those
# apart. MB/token stays near this arch's PLE design traffic (~5 KB/token of n-gram rows);
# tens of MB/token means the win is a spill.

set -uo pipefail

REPO="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/bench-models/bench-env.sh"

V="$REPO/vendor"
LEAN="$V/strata-lean"
SRC_CFG="${SRC_CFG:-$LEAN/strata-iq3_xxs-ple_q8.json}"
ENGINE="$LEAN/build/strata"
PY="$V/strata/.venv/bin/python"
PORT="${PORT:-8024}"
OUT="$REPO/logs/results/strata-tuning-ab.jsonl"
DIAG="$REPO/logs/results/strata-tuning-diag.jsonl"
# Engine logs have to outlive the run. The 2026-10-28 sweep wrote them under /tmp and a tmpfiles
# cleanup ate them before the numbers could be re-checked, so the acceptance figures had to be
# rebuilt from the JSONL instead. They land in the repo now.
TMP="$REPO/logs/results/strata-tune-$(date +%Y%m%d-%H%M%S)"
PP_SIZES="${PP_SIZES:-2048,8192,32768}"
TG_PROBES="${TG_PROBES:-4}"
TG_WARMUP="${TG_WARMUP:-6}"          # Strata is still climbing at the third measurement probe with 3
                                      # warm-ups; see tg_steady_mean / tg_steady_spread in the ab rows
TG_TOKENS="${TG_TOKENS:-256}"
DEEP_SIZES="${DEEP_SIZES:-}"           # e.g. 98304: decode throughput with a deep context live, which is
                                      # where an elastic K/V gives back the expert slots it borrowed
LOAD_WAIT_STEPS="${LOAD_WAIT_STEPS:-600}"     # 600 * 2 s = 20 min ceiling (huge-pages arm)
mkdir -p "$TMP" "$(dirname "$OUT")"

if [ ! -x "$ENGINE" ]; then echo "engine binary missing: $ENGINE"; exit 1; fi
if [ ! -f "$SRC_CFG" ]; then echo "run config missing: $SRC_CFG"; exit 1; fi

declare -A ARGS ENV MASK
mask_default="0-11"
# The default ladder is now the Exact-switch ladder measured against the kv-grow baseline. The
# 2026-10-28 switch hunt is still selectable:
#   ARMS="base reserve832 kv_grow ctx64k routeprior15 routeprior_kv draft_en grnormfast spec5 base"
DEFAULT_ARMS="kv_grow kg_attn kg_rows kg_mmvq kg_side kg_wide kg_exact1 kg_expdec kg_adapt kg_set kv_grow"
KG="--kv-grow"

arm_spec() { # label -> fills ARGS/ENV/MASK for one arm
  ARGS=""; ENV=""; MASK="$mask_default"
  case "$1" in
    # "base" is the production config as served, which since 2026-10-28 19:15 already carries --kv-grow.
    # The static (no elastic K/V) control is therefore nokvg: the overlay appends --no-kv-grow and the
    # engine's parser takes the last flag (src/program/generate.cpp:2339-2340).
    base)          ;;
    nokvg)         ARGS="--no-kv-grow" ;;
    reserve832)    ARGS="--vram-reserve-mib 832" ;;
    kv_grow)       ARGS="--kv-grow" ;;
    ctx64k)        ARGS="--max-context 65536" ;;
    routeprior15)  ENV="STRATA_ROUTE_PRIOR=1.5" ;;
    routeprior_kv) ARGS="--kv-grow"; ENV="STRATA_ROUTE_PRIOR=1.5" ;;
    routeprior10)  ENV="STRATA_ROUTE_PRIOR=1.0" ;;
    route_resident) ENV="STRATA_ROUTE_RESIDENT=4" ;;
    draft_en)      ARGS="--mtp-draft-vocab $LEAN/data/draft_vocab_en.bin" ;;
    # eddoursul/Lamina-Qwen3.8-Flash-Next-MTP: a drop-in --mtp runtime dir (same dense.txt layout and
    # dense.bin/experts.bin sizes as Strata's own tools/mtp_rt.py output) whose dense part is trained
    # further on Lamina's output over IQ3_S and UD-Q4_K_XL. Its own draft_vocab.bin (40,525 tokens) is
    # picked up from the dir, so this is a head swap, not a vocab swap. Last --mtp wins
    # (generate.cpp:2478), so the overlay replaces the config's head.
    mtp_lamina)    ARGS="--mtp $V/Strata-data/mtp/rt-lamina" ;;
    spec5)         ARGS="--spec 5" ;;
    specprob)      ARGS="--spec 5"; ENV="STRATA_SPEC_PROB=1" ;;
    no_suffix)     ARGS="--suffix-draft 0" ;;
    lookup)        ARGS="--lookup-chain 5" ;;
    prefill16k)    ARGS="--prefill auto:16384" ;;
    prefill_alias) ARGS="--prefill-head-alias" ;;
    imma)          ENV="STRATA_PROMPT_ATTN_IMMA=1" ;;
    hcq8)          ENV="STRATA_HC_Q8=1" ;;
    ple_mmap)      ARGS="--ple-io mmap" ;;
    aux_ecores)    ARGS="--aux-cpus auto"; MASK="0-15" ;;
    grnormfast)    ENV="STRATA_GR_NORM_FAST=1" ;;
    # The served baseline carries --kv-grow from 2026-10-29, so the Exact-switch arms stack on top of
    # it and their rows compare against the kv_grow control rows, not the 2026-10-28 base rows.
    kg)            ARGS="$KG" ;;
    kg_attn)       ARGS="$KG"; ENV="STRATA_ATTN_DAILY=1" ;;
    kg_rows)       ARGS="$KG"; ENV="STRATA_QSA_ROWS=1" ;;
    kg_mmvq)       ARGS="$KG"; ENV="STRATA_B6_MMVQ_ROWS=1" ;;
    kg_side)       ARGS="$KG"; ENV="STRATA_SIDE_MIXER=2" ;;
    kg_wide)       ARGS="$KG"; ENV="STRATA_DEQUANT_WIDE=1" ;;
    kg_exact1)     ARGS="$KG"; ENV="STRATA_CPU_IQ_ALL_EXACT1=1" ;;
    kg_expdec)     ARGS="$KG"; ENV="STRATA_EXPDEC=1" ;;
    kg_adapt)      ARGS="$KG"; ENV="STRATA_ADAPT_NOWAIT=1 STRATA_ADAPT_LAG=2" ;;
    kg_upmix)      ARGS="$KG"; ENV="STRATA_HC_UPMIX=1" ;;
    kg_set)        ARGS="$KG"; ENV="STRATA_ATTN_DAILY=1 STRATA_QSA_ROWS=1 STRATA_B6_MMVQ_ROWS=1 STRATA_SIDE_MIXER=2 STRATA_DEQUANT_WIDE=1" ;;
    *) return 1 ;;
  esac
}

list_arms() {
  cat <<'TXT'
base          (control, production config as served)
reserve832    --vram-reserve-mib 832
kv_grow       --kv-grow (the served baseline from 2026-10-29)
kg_attn       --kv-grow + STRATA_ATTN_DAILY=1
kg_rows       --kv-grow + STRATA_QSA_ROWS=1
kg_mmvq       --kv-grow + STRATA_B6_MMVQ_ROWS=1
kg_side       --kv-grow + STRATA_SIDE_MIXER=2
kg_wide       --kv-grow + STRATA_DEQUANT_WIDE=1
kg_exact1     --kv-grow + STRATA_CPU_IQ_ALL_EXACT1=1
kg_expdec     --kv-grow + STRATA_EXPDEC=1
kg_adapt      --kv-grow + STRATA_ADAPT_NOWAIT=1 STRATA_ADAPT_LAG=2
kg_upmix      --kv-grow + STRATA_HC_UPMIX=1
kg_set        --kv-grow + the stacked bitwise set (attn rows mmvq side wide)
ctx64k        --max-context 65536
routeprior15  STRATA_ROUTE_PRIOR=1.5
routeprior_kv --kv-grow + STRATA_ROUTE_PRIOR=1.5
routeprior10  STRATA_ROUTE_PRIOR=1.0 (opt-in, not in the default ladder)
route_resident STRATA_ROUTE_RESIDENT=4 (opt-in, not in the default ladder)
draft_en      --mtp-draft-vocab data/draft_vocab_en.bin
mtp_lamina    --mtp vendor/Strata-data/mtp/rt-lamina (Lamina draft head, 40,525-token vocab)
spec5         --spec 5
specprob      --spec 5 + STRATA_SPEC_PROB=1
no_suffix     --suffix-draft 0
lookup        --lookup-chain 5
prefill16k    --prefill auto:16384
prefill_alias --prefill-head-alias
imma          STRATA_PROMPT_ATTN_IMMA=1
hcq8          STRATA_HC_Q8=1
ple_mmap      --ple-io mmap
aux_ecores    --aux-cpus auto, taskset 0-15
grnormfast    STRATA_GR_NORM_FAST=1
TXT
}

if [ "${1:-}" = "--arms" ]; then list_arms; exit 0; fi

build_cfg() { # label -> writes $TMP/<label>.json from the production config plus this arm's args/env
  python3 - "$SRC_CFG" "$LEAN" "$TMP/$1.json" "$TMP/$1.engine.log" "$PORT" "$ARGS" "$ENV" <<'PY'
import json, os, shlex, sys
src, lean, out, log, port, extra, envi = sys.argv[1:8]
c = json.load(open(src))
c["exe"] = os.path.join(lean, "build/strata")
c["cwd"] = lean
c["log"] = log
c["port"] = int(port)

# An arm that re-declares a production flag REPLACES it (--max-context 65536 against the
# served 131072), because a repeated flag is a coin flip about which one the engine keeps.
BOOL = {"--kv-grow", "--prefill-head-alias", "--no-kv-grow"}
def strip(args, flag):
    out, i = [], 0
    while i < len(args):
        if args[i] == flag:
            i += 1 if flag in BOOL else 2
            continue
        out.append(args[i]); i += 1
    return out

args = list(c.get("args") or [])
tok = shlex.split(extra)
i = 0
while i < len(tok):
    f = tok[i]
    args = strip(args, f)
    if f in BOOL:
        args.append(f); i += 1
    else:
        if i + 1 >= len(tok):
            raise SystemExit(f"arm arg {f} has no value")
        args += [f, tok[i + 1]]; i += 2
c["args"] = args

cfg_env = dict(c.get("env") or {})
for kv in shlex.split(envi):
    k, _, v = kv.partition("=")
    cfg_env[k] = v
if cfg_env:
    c["env"] = cfg_env                       # serve/server.py:2867 hands the config's env to the engine

json.dump(c, open(out, "w"), indent=1)
print("   args:", " ".join(args))
if cfg_env:
    print("   env:", " ".join(f"{k}={v}" for k, v in cfg_env.items()))
PY
}

engine_pid() { pgrep -f "$ENGINE --serve" 2>/dev/null | head -n 1; }

io_read_bytes() { # pid
  awk '/^read_bytes:/{print $2}' "/proc/$1/io" 2>/dev/null || echo 0
}

# Pull the engine's own accounting out of its log: this is the half of the result a probe
# row cannot show (slots, hit rate, PCIe share, VRAM at request start, page backing).
emit_diag() { # label engine_log io_delta_bytes tokens id1 id2 args env
  python3 - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$DIAG" <<'PY'
import json, re, sys, time
label, log, io_delta, tokens_s, idfile, id2file, args_s, env_s, out = sys.argv[1:10]
tokens = int(tokens_s or 0)
io_delta = int(io_delta or 0)
try:
    txt = open(log, errors="replace").read()
except OSError:
    txt = ""

def last(pattern, cast=lambda s: s):
    # Cumulative engine counters repeat across a log; the last one is this arm's total.
    found = re.findall(pattern, txt)
    return cast(found[-1]) if found else None

def meminfo(key, kb=True):
    # /proc/meminfo values are in kB, except the HugePages_* rows which count 2 MiB pages.
    try:
        for line in open("/proc/meminfo"):
            if line.startswith(key + ":"):
                v = int(line.split()[1])
                return v // 1024 if kb else v
    except OSError:
        pass
    return None

windows = [int(x) for x in re.findall(r"capturing the \d+-token window \((\d+) MiB of VRAM free\)", txt)]
# Draft acceptance, which no probe row otherwise shows. The engine prints it per request, in two shapes:
# "drafts accepted 75 of 87, 1 checkpoints" on the serve path and
# "drafts: 2 windows, 4 of 6 drafts accepted" on the generate path. Summed over the arm's log.
draft_acc = (sum(int(x) for x in re.findall(r"drafts accepted (\d+) of \d+", txt))
             + sum(int(x) for x in re.findall(r"drafts: \d+ windows, (\d+) of \d+ drafts accepted", txt)))
draft_off = (sum(int(x) for x in re.findall(r"drafts accepted \d+ of (\d+)", txt))
             + sum(int(x) for x in re.findall(r"drafts: \d+ windows, \d+ of (\d+) drafts accepted", txt)))
arena = last(r"expert arena: (.*)")
row = {
  "label": label, "ts": time.strftime("%F %T"),
  "engine_version": last(r"session is up \(engine ([\d.]+)\)"),
  "loaded_gib": last(r"loaded ([\d.]+) GiB at", float),
  "load_rate_gibps": last(r"loaded [\d.]+ GiB at ([\d.]+) GiB/s", float),
  "expert_slots": last(r"expert cache (\d+) slots"),
  "cache_vram_gib": last(r"expert cache \d+ slots, ([\d.]+) GiB of VRAM", float),
  "auto_slots": last(r"expert cache auto: .*? -> (\d+) slots"),
  "profile_pairs": last(r"(\d+) ranked pairs"),
  "prefill_borrow_slots": last(r"the prompt path borrows (\d+) CUDA0 cache slots"),
  "hit_rate_pct": last(r"decode expert cache hit rate: ([\d.]+)%", float),
  "hits": last(r"hit rate: [\d.]+% \((\d+) hits"),
  "lookups": last(r"hit rate: [\d.]+% \(\d+ hits / (\d+) lookups"),
  "pcie_pct": last(r"\(([\d.]+)% of all \d+ routed\)", float),
  "free_vram_mib_start": last(r"only (\d+) MiB of VRAM free at the start"),
  "vram_free_all_loaded_mib": last(r"(\d+) MiB of VRAM free with everything loaded"),
  "warning_2065": "below 512 MiB" in txt,
  "pool_workers": last(r"(\d+) pool workers on logical processors"),
  "pool_placement": (last(r"pool workers on logical processors ([0-9,]+)") or "").rstrip(","),
  "arena_note": arena,
  "draft_vocab_tokens": last(r"draft head over (\d+) tokens"),
  "drafts_accepted": draft_acc,
  "drafts_offered": draft_off,
  "draft_accept_pct": round(100.0 * draft_acc / draft_off, 1) if draft_off else None,
  "window_captures": len(windows),
  "min_capture_free_mib": min(windows) if windows else None,
  "hugepages_total": meminfo("HugePages_Total", kb=False),
  "hugepages_free": meminfo("HugePages_Free", kb=False),
  "mem_available_mib": meminfo("MemAvailable"),
  "io_read_mb": round(io_delta / 1048576.0, 1),
  "gen_tokens": tokens,
  "io_mb_per_token": round(io_delta / 1048576.0 / tokens, 4) if tokens else None,
  # Does an elastic expert cache hand slots back as the context grows? The engine prints its slot
  # count whenever it changes, so every value it printed during this arm is kept.
  "slot_values_seen": sorted({int(x) for x in re.findall(r"expert cache (\d+) slots", txt)}),
  # --kv-grow prints its give-back as it happens: "K/V grown to 114688 cells (1.17 GiB); the expert
  # cache gave 634 slots for it ... (2642 of 3276 slots hold experts)". Those pairs are the trade.
  "kv_grown_cells": [int(x) for x in re.findall(r"K/V grown to (\d+) cells", txt)],
  "expert_slots_held": [int(x) for x in re.findall(r"\((\d+) of \d+ slots hold experts\)", txt)],
  "kv_notes": list(dict.fromkeys(re.findall(r"[^\n]*(?:KV|k8v4)[^\n]*MiB[^\n]*", txt)))[:4],
  # Most STRATA_* switches are silent when they take effect, so a null result cannot be told apart from
  # a switch that never armed. What the engine did say about a switch is kept verbatim; the ones that do
  # announce ("strata ab: GATE=... active=1") show up here.
  "arm_args": args_s, "arm_env": env_s,
  "activation_lines": list(dict.fromkeys(re.findall(r"strata ab: [^\n]*", txt)))[:8],
}
def merge(path, keys):
    # Gate columns are counts, not verdicts: the reference arm differs from itself too, so a switch is
    # judged against the control's divergence instead of against zero.
    try:
        d = json.load(open(path))
        for a, b in keys:
            row[a] = d.get(b)
        return True
    except (OSError, ValueError):
        return False

merge(idfile, [("identity_sha", "identity_sha"), ("identity_diff_n", "identity_diff_n"),
               ("identity_prompts", "identity_prompts"), ("identity_diffs", "identity_diffs")])
if merge(id2file, [("intra_diff_n", "identity_diff_n"), ("intra_sha", "identity_sha")]):
    pass
with open(out, "a") as f:
    f.write(json.dumps(row) + "\n")
show = {k: v for k, v in row.items() if k not in ("ts", "arena_note")}
print("   diag:", json.dumps(show))
if arena:
    print("   arena:", arena)
PY
}

run_arm() { # label
  local label="$1" cfg="$TMP/$1.json" log="$TMP/$1.log" elog="$TMP/$1.engine.log"
  arm_spec "$label" || { echo "== $label: unknown arm (see --arms)"; return 1; }
  echo "== arm: $label  args: ${ARGS:-none}  env: ${ENV:-none}  taskset: $MASK"
  build_cfg "$label" || return 1

  local srv="" pid="" io0=0 io1=0 tok=0
  # env goes in twice on purpose: the run config carries it (that is how a winner gets
  # served) and the shell exports it, so an arm still means the same thing if server.py
  # ever stops forwarding cfg["env"].
  ( cd "$LEAN" && env $ENV taskset -c "$MASK" "$PY" "$LEAN/serve/server.py" --engine strata \
      --config "$cfg" --port "$PORT" ) > "$log" 2>&1 &
  srv=$!

  local ok=""
  for _ in $(seq 1 "$LOAD_WAIT_STEPS"); do
    curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -qi '"status": *"ok"\|ok' && { ok=1; break; }
    kill -0 "$srv" 2>/dev/null || { echo "   server exited"; break; }
    sleep 2
  done
  if [ -z "$ok" ]; then
    echo "   FAILED to load:"; tail -6 "$log"; tail -6 "$elog" 2>/dev/null
    kill "$srv" 2>/dev/null; pkill -f "$ENGINE --serve" 2>/dev/null
    sleep 5; return 1
  fi

  # Identity gate first, twice, before the speed probes churn the expert cache: the load state is the
  # closest thing this engine has to a reproducible starting point.
  local idref="$TMP/reference.identity.json" id1="$TMP/$label.identity.json" id2="$TMP/$label.identity2.json" idargs=()
  [ -f "$idref" ] && idargs=(--compare "$idref")
  python3 "$REPO/bench-models/strata-identity-probe.py" --url "http://127.0.0.1:$PORT" \
      --label "$label" --out "$id1" "${idargs[@]}"
  [ -f "$idref" ] || cp "$id1" "$idref"
  python3 "$REPO/bench-models/strata-identity-probe.py" --url "http://127.0.0.1:$PORT" \
      --label "$label-repeat" --out "$id2" --compare "$id1"

  pid="$(engine_pid)"
  [ -n "$pid" ] && io0="$(io_read_bytes "$pid")"
  echo "   up (engine pid ${pid:-?}), VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

  python3 "$REPO/bench-models/bench-qwen38-refresh-probe.py" --url "http://127.0.0.1:$PORT" \
    --label "$label" --out "$OUT" --warmups "$TG_WARMUP" --tg-probes "$TG_PROBES" --tg-tokens "$TG_TOKENS" --pp-sizes "$PP_SIZES" \
    ${DEEP_SIZES:+--deep-sizes "$DEEP_SIZES"}

  if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
    io1="$(io_read_bytes "$pid")"
    tok=0
    if [ -f "$OUT" ]; then
    tok="$(python3 - "$OUT" "$label" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("label") == sys.argv[2]]
tot = 0
for p in (rows[-1].get("tg_probes") or []):
    tot += int(p.get("n") or 0)
print(tot)
PY
)" || tok=0
    fi
    emit_diag "$label" "$elog" "$((io1 - io0))" "$tok" "$id1" "$id2" "$ARGS" "$ENV"
  else
    echo "   engine pid gone before the io delta could be read (no diag row)"
  fi

  kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null
  pkill -f "$ENGINE --serve" 2>/dev/null
  pkill -f "serve/server.py --engine strata" 2>/dev/null
  sleep 5
}

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "arms this run would use: ${ARMS:-$DEFAULT_ARMS}"
  for a in ${ARMS:-$DEFAULT_ARMS}; do
    arm_spec "$a" && echo "  $a: args='${ARGS}' env='${ENV}' taskset='${MASK}'" || echo "  $a: UNKNOWN"
  done
  exit 0
fi

main() {
if systemctl --user is-active --quiet llama-swap.service && [ "${FORCE:-0}" != "1" ]; then
  cat <<'TXT'
llama-swap.service is active. This sweep measures one loaded model at a time; the router
would swap the tier out mid-run and every row would be noise. Stop it first:

  systemctl --user stop llama-swap.service
  bench-models/bench-strata-tuning-ab.sh
  systemctl --user start llama-swap.service

FORCE=1 runs anyway.
TXT
  exit 1
fi

bench_preflight_warn
bench_emit_frontmatter | tee "$TMP/frontmatter.yaml"

ARMS="${ARMS:-$DEFAULT_ARMS}"
for arm in $ARMS; do
  run_arm "$arm" || true
done
echo "done. rows in $OUT and $DIAG"
}

# Sourced (by a test or another script) this gives the arm table and the helpers without
# touching a GPU; run directly and it sweeps.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then main; fi
