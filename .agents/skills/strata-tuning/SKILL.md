---
name: strata-tuning
description: Measure and promote Strata serving switches on the RTX 4070 plat tier without rebuilding anything. Use when a flag, STRATA_* env var, taskset mask, profile or draft-vocab file needs an A/B verdict, when a bench row has to be judged (steady state, spill-free, comparable), when a winner gets promoted into vendor/strata-lean/strata-iq3_xxs-ple_q8.json and llama-swap.yaml, and whenever an agent has to benchmark the very model it is being served by.
---

# Strata tuning and measurement workflow

Companion to `strata-cherry-pick` (which owns re-slices, cherry-picks and rebuilds).
This skill owns the part of the loop that needs **no compile**: flags, env switches, CPU
masks, expert profiles and draft vocab files. Evidence and ranked backlog:
`docs/strata-perf-next-steps-2026-10.md`.

## When to use which lane

| Situation | Lane |
| :--- | :--- |
| A `STRATA_*` env var, a `--flag`, a taskset mask, a data file | this skill, `bench-models/bench-strata-tuning-ab.sh` |
| Code I wrote myself (kernel fusion, backend prune) | `strata-cherry-pick`, `build_lean.sh`, then `bench_strata.sh` (MODE=code) |
| Upstream moved and I want what it contains | `strata-cherry-pick` Step 1b re-slice, then a `strata_v027` arm in `bench-models/bench-qwen38-refresh-ab.sh` |

## The harness

```bash
./bench-models/bench-strata-tuning-ab.sh --arms          # print the ladder, touch nothing
DRY_RUN=1 ./bench-models/bench-strata-tuning-ab.sh       # expand every arm to args/env/taskset
bash bench-models/bench-strata-tuning-ab.sh              # default ladder (the kg_* Exact-switch arms)
ARMS="base kv_grow routeprior15 grnormfast base" bash bench-models/bench-strata-tuning-ab.sh
TG_WARMUP=6 TG_PROBES=4 DEEP_SIZES=98304 ARMS="base kv_grow kv_grow base" \
  bash bench-models/bench-strata-tuning-ab.sh            # deep-context decode: the kv-grow regime
MODE=tune .agents/skills/strata-cherry-pick/scripts/bench_strata.sh   # same run, with GPU handoff

# the run that measures the model this agent answers from, detached, with restart and health check
setsid nohup bench-models/run-strata-tuning-dark.sh > logs/strata-dark-launch.log 2>&1 &
tail -f logs/strata-dark-launch.log      # ends with a logs/strata-dark-<stamp>.status line, rc=0
```

Arms are **overlays on the production run config**
(`vendor/strata-lean/strata-iq3_xxs-ple_q8.json`), never a fresh cmd block. That is the whole
design: an arm is the served thing plus one change, so a winner becomes one config line.
`build_cfg` writes the override into the config's `args` and, for env arms, into its `env`
block, which is how a winner is promotable without rewriting anything else.

Three places the evidence lands, deliberately separate:

- `logs/results/strata-tuning-ab.jsonl`: probe rows (decode probes with `tg_steady_mean` and
  `tg_steady_spread`, pp sizes, `deep` when `DEEP_SIZES` is set), comparable across arms and engines.
- `logs/results/strata-tuning-diag.jsonl`: engine self-report per arm, plus the gate columns
  (`identity_sha`, `identity_diff_n`, `identity_diffs`, `intra_diff_n`) and the elastic-K/V columns
  (`slot_values_seen`, `kv_grown_cells`, `expert_slots_held`).
- `logs/results/strata-tune-<stamp>/`: everything raw and per arm, engine log, bench log, probe JSON and
  both identity passes. Keep it: an earlier sweep's engine logs lived in `/tmp` and a tmp cleanup destroyed
  the acceptance evidence, which is why acceptance had to be re-derived from the ab rows.

## The three patterns worth keeping

1. **Overlay, do not rebuild.** Every arm inherits the served config. If an arm needs a new
   cmd block, it is no longer a tuning arm and it cannot be promoted by editing one line.
2. **The engine's own words are a first-class measurement.** Speed alone is not evidence here.
   A row is read together with `expert cache N slots`, `decode expert cache hit rate`,
   `N more read by the GPU over PCIe`, `serve: N MiB of VRAM free with everything loaded`,
   the `#2065` warning, `window capture` free-MiB values, `pool workers on logical processors`,
   the arena line that says how the expert arena is page-backed, and `mtp: draft head over N
   tokens`. `emit_diag` pulls those out of the engine log so the comparison is not vibes.
3. **Diag before believe, and in the right regime.** Delta `/proc/<pid>/io read_bytes` across the
   arm's generation is the ground truth (AGENTS.md says the same). Steady state on this model is about
   5 KB/token of PLE traffic and tens of MB/token means hot expert pages are re-faulting from SSD. But a
   bench arm generates ~1,024 tokens right after boot, which is inside the cold page-in of the expert
   pool, not past it: every arm of the 2026-10-28 sweep landed between 0.0557 and 0.0998 MB/token. So
   judge io/token **relative across arms in the same boot** (the 2026-10-28 route-prior arm at 0.0775 sat
   inside its own controls' 0.0788 and 0.0557, which is what made the win believable), and only compare
   against the 5 KB band after 5-10 throwaway generations.

## Judging a row

- **Pair it.** `base` runs first and last in the ladder. Run-to-run spread on this tier is about
  2 t/s (69.60 then 67.88 on the same build and probe; the 2026-10-28 ladder's own pair read 70.70 and
  71.55), so anything under ~3 t/s is noise unless the paired control moved with it.
- **Engine rates are per window, not per run.** `emit_diag` reads the last decode window the engine
  printed. On 2026-10-28 the two control arms reported 57.4% and 57.7% hit there while their mid-run
  windows read 69-71%. Quote hit rate as "last window in the arm", and only compare it between arms
  measured the same way.
- **Steady state only, and check that you got it.** Discard the first 2 to 3 probes of a fresh load
  (ngram-mod pool warm-up) and report the rest; never quote a single burst as sustained throughput. With 3
  warm-ups this lied to me: on 2026-10-28 a `--kv-grow` arm read 70.0, 76.1, 85.1 t/s with draft acceptance
  0.680, 0.715, 0.828 while its identical twin 30 minutes later read 67.2, 68.7, 72.4 with 0.681, 0.681,
  0.760, a 10.05 t/s phantom win with `mem_available_mib` flat at 12.8 to 14.3 GiB the whole ladder. Read
  `tg_steady_mean` / `tg_steady_spread` off the row (last three probes) rather than `tg_mean`, run
  `TG_WARMUP=6 TG_PROBES=4`, and treat a row whose spread is over ~3 t/s as not yet steady.
- **The identity gate is comparative, not absolute.** Greedy output on this engine is not reproducible: two
  boots of the same config diverged on 4 of 7 fixed prompts and two passes in one boot on 3 of 7, at
  identical token counts, because which cached expert pairs are resident moves between passes and the cached
  and streamed paths are not bitwise the same arithmetic. `identity_diff_n` means something only next to the
  control pair's own divergence, with `intra_diff_n` (second pass against first, same boot) as the arm's
  noise floor. Run the probe twice per arm and **before** the speed probes, since the speed probes are what
  churn the cache.
- **pp needs fresh text.** The probe script varies the prompt prefix; a repeated prompt measures
  KV reuse, not prefill.
- **Compare like for like.** Same quant, same probe script, same boot. Never compare across power
  governors (`powersave` costs 15 to 25% decode against `performance`) and never against an
  unverified RAM clock state. Measured 2026-10-29 up to 14:35:35: this box was in `powersave` on all 16
  logical CPUs with `intel_pstate min_perf_pct=17` (active-mode kernel default; power-profiles-daemon running
  with `Profile=performance` but writing EPP only, tuned not installed), so **every row in
  `logs/results/strata-tuning-ab.jsonl` predates the fix and is a powersave row.** Route 1 is applied from
  14:35:35: `cpupower.service` enabled, governor and EPP `performance`, and a 4-core busy loop measured
  4.50 GHz from the first sample with no ramp even though `min_perf_pct` stays 17 (that floor is idle depth,
  not a loaded-core cap). `maintenance/cpu-power-state.sh show` prints the layering, `check` is the gate
  (modes `apply-cpupower --yes`, `install-tuned`, `apply-tuned --yes --ppd`, all needing root; once
  `/usr/local/bin/l3ms-cpu-power` is installed with its sudoers grant the script prints
  `sudo /usr/local/bin/l3ms-cpu-power apply` and exits 4 instead). Re-run the
  paired control in the new state before quoting any row taken before 14:35:35.
- **Quality gate before promoting** any arm that changes what gets routed or which tokens the
  draft head can propose (`STRATA_ROUTE_PRIOR`, a custom `--expert-profile`, a custom
  `--mtp-draft-vocab`): read a real Pi session's output before it goes into the served config.
  Cap `STRATA_ROUTE_PRIOR` at 1.5.

## Promoting a winner into serving

```bash
ts=$(date +%Y%m%d-%H%M%S)                       # house snapshot convention
cp vendor/strata-lean/strata-iq3_xxs-ple_q8.json "vendor/strata-lean/strata-iq3_xxs-ple_q8.json.bak-$ts"
cp llama-swap.yaml "llama-swap.yaml.bak-$ts"
# add the arg to the config's "args", or the switch to its "env" block (env: null means add it)
kill -HUP $(pgrep '[l]lama-swap')              # hot reload; the reload still swaps the resident model
curl -s -H "Authorization: Bearer $LLAMA_SWAP_API_KEY" http://127.0.0.1:8080/v1/chat/completions \
  -d '{"model":"qwen38-flash-next-plat","max_tokens":64,"messages":[{"role":"user","content":"say ok"}]}'
```

A reasoning model asked for fewer than 64 tokens answers with empty `content` and everything in
`reasoning_content`. That is not a failure; check `reasoning_content` before calling the tier down.
Then write the measured row into `docs/strata-perf-next-steps-2026-10.md`, `AGENTS.md` (tier line)
and `CHANGELOG.md`, and only then close the Linear ticket.

## Benching the model you are running on

This is the awkward part of tuning a stack that serves the agent doing the tuning. Verified facts:

- `PI_PROVIDER=llama-swap` and `PI_MODEL=qwen38-flash-next-plat-vision` in my own environment, so
  stopping `llama-swap.service` removes the server that generates my next token.
- The card holds 12,282 MiB and the resident Strata tier takes 11.5 to 11.7 GiB of it. A second
  Strata engine cannot boot beside it, which is why the harness refuses to run while
  `llama-swap.service` is active (`FORCE=1` overrides, and a mid-sweep swap invalidates every row).
- Idle VRAM is about 44 MiB. That is the "the card is actually free" signal, per the `gpu-handoff`
  skill; stopping the unit can leave the engine child alive holding the card.

So the safe shape is:

1. Say what is about to happen and how long it will take (about 2.5 to 5 minutes per arm: a boot is 60 to
   90 s, the two identity passes 60 s, the probe set the rest), because the agent goes dark while the card is
   busy. Nothing else should be asked for during the run: a request to the router mid-sweep either fails (no
   server) or evicts the arm.
2. Use `bench-models/run-strata-tuning-dark.sh` rather than an ad hoc `sleep 30; ...` line. It first
   runs `maintenance/cpu-power-state.sh check` and refuses with `rc=3` (`cpu-gate=failed` in the status
   line) unless `FORCE_CPU=1`, so a ladder cannot spend 20 dark minutes measuring a power state no
   other row was taken in; `FORCE=1` is the harness's separate "run even with the router up" override,
   and the two are different variables on purpose. It takes a
   `GRACE` delay (default 75 s) so the current turn finishes generating before the router stops, stops the
   user unit, kills any engine child still holding the card, waits for VRAM to read under 300 MiB, runs the
   ladder, starts the unit again, and confirms the restart by polling the **engine's own** `/health` on the
   port it was just given (discover it from `/proc/<server.py pid>/cmdline`, `--port`), because
   `systemctl --user is-active` says `active` a second after the unit starts while the tier is still loading.
   It writes `logs/strata-dark-<stamp>.log` and a one-line `.status` (`rc=0` means ladder clean and engine
   answering; the wrapper adds 10 to `rc` if health never answered).
3. Verify `systemctl --user is-active llama-swap.service` reads `active`, a `Health check passed` line
   in `journalctl --user -u llama-swap.service`, and VRAM back at about 11.7 GiB.
4. Read `logs/results/strata-tuning-ab.jsonl` and `logs/results/strata-tuning-diag.jsonl` after
   the card is back. Rows are appended per arm, so a sweep that dies halfway still leaves usable
   data.

## Cost accounting

Each arm is one engine boot (39.97 GiB loaded at about 0.64 GiB/s, so 60 to 90 s) plus the probe
set. Budget 3 to 5 minutes per arm. The CPU-only prep (draft vocab subset, expert profile ranking)
is cheap; collecting expert-profile traces is not, because `tools/collect_profile_traces.py` boots
the engine per prompt (20 to 60 s start plus about 10 s decode, 45 prompts is about 30 minutes of
card time).
