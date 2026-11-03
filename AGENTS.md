# AGENTS

Agent and contributor guide for `L3MS`.

## Purpose

Build a keyboard-first, script-first homelab model stack with strong operational
ergonomics.

Scope: one product, three lanes, served by a few shared engines. The former `l2m2`
(media) and `l3ds` (decisions) toolkits were folded into this tree on 2026-10-15;
their old guides are kept in `docs/migration/history/`.

- Language: the Rust workbench (`src/`), llama-swap, llama.cpp, vision-capable
  LLMs and embeddings. `llama-swap.yaml` is the serving source of truth.
- Media: `media-runtimes.json`, `media-models/`, `workflows/` and `maintenance/`.
  Images and LTX/YuE2 run on ComfyUI (`comfyui_auto`), speech and music on audio.cpp
  (`audiocpp_auto`), both through llama-swap. Setup guides are in `docs/` (start at
  `docs/media-models.md`).
- ComfyUI through the router, measured 2026-10-15/17 (details:
  `docs/migration/media-through-llama-swap.md`):
  - Cold start is **~246 s** (`--check` and any client must budget the boot, not a request).
  - Requesting `/comfyui/` evicts the resident language/vision model — including the one this
    agent runs on. A batch is **one blocking call**: no LLM request until it returns, or the
    router swaps ComfyUI off the card mid-batch (`NVRM: Out of memory`).
  - `TextEncodeQwenImage21` autogrow reference slots are addressed by their **dotted finalized
    id** (`"images.image_1"`); bare `image_1` is an unexpected kwarg and a nested dict arrives
    empty (`_io.py:1183/2022` + `execution.py:295`).
  - ComfyUI's input dir is `vendor/ComfyUI/input/`, not `models/input/`.
  - `systemctl is-active llama-swap.service` needs `--user` — without it a running unit reads
    as `inactive`.
- Decisions: `decision-runtimes.json`, `decision-models/` and `suites/`. The
  preferred local models run on llama.cpp through llama-swap; the Python runner
  serves the CPU and hosted profiles (`docs/decision-toolkit.md`).

Direct scripts are a last resort (today only Blender headless). Prefer a shared
engine route over a new executor; see `docs/migration/runtime-policy.md`.

## Engineering Rules

- Prefer deterministic script orchestration over hidden automation.
- **Strata Anti-Bloat Policy**: Never blindly rebase or "re-slice" `carteakey/strata-lean` (`strata-lean`) onto upstream. Upstream commit counts (e.g. 2,252 commits to v0.2.7) are dominated by non-Ada backends (Metal, SYCL, ROCm), experimental engines, and test suites. The fork's mandate is lean, zero-bloat Ada `sm_89` on RTX 4070.Headroom hierarchy: 1. Runtime switches (e.g. `--kv-grow`); 2. Cherry-picking isolated CUDA kernel optimizations; 3. Re-slicing is rejected.
- Serving is declared in `llama-swap.yaml`; benching is declared in
  `bench-models/*.sh`. Both are editable text, not hard-parameterized UI
  forms. When serving and benching flags drift, update them together.
- Standardize bench environment: all bench runs log system frontmatter
  via `bench-models/bench-env.sh` so performance numbers are contextualized
  against CPU governor, RAM speed, available memory, and thermals.
- Treat keyboard control as first-class; mouse workflows are optional.
- Preserve existing files by default for downloads unless explicitly overridden.
- Adult (NSFW) prompt text, per-render verdicts and experiment write-ups live in the
  sibling `~/repos/l2m2-nsfw` checkout, which depends on this tree through `L2M2_ROOT`.
  Keep explicit prompt strings and adult grading notes out of this repository.
- Keep version snapshots for both model configs and scripts before writes.
- Publishing & Blog Conventions: all public posts, mirrored blog documentation (`docs/*post*.md`), and changelogs must follow the author's writing conventions and the Human Writer skill in `~/repos/carteakey.dev/.agents/skills/human-writer/SKILL.md`:
  - Zero em dashes (`—`): use commas, parentheses, or spaced hyphens (` - `).
  - First-person voice (`I`, `my`), not corporate plural (`"We evaluated"`).
  - No synthetic framing, dramatic one-line closers, or AI marketing hype (`"Not X but Y"`, `"The physics of X"`, `"game-changer"`, `"blazing"`).
  - Natural contractions (`it's`, `didn't`, `doesn't`).
  - Report ground-truth empirical numbers with system context; never publish single-burst anomalies (e.g. the 90.2 t/s burst) as sustained throughput.

## llama-swap Operations

- Config hot-reload: `kill -HUP $(pgrep '[l]lama-swap')`. New/edited
  `llama-swap.yaml` entries are picked up without restarting the router
  (verified: no dropped connections). A reload still evicts the resident model and
  runs the config's startup preload (2026-10-15: the abliterated tier was replaced
  by `qwen38-flash-next-plat`), so reload only when a model swap is acceptable.
- Full-VRAM experiments (bench ladders, OOM hunting, fitting runs): pause the
  router and its spawned servers with `systemctl --user stop llama-swap.service`,
  run the experiment, then `systemctl --user start llama-swap.service`. A 12 GB
  card cannot hold two loaded models; llama-swap models claim the whole card.
- Requesting a model through the router swaps out whatever is loaded (server
  for the previous model is killed, new one spawned with its declared flags).
  `globalTTL: 600` unloads idle models after 10 minutes.
- New model entries: snapshot the yaml first
  (`llama-swap.yaml.bak-YYYYmmdd-HHMMSS`), add a `${...}_server` macro for any
  non-default binary under `vendor/`, then SIGHUP and smoke-test with a
  chat request before considering it wired. Reasoning models need
  `max_tokens >= 64` in the smoke request: smaller budgets land entirely in
  `<think>` and return empty `content`, which is not a failure (check
  `reasoning_content` instead).
- Flag generation matters when porting cmd blocks between entries:
  master-based builds take `--lazy-mode on`; pre-rename PR builds (e.g. the
  pr-test-27836 era) take `--tensor-read-lazy on`. Verify with
  `llama-server --help` before reusing a flag.

## Qwen3.8-Flash-Next (qwen4exp) Tiers

Deep detail, decisions and next steps: `docs/qwen38-flash-next-internal.md` and `docs/strata-evaluation-internal.md`.

Seven router tiers:

- plat `qwen38-flash-next-plat` (alias `strata`): Strata strata-lean (`carteakey/strata-lean`,
  branch `strata-lean`, cut from upstream `v0.1.45`; upstream is at `v0.2.10`, audit it with
  `.agents/skills/strata-cherry-pick/scripts/audit_upstream.sh`, analysis in
  `docs/strata-perf-next-steps-2026-10.md` and `docs/strata-lean.md`) with ISTA-DASLab GSQ-RCO IQ3_XXS quant,
  unquantized Q8_0 PLE table on NVMe, dynamic VRAM expert caching (2,583 hot expert pairs in
  4.24 GiB at the 128k config; 2,444 to 3,086 across configs) with **`--kv-grow` served since 2026-10-28
  19:15** (elastic K/V: 16,384 of 131,072 cells mapped at load, 0.21 of 1.32 GiB; the expert cache takes
  the room first, so 3,276 slots / 5.35 GiB at short context, measured 2,642 slots holding experts at
  114,688 cells in a live session), native MTP speculative decoding
  (Q2_0 draft head over 106,299 draft tokens, 178 MiB), fused hyper-connections (down GEMV
  inject folding and gate/mean fusion reducing kernel launches from 6 to 4), fused QSA RMSNorm + RoPE
  (`STRATA_QSA_FUSE_NR`, cuts memory traffic from 1.1 GB to 0.4 GB a layer with assembly barrier exactness),
  strict mathematical routing (`STRATA_ROUTE_TAIL_SKIP=0` default in `strata-lean`, rejecting upstream's
  rank >= 7 expert dropping to prevent KL 0.003-0.006 drift), thinking wrap-up
  on repeat loops inside <think>, K8V4 KV cache @ 128k ctx (`131072`), expert arena pinned with
  `cudaHostRegister` on **4 KB pages** (`vm.nr_hugepages=0` so `MAP_HUGETLB` fails; upstream
  stopped requesting `MADV_HUGEPAGE` while THP is `always`, #771 and #2126 - the lever is a
  reserved hugetlb pool, not `MADV_HUGEPAGE`), 4-slot / 3 GiB host-RAM conversation parking,
  and `ttl: 3600` (1-hour idle timeout).
  **69.60 t/s steady decode on real code (65.4 to 76.1 range; a same-day repeat measured
  67.88, so run-to-run spread is about 2 t/s), pp 1,876 to 1,891 t/s @ 8k**, 11.7 GB VRAM,
  47 GB RAM (14 GB headroom). The engine reports 410 MiB of VRAM free at request start, under
  its own 512 MiB floor (#2065, it asks for `--vram-reserve-mib 832`), and a 58.6% decode
  expert-cache hit rate: those two lines are where the remaining headroom is.
  `STRATA_GR_NORM_FAST` is opt-in on CUDA (`fused_gr.cu:1179`, upstream's comment: "measured on sm_70
  only") and the tier's run config has no `env` block, so it is NOT on; an earlier revision of this file
  listed it as a tier feature, and measuring it on 2026-10-28 settled it: 64.85 against a 71.13 base pair
  mean, i.e. 6.3 t/s **slower** on Ada.
  First tuning sweep (`bench-models/bench-strata-tuning-ab.sh`, rows in `logs/results/strata-tuning-ab.jsonl`,
  telemetry in `logs/results/strata-tuning-diag.jsonl`): paired control 70.70 and 71.55; **`STRATA_ROUTE_PRIOR=1.5`
  measured 102.23 t/s** with hit rate 93.0% (from 57.4/57.7) and PCIe 1.3% (from 9.9%) at unchanged slots and
  acceptance, **measured but not served** - it re-picks experts, upstream's 0.2.7 note flags agent-task
  completion at 1.5 (4.5 of 10 at 1.5 against 5.6 off), and this agent's work is agent tasks; `--kv-grow`
  73.53 with 3,277 slots (promoted into serving 2026-10-28 19:15); `--vram-reserve-mib 832` 68.75 for +134
  MiB of headroom. Rejected: `--spec 5` (61.00) and `--mtp-draft-vocab data/draft_vocab_en.bin` (65.78).
  Second sweep 17:05 to 17:36, 11 arms stacked on `--kv-grow`, run dark with
  `bench-models/run-strata-tuning-dark.sh`: **nothing promoted**. The paired control itself read 77.07 and
  69.43 (80.60 / 70.55 on last-two probes), so `kg_exact1` 81.70 and `kg_side` 79.80 are undecided rather
  than fast; the only signal that survived is prefill - `STRATA_DEQUANT_WIDE=1` at 1,978 t/s @ 8k and 1,992
  @ 32k against the controls' 1,938/1,950 and 1,932/1,933. Table and method: `docs/strata-perf-next-steps-
  2026-10.md` §8.
  All non-Ada backends (Metal, SYCL, ROCm tuning headers) and unreleased Qwen3.5 engine code stripped.
- plat-vision `qwen38-flash-next-plat-vision` (alias `strata-vision`): Strata v0.1.44.1
  with MTMD vision support (`strata-vision` CPU encoder with `mmproj-F16.gguf`), `--vision` position table
  allocation, and `--fit-max-tokens`. Zero VRAM overhead for vision; retains all GPU experts and K8V4 KV cache.
  512px warmup patch slashes cold boot from 8m to 1.8s. 53–60 t/s decode @ 128k ctx. Serves `--kv-grow`
  since 2026-10-28 19:15: expert cache 2,581 slots / 4.23 GiB became 3,276 / 5.35 GiB, the whole 128k K/V
  is only 1.32 GiB, and the cache hands slots back as a context grows (floor 128 slots,
  `src/program/generate.cpp:7350-7372`); roughly neutral above ~110k cells.
- plat-abliterated-vision `qwen38-flash-next-plat-abliterated-vision` (alias `abliterated-vision`):
  Strata v0.1.44.1 uncensored/unaligned SC117 weights with CPU-offloaded `strata-vision` image encoder,
  4-slot / 3 GiB host-RAM conversation cache, 128k context, and `ttl: 3600`. Primary multimodal daily driver for Pi.
- gold `qwen38-flash-next`: plain upstream master (`vendor/llama.cpp-master`,
  macro `qwen38_master_server`, commit `d1044bc82`, tag `b12073-15-gd1044bc82`). 19.95-22.1 t/s steady state
  decode, pp 592.6 t/s @ 8k ctx. Natively includes PR #29393 (RMS_NORM+SCALE CUDA fusion),
  PR #28770 (CUDA sparse Flash Attention), and kernel fusions (#28896, #28901).
- vision `qwen38-flash-next-vision`: master with `mmproj-F16.gguf`
  vision projector (`models/unsloth/Qwen3.8-Flash-Next-GGUF/mmproj-F16.gguf`),
  `-ngl 99 -ncmoe 45` @ 16k ctx, 18.2–18.6 t/s, 10.4 GB VRAM (1.8 GB headroom).
- MTP `qwen38-flash-next-mtp`: Unified QSA Sparsity Stack V2 (#28770 [merged upstream b11062] + #28699 + #28213 + #29166 multi-seq fix
  + #29599 NVMe madvise prefetch + #28671 TOP_K radix select + #28569 tensor scheduler)
  + Daniel Han PR #28243 with compact `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` (1.78 GB),
  `vendor/llama.cpp-exp-stack-v2/build/bin/llama-server`, macro `qwen38_mtp_server`,
  `-ngl 99 -ncmoe 46 --spec-draft-n-max 2 --spec-draft-p-min 0.7` @ 16k ctx,
  **25.32–27.06 t/s aggregate** (97–100% acceptance, 2.76–2.93 tok/step), 11,025 MiB VRAM.
- exp `qwen38-flash-next-exp`: Unified QSA Sparsity Stack V2 (#28770 + #28699 + #28213 + #29166 + #29599 + #28671)
  (`build/bin/llama-server` inside `vendor/llama.cpp-exp-stack-v2`, macro `qwen38_exp_server`).
  20.52–20.81 t/s plain decode, cold prefill 135.2 t/s (3x boost from Aman Gupta's PR #29599 `madvise` row prefetching),
  9,953 MiB VRAM. Same flags as gold for A/B.

Note: Legacy AtomicChat `AD-4.27bpw` quant (89 GB, 33 shards) is archived to NAS (`/mnt/storage/models/qwen38-flash-next/AD-4.27bpw-Q4_K_M-M64/`).
Active pipeline uses ISTA-DASLab GSQ-RCO quants (`IQ3_XXS`, `IQ3_S`, and `Coder IQ1_M` with shared byte-identical PLE Shard 2).

MTP upgrade status (2026-09-29): Unified QSA Sparsity Stack V2 on master b11242
with Daniel Han PR #28243, compact 1.78 GB `shared-Q4_K_M` head, and `-ncmoe 46`
reaches **25.32–27.06 t/s** (up to 31% speedup over previous 20.65 t/s),
achieving 97–100% draft acceptance while fitting safely in 11,025 MiB VRAM.

Gotchas:

- The exp clone's build dir serves two branches: `llama-server` = MTP binary,
  `llama-server-exp` = exp binary. Switching branches there requires
  rebuilding both.
- Vendor clones can carry uncommitted local patches — check `git status`
  before trusting or rebuilding a clone (the MTP detached-head patch once
  lived only as a dirty file).
- PR stacks: `maintenance/llama-test-pr.sh <pr>...` merges PR heads onto
  master and builds with house flags; `vendor/llama.cpp` is a stale bisect
  checkout, not a master build — use `vendor/llama.cpp-master` for master.
- Memory rule: the PLE/n-gram table (~30-38 GB) must stay on SSD. AtomicChat
  quants do it natively (separate shard + lazy reads); unsloth quants need
  `-ot "per_layer_token_embd\.weight=CPU"` with mmap (never `--no-mmap` or
  `--mlock`, which force the table into RAM). Non-PLE weights must fit fast
  memory (~72 GB here: 60 RAM + 12 VRAM) — UD-Q4_K_XL does not fit; IQ4_XS
  fits but is quality-lateral to AD-4.27bpw (KLD 0.0836 vs 0.0842). Preflight
  big downloads by range-fetching GGUF shard headers and parsing the
  PLE/non-PLE split before committing disk.
- Bench hygiene: tg has a warm-up transient (ngram-mod pool, and draft acceptance climbs with it) — discard
  the first 2-3 fresh-load probes and report steady state; pp probes need unique
  prefixes to defeat KV reuse. 2026-10-28 proof of how bad this gets: with 3 warm-ups a `--kv-grow` arm read
  70.0, 76.1, 85.1 t/s with acceptance 0.680, 0.715, 0.828 while its identical twin 30 minutes later read
  67.2, 68.7, 72.4, a 10 t/s phantom win. Use `TG_WARMUP=6` / `TG_PROBES=4` and read `tg_steady_mean` and
  `tg_steady_spread` (last three probes) off the row instead of `tg_mean`.
- CPU power state is part of a bench row. Until 2026-10-29 14:35:35 this box ran
  `scaling_governor=powersave` on all 16 logical CPUs (intel_pstate active default, `min_perf_pct=17`, EPP
  already `performance`), power-profiles-daemon writes EPP only, and a governor choice is not persistent, so
  the 2026-08-31 `governor=performance` reading was lost at a reboot and **every t/s row recorded here,
  sweeps included, is a powersave row**. Route 1 is now applied: `cpupower.service` is enabled and writes
  governor+EPP `performance` from `/etc/default/cpupower-service.conf` at every boot, and a 4-core busy loop
  measured 4.50 GHz from the first sample with no ramp, which is the bench state even though `min_perf_pct`
  stays 17 (that floor only sets how deep HWP may idle). `maintenance/cpu-power-state.sh check` is the gate,
  `bench-models/run-strata-tuning-dark.sh` refuses with `rc=3` unless `FORCE_CPU=1` (not the harness's
  `FORCE=1`), and `docs/bench-runbook.md` carries the routes (route 2 is `tuned-cachy`/`tuned-cachy-ppd` plus
  the repo's `l3ms-bench` profile, the only one that pins the floor at 100). Re-run the paired control in the
  new state before quoting any row taken before 14:35:35. Write access is delegated through a root-owned
  helper (`sudo install -m 0755 -o root -g root maintenance/l3ms-cpu-power-helper.sh
  /usr/local/bin/l3ms-cpu-power` plus `/etc/sudoers.d/l3ms-cpu-power`, four fixed verbs), never through
  `systemctl *` or a repo path; until it is installed the script prints the sudo line and exits 4.
- Output reproducibility: greedy output on this engine is **not reproducible**. Two boots of the same config
  differed on 4 of 7 fixed greedy prompts, two passes inside one boot of the served tier differed on 3 of 7,
  always at the same token count, because which of the cached expert pairs are resident changes between
  passes and the cached and streamed paths are not bitwise the same arithmetic. `bench-models/
  strata-identity-probe.py` (twice per arm, before the speed probes) therefore yields counts, not a verdict:
  compare `identity_diff_n` with the control pair's own cross-boot divergence and use `intra_diff_n` as each
  arm's noise floor. Upstream's "Exact = identity gate passed" rule cannot be met by any arm here, mine
  included. A full-VRAM ladder can be run detached with `bench-models/run-strata-tuning-dark.sh`, which does
  the grace delay, the unit stop, the ladder, the unit start and the engine `/health` confirmation, and
  writes `logs/strata-dark-<stamp>.log` plus a `.status` line. A/B harness:
  `bench-models/bench-llama-qwen38-flash-next-build-ab.sh` (stop llama-swap
  first). Standardize system context: all bench scripts must capture the
  standard system frontmatter via `bench-models/bench-env.sh` (RAM MT/s,
  topology, CPU governor/EPP, PPD profile, free/available RAM, active anon,
  ZRAM usage, baseline VRAM, THP, CPU/GPU temperatures). Never compare runs
  across differing power governors (`powersave` drops decode ~15–25% vs
  `performance`) or unverified RAM clock states. Runners (`run-llama-bench.sh`,
  `run-ik-llama-bench.sh`, `run-llama-fit-bench.sh`) write YAML frontmatter to
  log headers and embed `"sys"` in `logs/results/<model>.jsonl`.
- Memory-spill check for SSD-offloaded models: with the model loaded, delta
  `/proc/<llama-server-pid>/io read_bytes` across a ~256-token generation.
  Expected ≈ PLE design traffic only (~5 KB/token for qwen4exp n-gram rows);
  tens of MB/token means hot expert pages are re-faulting from SSD (decode
  drops ~35%+). Seen live on a full box (free ≈ 1 GiB + 20 GiB zram): the
  54.5 GB fast-memory model thrashes when the desktop stack competes for
  RAM. Note zram swap metrics are useless for this check (in-RAM, churns
  wildly); read_bytes is the ground truth. Fix: free RAM before long
  sessions. `--fit-target` is MiB of VRAM to leave FREE, so a LOWER value
  packs more weights onto the GPU; only relevant if VRAM has real slack
  (at 64k/12 GB it does not — VRAM is already ~11.3/12.28 GiB). Also: most
  of a mid-session "spill" is actually cold page-in of the ~46 GB expert
  pool — reads converge (8.6 GB → 0.5 GB per 256-token gen) over ~5-10
  generations after load; run 2-3 throwaway generations before judging.
- Headless mode (reclaim desktop RAM for the model): `llama-swap.service` is
  a `systemctl --user` unit, so the user manager must survive session teardown
  — `loginctl enable-linger kchauhan` is set (Linger=yes). Then
  `sudo systemctl isolate multi-user.target` drops the GUI (recover with
  `sudo systemctl isolate graphical.target`); SSH sessions are unaffected.
  Even headless, the AD-4.27bpw model (~45.5 GB RAM-side weights) + live
  anon + zram compressed stores leave the box only marginally above its
  working set — re-verify with the read_bytes spill check; a fresh boot
  (empty zram, no GUI apps) is the clean-state reset.
- `target/` can contain artifacts built in another tree (deploy snapshots
  carry it) — test binaries then bake a foreign `CARGO_MANIFEST_DIR` and
  repo-detection tests fail with `left: None`. Fix: `cargo clean -p l3ms`.

## Decision Models

Native decision models (typed choice/noul/score probabilities, no generation) run on
`vendor/llama.cpp-d3` (macro `d3_server`) as `vllm-d3-flash` and
`cloudflare-clef-flash`, called via `POST /upstream/<id>/v1/systemone` (llama-swap does
not route `/v1/systemone` directly). They need `-c = -b = -ub` (whole prompt in one
ubatch). Before downloading, verify `<arch>.decision.type` with
`maintenance/gguf-decision-check.py`: most community GGUFs of leaderboard models
are plain LLM conversions. Onboard with `/add-decision-model`. Latency, leaderboard
snapshot and candidates: `docs/decision-models.md`. The decision code (the former L3DS repo) lives in
`decision-models/`, `decision-runtimes.json` and `suites/`.

## Structure

- `src/app.rs`: TUI state, keybindings, views, workflows, and process supervision
- `src/ui/`: workbench shell (frame layout, sidebar, toolbar, status bar and its
  thresholds); draws plain data handed over by `app.rs`
- `src/cli.rs`: launcher + interactive CLI (`--run`, `--bench`, `--list`)
- `src/param_registry.rs`: curated `llama-server` flag table (control kinds,
  categories, house presets) for the Param Builder tab
- `src/param_builder.rs`: Param Builder state — flag import/export
  (quote-aware), cmd-block generation, and `llama-swap.yaml` cmd replacement
- `src/llama_swap.rs`: authenticated llama-swap HTTP boundary
- `src/config_store.rs`: download config CRUD + validation + snapshots
- `src/script_store.rs`: script CRUD + snapshots
- `l3ms/` and `l3ms.py`: legacy Python TUI retained during parity work
- `model_downloader/`: Python downloader compatibility boundary
  (`media_models_config.json` lists the media downloads)
- `media-runtimes.json`, `media-models/`, `workflows/`, `models/*-files.txt`,
  `prompts/`, `runtime-locks/`: the media lane (former `l2m2`)
- `decision-runtimes.json`, `decision-models/`, `suites/`, `requirements/`: the
  decision lane (former `l3ds`)
- `audiocpp-server.json`: audio.cpp models served as `audiocpp_auto`
- `.venv` (downloader), `.venv-comfy*`, `.venv-decisions`, `.venv-hosted`, `vendor/`,
  `.toolkit/`: ignored per-host runtime state; clone or rebuild, never commit
- `maintenance/codemode/`: pi `codemode` scripts — Jev classifier triage for bench
  results and serving logs (`/jev-bench`, `/jev-log`)

## Classifier Triage (Jev)

Typed-judgment triage runs through pi's `codemode` tool with TypeSafe
`typesafe/jev-latest` (credential in `~/.pi/agent/auth.json`, `codemode` enabled
via `defaultTools: ["+codemode"]`). `/jev-bench` judges whether a bench row is
steady state, strong enough to quote, comparable to other runs, and spill-free;
`/jev-log` classifies a serving failure and picks the llama-swap action (hot
reload, router restart, RAM reclaim, flag fix). Code gathers evidence and applies
thresholds; Jev only answers the questions. Details, verified numbers, and limits:
`docs/pi-jev-classifier-workflow.md`. Treat a low `next_action` confidence as
"gather more evidence", not "the answer is wrong".

## Versioning

Use semantic versioning (`MAJOR.MINOR.PATCH`):

- `MAJOR`: breaking workflow or API changes
- `MINOR`: new features (tabs, actions, commands)
- `PATCH`: bug fixes, UX polish, non-breaking improvements

Update `CHANGELOG.md` on every user-visible change.

## Release Intent

The Rust port is active under `CAR-97` and starts at version `0.7.0`.
Retain the legacy Python TUI and downloader until their Rust replacement or
compatibility boundary has explicit parity coverage.

## Planning Authority

Linear is authoritative for committed L3MS work. The repository and Markdown
remain authoritative for code and canonical documentation. The single Linear
project is `L3MS` with milestone `Operations backlog completion — 2026-08-10`.
The reconciled TODO groups are tracked by `CAR-140` through `CAR-146`, while
`CAR-97` remains the parent Rust-port history and is attached to the same
project/milestone. `TODO.md` is reserved for speculative or research notes
that have not been intentionally committed.
