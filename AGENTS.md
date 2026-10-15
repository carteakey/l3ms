# AGENTS

Agent and contributor guide for `L3MS`.

## Purpose

Build a keyboard-first, script-first homelab model stack with strong operational
ergonomics.

Scope: this repository contains three independently operated inference lanes:

- Language: the existing root L3MS Rust package, llama-swap service, vision-capable
  LLMs, and embeddings.
- Media: `toolkits/l2m2` (Local Latent Media Machine), including image, audio,
  speech, music, and video workflows.
- Decisions: `toolkits/l3ds` (Local Low-latency Decision System), including its local
  and hosted decision runners.

The first import preserves the existing L3MS package and runtime behavior. The
root guide governs repository-level work; keep each toolkit's setup, runtime,
and task-specific instructions in its local `AGENTS.md`. Do not add media or
decision runtime dependencies to the root L3MS package as part of source import.

## Engineering Rules

- Prefer deterministic script orchestration over hidden automation.
- Serving is declared in `llama-swap.yaml`; benching is declared in
  `bench-models/*.sh`. Both are editable text, not hard-parameterized UI
  forms. When serving and benching flags drift, update them together.
- Standardize bench environment: all bench runs log system frontmatter
  via `bench-models/bench-env.sh` so performance numbers are contextualized
  against CPU governor, RAM speed, available memory, and thermals.
- Treat keyboard control as first-class; mouse workflows are optional.
- Preserve existing files by default for downloads unless explicitly overridden.
- Keep version snapshots for both model configs and scripts before writes.

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

- plat `qwen38-flash-next-plat` (alias `strata`): Strata v0.1.44.1 with ISTA-DASLab
  GSQ-RCO IQ3_XXS quant, dynamic VRAM expert caching (2,444–3,086 hot experts in VRAM),
  native MTP speculative decoding (Q2_0 draft head), K8V4 KV cache @ 128k ctx (`131072`),
  StrataGP 2 MB Transparent Huge Page (`MADV_HUGEPAGE`) kernel acceleration, 4-slot / 3 GiB
  host-RAM conversation parking, and `ttl: 3600` (1-hour idle timeout). **59–65 t/s steady decode on real code, 90.2 t/s peak, pp=1,138–2,013 t/s**,
  11.5 GB VRAM, 47 GB RAM (14 GB headroom).
- plat-vision `qwen38-flash-next-plat-vision` (alias `strata-vision`): Strata v0.1.44.1
  with MTMD vision support (`strata-vision` CPU encoder with `mmproj-F16.gguf`), `--vision` position table
  allocation, and `--fit-max-tokens`. Zero VRAM overhead for vision; retains all GPU experts and K8V4 KV cache.
  512px warmup patch slashes cold boot from 8m to 1.8s. 53–60 t/s decode @ 128k ctx.
- plat-abliterated-vision `qwen38-flash-next-plat-abliterated-vision` (alias `abliterated-vision`):
  Strata v0.1.44.1 uncensored/unaligned SC117 weights with CPU-offloaded `strata-vision` image encoder,
  4-slot / 3 GiB host-RAM conversation cache, 128k context, and `ttl: 3600`. Primary multimodal daily driver for Pi.
- gold `qwen38-flash-next`: plain upstream master (`vendor/llama.cpp-master`,
  macro `qwen38_master_server`, commit `1c4729414`, tag `b11241`). 20.5–22.1 t/s steady state
  decode, pp 180–385 t/s @ 96k ctx (`-c 98304`). Natively includes PR #29393 (RMS_NORM+SCALE CUDA fusion),
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
- Bench hygiene: tg has a warm-up transient (ngram-mod pool) — discard the
  first 2-3 fresh-load probes and report steady state; pp probes need unique
  prefixes to defeat KV reuse. A/B harness:
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
snapshot and candidates: `docs/decision-models.md`. The decision toolkit itself is
L3DS (`~/repos/l3ds`, moving to `toolkits/l3ds/`).

## Structure

- `src/app.rs`: Ratatui layout, keybindings, workflows, and process supervision
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
