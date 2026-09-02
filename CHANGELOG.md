# Changelog

## [Unreleased]

### Added
- Winamp-classic skin for the Rust TUI (0.8.0): new `src/theme.rs` palette module — playlist green (#00FF00) body text on black, dim-green borders/titles, classic dark-blue (#0000C6) selection bars with white bold rows, amber accents/warnings, red errors, and an inverted LED badge style (black-on-green) for the header logo, footer status bar, and focused download fields. All inline `Color::Cyan`/`Yellow`/`DarkGray` styling in `src/app.rs` (34 sites, 27 blocks) rewired through the theme helpers; chat user lines render amber and assistant lines playlist green.

### Changed
- Model inventory reconciliation (2026-08-31 storage cleanup follow-up), via `codex-skills/model-inventory-manager`: 11 serving entries referencing deleted files commented out of `llama-swap.yaml` (gpt-oss-120b ×3, qwen3-coder-next ×3, nomic-embed, qwen3-6-mtp + vision, gemma-4-26b-qat-mtp-vision, sarvam-30b; pre-edit snapshot `llama-swap.yaml.bak-20260831-<ts>`); 7 orphan files deleted (~24 GB — Ridge 3.7bpw 13 GB superseded by IQ3_XXS, 4 spare qwen38 MTP heads, unsloth Qwen3.8 MTP head dir, gemma QAT Q8_0 head). Audit script fixed: model store path fell back from the dead `/mnt/lab/models` mount to `/home/kchauhan/models`, sharded-GGUF false orphans (a served `-00001-of-N` shard now covers its set), `--spec-draft-model` path extraction, and media models (MiniMax/HeartMuLa/LTX) excluded per the skill's preservation rules. Router restarted clean with 7 active models (`/v1/models` verified).
- Dashboard (l3ms.carteakey.dev) restored to regenerable state and history preserved: `docs/dashboard-meta.json` was stale (generator hard-failed — missing qwen3.8 family + qwen3-8-27b, still listing the retired gemma-4-12b trio). Active set now matches the 7 served entries exactly (added qwen38-flash-next gold/exp/MTP with the internal-doc measurement ledger as evidence fields, and qwen3-8-27b-iq3-xxs @ 262k); 11 retired profiles moved to the `archived` section **with their last measured tps preserved** (gemma-12b 120.8, QCN 39.6, gpt-oss-120b 28.5, Qwen3.6 Q4 60.3, Gemma vision 39.0, Sarvam 15.0); system block corrected to 64 GB DDR5 @ 5000 MT/s; `index.html` archive card now renders tps/quant/profiles with an explicit "archived numbers are not reproducible" caveat; generator + tests pass, `generated-models.js` regenerated.
- Bench docs split (was a 1260-line `docs/bench-runbook.md`, now three docs with verified no-loss migration — all 939 non-blank source lines accounted for, 934 verbatim, 5 justified rewordings: two "section above" → explicit doc reference after the move, three headings re-leveled under the new structure; a stray unclosed ` ``` ` fence artifact at EOF was dropped):
  - `docs/bench-runbook.md` (639 lines): operational core only — overview, prerequisites, system-state check, CPU power layering, script layout, experiment sequence, reading results, known constraints, adding a new model, generic quick reference.
  - `docs/bench-results.md` (new, 521 lines): per-model measured results (Qwen3-Coder-Next, Gemma-4-26B-A4B stub, Gemma/Qwen3.6 MTP, Qwen3.6-35B, Qwen3.5-122B, gpt-oss-120b), per-model quickstarts, winner cheatsheet, KV cache quant recommendation. This is where `/add-bench-result` writes now.
  - `docs/bench-troubleshooting.md` (new, 152 lines): tg-variability root cause, archived 22-item checklist, common troubleshooting table, GGML CUDA memory pools (VMM growth, fit headroom, mid-session OOM fixes).
  - Cross-references updated in: `llama-swap.yaml` model descriptions (§8 → `docs/bench-results.md`; pre-edit snapshot `llama-swap.yaml.bak-20260831-203159`), `.claude/commands/{add-bench-result,new-model-config,model-status,bench-model,optimize-model}.md`, `docs/model-onboarding-playbook.md`, `docs/qwen38-flash-next-internal.md`. `.bak-*` snapshots and `thread.md` transcript left untouched (historical record).
- Bench runbook: added a "CPU power layering" section (docs/bench-runbook.md) disentangling governor vs EPP vs HWP vs profile managers — on `intel_pstate` the governor name is mostly a label and EPP is the knob that matters (`powersave` + EPP=performance is a valid high-perf setup); Tuned is not a CachyOS default (PPD is; tuned-ppd is the shim bridging them) and its `throughput-performance` profile is functionally equivalent to the manual setup; I/O scheduler noted as an unrelated block layer. Recorded current box state (2026-08-31): PPD 0.30 active at `performance` profile, tuned/tuned-ppd not installed (tuned-ppd swap reverted), governor/EPP=performance verified live; added a status-update note to the historical root-cause section.

### Fixed
- llama-swap v247 API alignment (src/llama_swap.rs): model state now parses the nested `status.value` field from `/v1/models` (top-level `state`/`status` strings still honored for older routers) — states no longer show "unknown"; model load switched from the nonexistent `POST /models/load` to `GET /upstream/{model}/load` (verified live: triggers the on-demand startup path and holds until health-check); per-model unload moved to `POST /api/models/unload` with the existing `{"model": id}` body (verified live: `DELETE /upstream/{model}` does not evict a still-starting process, the API endpoint does).

- Model Browser auto-scan: entering the Browser tab (F4, palette, or tab cycling) now scans the GGUF path automatically when it has never been scanned or changed since the last scan (mirrors the Chat tab auto-connect); manual `r` rescan unchanged. Previously the table sat empty until a manual scan.

## [0.7.0] - 2026-08-31

The Rust port remains in progress under `CAR-97`; these entries describe
implemented slices and do not assert full legacy parity or a fully green
verification matrix.

### Added
- CPU power profile switched to `performance` on the serving box and documented (`docs/qwen38-flash-next-internal.md` §5/§9.4): ~+3% on the ngram-mod spec-warm path (25.7-26.3 vs 24.8-25.4 t/s — ramp penalty between speculative verify rounds), neutral on sustained prose (19.5 t/s floor unchanged, memory-latency bound). Reboot checklist gained a governor-verification step; internal doc also records that a single cold speculation-pool reading is pool state, not a regression.
- Three-tier qwen3.8-flash-next layout (gold = master, exp = stack, MTP = exp+MTP), all smoke-tested through the router (content "OK", reasoning intact): `qwen38-flash-next` now runs plain upstream master (`vendor/llama.cpp-master`, 85c55223c — dedicated clone; the shared `vendor/llama.cpp` is a stale bisect checkout and was left untouched); new `qwen38-flash-next-exp` entry runs the exp stack (e1748dbd5) as `build/bin/llama-server-exp`; `qwen38-flash-next-mtp` rebuilt as exp+MTP (0b7d6d57d) by cleanly merging #27836 onto the stack and finally committing the previously-uncommitted 57-line detached-head patch from the pr-test-27836 working tree (saved at /tmp and applied; that clone is now obsolete). Flag rename applied everywhere on master-based builds (`--tensor-read-lazy` → `--lazy-mode on`); the A/B bench harness updated to prod=master vs stack=exp. Maintenance note: the stack clone's `build/bin/llama-server` is the MTP binary and `build/bin/llama-server-exp` the exp binary — branch switches in that clone require rebuilding both.
- Unsloth-quant evaluation for Qwen3.8-Flash-Next, concluded without adopting (bench-models/bench-llama-qwen38-flash-next-build-ab.sh repurposed as a binary A/B harness): remote-parsed UD-IQ4_XS and UD-Q4_K_XL shard headers (PLE = 29.8 GiB at ~4.7 bpw in both; non-PLE weights 62.5 GiB / 74.7 GiB) and cross-checked KLD tables — UD-IQ4_XS (0.0836) is a lateral move vs AtomicChat AD-4.27bpw (0.0842), and the only quality step up (UD-Q4_K_XL, 0.0469) needs ~78 GiB fast memory vs ~72 GB available (60 RAM + 12 VRAM), leaving zero headroom for long-conversation KV growth, so expert paging would degrade decode (the exact Reddit failure mode). The `-ot "per_layer_token_embd\.weight=CPU"` + mmap technique (replicates AtomicChat's separate-shard PLE offload on unsloth quants; mmap is load-bearing) is documented here for a future bigger-RAM box. Dangling router entry removed; no unsloth download retained.
- PR-stack test build for Qwen3.8-Flash-Next (`vendor/llama.cpp-pr-test-28023-28068-27941`, built with `maintenance/llama-test-pr.sh` convention): current master (qwen4exp is upstream now; TENSOR_READ_LAZY landed via #27794 so production flags are safe) + #28023 (indexer heads summed by slices, approved) + #28068 (GDN l2norm max→rsqrt correctness fix across delta-net models) + #27941 (kvu NaN-collapse + indexer restore fixes; test-file conflict vs master's newer test suite resolved by taking master's side — test file is not in our build targets). To be A/B'd against the production pr-test-27742 binary via `bench-models/bench-llama-qwen38-flash-next-build-ab.sh`; #27977 (tg slowdown as ctx grows) and #27992 (O(log n) n-gram lookups) deferred to a second stack.
- Binary A/B run (`bench-llama-qwen38-flash-next-build-ab.sh`, AtomicChat AD-4.27bpw, fit64, 64k, q8 KV, ngram-mod spec): production pr-test-27742 (tg 25.0/25.4, hot pp 198) vs master+stack e1748dbd5 (tg 25.1/20.2, hot pp 200) — speed is a wash within noise; the stack's value is correctness (#28068 GDN l2norm, #27941 NaN-collapse/restore fixes). Note for future stacks: upstream renamed `--tensor-read-lazy` to `--lazy-mode on|auto|off`, and qwen4exp itself is upstream via the #27742 squash merge (6c84c7d5d), so PR-branch vendoring is only needed for PRs not yet merged. Router restarted after the full-VRAM experiment. Production macro swap pending a long-context tg probe and/or quality sign-off.
- Production swap: `qwen38_server` macro now points at the PR-stack build (`pr-test-28023-28068-27941`, e1748dbd5) and the `qwen38-flash-next` entry's `--tensor-read-lazy on` became `--lazy-mode on` (upstream flag rename; the pr-27836 MTP build keeps the old flag and is untouched — gold still speculates with ngram-mod only, no MTP). tg-variance follow-up on the stack build: fresh-load tg 13.3 → 15.2 → 24.4 → 24.8 → 24.8 t/s — the A/B dips were the ngram-mod pool warm-up transient, not a regression; steady state 24.8 t/s ≈ prod 25.0-25.4, hot pp 200 ≈ 198. SIGHUP + router smoke test passed (content "OK", reasoning trace intact). Rollback: restore `pr-test-27742` macro from `llama-swap.yaml.bak-*`.
- Prefill (pp) investigation for Qwen3.8-Flash-Next (`bench-llama-qwen38-flash-next-pp.sh`): first prompt after model load pays ~7.5 s of expert-page NVMe fault-in (SN770 Gen4, ~6 GB/s); repeat prompts hit KV prefix reuse (prompt_n collapses — bench probe fixed to force full prefill and print prompt_n); true warm full-prefill pp measured at ~200 t/s (fit-on, 64k, THP=always) on a fresh quiet boot. THP=always + `defrag=defer+madvise` kept (neutral-to-positive). Also added `maintenance/gguf_tensor_types.py` (pure-python GGUF tensor-type dump) after finding the AtomicChat file's routed experts are actually IQ2_XS + Q2_K_S.
- Attribution 2×2 for Qwen3.8-Flash-Next (`bench-llama-qwen38-flash-next-graphopt.sh`): `GGML_CUDA_GRAPH_OPT=1` is a no-op on current master (graphs already reused without it); `--fit on --fit-target 512` gives +2.8% tg over `-ncmoe 46` at 64k — base router entry switched to fit-on. Warm ngram-mod pool measured at 18.9 → 35.9 t/s (+90%) on repeated identical prompts; novel-content decode ~20 t/s.
- Benchmarked Qwen3.8-Flash-Next MTP via llama.cpp PR #27836 (+ detached-head patch): sidecar Q4_K_M draft head (requantized from Q8_0) fits 12 GB VRAM at 32k ctx. Ungated dn=2 is net-negative (prose −12..−22%); with `--spec-draft-p-min 0.7` gating code tg +15..26% at prose parity, acceptance 0.81–0.97. Wired as a separate `qwen38-flash-next-mtp` router entry (32k ctx cap) plus `bench-models/bench-llama-qwen38-flash-next-mtp.sh` A/B harness.
- Benchmarked Gemma 4 26B QAT (UD-Q4_K_XL) with external MTP drafter: tg 69.1 t/s (vs 45.03 no-MTP fit, ~1.5x). Placement derived via llama-fit-params (fit with external draft loops in 571d0d5). Logged in `bench-models/logs/results/gemma-4-26b-qat-mtp.jsonl`; runbook §8 updated.
- Benchmarked Qwen3.6-35B-A3B UD-Q6_K with self-drafted MTP (`--spec-type draft-mtp --spec-draft-p-min 0.75 --spec-draft-n-max 2`): tg 51.3 t/s (vs 26.92 no-MTP fit, ~1.9x). Logged in `bench-models/logs/results/qwen3-6-mtp-q6.jsonl`; runbook §8 updated.
- Added a bounded benchmark result browser with JSONL/Markdown parsing,
  deterministic sorting, and `--compare-results` metric diffs.
- Added persisted operator settings, atomic non-secret profile import/export,
  and a bounded Chat system-prompt library with a TUI picker.
- Added paged model inventories, Hugging Face progress/ETA feedback, optional
  shellcheck-on-save diagnostics, serving/bench flag drift checks, and a safe
  explicit action to stop a freshly detected external llama-server.
- Added disk-free and network byte counters to runtime telemetry.
- Added `maintenance/run-l3ms-kpc.sh` to launch the deployed Rust binary with
  the existing KPC llama-swap service and downloader environment.
- Added the initial Rust `0.7.0` application: a Clap launcher, authenticated
  llama-swap client, typed configuration/script stores, and seven-view Ratatui
  workbench.
- Added atomic, collision-safe configuration and script snapshots with strict
  validation, traversal protection, runtime-root-aware storage, and Unix mode
  preservation.
- Added supervised bench, download, and maintenance processes with bounded
  output, in-session jobs, and process-group shutdown.
- Added streamed chat with bounded SSE parsing, system/temperature/token and
  thinking controls, token-rate feedback, and authenticated requests.
- Added a typed command registry, executable `Ctrl+P` palette, and contextual
  help generated from the same command metadata.
- Added live process-tree CPU/RAM telemetry with optional NVIDIA memory
  reporting.
- Added tested standalone modules for bounded GGUF v2/v3 metadata parsing and
  legacy-compatible atomic job/chat-session persistence.
- Added reusable bench, maintenance, and download editor state with validation,
  dirty tracking, safe selection, and snapshot save/reload/restore operations.
- Added inline UTF-8 bench and maintenance script editors with persistent
  viewports, dirty-change guards, snapshot browsing, and explicit save/reload
  controls.
- Added the current keyboard-driven Download editor surface: config-path and
  runtime controls, model CRUD and all typed fields, validation, atomic save,
  snapshot restore, dedicated logs, disk-space feedback, and supervised
  selected/enabled launches.
- Added a schema-versioned Hugging Face dry-run estimator and a strict Rust
  preflight boundary with bounded output, timeout/cancellation, child cleanup,
  cache-aware byte totals, and platform-aware target disk probing.
- Added a reusable Unicode-safe text buffer with character-boundary editing,
  terminal-cell cursor positioning, and vertical/page navigation.
- Added a pinned Rust toolchain, locked dependencies, unit coverage, strict
  Clippy checks, and GitHub Actions verification.
- Added an on-demand `nomic-embed-text-v1.5` llama-swap profile with
  authenticated `/v1/embeddings` routing and a five-minute idle unload TTL.
- Added an enabled targeted downloader profile and embedding deployment checks.
- Added Rust Chat endpoint editing, authenticated Connect/Detect, independent
  model selection, request-ID guarded streaming, and responsive cancellation.
- Added declarative media-generation profiles and an argv-safe `--media` CLI
  for MiniMax-H3 and HeartMuLa through audio.cpp plus the LTX-2.5
  DistilledPipeline.
- Added Yeti-focused media setup and readiness checks, including the audio.cpp
  MiniMax-H3 Q4_K and HeartMuLa Q8_0 installers plus the gated LTX-2.5
  bootstrap.
- Added explicit still-image conditioning to the LTX-2.5 media profile while
  keeping H3's audio.cpp input surface text-first and shell-safe.
- Documented LTX-2.5 quantization compatibility: BF16 plus runtime FP8 for
  the Ada Yeti target, rather than the ComfyUI-only INT8 or Blackwell-only
  NVFP4 files.
- Added prompt-file inputs and playable MP4 muxing for H3 video output when
  the host provides ffmpeg, while retaining the raw RGB24 artifact for
  inspection and recovery.
- Added a local HeartMuLa music profile through audio.cpp using the published
  Q8_0 GGUF package, with prompt/tag, lyrics-file, instrumental, duration, and
  memory-saver controls.
- Added an explicit LTX-2.5 transformer checkpoint override so a future
  loader-compatible pre-quantized artifact can be selected without changing
  the wrapper.

- Added a reproducibility evidence record to every published local profile,
  including explicit historical gaps for PP, tested context, cache state,
  draft acceptance, llama.cpp commit, and benchmark command.
- Added a validated community-run schema, submission guide, generator tests,
  and a separate unranked dashboard view for non-local hardware.
- Added keyboard and palette-backed Bench script filtering with stable editor
  and run selection mapping plus an explicit no-results state.

### Changed
- Kept community runs structurally separate from the local RTX 4070 ranking
  and updated the dashboard methodology to make that boundary visible.
- `qwen38-flash-next` router entry now caps thinking by default: `--reasoning-effort medium --reasoning-budget 4000 --reasoning-preserve` (Qwen3.8 ships xhigh thinking; the cap trades a small quality delta for large effective-throughput gains in agentic use).
- Made the Rust launcher use llama-swap for `--run` and `--list run`, while
  preserving `bench-*.sh` files as the benchmark entry points and retaining the
  Python downloader as a compatibility child process.
- Made media prompts, lyrics, and output paths travel through a shell-free
  process boundary; media runtime scripts remain the editable upstream-facing
  layer.
- Replaced the hosted MiniMax Music/mmx profile with offline HeartMuLa Q8_0
  generation through the same pinned audio.cpp CUDA runtime used by H3.
- Connected jobs and chat sessions to their legacy-compatible persistent state,
  including stale-job reconciliation, bounded history, safe retry reconstruction,
  and saved-session browsing.
- Replaced the Rust GGUF view's filename-only inventory with bounded metadata
  scanning, recursive/top-level modes, filtering, deterministic sorting, file
  details, and per-file parse warnings.
- Made explicit global download workers override per-model workers, with the
  per-model value taking precedence over the slow preset when no global value
  is supplied.
- Made Rust downloader launches construct argv without a shell and select
  Python from `L3MS_DOWNLOADER_PYTHON`, the platform repository venv, or
  `python3`, in that order. The downloader script now has a portable
  `#!/usr/bin/env python3` shebang for direct CLI use.
- Made Download launches preflight the immutable selected/enabled command off
  the rendering thread, report cached/remaining size against free space, allow
  `Esc` cancellation, and preserve legacy launch behavior when estimation is
  unavailable.
- Made the Download view responsive across wide, compact, and focused-pane
  terminal layouts while keeping the active field and cursor visible.
- Isolated new Download histories with stable path-hashed snapshot namespaces
  while retaining list and restore compatibility with legacy snapshot
  directories.
- Added repeat-action confirmation before dirty Download reload or restore, and
  reported post-operation snapshot-list failures as secondary warnings rather
  than failures of completed load, save, or restore operations.

### Fixed
- Treat llama-swap load and unload HTTP failures as failed operations instead
  of successful jobs.
- Correct selected-download handling for multiple allow/ignore patterns and
  `base_models_dir`.
- Validate Download snapshots before mutation and save the exact displaced
  config bytes as an undo snapshot before atomic restore, preventing invalid or
  non-undoable replacements and the post-restore disk/memory split-brain window.
- Prevent dirty-quit confirmation from being hidden behind the command palette
  and prevent large-script cursor placement from disagreeing with Ratatui's
  rendered scroll window.
- Apply batch download slow/global worker controls when normalized model rows
  contain `max_workers: null`.
- Prevent synchronous disk probing and stale preflight events from blocking or
  mutating the active Download view, and reap canceled estimator descendants on
  Unix even when they retain inherited output pipes.

## [0.6.0] - 2026-07-03

### Added
- Added a Workbench-first llama-swap launcher with live model filtering, load,
  unload, chat, bench, browser, download, and jobs actions on the first screen.
- Added `LLAMA_SWAP_API_KEY` support for authenticated llama-swap model and
  chat requests.

### Changed
- Replaced the static Start hub with the Workbench tab as the default entry
  point.
- Routed Workbench model loads through Model Ops so job history and resource
  telemetry stay consistent.
- Reordered top-level shortcuts around the fast path: F1 Workbench, F2 Ops,
  F3 Chat, F4 Browser, F5 Download, F6 Jobs, F7 Maintenance.
- Simplified Model Ops run mode by hiding bench-only binary, extra-args,
  version, and script-save controls until bench mode is active.

## [0.5.3] - 2026-06-17

### Added
- Added `maintenance/systemd/nanobot-gateway.service` to manage and run the
  `nanobot gateway` command persistently.
- Added `maintenance/setup-nanobot-gateway-service.sh` to install, start,
  stop, restart, enable, disable, and monitor the gateway service.

## [0.5.2] - 2026-06-22

### Added
- Added a machine-testing handoff for n-gram speculation, target/draft KV precision, CUDA graphs, ubatch sizing, prompt-cache reuse, imatrix quants, and future multi-GPU work.

### Changed
- Removed the inactive `LLAMA_SET_ROWS` environment variable from current serving profiles, benchmark scripts, and Windows launchers, then regenerated the dashboard commands.
- Marked its appearance in an older benchmark result as historical context rather than an active recommendation.

## [0.5.1] - 2026-06-21

### Fixed
- Corrected the dashboard test GPU from RTX 4070 Super 12GB to RTX 4070 12GB.

## [0.5.0] - 2026-06-21

### Added
- Expandable served-profile details with portable, copyable `llama-server`
  commands generated from `llama-swap.yaml`.
- Direct links from each profile to its serving config and benchmark source.
- A compact methodology note explaining the difference between profile-level
  throughput and task-level MTP benchmark runs.
- CI validation that rejects stale generated dashboard data.

### Changed
- Rebuilt the public dashboard around generated data instead of a handwritten
  duplicate of the active model catalog.
- Replaced the static online indicator with an honest dated profile snapshot.
- Sorted profile rankings by measured TG and corrected ranks 10–12.
- Replaced the crushed mobile table with stacked model cards and expandable
  command details.
- Clarified benchmark descriptions where headline and task-level measurements
  come from separate runs.

### [2026-06-16]
- **Qwen 3.6 MTP Optimization & Cleanup**:
  - Benchmarked Qwen 3.6 35B MTP configurations on RTX 4070 (`UD-Q4_K_XL` and `UD-Q6_K` variants).
  - Aligned `llama-swap.yaml` and benching scripts with optimal `spec-draft-n-max` parameters (n-max=2 for `UD-Q4_K_XL` yielding **60.3 tok/s**; n-max=2 for `UD-Q6_K` yielding **43.1 tok/s**).
  - Fixed hanging issue in `bench-llama-qwen3-6-reddit-baseline.sh` and `bench-ik-qwen3-6-reddit-test.sh` by adding `-st` (single-turn) flag.
  - Evaluated `nothink` variants (`enable_thinking: false`) and confirmed they generate slightly slower (**58.0 tok/s** on Q4; **40.9 tok/s** on Q6) due to less structured output reducing speculative decoding acceptance rates compared to thinking mode.
  - Safely archived (took version snapshots of) and removed 5 obsolete non-MTP Qwen3.6 bench and run scripts from the codebase.
  - Removed the obsolete `unsloth/Qwen3.6-35B-A3B-GGUF` model download profile from `models_config.json`.

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

### [2026-06-12]
- **Gemma 4 QAT & MTP Integration**:
  - Rebuilt mainline `llama.cpp` using the system GCC 16 compiler to resolve segmentation faults in CUDA template compilation.
  - Added new Gemma 4 26B and 12B QAT & MTP profiles to `llama-swap.yaml` (`gemma-4-26b-qat`, `gemma-4-26b-qat-mtp`, `gemma-4-12b-qat`, `gemma-4-12b-qat-mtp`).
  - Deprecated and removed the older, redundant non-QAT Gemma 4 26B profiles (`gemma-4-26b-a4b`, `gemma-4-26b-a4b-vision`, `gemma-4-26b-mtp`, `gemma-4-26b-mtp-vision`, `gemma-4-26b-a4b-q6-k-xl`, and `gemma-4-26b-mtp-q6`) as the new QAT + MTP configurations fully replace them at double the speed.
  - Created blog posts documenting benchmarks for Gemma 4 26B and 12B under QAT and MTP configurations, showing local speeds up to 100.6 tok/s (26B) and 120.8 tok/s (12B) on an RTX 4070.
- **TPS Leaderboard & Dashboard**:
  - Implemented a premium, glassmorphism-based static HTML dashboard (`docs/index.html`) served as a GitHub Page to showcase currently active/served models, performance ranges across tasks, and archived/deprecated models.

### [2026-05-19]
- **MTP Improvements**: Rebuilt `llama.cpp` with [PR #23269](https://github.com/ggml-org/llama.cpp/pull/23269) for enhanced Multi-Token Prediction performance.
- **Qwen 3.6 35B**: Added Q6_K variant and initialized benchmarking to evaluate performance vs Q4_K_XL.
- **Documentation**: Updated MTP-related posts with latest upstream status.


### Changed
- **Qwen 3.6 MTP Mainline**: Updated `llama-swap.yaml` configuration to use the mainline `llama-server` and the updated `--spec-type draft-mtp` flag since MTP support is now merged into `llama.cpp` master.
- **TUI workbench pass**:
  - Added a persistent command bar with context-specific shortcut hints.
  - Reworked the Start tab into a compact workbench hub grouped around
    Operate / Inventory / Command flows.
  - Upgraded the command palette with a shortcut column and token-based
    filtering.
  - Reduced Run/Model Ops table churn by avoiding column rebuilds on every
    filter refresh.
  - Moved repeated shortcut hint rows into a reusable `ShortcutStrip`.
  - Moved llama-swap model refresh onto a named Textual worker and refresh the
    command bar on direct tab activation.
- **Maintenance updater**: new `maintenance/update-llama-stack.sh` snapshots
  the current llama-swap binary and llama.cpp build metadata, updates
  llama-swap, rebuilds mainline llama.cpp, validates `llama-swap.yaml`, and
  restarts `llama-swap.service` only if it was already active.
- **llama.cpp build script**: dropped the removed `llama-sweep-bench` target
  from the default build target list.
- **Bench result logging**: fixed fitted `-ot "..."` placement strings so
  `bench-models/log-result.sh` records JSONL results instead of tripping over
  shell quotes. Structured results now live under `bench-models/logs/results/`
  alongside the raw bench logs.
- **Codex skill**: added `codex-skills/l3ms-prepost` for repeatable
  before/update/after llama-swap + llama.cpp maintenance checks.
- **Serving architecture switched to llama-swap**:
  - New `llama-swap.yaml` is the single source of truth for every servable
    model (28 previous `run-models/*.sh` scripts collapsed into YAML entries
    + aliases + reasoning-effort variants).
  - New `maintenance/systemd/llama-swap.service` user-level unit runs
    `llama-swap -config llama-swap.yaml -listen :8080`.
  - Startup preload is `gemma-4-26b-a4b-vision` (previously the dedicated
    `gemma-vision.service` default).
  - New `docs/llama-swap-runbook.md` covers install, start/stop, curl,
    and how to add a model.
  - **Breaking**: `run-models/` directory removed. Clients previously hitting
    per-model ports (mostly `:8001`) now hit the single `:8080` endpoint and
    pass the model ID in the OpenAI `model` field. Update the TUI Chat tab's
    base URL to `http://<host>:8080/v1`.
- **TUI Model Ops Run mode now talks to llama-swap**:
  - New `l3ms/llama_swap.py` HTTP client (`list_models`, `load_model`,
    `unload_model`, `probe`). `LLAMA_SWAP_URL` env override supported.
  - Run mode table lists models from `/v1/models` (with state column).
    Start (`Ctrl+R`) calls `POST /models/load`; Stop (`Ctrl+S`) calls
    `POST /models/unload`. Editor becomes a read-only detail pane with
    ready-to-copy curl snippets.
  - Bench mode is unchanged: still globs `bench-models/*.sh` and spawns
    subprocesses.
  - Jobs tab retry: for `run` mode, retries now resolve as model IDs
    (not script paths).
  - `l3ms.py --run`: picks a model from llama-swap and POSTs `/models/load`.
    `l3ms.py --list run`: prints models from `/v1/models`.
- **Installer**: `maintenance/install-llama-swap.sh` fetches the release
  binary into `~/bin/` with OS/arch auto-detection and `FORCE` / version
  pinning. Replaces the copy-paste curl snippet in the runbook.

- **Polish pass on the migration**:
  - Fix `--fit-ctx 32678` → `32768` typo in `gpt-oss-120b-legacy` and
    `gpt-oss-120b-low` (copied verbatim from the original shell scripts).
  - Switch `ik-qwen3-5-122b-thinking-coding` to `${ik_server}` (the original
    ik- shell script used the vanilla binary — required the ik fork for
    `-merge-qkv`).
  - `llama-swap.service` now uses `%h` + env vars (`L3MS_ROOT`,
    `LLAMA_SWAP_BIN`, `LLAMA_SWAP_LISTEN`) so the unit runs unmodified on any
    account; documented drop-in override flow in the runbook.
  - RunPanel in run mode now tracks `loaded_model_id` separately from the
    cursor; Ctrl+S unloads the model that's actually loaded, not whatever
    row the user last clicked. Friendly message when nothing is loaded.
  - Run-mode live resource telemetry restored: `_find_llama_swap_pid` +
    `ps --ppid` aggregate CPU/RAM of llama-swap upstream processes, polled
    every 2s. `nvidia-smi` still feeds the GPU column when available.
  - ChatPanel: replaced the read-only model label with a `Select` populated
    from `/v1/models` on connect/detect; requests use the selected model ID
    instead of the hard-coded `"default"` string (which llama-swap rejects).
  - Jobs-tab retry for run mode now carries the model ID through
    `JobStarted.script_path` so "retry" reloads the right model.
- **`gemma-vision.service` retired**:
  - Unit moved to `maintenance/systemd/archive/gemma-vision.service`.
  - Installer helper moved to
    `maintenance/archive/setup-gemma-vision-service.sh`.
  - `maintenance/archive/README.md` explains the migration + disable steps.
  - `docs/bench-runbook.md` no longer documents that flow.

### Added
- **Qwen3.6-35B-A3B workflow support**:
  - `bench-models/run-llama-cpp-qwen3-6-35b-a3b.sh` direct serve helper for local tuning outside llama-swap
  - `bench-models/run-llama-cpp-qwen3-6-35b-a3b-vision.sh` vision preset wrapper (`mmproj-F16`, 64k ctx, safer fit/batch defaults)
  - `bench-models/bench-llama-cpp-qwen3-6-35b-a3b.sh` baseline bench script (safe all-experts-on-CPU default via `-ot`)
  - `bench-models/bench-llama-cpp-qwen3-6-35b-a3b-strategies.sh` strategy sweep bench script (`all-cpu-moe`, `partial-cpu`, `up-down-cpu`, `up-cpu`)
  - `bench-models/bench-llama-cpp-qwen3-6-35b-a3b-fit.sh` fit-based bench script
  - `llama-swap.yaml` model entries `qwen3-6-35b-a3b` (text) and `qwen3-6-35b-a3b-vision` (multimodal)
  - `model_downloader/models_config.json` Qwen3.6 profile now fetches both `UD-Q5_K_XL` and `mmproj-F16`
  - `docs/bench-runbook.md` quickstart + measured pp/tg results (fit winner on RTX 4070 12 GB), including vision serving flow
  - `docs/qwen3-6-35b-a3b-post.md` draft blog post for text + vision setup and benchmark outcomes
- **Start tab + accessibility navigation pass**:
  - `Start` tab now opens by default and provides guided core actions (Download, Model Ops, Chat, Browser, Maintenance, Jobs) plus direct Help/Palette entry points
  - tab navigation fallback keys: `Alt+1..Alt+7` (direct tab switch) and `Alt+←/Alt+→` (cycle tabs)
  - `l3ms.py --quickstart` prints a no-TUI quick-start guide for first-time users or remote terminals
  - Jobs panel now surfaces history load/save status instead of silently swallowing history file failures
  - key-hint copy across panels is standardized around "core actions + ? full shortcuts" to keep dense layouts but improve scanability

- **Gemma-4-26B-A4B workflow support**:
  - `run-models/run-llama-cpp-gemma-4-26b-a4b.sh` run script targeting mainline `vendor/llama.cpp/build/bin/llama-server`
  - `run-models/run-llama-cpp-gemma-4-26b-a4b-vision.sh` dedicated vision preset wiring `mmproj-BF16.gguf`
  - default contexts now aligned to this local profile: text `128k`, vision `64k`
  - `bench-models/bench-llama-cpp-gemma-4-26b-a4b.sh` baseline bench script
  - `bench-models/bench-llama-cpp-gemma-4-26b-a4b-strategies.sh` strategy sweep bench script
  - `bench-models/bench-llama-cpp-gemma-4-26b-a4b-fit.sh` fit-based bench script
  - `model_downloader/models_config.json` profile for `unsloth/gemma-4-26B-A4B-it-GGUF` (`UD-Q5_K_XL` + `mmproj-BF16`)
  - `docs/bench-runbook.md` quickstart section for Gemma-4-26B-A4B on mainline llama.cpp
- **gemma-4-26b-a4b-q6-k-xl onboarding**:
  - `llama-swap.yaml` model entry `gemma-4-26b-a4b-q6-k-xl` using `--fit` defaults for first-pass tuning
  - `bench-models/bench-llama-cpp-gemma-4-26b-a4b-q6-x-l.sh`, `bench-models/bench-llama-cpp-gemma-4-26b-a4b-q6-x-l-strategies.sh`, `bench-models/bench-llama-cpp-gemma-4-26b-a4b-q6-x-l-fit.sh`
  - `model_downloader/models_config.json` disabled profile for `unsloth/gemma-4-26B-A4B-it-GGUF` with `*gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf*`
  - `docs/bench-runbook.md` §1 hardware table placeholder row + §8 benchmark stub for Gemma UD-Q6_K_XL
- **Gemma vision user service support**:
  - `maintenance/systemd/gemma-vision.service` user-level systemd unit for always-on startup
  - `maintenance/setup-gemma-vision-service.sh` helper for `install/start/stop/restart/enable/disable/status/logs`
  - `docs/bench-runbook.md` usage section for managing Gemma vision as a startup service
- **gpt-oss-puzzle-88B workflow support**:
  - `maintenance/build-gpt-oss-puzzle-llama-cpp.sh` wrapper for upstream PR merge build flow (defaults to PR `#21032` via `llama-test-pr.sh`)
  - `run-models/run-llama-cpp-gpt-oss-puzzle-88b.sh` run script targeting puzzle-compatible llama.cpp build output
  - `bench-models/bench-llama-cpp-gpt-oss-puzzle-88b.sh` baseline bench script
  - `bench-models/bench-llama-cpp-gpt-oss-puzzle-88b-strategies.sh` strategy sweep bench script (defaulted to fit-shaped partial split, `ngl=37`, semicolon-delimited `-ot` patterns)
  - `bench-models/bench-llama-cpp-gpt-oss-puzzle-88b-fit.sh` fit-based bench script
  - `model_downloader/models_config.json` profile for `SamPurkis/gpt-oss-puzzle-88B-GGUF` (`*MXFP4_MOE*` primary pattern)
  - `docs/bench-runbook.md` quickstart section + recorded benchmark results (`pp/tg`) for baseline, all-cpu-moe, and fit/fit-shaped partial split
- **Model onboarding playbook**:
  - new `docs/model-onboarding-playbook.md` documenting end-to-end model-family onboarding (build wrapper, downloader profile, run/bench scripts, docs/changelog, validation, and targeted download flow)
- **GGUF Model Browser tab**:
  - New `Model Browser` tab to scan any local directory for `.gguf` files (recursive or top-level)
  - Sortable/filterable table with quantization, size, parameter count, architecture, and modified time
  - Lightweight GGUF header parser for per-file metadata (model name, architecture, tokenizer, tensor count)
  - New keyboard actions: `Alt+R` scan, `Alt+G` focus path, `Alt+J` focus table, and `F7` tab switch
- **Sarvam 30B workflow support**:
  - `maintenance/llama-test-pr.sh` now defaults to PR-specific vendor folders: `vendor/llama.cpp-pr-test-<joined-prs>`
  - `maintenance/build-sarvam-llama-cpp.sh` is now a thin wrapper over `maintenance/llama-test-pr.sh` with default `SARVAM_PR_NUMBER=20275`
  - `run-models/run-llama-cpp-sarvam-30b.sh` updated with cleaner server flags and Sarvam-aligned sampling defaults (`temp=1.0`, `top_p=1.0`, `top_k=20`)
  - `bench-models/bench-llama-cpp-sarvam-30b.sh` for repeatable `llama-bench` runs against the Sarvam build
  - `bench-models/bench-llama-cpp-sarvam-30b-fit.sh` for automatic fit-based benching
  - `bench-models/fit-params-sarvam-30b.sh` for printing fitted `-ngl/-ts/-ot` placement args
  - `model_downloader/models_config.json` entries for `Sumitc13/sarvam-30b-GGUF` (Q6_K) and `limegreenpeper1/sarvam-105B-GGUF` (Q4_K_M default)
  - `docs/sarvam-local-post.md`: first natural-language draft post for running Sarvam locally (30B workflow + 105B download profile)
- **gpt-oss-120b bench suite**: `bench-llama-cpp-gpt-oss-120b.sh`, `bench-llama-cpp-gpt-oss-120b-strategies.sh`, `bench-llama-cpp-gpt-oss-120b-fit.sh`, `bench-ik-llama-cpp-gpt-oss-120b.sh` — full runbook coverage for gpt-oss-120b mxfp4 on 64 GB RAM systems
- **Qwen3.5-122B-A10B bench suite**: `bench-llama-cpp-qwen3-5-122b-a10b.sh`, `bench-llama-cpp-qwen3-5-122b-a10b-strategies.sh`, `bench-llama-cpp-qwen3-5-122b-a10b-fit.sh`, `bench-ik-llama-cpp-qwen3-5-122b-a10b.sh` — full bench coverage; documents shared-expert `(ch|)exps` pattern gotcha
- **`run-llama-cpp-gpt-oss-120b-optimized.sh`**: optimized server run script with static `-ngl 37 --override-tensor` (fit-derived), `--parallel 1`, explicit `--ctx-size 32768`; drops `--fit` startup overhead; confirmed +540 MiB more model weight on GPU and 28 t/s tg vs 27 t/s with original script
- **`maintenance/build-llama-cpp-cublas.sh`**: builds llama.cpp with `GGML_CUDA_FORCE_CUBLAS=ON` + `GGML_CUDA_FORCE_DMMV=OFF` into a separate `build-cublas/` dir; tested against gpt-oss-120b, found slower than default build (GGML MMQ mxfp4 kernel wins at decode-batch sizes)
- **`docs/bench-runbook.md §8`**: bench results for Qwen3.5-122B-A10B and gpt-oss-120b; documents shared-expert OOM root cause, RAM ceiling constraints, cuBLAS/ik_llama findings, static-ot vs fit VRAM breakdown comparison, and active-parameter tg scaling table
- **`2025-09-21-optimizing-gpt-oss-120b-local-inference.md`**: updated TL;DR tg to 28 t/s, pp to 420+; added `llama-fit-params` workflow section; expanded `--override-tensor` section with shared-expert gotcha and RAM ceiling warning; updated run script to static placement + `--parallel 1`; added l3ms repo link; closed out cuBLAS and ik_llama experiments
- Zellij-style `?` help overlay: footer now shows only tab-switch keys (`F1–F7`), `q`, `?`, and `Ctrl+P`; all `Ctrl+*`/`Alt+*` shortcuts moved to a `HelpScreen` modal grouped by context (Global / Jobs / Run / Chat / Download)
- **Command palette** (`Ctrl+P`): fuzzy-filtered `CommandPaletteScreen` modal lists every app action; type to narrow, Enter to run, Esc to cancel
- **Jobs tab stop + retry**: `■ Stop Running` and `↺ Retry Selected` buttons added to Jobs panel; running job shown with `▶` indicator; `s` / `r` key shortcuts when Jobs tab is active; `StopRequest` / `RetryRequest` messages routed through `L3MSApp` to `RunPanel`; `script_path` and `mode` now persisted in job history for reliable retry
- **Chat history persistence**: `save_chat` now writes both `.md` (human-readable) and `.json` (machine-loadable) to `~/.l3ms/chats/`; new `Sessions` / `Load` buttons open `ChatHistoryScreen` modal to browse and restore saved sessions
- **Graceful shutdown** (`action_quit`): on `q`, all running subprocesses are `terminate()`d and async resource/task loops are cancelled before exit — no more orphaned `llama-server` processes on quit
- `run-llama-cpp-nemotron-super-120b.sh`: run script for NVIDIA Nemotron 3 Super 120B-A12B UD-Q3_K_XL (latent-MoE, 12B active params, port 8001, ctx 32768)
- `NVIDIA-Nemotron-3-Super-120B-A12B-GGUF` UD-Q3_K_XL entry added to `models_config.json` (~62.6 GB, 3 split shards, enabled)
- **Mistral Small 4 (119B) script set**:
  - `run-models/run-llama-cpp-mistral-small-4-119b.sh` standard fit-based server script (safe defaults)
  - `run-models/run-llama-cpp-mistral-small-4-119b-optimized.sh` throughput-oriented static-placement script (`-ngl`, `--override-tensor`, `q8_0` KV, `--parallel 1`)
  - `run-models/run-llama-cpp-mistral-small-4-119b-optimized-no-vision.sh` explicit non-vision optimized preset (same tuned defaults as current non-vision path)
  - `run-models/run-llama-cpp-mistral-small-4-119b-optimized-vision.sh` explicit vision optimized preset with `--mmproj` and one fewer GPU layer by default (`-ngl 8`)
  - `bench-models/bench-llama-cpp-mistral-small-4-119b-strategies.sh` strategy sweep script that compares tg across offload presets and reports the best strategy
- **Bench + run script expansion**:
  - Added model-specific strategy/fit benches for Nemotron 120B, Qwen3.5-122B-A10B, Sarvam 30B, and gpt-oss-120B under `bench-models/`
  - Added optimized launch presets `run-llama-cpp-qwen3-coder-next-optimized.sh` and `run-llama-cpp-mistral-small-4-119b-optimized-no-vision-thinking.sh`
  - Added `preflight-check.sh` and maintenance helpers (`llama-sweep.sh`, `llama-test-pr.sh`) for repeatable local validation workflows
- **Docs additions**:
  - Added `docs/vibe_configuration.md` and expanded bench/run workflow guidance for current local model ops

### Fixed
- `MarkupError` crash in `refresh_disk_space`: paths like `/home/user` inside `[…]` were parsed as Rich closing tags; fixed by passing the full `[path]` token through `markup_escape()`
- **Maintenance tab output capture**: `run_script` no longer `await`s its own task — script output now streams live to `maint_log` without blocking the event loop
- `on_run_panel_job_started` and `on_run_panel_job_finished` now forward `mode` correctly to `JobsPanel`
- `ctrl+p` remapped from `run_save_script` (moved to `alt+p`) to `show_command_palette` for consistency with editor convention

- Launcher CLI interactive modes: `--run`, `--bench`, `--list`, `--extra`
- Model Ops runtime status with current model and resource telemetry
- Keyboard-first action scoping to active tab to prevent unwanted tab switching
- Project docs set: `TODO.md`, `AGENTS.md`, and app description updates
- New Qwen3.5 run config scripts for four modes: thinking-general, thinking-coding, instruct-general, instruct-reasoning

### Changed

- Renamed tab label from `Run Models` to `Model Ops`
- Updated docs branding to `L3MS`

### Fixed

- Global shortcut actions no longer force-switch to the Run tab

## [0.4.0] - 2026-02-24

### Added

- Keyboard-first Download controls and shortcuts
- Run script editor with save/restore snapshots in `.toolkit/script_versions/`

## [0.3.0] - 2026-02-24

### Added

- Keyboard-first Run tab with run/bench mode, script filter, start/stop, and live logs

## [0.2.0] - 2026-02-24

### Added

- Renamed toolkit TUI to `L3MS`

## [0.1.0] - 2026-02-24

### Added

- Initial Textual TUI with Download config editor/updater
- Config validation and snapshot history for model config files
