# AGENTS

Agent and contributor guide for `L3MS`.

## Purpose

Build a keyboard-first, script-first homelab LLM toolkit with strong operational ergonomics.

## Engineering Rules

- Prefer deterministic script orchestration over hidden automation.
- Serving is declared in `llama-swap.yaml`; benching is declared in
  `bench-models/*.sh`. Both are editable text, not hard-parameterized UI
  forms. When serving and benching flags drift, update them together.
- Treat keyboard control as first-class; mouse workflows are optional.
- Preserve existing files by default for downloads unless explicitly overridden.
- Keep version snapshots for both model configs and scripts before writes.

## llama-swap Operations

- Config hot-reload: `kill -HUP $(pgrep '[l]lama-swap')`. New/edited
  `llama-swap.yaml` entries are picked up without restarting the router
  (verified: no dropped connections, running models keep serving).
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

Deep detail, decisions and next steps: `docs/qwen38-flash-next-internal.md`.

Three router tiers share one AtomicChat AD-4.27bpw quant:

- gold `qwen38-flash-next`: plain upstream master (`vendor/llama.cpp-master`,
  macro `qwen38_master_server`). Refresh = `git fetch` + rebuild; stays
  current as qwen4exp PRs merge upstream.
- exp `qwen38-flash-next-exp`: master + unmerged qwen4exp PRs
  (`build/bin/llama-server-exp` inside `vendor/llama.cpp-pr-test-28023-28068-27941`,
  branch `pr-test-28023-28068-27941`). Becomes redundant when its PRs merge;
  delete the delta then.
- MTP `qwen38-flash-next-mtp`: exp + #27836 draft head + detached-head patch
  (`build/bin/llama-server` in the same clone, branch `pr-test-exp-mtp`),
  32k ctx cap, opt-in.

MTP upgrade status (2026-09-01): every alternative tested and closed —
see the internal doc §0/§9.1. The unlock is upstream #28104 (or
#27836+#28097+#28118) merging; until then 0b7d6d57d is the fastest MTP
that exists on 12 GB. Do not hand-merge MTP PR branches — three separate
hand-merges all failed for named reasons.

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
  first).
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

## Structure

- `src/app.rs`: Ratatui layout, keybindings, workflows, and process supervision
- `src/cli.rs`: launcher + interactive CLI (`--run`, `--bench`, `--list`)
- `src/llama_swap.rs`: authenticated llama-swap HTTP boundary
- `src/config_store.rs`: download config CRUD + validation + snapshots
- `src/script_store.rs`: script CRUD + snapshots
- `l3ms/` and `l3ms.py`: legacy Python TUI retained during parity work
- `model_downloader/`: Python downloader compatibility boundary

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
