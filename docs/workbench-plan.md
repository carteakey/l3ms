# Workbench rebuild plan

Implementation plan for [workbench design](workbench-design.md). Written
2026-10-15 so work can resume after a context reset.

## Starting point

- Dev worktree `~/repos/l3ms-monorepo` (branch `feature/l3ms-monorepo`);
  deployed checkout `~/repos/l3ms` (`main`). Deploy: commit in the worktree,
  then `git merge --ff-only feature/l3ms-monorepo` in `~/repos/l3ms`. The
  deployed tree carries the user's uncommitted work (bench logs,
  `llama-swap.yaml` edits, untracked docs): back it up under
  `~/repos/_backups/l3ms-wip-*` and stash or pop around merges that touch it.
- After deploying: `cargo build --release` and `install -m755
  target/release/l3ms ~/bin/l3ms`.
- Gates for every step: `cargo fmt`, `cargo clippy --all-targets -- -D
  warnings`, `cargo test` (556 passing at plan time), the Python suites
  (`python3 -m unittest discover -s <dir> -p 'test_*.py'` for
  `model_downloader`, `media-models`, `maintenance`, `decision-models`),
  `l3ms --list stack` still lists 44 profiles in the deployed tree.
- Screen check: a temporary `#[ignore]` test that draws each view to a
  `TestBackend` at 110×30 and 80×24 and writes the rows to a scratch file
  (pattern used on 2026-10-15; delete it before committing).
- Live router: llama-swap v265 on `:8080`, `llama-swap.service`. Never reload
  or swap models without asking; a reload evicts the resident model and kills
  in-flight requests (check `/running` and the journal first).

Current UI: `src/app.rs` (13.8k lines) holds state, keys, workflows and
drawing; `src/commands.rs` is a UI-free command registry (102 commands,
`CommandContext` per tab); `src/theme.rs` holds Signal / 03 tokens;
`src/telemetry.rs` has `GpuTelemetry`, `SystemMemoryTelemetry`,
`ServerIoTelemetry`, polled by `App` with a GPU-utilisation history.

## Step 1 — shell

Replace the tab bar and per-tab chrome with the new frame. Keep existing views
working inside the pane.

1. New `src/ui/` module: `shell.rs` (frame layout), `sidebar.rs`,
   `toolbar.rs`, `statusbar.rs`. `app.rs` calls `shell::draw` and passes the
   pane renderer for the current view.
2. Sidebar sections: **Models** (Stack catalog rows grouped by lane, state dot
   from `router_state`, sub-line runtime · last steady t/s from
   `logs/results/<model>.jsonl` or "not benched") and **Jobs** (from
   `job_history`, newest first, dot for running/done/failed). `Tab`/`Shift+Tab`
   moves focus sidebar ↔ pane; `j/k` within a section; `[`/`]` jumps sections;
   sections collapse with `Space` on the header.
3. Toolbar row: buttons for the toolbar utilities (step 2 fills them in;
   initially reload, bench, tune map to existing commands). Keys shown on the
   buttons; mouse clicks optional.
4. Status bar: see System monitoring below.
5. View routing: Enter on a model opens its default view (Chat for language
   models, task form for media and decisions); Enter on a job opens its output.
   The old tabs remain reachable as views from the palette (`Ctrl+P` → "open
   view") until step 4 folds them.
6. Delete the Workbench and Model Ops tabs and their commands once each command
   they offered maps to a sidebar row action, toolbar button or palette entry.
7. Open on the shell with the Models section focused.

Gate: the render-at-awkward-sizes test covers the shell; a test asserts every
command formerly in `Workbench`/`ModelOps` contexts is still reachable.

## Step 2 — toolbar utilities

1. `src/utilities.rs`: a declaration per utility — id, label, key, inputs
   (typed fields reusing `control/task_form.rs` field kinds), preconditions
   ("router stopped", "model selected", "no job running"), what it writes, and
   the command it runs. One declaration serves the toolbar, palette and a new
   CLI verb `l3ms do <utility> [--field value]`.
2. Every run is a job in the sidebar with live output in a pane, a dry-run
   preview before anything is written, and the exit recorded in history.
3. Order:
   - **Reload router**: shows the resident model and in-flight requests from
     `/running` before confirming; SIGHUP; waits for `configuration reloaded`.
   - **Bench**: on a model; ladder choice (tg, pp, ctx sweep); stops llama-swap
     when the ladder needs the full card and restarts it after; wraps the
     existing `bench-models/run-*.sh` runners and `bench-env.sh`; result view
     compares steady state with earlier runs in `logs/results`.
   - **Tune**: on a language model; curated flags from `param_registry.rs`;
     snapshot `llama-swap.yaml.bak-*`, write via `param_builder.rs`, reload
     (with the Reload router warning), smoke chat with `max_tokens >= 64`,
     one-key revert to the snapshot.
   - **Update engines**: llama.cpp master, ik_llama.cpp, Strata, audio.cpp,
     ComfyUI; fetch, build with house flags, smoke test, then repoint the
     macro in `llama-swap.yaml` (snapshot first). Check `git status` of the
     vendor clone first; refuse on dirty trees unless confirmed.
   - **Download model**: wraps `model_downloader`; header preflight
     (`download_preflight.rs`) and disk check before starting.
   - **Health**: the System monitoring checks below, on demand.
   - **Media setup**: per media profile, check weights and ComfyUI model links
     (`setup-*.sh check`), install missing pieces after confirmation.
4. Remove the Maintenance script editor once every script it served is behind
   a utility or confirmed unused (list them first; ask the user).

## Step 3 — extract the core

Move state and behaviour out of `app.rs` into `src/workbench/`
(`state.rs`, `action.rs`, `update.rs`, `effects.rs`): `update(&mut State,
Action) -> Vec<Effect>`; effects (HTTP, process spawn, file write, telemetry
polls) run off the UI thread and return Actions. `app.rs` keeps key mapping
and drawing only. Tests move with the logic. No behaviour change.

## Step 4 — fold the remaining views

Chat, job output, media and decision results, GGUF picker and download config
become pane views. Delete the fixed `Tab` enum and its F-key bindings.

## Step 5 — crate split

Cargo workspace: `l3ms-core` (everything without ratatui) and `l3ms`
(CLI + TUI). CLI flags and tests unchanged.

## Step 6 — tabs and splits

User workspaces as tabs; each tab a tree of horizontal/vertical splits holding
views. Keys: new tab, close, split right/down, move focus, resize, zoom (`z`).
Persist tabs and layouts in the app data root next to job history.

## Step 7 — GPUI prototype

Separate optional package on `l3ms-core`; pin a GPUI revision and prove build
and launch on this desktop first. Then read-only sidebar and job views, then
actions, then inline image, audio and video previews of job artifacts.
Measure its VRAM use with a model loaded before going further.

## System monitoring

Goal: the user can see at a glance whether the machine is in a state where
serving and bench numbers are trustworthy, without leaving L3MS.

**Sources** (extend `src/telemetry.rs`; poll off the UI thread, 1–2 s for fast
values, 30 s for slow ones):

| Signal | Source |
| --- | --- |
| GPU util, VRAM, temperature, power | `nvidia-smi` (existing `GpuTelemetry`) |
| RAM used/available, zram | `/proc/meminfo`, `/sys/block/zram*` (existing `SystemMemoryTelemetry`, add available) |
| CPU load per core, governor, EPP | `/proc/stat`, `/sys/devices/system/cpu/cpu*/cpufreq/` |
| Power profile | `powerprofilesctl get` (slow poll) |
| CPU temperature | `/sys/class/hwmon` |
| Disk free | `statvfs` on `/home`, `/mnt/dev` and the models root |
| THP mode | `/sys/kernel/mm/transparent_hugepage/enabled` |
| Model server I/O and RSS | `/proc/<pid>/io` `read_bytes` delta (existing spill check), `/proc/<pid>/status` VmRSS |
| Router state | llama-swap `/running` (resident model, state, in-flight) |
| Decode speed | llama-swap request log or server `/metrics` if the engine exposes it; otherwise last bench result |

**Where it shows:**

- **Status bar** (always): resident model, VRAM bar, RAM bar, CPU %, governor,
  disk free of the fullest watched mount, router endpoint. Values go amber at
  their warning threshold and red at the critical one, always with the number.
- **System view** (pane, toolbar button `♥`): every signal above with short
  history sparklines (GPU util, VRAM, RAM, read_bytes rate, CPU temperature)
  and the bench-readiness checklist.
- **Alerts** become entries in the sidebar Jobs section (kind `alert`) so they
  are seen and dismissible: hot-expert spill (read rate over 1 MB/s for 10 s
  after warm-up), RAM available under 4 GiB, governor not `performance`,
  disk under 10 % free, GPU over 85 °C.
- **Bench gate**: the Bench utility runs the Health checks first and refuses on
  failures that invalidate numbers (governor, spill, low RAM) unless the user
  overrides; the override is recorded in the bench row's `sys` frontmatter.

Thresholds live in one table in code with tests; no thresholds in the views.

## Open questions for the user

- Utilities beyond the list (headless mode, NAS archive, PR-stack builds via
  `llama-test-pr.sh`)?
- Tune: curated flags only, or every flag with the curated set first?
- Which scripts in `maintenance/` are still used (decides what the script editor
  removal may drop)?
