# Workbench rebuild plan

Implementation plan for [workbench design](workbench-design.md). Written
2026-10-15 so work can resume after a context reset.

## Finish line

**The release is steps 1 and 2.** It is done when a user can: select a model →
bench, tune or run a media task → inspect the output and history → safely
revert. One sidebar, one pane, useful utilities, compact readiness monitoring.

Steps 3–7 (core extraction as a project, folding every view, crate split, tabs
and splits, GPUI) are optional follow-ups, not completion requirements. Logic
leaves `app.rs` incrementally, as each action moves, not as a separate step.

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
  (pattern used on 2026-10-15; delete it before committing). Permanent tests
  assert the shell and every view render readably at 80×24 and 60×18, not just
  without panicking.
- Live router: llama-swap v265 on `:8080`, `llama-swap.service`. Never reload
  or swap models without asking; a reload evicts the resident model and kills
  in-flight requests (check `/running` and the journal first).

Current UI: `src/app.rs` (13.8k lines) holds state, keys, workflows and
drawing; `src/commands.rs` is a UI-free command registry (102 commands,
`CommandContext` per tab); `src/theme.rs` holds Signal / 03 tokens;
`src/telemetry.rs` has `GpuTelemetry`, `SystemMemoryTelemetry`,
`ServerIoTelemetry`, polled by `App` with a GPU-utilisation history.

## References and what upstream already does

Three UIs on this machine set the bar (screenshots shared 2026-10-15):

- **herdr**: sidebar with sections, user tabs, split panes, status bar.
- **llama-swap v265 web UI** (`/ui/`): sidebar (Activity, Playground, Models
  with state dots, Logs, Performance, Hardware); Playground tabs Chat, Images,
  Speech, Transcription, Rerank and **Load Test** (queue models, fire
  concurrent streaming requests, live per-request panels).
- **Strata monitor** (`/upstream/<strata-model>/#monitor`): model state strip
  (Idle, Reading, Generating, Queued, Error, with the current activity and a
  progress bar); metric cards with a big number and a sparkline (decode and
  prefill t/s, GPU load, VRAM with experts cached, GPU temperature, power of
  limit, PCIe generation and throughput, CPU, disk read and write); context fill
  gauge; experts-in-VRAM, system RAM and GPU temperature bars; recent requests
  table (time, status, prompt, reused, output, prefill t/s, decode t/s, VRAM hit
  rate, duration) with a since-start summary.

Consequence: these are references, not things to copy. L3MS does not rebuild a
playground, log viewer or hardware page. Its views read llama-swap's data and
link out to the web UI: `o` opens the matching page in a local browser, or
prints and copies the URL when there is none (SSH). L3MS spends effort on what neither
has: bench with readiness gating and history, tune with snapshot and revert,
engine updates, downloads with preflight, media workflows with artifacts, job
history across lanes, and keyboard-first use over SSH.

Read-only APIs verified on the live router (2026-10-15, no model dispatched):

| Endpoint | Gives |
| --- | --- |
| `GET /running` | resident models and state |
| `GET /api/events` (SSE) | `modelStatus`, `inflight`, `profileChanged`, `uiConfig` events |
| `GET /api/performance` | `gpu_stats` and `sys_stats` histories (720 samples: GPU util, VRAM, temperature, power, fan; per-core CPU, RAM, swap, load, net I/O) |
| `GET /api/hardware` | host description (CPU, memory, board, OS) |
| `GET /logs` | router log tail |
| `GET /upstream/<id>/status` (Strata) | `busy`, `queued`, `phase`, prompt and generated tokens, current tool, output tail |
| Strata `/metrics`, `/metrics/prometheus`, `/api/requests` (path under `/upstream/<id>/` to be confirmed; `/api/requests` returned 404 there) | engine counters and recent requests |

Hazard: any `/upstream/<id>/…` request for a model that is not running makes
llama-swap load it, evicting the resident model. Checking `/running` first is
not enough: the model can unload (TTL, another client) between the check and
the request. Rules:

- No automatic polling of `/upstream/…` paths. Monitoring uses only router
  endpoints that never load a model (`/running`, `/api/events`,
  `/api/performance`, `/api/hardware`, `/logs`) until a genuinely
  non-autoloading engine route exists.
- Engine detail (Strata status) is fetched only on an explicit user action,
  behind the same warning as a model switch.
- Opening an engine's web page (`o` on Strata's monitor) can load the model too
  and carries the same warning.

Privacy: Strata `status` exposed live conversation text (`tail`). Telemetry is
whitelisted: read phase, counts, timings and states only; never store or show
prompts, output tails or tool arguments in monitoring or history. Every value
shows its source host and age; a stale or remote reading is labelled as such.

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
   buttons; mouse clicks optional. Buttons are `commands.rs` entries rendered
   in a row, not a separate list.
4. Status bar: see System monitoring below.
5. View routing: Enter on a model opens its default view (Chat for language
   models, task form for media and decisions); Enter on a job opens its output.
   The old tabs remain reachable as views from the palette (`Ctrl+P` → "open
   view"); folding them all is optional (step 4).
6. Delete the Workbench and Model Ops tabs and their commands once each command
   they offered maps to a sidebar row action, toolbar button or palette entry.
7. Open on the shell with the Models section focused.

Gate: the render-at-awkward-sizes test covers the shell; a test asserts every
command formerly in `Workbench`/`ModelOps` contexts is still reachable.

## Step 2 — toolbar utilities

1. Extend the existing `src/commands.rs` registry, not a competing framework:
   add to `CommandSpec` the inputs (typed fields reusing `control/task_form.rs`
   field kinds), preconditions ("router stopped", "model selected", "no job
   running"), what it writes, and how it executes. The same entry serves the
   toolbar, palette, help and a new CLI verb `l3ms do <command> [--field
   value]`. Move each action's logic out of `app.rs` when its utility is built.
2. Every run is a job in the sidebar with live output in a pane, a dry-run
   preview before anything is written, and the exit recorded in history.
3. Order:
   - **Reload router**: shows the resident model and in-flight requests from
     `/running` before confirming; SIGHUP; waits for `configuration reloaded`.
   - **Bench**: on a model; ladder choice (tg, pp, ctx sweep); when the ladder
     needs the full card, record whether llama-swap was running, stop it, and
     afterwards restore that prior state (do not blindly start it); wraps the
     existing `bench-models/run-*.sh` runners and `bench-env.sh`; result view
     compares steady state only with earlier runs in `logs/results` whose
     model, build, settings and system conditions match.
   - **Tune**: on a language model; curated flags from `param_registry.rs`;
     snapshot `llama-swap.yaml.bak-*` and record its hash; write via
     `param_builder.rs` only if the file still matches what was read; reload
     (with the Reload router warning), smoke chat with `max_tokens >= 64`.
     Revert restores the snapshot only if the file still matches what Tune
     wrote; if someone edited it in between, show the diff and ask.
   - **Update engines**: llama.cpp master, ik_llama.cpp, Strata, audio.cpp,
     ComfyUI; check `git status` of the vendor clone first and refuse on dirty
     trees unless confirmed; build beside the active installation (new
     directory), smoke test it, then repoint the macro in `llama-swap.yaml`
     (snapshot and conflict check as in Tune). The old build stays for revert.
   - **Download model**: wraps `model_downloader`; header preflight
     (`download_preflight.rs`) and disk check before starting.
   - **Health**: the System monitoring checks below, on demand.
   - **Media setup**: per media profile, check weights and ComfyUI model links
     (`setup-*.sh check`), install missing pieces after confirmation.
4. Remove the Maintenance script editor once every script it served is behind
   a utility or confirmed unused (list them first; ask the user).

## Optional follow-ups (not part of the release)

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

**Sources.** Subscribe to llama-swap first, probe locally only for the gaps:

| Signal | Source |
| --- | --- |
| Model state, in-flight requests | `/api/events` (`modelStatus`, `inflight`) |
| GPU util, VRAM, temperature, power, fan; CPU per core, RAM, swap, load, net I/O | `/api/performance` (poll the tail; histories feed sparklines) |
| Engine phase, tokens, context, recent requests | engine endpoints under `/upstream/<id>/` for ready models only (Strata `status`, `metrics`) |
| Governor, EPP, power profile | `/sys/devices/system/cpu/cpu*/cpufreq/`, `powerprofilesctl get` |
| CPU temperature | `/sys/class/hwmon` |
| Disk free | `statvfs` on `/home`, `/mnt/dev` and the models root |
| THP mode | `/sys/kernel/mm/transparent_hugepage/enabled` |
| Spill check, server RSS | `/proc/<pid>/io` `read_bytes` delta, `/proc/<pid>/status` VmRSS |

Retire L3MS's own `nvidia-smi` and `/proc/meminfo` polling in
`src/telemetry.rs` once `/api/performance` covers them; keep a local fallback
only for when the router is down.

**Where it shows:**

Monitoring in the release is compact: the status bar plus the Health checks.
The full System view below is optional.

- **Status bar** (always): resident model, VRAM bar, RAM bar, CPU %, governor,
  disk free of the fullest watched mount, router endpoint. Values go amber at
  their warning threshold and red at the critical one, always with the number.
- **System view** (optional; pane, toolbar button `♥`), laid out like the Strata
  monitor: a model state strip (idle, reading, generating, queued, error, with
  the current activity), metric cards with a value and sparkline (decode and
  prefill t/s, GPU load, VRAM, GPU temperature, power, CPU, disk read, RAM),
  context fill, a recent-requests table, and the bench-readiness checklist. In
  the TUI the cards are compact rows with braille sparklines.
- **Alerts** become entries in the sidebar Jobs section (kind `alert`) so they
  are seen and dismissible: sustained hot-expert spill after warm-up, low
  available RAM, disk nearly full, GPU running hot.
- **Bench readiness** is contextual, not one fixed rule. Each check returns
  **fail** (numbers would be meaningless: router serving another request on the
  card, disk full), **warn** (numbers are comparable only to runs with the same
  condition: governor not `performance`, spill above the model's own baseline,
  low RAM) or **unknown** (signal unavailable, stale, or from a remote router
  host — local checks never judge a remote router). Fails block unless
  overridden; warns and unknowns are shown and recorded. Every override and
  condition goes into the bench row's `sys` frontmatter, and comparisons only
  pair rows with matching conditions.

Thresholds live in one table in code with tests; no thresholds in the views.

## Open questions for the user

- Utilities beyond the list (headless mode, NAS archive, PR-stack builds via
  `llama-test-pr.sh`)?
- Tune: curated flags only, or every flag with the curated set first?
- Which scripts in `maintenance/` are still used (decides what the script editor
  removal may drop)?
