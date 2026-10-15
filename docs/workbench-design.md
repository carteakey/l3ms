# Workbench design: sidebar, toolbar, tabs, sections

Proposal, 2026-10-15. Not implemented. Supersedes the tab layout; keeps the
Signal / 03 look from [interface direction](interface-direction.md).

## Problem

The TUI has eight tabs that mirror how the code grew, not what the user does:

| Tab | What it really is |
| --- | --- |
| Workbench | a model list, GPU stats and bench scripts |
| Model Ops | the same model list, with load and unload |
| Stack | the same models again, plus media and decision profiles |
| Chat | a view of one language model |
| Browser | GGUF files on disk |
| Download | the downloader config |
| Maintenance | script editor |
| Jobs | run history |

Three tabs list models. Four are file management. All 102 commands live in one
13.8k-line `app.rs` that holds state, key handling, workflows and drawing together,
so a second frontend would have to copy it.

## Shape: sidebar, toolbar, tabs, sections

Reference: herdr, the workspace app this machine already runs agents in. Not a
copy; the same building blocks:

- **Sidebar** with **sections**: models (by lane), then jobs. Collapsible.
- **Toolbar**: the utilities as buttons (reload router, bench, tune, update
  engines, download, health), acting on the selected model where that applies.
- **Tabs**: workspaces the user opens, each holding split panes.
- **Status bar**: live machine and router state.

How herdr's parts translate:

| herdr | L3MS |
| --- | --- |
| sidebar `spaces`: repos with branch and ahead count | **models**, grouped by lane, with a state dot and one sub-line (runtime · last steady t/s or "not benched") |
| sidebar `agents`, grouped, with status dots | **jobs**: running and recent chats, benches, media tasks and utility runs, grouped by model |
| `new` / `menu` under the spaces list | **new pane** picker and the **utilities** menu |
| top tabs `○ 1  ● 2  +` | **workspaces** the user creates and names; not fixed app sections |
| split panes, each a terminal | split panes, each a **view**: chat, bench run, tune, job output, router log, media result |
| bottom bar: usage meters, CPU, MEM, theme, shortcuts | **status bar**: loaded model, VRAM and RAM bars (red over 90%), CPU, governor, disk free, router endpoint, shortcuts |

```
┌ ○ 1 chat  ● 2 bench  +        ⟳ reload  ▷ bench  ⚙ tune  ⇣ update  ⇩ get  ♥ ┐
│ models               │ chat · qwen38-flash-next-plat   │ bench · ctx sweep  │
│ language             │ > …                             │ 16k  62.1 t/s      │
│ ● plat               │                                 │ 32k  58.4 t/s      │
│   strata · 61 t/s    │                                 │ 64k  ⏵ running     │
│ ○ gold               │                                 │                    │
│   master · 21 t/s    │                                 │                    │
│ media                │                                 │                    │
│ ○ flux-3-klein       │                                 │                    │
│ ○ breeze-tts         │                                 │                    │
│ decisions            │                                 │                    │
│ ○ vllm-d3-flash      │                                 │                    │
│ new          menu    │                                 │                    │
│ jobs         grouped │                                 │                    │
│ ● bench · plat  2m   │                                 │                    │
│ ○ chat · plat   5m   │                                 │                    │
│ ✓ image · flux  1h   │                                 │                    │
├──────────────────────┴─────────────────────────────────┴────────────────────┤
│ ● plat  VRAM ████████░ 11.2/12  RAM █████████ 93%  CPU 67%  perf  :8080   ? │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **Models** (sidebar, top) is the only model list: today's Stack catalog with
  live router state. Enter opens the model's default view (chat, or its task
  form) in the focused pane; the verbs (load, unload, bench, tune) are on the
  row and in the palette.
- **Jobs** (sidebar, bottom) replace the Jobs tab and the Activity box. A dot
  shows running, done or failed; Enter opens the job's output or artifacts in
  a pane.
- **Workspaces** answer "are tabs useful": yes, but as the user's own
  arrangements (a chat beside a bench, a media result beside its prompt), not
  as one tab per app feature. Workspaces and their pane layout persist.
- **Panes** split horizontally or vertically, resize and close. Every view can
  go in any pane.
- **Status bar** carries what every tab used to repeat. Verbs come from the
  existing command registry, so the palette, help and row actions never
  disagree.

### Utilities, not scripts

L3MS is a wrapper on llama-swap, so the user should press buttons, not edit
shell scripts. Each utility is a typed action with a short form, a dry-run
preview of what it will change, live output and a recorded result. The scripts
under `bench-models/` and `maintenance/` become implementation details behind
those actions; the in-app script editor goes away.

| Utility | What it does |
| --- | --- |
| Update engines | pick an engine (llama.cpp master, ik_llama.cpp, Strata, audio.cpp, ComfyUI); fetch, build with house flags, smoke-test, then swap the binary path in `llama-swap.yaml` |
| Bench | on a model: pick a ladder (tg, pp, ctx sweep); captures `bench-env.sh` frontmatter, discards warm-up probes, shows steady-state t/s next to earlier runs |
| Tune | on a model: typed controls for its serving flags (ctx, `-ncmoe`, KV type, draft settings); snapshot the yaml, write, reload, smoke-test, with one-key revert |
| Download model | repo and quant picker, header preflight for size and PLE split, disk check, then download |
| Reload router | SIGHUP with a warning naming the model it will evict and any in-flight request |
| Health | spill check (`read_bytes` delta), free RAM, governor, VRAM; flags anything that would skew a bench |
| Media setup | per media profile: check weights and links, install what is missing after confirmation |

Each utility is declared once (inputs, preconditions such as "router must be
stopped", what it writes) so the TUI, GUI and CLI (`l3ms do bench …`) share it.

Navigation: move in the sidebar, act from the toolbar or the row, arrange
views in tabs. `Ctrl+P` still finds any command.

### Where each tab goes

| Today | Becomes |
| --- | --- |
| Workbench, Model Ops | deleted; sidebar models section plus status bar |
| Stack | the sidebar models section |
| Chat | a pane view of a language model |
| Param Builder | the Tune toolbar button |
| Jobs, Activity | the sidebar jobs section |
| Download, Browser | the Download toolbar button; the GGUF list becomes its picker |
| Maintenance (script editor) | deleted; scripts sit behind toolbar utilities |
| Workbench bench list | the Bench toolbar button |

## Architecture: core plus two thin frontends

```
l3ms-core   catalog, llama-swap client, chat client, job history, control/*,
            command registry, and a Workbench model:
              State  — models, jobs, tabs and panes, selection, telemetry
              Action — every CommandId plus input events, frontend-neutral
              update(&mut State, Action) -> Vec<Effect>
              Effect — HTTP call, spawn, file write; run off the UI thread,
                       results come back as Actions
l3ms        CLI and ratatui TUI: key map → Action, State → frame
l3ms-gui    optional GPUI app: same Actions, same State; not in the default build
```

Rules that already hold and stay: llama-swap owns engines and swapping; no
frontend spawns engines, retries an ambiguous submission or blocks on inference.

What each frontend adds:

- **TUI:** the same sidebar, workspaces and splits with keyboard resizing;
  `z` zooms a pane; the sidebar collapses below 100 columns. Works over SSH and
  headless, and inside herdr itself.
- **GUI:** the herdr look with mouse-draggable splits; inline image, audio and
  video previews of job artifacts, which a terminal cannot do well.

## Migration, each step shippable

1. **Shell.** Sidebar (models and jobs sections), toolbar, status bar and one
   pane, with today's tabs temporarily opening as pane views; delete Workbench
   and Model Ops. Gate: every command they offered is reachable from the
   sidebar, toolbar or palette.
2. **Utilities.** Declare the utility schema; ship Reload router, Bench and
   Tune first (the daily ones), then Update engines, Download model, Health and
   Media setup. Delete the script editor once each script it served has a
   utility or is plainly unused.
3. **Extract the model.** Move state, `update` and effects out of `app.rs`
   into `src/workbench/`; `app.rs` keeps keys and drawing. Tests move with the
   logic they cover. No behaviour change.
4. **Fold the rest.** Chat, job output and media results become pane views; the old fixed tabs go.
5. **Crate split.** `l3ms-core` and `l3ms` crates; CLI and tests unchanged.
6. **Tabs and splits.** User workspaces with split panes, persisted.
7. **GPUI prototype.** Pin a GPUI revision and prove build and launch on this
   desktop first; then read-only sidebar and job views; then actions and
   artifact previews. Measure the GUI's VRAM use: a 12 GB card running a model
   has little to spare.

## Open questions

- Which reference (herdr-gpui or otherwise) defines the window behaviour to aim for?
- Which utilities beyond the list above do you reach for (for example headless
  mode, NAS archive, PR-stack builds via `llama-test-pr.sh`)?
- Should Tune expose every flag or only a curated set per engine (today's
  `param_registry.rs` is the curated set)?

Implementation steps, gates and system monitoring: [workbench plan](workbench-plan.md).
