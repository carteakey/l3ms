# Workbench design: one inventory, two frontends

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

## Shape: one inventory, a focus pane, an inspector

```
┌ L3MS ─ ● qwen38-flash-next-plat ─ VRAM ███████░░ 11.2/12 ─ RAM ████░░ 31/62 ─ :8080 ┐
│ INVENTORY              │ FOCUS                                │ INSPECTOR         │
│ language               │  chat with qwen38-flash-next-plat    │ runtime  strata   │
│ ● qwen38-flash-next-pl │  > …                                 │ ctx      131072   │
│ ○ qwen38-flash-next    │                                      │ router   loaded   │
│ media                  │                                      │ last job  3m ago  │
│ ○ flux-3-klein         │                                      │                   │
│ ○ breeze-tts           │                                      │ verbs             │
│ decisions              │                                      │ c chat  l load    │
│ ○ vllm-d3-flash        │                                      │ u unload  b bench │
│ utilities              │                                      │ t tune            │
│   update engines       │                                      │                   │
│   download model       │                                      │                   │
│   reload router        │                                      │                   │
│ jobs · 12              │                                      │                   │
├────────────────────────┴──────────────────────────────────────┴───────────────────┤
│ : command   ⏵ job 4 running · yue2 music · 42s                       ? keys  q quit│
└────────────────────────────────────────────────────────────────────────────────────┘
```

- **Inventory** (left) is the only list. It is today's Stack catalog with live
  router state, grouped by lane, plus a `utilities` group (below) and job
  history. Everything you can act on is a row here.
- **Focus** (centre) shows what the selected row does: chat for a language
  model, a task form and its result for media and decisions, the output and
  artifacts of a job, a model's tuning controls or bench results, or a
  utility's form and live output.
- **Inspector** (right) shows facts and the verbs that apply to the selection.
  Verbs come from the existing command registry, so the palette, help and
  inspector never disagree.
- **Status line** (top) carries what every tab used to repeat: loaded model,
  VRAM and RAM gauges, router endpoint. **Command line** (bottom) shows the
  running job and takes `:` commands; the Activity box becomes a drawer opened
  with `a`.

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

There are no tabs. Navigation is: move in the inventory, act on the selection.
`Ctrl+P` still finds any command.

### Where each tab goes

| Today | Becomes |
| --- | --- |
| Workbench, Model Ops | deleted; inventory plus status line |
| Stack | the inventory itself |
| Chat | focus view of a language profile |
| Param Builder | the Tune utility on a model |
| Jobs | job history in the inventory; the running job also shows in the command line |
| Download, Browser | the Download model utility; the GGUF list becomes its picker |
| Maintenance (script editor) | deleted; scripts sit behind utilities |
| Workbench bench list | the Bench utility on a model |

## Architecture: core plus two thin frontends

```
l3ms-core   catalog, llama-swap client, chat client, job history, control/*,
            command registry, and a Workbench model:
              State  — inventory, selection, focus, jobs, telemetry
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

- **TUI:** fixed three panes; `z` zooms the focus pane, `<` `>` resize the
  inventory, panes collapse below 100 columns (inventory and focus) and 70
  columns (focus only). Works over SSH and headless.
- **GUI:** dockable, resizable panes and several focus windows (two chats side
  by side, a job next to its source profile); inline image, audio and video
  previews of job artifacts, which a terminal cannot do well.

## Migration, each step shippable

1. **Inventory-first TUI.** Add the inspector and job history to the Stack
   screen; delete Workbench and Model Ops. Gate: every command they offered is
   reachable from the inventory.
2. **Utilities.** Declare the utility schema; ship Reload router, Bench and
   Tune first (the daily ones), then Update engines, Download model, Health and
   Media setup. Delete the script editor once each script it served has a
   utility or is plainly unused.
3. **Extract the model.** Move state, `update` and effects out of `app.rs`
   into `src/workbench/`; `app.rs` keeps keys and drawing. Tests move with the
   logic they cover. No behaviour change.
4. **Fold the rest.** Chat and job views become focus views; tabs go.
5. **Workspace split.** `l3ms-core` and `l3ms` crates; CLI and tests unchanged.
6. **GPUI prototype.** Pin a GPUI revision and prove build and launch on this
   desktop first; then read-only inventory and job views; then actions and
   artifact previews. Measure the GUI's VRAM use: a 12 GB card running a model
   has little to spare.

## Open questions

- Which reference (herdr-gpui or otherwise) defines the window behaviour to aim for?
- Which utilities beyond the list above do you reach for (for example headless
  mode, NAS archive, PR-stack builds via `llama-test-pr.sh`)?
- Should Tune expose every flag or only a curated set per engine (today's
  `param_registry.rs` is the curated set)?
