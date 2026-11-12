# Workbench review, 2026-11-12

Review of the Rust TUI as a tool someone would open every day, plus a proposal for
image and video generation without hand-edited ComfyUI graphs. Proposal only,
nothing here is implemented.

## How this was reviewed

- Read `src/ui/`, `src/theme.rs`, the outline of `src/app.rs`, `src/commands.rs`,
  `media-runtimes.json`, `media-models/*.py`, and the existing direction docs
  ([interface direction](interface-direction.md), [workbench design](workbench-design.md),
  [workbench plan](workbench-plan.md)).
- Built the debug binary and drove it in a pty at 140x40, 130x34 and 80x24,
  capturing every view (F1 to F9), help, palette and the media task form as text.
- Limits: the sandbox has no llama-swap router, no GPU and no model files. Every
  live state (loaded model, VRAM bar, running job, chat stream) was not observed.
  Colour was not observed either, only layout and text. Findings about live
  behaviour are marked "check on the box".

## Verdict

The scripts are the product. They encode the hard-won workflows: the dark ladder
with its grace delay and unit stop/start, the CPU power gate, warm-up discard and
`tg_steady_mean`, the identity probe, the spill check, Jev triage, download
preflight. The TUI exposes almost none of that. It exposes **files**: a list of 37
catalog rows, a list of 66 bench scripts with a source editor, a list of
maintenance scripts with a source editor, a download config table, a GGUF
inventory.

That is why it does not get used. Every question I'd actually open a tool for is
answered faster by the shell or by llama-swap's own `/ui/`:

| Daily question | Where the TUI answers it today |
| --- | --- |
| What is loaded and is it healthy? | One status-bar segment. No decode rate, no context fill, no cache hit rate, no request phase. |
| How fast is the plat tier right now, against last week? | Nowhere. The sidebar says "not benched" for every Strata tier (the plan notes why: path matching against `bench-models/logs/results` only). The headline 73.00 t/s exists only in `AGENTS.md`. |
| Run the paired control safely | Pick a `.sh` from 66 names and press run. None of the gating the scripts carry is visible. |
| Change `--kv-grow` or an `env` switch and be able to revert | "Tune (coming)". Param Builder is in the palette with no key (`—`). |
| Make an image | Paste an API-format ComfyUI graph into a text field (see the media section). |
| Why did it get slow? | Open F8 for two sparklines, or leave and run the spill check by hand. |

The shell from step 1 of the plan is sound. Sidebar, toolbar, status bar, palette,
confirm-before-evict, render tests at 80x24 and 60x18: all good bones. What is
missing is what goes **in** the pane. The plan's own finish line says the release
is steps 1 and 2, and step 2 (the utilities) is the part that makes this a tool.
My main recommendation is to stop at the current shell, skip workspaces, splits,
crate split and GPUI for now, and spend everything on six loops.

## The six loops

Make each of these reachable in three keys or fewer from launch, and the tool
earns its place:

1. **Now**: what is resident, its phase, decode and prefill rate, VRAM, context fill.
2. **Swap**: load a tier by alias, with the eviction warning that already exists.
3. **Bench**: run a named ladder on the selected model with the gates applied, then
   see the steady-state row next to earlier comparable rows.
4. **Tune**: edit curated flags and `env`, snapshot, reload, smoke, one-key revert.
5. **Generate**: prompt, size, steps, seed, go. Open the result.
6. **Diagnose**: readiness checklist, spill check, last router errors.

Everything below is in service of those.

## Findings from the captures

### F1. The first screen is a catalog, not a cockpit

On launch the pane shows the first alphabetical entry (`cloudflare-clef-flash`), a
paragraph of description prose, and a raw flag string. On a 140x40 terminal about
60% of the pane is empty. Nothing on screen says what the machine is doing.

Do: make the default pane a **Now** view. One state strip (idle, reading,
generating, queued, error), then four to six number cards with braille sparklines
(decode t/s, prefill t/s, VRAM, RAM available, GPU temperature, context fill), then
the last few requests, then the bench-readiness checklist. The plan already
verified the sources that never autoload a model: `/running`, `/api/events`,
`/api/performance`. Keep the rule that `/upstream/<id>/...` is only fetched on an
explicit key.

This is also where "futuristic" should come from. The interface direction is right
to rule out scanlines and glow. An instrument looks advanced because it is dense,
live and exact: big numerals, a sparkline under each, one lane-coloured routing
line that lights only for the route that is really executing, and a token-rate
readout that moves because tokens are moving. Ratatui's `Sparkline`, `Gauge`,
`Chart` and `Canvas` (braille) cover all of it. Today they appear only in F8.

### F2. Router offline is invisible

With no router, every model reads "not listed by the router", the status bar shows
a dim `:8080`, and the connection error is only in F9. It reads as 37 broken
models rather than one unreachable service.

Do: give the endpoint segment a state (`● :8080`, `✕ :8080 offline`), put "router
offline" in the pane title once instead of on every profile, and surface the last
error on the message line. Offer the fix as a key: retry, and (local host only)
the `systemctl --user start llama-swap.service` line, remembering `--user`.

### F3. The sidebar is long, flat and ambiguous

- 37 rows in alphabetical order. The five tiers used daily sit between
  `humanizer-12b` and `vllm-d3-flash`.
- At width 32 two different tiers render as the same string,
  `qwen38-flash-next-plat-abli…`.
- Decision models (`cloudflare-clef-flash`, `vllm-d3-flash`) appear under
  `language` and again under `decisions`.
- Only one row carried a rate (`gemma-4-26b-qat-mtp 77 t/s`).
- Profiles whose weights are not on disk ("file not found") look the same as
  installed ones.

Do:

- Order each lane: resident, then pinned, then recently used, then the rest.
- Show the alias when one exists (`strata`, `strata-vision`, `abliterated-vision`)
  and truncate in the middle, never at the end, so suffixes survive.
- Strip the shared family prefix inside a group (`plat`, `plat-vision`, `gold`,
  `mtp`, `exp` under a `qwen38-flash-next` sub-header), as the design sketch did.
- List a profile once, under its lane.
- Dim or fold "not installed" behind a toggle.
- Read rates from every results file, including `logs/results/strata-tuning-ab.jsonl`,
  keyed by router id rather than GGUF path, and show `tg_steady_mean` with its spread.

### F4. Bench and Maintenance are file browsers with an editor attached

F4 lists 66 scripts and shows the selected script's source. F7 does the same for
`maintenance/`. Results, which are the point of a bench, are not on screen.
`src/bench_results.rs` and `--compare-results` already exist in the CLI.

Do: make Bench result-first. A table per model: date, arm, `tg_steady_mean`,
spread, prefill at 2k and 8k, and condition badges from the row's `sys` block
(governor, RAM available, temperature). Rows whose conditions differ from the
selected row are dimmed, since `AGENTS.md` forbids comparing across governors.
`Enter` on a row shows the frontmatter; `b` starts a ladder. The script list moves
behind the ladder picker. Jev's `/jev-bench` verdict (steady, quotable, comparable,
spill-free) is a natural fourth column.

The bench utility itself should wrap `run-strata-tuning-dark.sh` and show its
gates as a checklist before the run: CPU power state (`cpu-power-state.sh check`),
router unit state and whether it will be restored, `TG_WARMUP`/`TG_PROBES`, free
RAM. A failing gate blocks, a warning is recorded into the row.

### F5. Tune is the highest-value missing feature

Serving lives in `llama-swap.yaml` and every change today is a hand edit, a manual
snapshot, a SIGHUP and a smoke request. `param_registry.rs` and `param_builder.rs`
(about 1,900 lines) already do import, export and cmd replacement. The plan's Tune
spec (snapshot with hash, write only if unchanged, reload behind the eviction
warning, smoke with `max_tokens >= 64`, revert only if the file still matches) is
right. For the Strata tiers it also needs the run config's `env` block, since most
of the levers there are `STRATA_*` variables, not flags. Link each switch to its
row in the [tuning matrix](strata-tuning-matrix.md) so the UI shows "rejected,
-6.3 t/s on Ada" next to `STRATA_GR_NORM_FAST` before anyone flips it.

### F6. Chat asks for a ritual

The Chat header carries five hotkeys (`e`, `Ctrl+G`, `Ctrl+B`, `K`, arrows) and
starts "disconnected", although the app already queried `/v1/models` on the same
endpoint at startup. Model choice in Chat is independent of the sidebar selection.

Do: connect automatically to the configured endpoint, take the model from the
sidebar selection, and keep endpoint editing in settings. Chat here only needs to
be a good smoke test: send, stream, and print prompt tokens, generated tokens,
prefill and decode t/s under each reply. llama-swap's playground remains the full
chat client, as the plan says.

### F7. Three rows of chrome, with the same hints three times

The message line shows `Enter open · L load · ? help` on the left and
`j/k · [/] · Enter open · L load · Tab pane` on the right; the status bar ends in
`? help`; the toolbar repeats keys again. At 80x24 that is 3 of 24 rows, and the
pane is hidden entirely until `Tab`.

Do: one hint line, context only, right-aligned. Left side is for messages. At
narrow widths consider showing the Now strip above the sidebar instead of nothing.

### F8. The palette should be the main verb surface

It opens on `Quit L3MS`, lists 101 commands flat, and takes no arguments. Help is
a static overlay that closes on any key and cannot scroll.

Do:

- Rank by current context, then recency. Never open on Quit.
- Fuzzy match on name and aliases, with group headers.
- Accept arguments: `load strata`, `bench plat tg`, `gen flux a red fox`,
  `open results plat`. This is the same grammar as the planned `l3ms do <command>`,
  so the CLI and the palette stay one registry.
- Fold help into the palette (it is the same list with keys shown).

### F9. Model pane: prose first, numbers last

The facts card leads with the description and a wrapped flag string. Lead with
what is scanned: state, last steady t/s with date, VRAM and RAM footprint, context,
TTL, engine build. Then recent bench rows. Then flags, folded. For media profiles
show the last few artifacts instead.

### F10. Jobs need to carry results

The Jobs view is an ID/kind/status table. For this tool a job's value is its
output: a bench row, an image path, a log tail. `Enter` should open that, and `o`
should hand an artifact to `xdg-open` locally or print and copy the path over SSH.

### F11. Structure

`src/app.rs` is 14,423 lines, about 9,900 of them non-test, holding state, keys,
workflows and drawing. I would not schedule the extraction (plan step 3) as its
own project. Move each loop's logic out as that loop is built, which is what the
plan already says. The 100 ms `event::poll` loop is fine; just keep every new
network read off it.

### Small things

- The media task form defaults to `Timeout (ms): 30000` (`src/app.rs:433`) while
  the CLI default is 300,000 and `AGENTS.md` measures a ComfyUI cold start at about
  246 s. A first run from the form would time out during boot unless that timeout
  is scoped to something narrower. Check on the box.
- `Open Param Builder tab` has no binding while the toolbar shows `⚙ tune t`.
- The Download view opens on 19+ disabled rows from older models. Sort enabled and
  recent first.
- The pane title "not listed by the router" for a media profile whose engine is
  `comfyui_auto` is confusing; show the engine's state there.

## Image and video without ComfyUI graphs

### What exists

There are three surfaces, and they disagree:

1. **Python generators** (`media-models/flux_3_klein.py`, `krea_2.py`,
   `qwen_image_21.py`, `qwen_image_21_turbo.py`, `minimax_h3_video.py`). These are
   already the right idea: typed flags in, graph built in code by
   `build_workflow()`, artifact out. No graph editing.
2. **The TUI task form**. For every ComfyUI profile it offers three fields:
   timeout, declared task, and "Workflow graph JSON: (empty)". The user has to
   paste an API-format graph. This is the complex path, and it is the only one the
   workbench offers.
3. **`workflows/*.json`** exports, and the LTX-2.5 and YuE2 manifest entries, which
   say outright "supply API-format workflow JSON" and "edit the graph inputs for
   style, lyrics, seeds and duration".

Other friction:

- `request_bytes`, `check_api`, `check_gpu_exclusive` and the wait loop are copied
  into five files. `configure_api` (router awareness) is in two.
- Flags are inconsistent: FLUX takes a positional prompt and a 300 s timeout, the
  rest take `--prompt`/`--prompt-file` and 3,600 s; `--dry-run` exists only for H3;
  reference-image flags differ per model.
- `media-runtimes.json` has no parameter schema (profiles carry only id, runtime,
  tasks, inputs, script, notes), so the Rust side cannot draw a real form.

### Proposal: recipes

A recipe is a profile plus a declared parameter list plus a way to turn parameters
into a graph. The graph becomes an implementation detail nobody edits.

1. **One client, many builders.** Add `media-models/comfy_client.py` with the
   transport, router awareness, the 246 s cold-start budget, reference staging into
   `vendor/ComfyUI/input/`, the wait loop, artifact download and the exclusive-GPU
   check. Each model module keeps only `PARAMS`, `validate()` and
   `build_workflow()`. One entry point, `media-models/generate.py <profile>`, with
   one flag vocabulary: `--prompt`, `--prompt-file`, `--negative`, `--size` or
   `--aspect`, `--steps`, `--seed`, `--ref`, `--count`, `--preset`, `--out`,
   `--check`, `--dry-run`. The nine-line `generate-*.sh` wrappers can stay as shims.

2. **Declare parameters in the manifest.** Per profile in `media-runtimes.json`:

   ```json
   "params": [
     {"name": "prompt", "kind": "text", "required": true},
     {"name": "size", "kind": "choice", "values": ["768x768", "1024x1024", "1280x720"], "default": "1024x1024"},
     {"name": "steps", "kind": "int", "min": 4, "max": 50, "default": 16},
     {"name": "seed", "kind": "int", "default": 42},
     {"name": "ref", "kind": "path", "repeat": true}
   ],
   "presets": {
     "draft": {"size": "768x768", "steps": 8},
     "final": {"size": "1024x1024", "steps": 16}
   }
   ```

   The task form renders this with the field kinds it already has
   (`TaskFormField`, selectors, text). "Workflow graph JSON" stays as an advanced
   escape hatch, off by default. A test asserts the manifest params match the
   Python `PARAMS`, in the spirit of `check-serve-bench-drift.sh`.

3. **Bindings for models that only have an official graph.** For LTX-2.5 and YuE2
   there is no Python builder, only an exported graph. Do not write one. Keep the
   API-format JSON in `workflows/` and add a small sidecar that maps parameters to
   node inputs:

   ```json
   {"prompt": ["6", "text"], "seed": ["31", "noise_seed"], "frames": ["12", "length"]}
   ```

   The client loads the graph, applies the bindings, validates and submits.
   Onboarding a new model becomes: export one graph from ComfyUI once, write ten
   lines of bindings, add params to the manifest. The autogrow gotcha from
   `AGENTS.md` (dotted ids such as `"images.image_1"`) lives in the binding, once.

4. **Draft, then promote.** Video is where time goes (117 s for an H3 row with a
   7 s boot). Give every video recipe a `draft` preset (the quality notes already
   name one: few steps, about 0.2 MP) and a `final` preset. A job row gets
   `p promote`: rerun the same prompt and seed with `final`. Images get
   `--count N` as a seed sweep.

5. **Queue the whole batch as one call.** `AGENTS.md` is explicit that requesting
   `/comfyui/` evicts the resident language model and that a batch must be one
   blocking call. So the form should collect N jobs (seed sweep, prompt file, draft
   set) and submit them together, and the confirm line should say what it evicts
   and the expected cold start, exactly like the load confirmation does now.

6. **Chain the lanes in swap order.** The useful pipeline is: expand a short prompt
   with the language tier, generate N candidates, grade them with the vision tier
   (`maintenance/grade-images-with-vision-tier.py` exists), keep the best, and
   optionally use it as the first frame for H3 (`--first-frame` exists). Each stage
   is a model swap, so run stages in bulk: all expansions, then all generations,
   then all grading. Three swaps per batch instead of three per image. Adult prompt
   text and grading stay in the sibling checkout per the repo rule; the pipeline
   only needs to accept a prompt file path.

7. **Sidecars and row actions.** Write `<artifact>.json` beside every output:
   profile, params, seed, preset, graph hash, engine build, timings. That makes
   "rerun", "rerun with seed + 1", "promote to final" and "use as video reference"
   one-key actions on a Jobs row, and it makes results reproducible without keeping
   prompts in job history (which the form already avoids).

8. **Preview.** `o` opens the artifact with `xdg-open` locally, or prints and
   copies the path over SSH. Inline preview through the kitty graphics protocol or
   sixel is a nice optional extra where the terminal supports it. Not a blocker.

### Worth a bounded look: an images endpoint instead of graphs

llama-swap's playground has an Images tab, which means the router can front an
OpenAI-style `/v1/images/generations` server. If one of the image models can be
served by such a server (stable-diffusion.cpp's `sd-server` is the usual
candidate), image generation becomes a prompt-only HTTP call with no graph at all,
and likely a much shorter cold start than ComfyUI's 246 s. I have not checked
whether any of the installed checkpoints (the INT8 ConvRot safetensors in
particular) load there, so treat this as a one-evening experiment for a single
model, not a plan. The recipe layer above is engine-neutral either way: a recipe
could target ComfyUI or an images endpoint without the form changing.

## Suggested order

P0, small and independent:

1. Router-offline state in the status bar and pane title (F2).
2. Sidebar: aliases, middle truncation, de-duplicated lanes, resident and recent
   first (F3).
3. Rates from all results files, keyed by router id (F3).
4. One hint line (F7); palette ranking and no Quit on top (F8).
5. Check the 30 s form timeout against the ComfyUI cold start.

P1, the loops:

6. Now view as the default pane (F1).
7. Media recipes: shared client, manifest params, form rendering, draft and final
   presets (media 1, 2, 4, 5).
8. Bench result table, then the gated bench utility (F4).
9. Tune, including the Strata `env` block (F5).

P2:

10. Graph bindings for LTX-2.5 and YuE2, sidecars and row actions, the
    expand-generate-grade chain (media 3, 6, 7).
11. Palette arguments and `l3ms do` (F8).
12. Readiness checklist and alerts as job rows (plan, System monitoring).

Defer: user workspaces and split panes, the crate split, and GPUI. They are the
expensive part of the herdr look and they do not shorten any of the six loops.
Revisit once the tool is open all day.

## Open questions

- Which of the six loops do you run most? That decides whether Tune or Bench goes
  first after the Now view.
- Should the Now view poll `/api/performance`, or keep the local `nvidia-smi` and
  `/proc` readers as primary because they also work when the router is stopped for
  a dark ladder?
- For media, is ComfyUI staying as the single image and video engine, or is an
  images-endpoint experiment welcome?
