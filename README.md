# L3MS

`L3MS` (Local Large Language Model System) is a keyboard-first terminal toolkit for homelab LLM workflows.

Image generation/editing, audio/music, speech, and video tooling now live in the
sibling `l2m2` (Local Latent Media Machine) repository. Vision-capable LLMs,
including Gemma vision, and embeddings remain in L3MS.

## App Description

L3MS is built for developers who want script-first control with better ergonomics:

- Manage model download configs with validation and version history
- Run and bench llama.cpp models from curated script folders
- Browse local GGUF inventories with size, quantization, and metadata details
- Edit run/bench scripts in-place with snapshot restore
- Track live process output and runtime resource usage while a model is running
- Use either a full TUI or interactive CLI commands (`--run`, `--bench`)

## Public Dashboard

The [L3MS profile dashboard](https://l3ms.carteakey.dev) publishes the active
12 GB homelab configurations, measured generation throughput, task-level MTP
comparisons, and the reasons older profiles were retired.

Open any served profile to get a portable `llama-server` command generated
from `llama-swap.yaml`. Public commands replace local model paths with shell
variables and bind to `127.0.0.1` by default.

## Install

The active Rust port uses the pinned toolchain in `rust-toolchain.toml`:

```bash
rustup show
cargo build --release --locked
```

Downloads continue to use the repository's Python compatibility boundary.
Install that dependency into the repository virtual environment so the Rust
TUI can discover it automatically:

```bash
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements-downloader.txt
```

Set `L3MS_DOWNLOADER_PYTHON` to another Python executable when the dependency
lives elsewhere.

## Start TUI

```bash
cargo run --release --locked
# or: ./target/release/l3ms
```

On the KPC test machine, run the deployed checkout with its existing
llama-swap service configuration:

```bash
./maintenance/run-l3ms-kpc.sh
```

The script derives the service port, API key, Rust toolchain environment, and
downloader Python interpreter. Arguments are passed through to the Rust binary,
for example `./maintenance/run-l3ms-kpc.sh --list run`.

Show quick-start instructions without opening TUI:

```bash
cargo run --locked -- --quickstart
```

## Interactive CLI Modes

List llama-swap models and benchmark entry points:

```bash
cargo run --locked -- --list all
```

Inspect benchmark history and compare two result files without opening the
TUI:

```bash
cargo run --locked -- --list results
cargo run --locked -- --compare-results bench-results/left.md bench-results/right.md
```

Saved Chat system prompts live under `~/.l3ms/prompts/` and can be listed with
`--list prompts` (the Chat tab's `l` action opens the same picker). Operator
settings and non-secret profile bundles use `--settings`,
`--export-profile DIRECTORY`, and `--import-profile DIRECTORY`.

Interactively select and load a llama-swap model:

```bash
cargo run --locked -- --run
```

Interactively run a bench script:

```bash
cargo run --locked -- --bench
```

Filter the bench picker and pass extra arguments without reparsing them through
a shell:

```bash
cargo run --locked -- --bench qwen --extra "--ctx-size 32768"
```

## Rust Migration Status

The Rust `0.7.0` foundation now covers the llama-swap run/list CLI, benchmark
execution, typed stores, atomic snapshots, authenticated Chat endpoint
connect/detect, independent Chat model selection, cancellable streaming, runtime
telemetry, and a keyboard-first seven-view Ratatui workbench with a searchable
command palette. Jobs and chat sessions now persist through the compatible
legacy formats. The GGUF view uses bounded metadata parsing with recursive
inventory, filtering, deterministic sorting, detailed metadata, and per-file
warnings.
Bench and maintenance now have inline UTF-8 script editors with dirty guards
and snapshot save/reload/restore. The Download view exposes the legacy JSON
fields, runtime speed controls, model CRUD, strict validation, atomic snapshots,
dedicated output, responsive narrow-terminal rendering, and supervised
selected/enabled launches. Each launch first runs a bounded, cancellable,
cache-aware size and disk-space preflight off the rendering thread. The Python
downloader remains the download implementation behind a portable, shell-free
Rust command boundary.

Run `maintenance/check-serve-bench-drift.sh` after changing serving or bench
flags. It reports drift in `-ngl` and tensor override flags without rewriting
either source file. New bench work can start from
`bench-models/templates/bench-script.sh`; named preset conventions are
documented in `bench-models/presets/README.md`.

The privileged KPC slow-TG comparison is documented in
[`docs/slow-tg-investigation.md`](docs/slow-tg-investigation.md). Platform and
WebAssembly boundaries are recorded in
[`docs/platform-portability.md`](docs/platform-portability.md).

`CAR-97` remains a work in progress. These implemented slices do not yet imply
full legacy parity, complete live smoke coverage, or a fully green verification
matrix.

The Python TUI remains available during the parity period:

```bash
python3 -m pip install -r requirements-tui.txt -r requirements-downloader.txt
python3 l3ms.py
```

The TUI feature and key list below describes the shared target surface. In the
Rust binary, `?` opens contextual help and `Ctrl+P` opens the executable command
palette. See [ARCHITECTURE.md](ARCHITECTURE.md) for the compatibility boundary
and Linear issue `CAR-97` for the authoritative remaining-work checklist.

## TUI Scope

- Workbench shell:
  - persistent command bar with context-specific shortcut hints
  - richer `Ctrl+P` command palette with shortcut column and token filtering
- `Workbench` tab:
  - default first screen for quick llama-swap model loading
  - live model table from the configured llama-swap endpoint
  - fast actions for load, unload, chat, bench, browser, downloads, and jobs
  - delegates model loads to Model Ops so job history and resource telemetry stay consistent
- `Download` tab:
  - config load/save/validate/restore
  - model row add/apply/delete
  - download selected or enabled models
  - asynchronous cache-aware download-size and target-disk preflight
  - responsive wide, compact, and focused-pane layouts
  - repeat-action confirmation before dirty reload or restore
  - restore validates the snapshot and saves displaced config bytes for undo
    before atomic replacement
  - per-config snapshot namespaces in `.toolkit/download_config_versions/`,
    with existing legacy history still visible and restorable
  - snapshot-list failures surface as warnings without converting a completed
    load, save, or restore into a failure
- `Model Ops` tab:
  - llama-swap run mode and bench script mode
  - live run logs + start/stop
  - current running model + resource telemetry (CPU/RAM/GPU when available)
  - script editor and per-script snapshots in `.toolkit/script_versions/` for bench mode
  - paged model tables (`[` / `]`) keep large llama-swap inventories responsive
- `Model Browser` tab:
  - scan any local directory for `.gguf` files
  - inspect size, quantization, params, architecture, and modified timestamp in a table
  - filter/sort results and inspect per-file metadata details
- `Chat` tab:
  - edit and connect an authenticated OpenAI-compatible endpoint
  - detect local llama-server ports and choose a Chat model independently
  - `K` terminates a freshly detected external llama-server only after PID and
    command-line identity checks
  - `l` opens the bounded local system-prompt library
  - stream, stop, clear, save, and restore conversations with stale-result guards
- Additional tabs: `Maintenance`, `Jobs`

Download output parses Hugging Face progress lines when available and displays
an ETA beside the disk-space status; the preflight estimate remains the source
of truth for total and remaining bytes when progress callbacks are unavailable.
Saving a bench or maintenance script runs `shellcheck` when installed and
retains warnings in the activity log without blocking a safe snapshot save.

## Keyboard-first Controls

Global:

- `Ctrl+P`: command palette
- `?`: key binding help
- `F1`: Workbench tab
- `F2`: Model Ops tab
- `F3`: Chat tab
- `F4`: Model Browser tab
- `F5`: Download tab
- `F6`: Jobs tab
- `F7`: Maintenance tab
- `Alt+1..Alt+7`: tab fallback when F-keys are unreliable
- `Alt+←` / `Alt+→`: previous / next tab

Workbench:

- `Ctrl+R` / `Enter`: load selected llama-swap model
- `Ctrl+S`: unload selected llama-swap model
- `Ctrl+F`: focus model filter
- `Ctrl+J`: focus model table
- `Ctrl+L`: clear workbench log
- `F3`: chat with the loaded model endpoint
- `F2`: open full Model Ops

Download (active only on Download tab):

- `Alt+T`: focus models table
- `Alt+I`: focus model editor
- `Alt+O`: load config
- `Alt+W`: save config
- `Alt+V`: validate config
- `Alt+R`: choose or restore a config snapshot
- `Alt+N`: add model
- `Alt+A`: apply model edit
- `Alt+K`: delete selected model
- `Alt+D`: download selected model
- `Alt+E`: download enabled models
- `Alt+Y`: clear download log
- `Esc`: cancel a pending download preflight

Model Ops (active only on Model Ops tab):

- `Ctrl+F` / `/` (Bench): focus the model or benchmark-script filter
- `Ctrl+J`: focus model/script table
- `Ctrl+U`: focus detail/script editor
- `Ctrl+M`: toggle run/bench mode
- `Ctrl+R`: load selected llama-swap model or run selected bench script
- `Ctrl+S`: unload selected llama-swap model or stop running bench
- `Alt+P`: save edited bench script snapshot
- `Alt+O`: reload the selected bench script (`Alt+O` twice discards dirty edits)
- `Alt+V`: choose or restore a bench script snapshot
- `Ctrl+L`: clear run log

Chat (active only on Chat tab):

- `e`: edit the endpoint draft
- `Ctrl+G`: connect to the draft endpoint and refresh its model list
- `Ctrl+B`: detect a local llama-server and connect to it
- `r`: refresh models on the committed Chat endpoint
- `↑` / `↓` or `k` / `j`: choose the Chat model independently of Workbench
- `i` / `Enter`: focus the message composer
- `Esc`: stop an active response, or leave the current Chat editor
- `Ctrl+X` / `x`: clear the transcript and stop an active response
- `Alt+S`: save the current session
- `o`: browse saved sessions

Maintenance (active only on Maintenance tab):

- `Ctrl+U`: focus or leave the script editor
- `Ctrl+R` / `Enter`: run selected maintenance script
- `Ctrl+S`: stop the active script
- `Alt+P`: save edited script with a snapshot
- `Alt+O`: reload selected script (`Alt+O` twice discards dirty edits)
- `Alt+V`: choose or restore a script snapshot
- `Ctrl+L`: clear activity output

Model Browser (active only on Model Browser tab):

- `Alt+R`: scan selected GGUF directory
- `Alt+G`: focus directory path input
- `Alt+J`: focus GGUF table

## Project Layout

- `model_downloader/`: Hugging Face downloader + model config
- `llama-swap.yaml`: single source of truth for servable models (see `docs/llama-swap-runbook.md`)
- `bench-models/`: editable `bench-*.sh` benchmark entry points and helpers
- `maintenance/`: system/build scripts
- `maintenance/systemd/`: user service units (including `llama-swap.service` and `nanobot-gateway.service`)
- `src/`: Rust CLI, llama-swap client, stores, process supervisor, and Ratatui app
- `Cargo.toml` / `Cargo.lock`: Rust package and locked dependency graph
- `requirements-downloader.txt`: Python Hugging Face downloader dependency
- `l3ms/`: legacy Python TUI and stores retained during parity work
- `l3ms.py`: legacy Python launcher
- `ARCHITECTURE.md`: source-of-truth boundaries and Rust migration status
- `docs/llama-swap-runbook.md`: install, start/stop, curl, add-a-model
- `docs/model-onboarding-playbook.md`: end-to-end checklist for adding new model families

## Serving

Models are served by [llama-swap](https://github.com/mostlygeek/llama-swap) on
a single OpenAI-compatible endpoint (`http://<host>:8080`). The daemon
hot-swaps models on demand and exposes every entry in `llama-swap.yaml`
under `/v1/models`. See `docs/llama-swap-runbook.md`.

The same endpoint serves on-demand embeddings through
`nomic-embed-text-v1.5`. It starts on the first `/v1/embeddings` request and
unloads after five idle minutes; see the runbook for the model path, task
prefixes, and verification request.

## Downloader CLI (direct)

Run with config file (downloads all enabled models):

```bash
./model_downloader/download_hf_model.py --config model_downloader/models_config.json
```

Pull updates for already-downloaded models (skips models with no local files):

```bash
./model_downloader/download_hf_model.py --config model_downloader/models_config.json --update
```

Single model with pattern filter:

```bash
./model_downloader/download_hf_model.py --repo-id Qwen/Qwen3-32B-GGUF --allow-patterns "*Q6_K*"
```

Throttle concurrency (useful on metered connections):

```bash
./model_downloader/download_hf_model.py --config model_downloader/models_config.json --slow
./model_downloader/download_hf_model.py --config model_downloader/models_config.json --max-workers 2
```

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to JSON config file |
| `--repo-id` | `-r` | Single repo to download |
| `--allow-patterns` | `-a` | File glob patterns to include |
| `--ignore-patterns` | `-i` | File glob patterns to exclude |
| `--local-dir` | `-d` | Override local destination directory |
| `--revision` | | Specific branch/tag/commit to pin |
| `--update` | `-u` | Sync updates for models already on disk (skips new ones) |
| `--force-download` | | Re-download all files even if already present |
| `--slow` | | Throttle to `max_workers=4` |
| `--max-workers` | | Explicit worker count |
| `--base-models-dir` | | Override base directory for auto-organized downloads |

When the Rust TUI starts the downloader, it constructs argv directly without a
shell and selects the interpreter in this order:

1. Non-empty `L3MS_DOWNLOADER_PYTHON`.
2. The repository virtual environment (`.venv/bin/python3`, or
   `.venv/Scripts/python.exe` on Windows) when that path is a file.
3. `python3` from `PATH`.

The interpreter is followed by
`model_downloader/download_hf_model.py` and its arguments.
`L3MS_DOWNLOADER_PYTHON` must therefore be one executable path or command name;
it is not parsed as a shell fragment and cannot contain additional flags.

For direct CLI use, the script's portable `#!/usr/bin/env python3` shebang uses
the first `python3` on `PATH`. Activate the intended virtual environment first,
or invoke its Python executable explicitly.

## Dashboard Development

The served-profile data and commands in `docs/generated-models.js` are
generated from `llama-swap.yaml`. Presentation-only values such as measured
throughput and benchmark comparisons live in `docs/dashboard-meta.json`. Every
published local profile receives the same evidence fields; missing historical
values are rendered as `not recorded` rather than inferred from another run.

Reviewed external results live in `docs/community-runs.json` and follow
`docs/community-runs.schema.json`. They render in a separate unranked view and
are never merged into the local RTX 4070 profile array. See
`docs/community-runs.md` for the submission format.

Regenerate after changing either source:

```bash
python3 docs/generate_dashboard_data.py
python3 -m unittest docs/test_generate_dashboard_data.py
```

Preview locally:

```bash
python3 -m http.server 8080 -d docs
```

Do not edit `docs/generated-models.js` by hand.

## Versioning

This project uses semantic versioning. See [CHANGELOG.md](CHANGELOG.md).

## Roadmap Note

The Rust port is active under `CAR-97`. Python remains as the deliberate
Hugging Face downloader compatibility boundary, while the Textual TUI remains
available as a fallback until live Rust Chat smoke verification and release
closure are complete.
