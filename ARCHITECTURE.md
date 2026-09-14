# L3MS Architecture

L3MS is a keyboard-first terminal application around editable homelab LLM
configuration and scripts. The Rust port is being developed incrementally so
the operational files remain useful throughout the migration.

## Sources of truth

- `llama-swap.yaml` declares models that can be served.
- `bench-models/bench-*.sh` declares benchmark entry points.
- `model_downloader/models_config.json` declares model downloads.
- The repository is authoritative for code and public documentation.
- Linear issue `CAR-97` is authoritative for the committed Rust-port work.

The terminal application orchestrates these files; it does not replace them
with an internal database or generated UI-only configuration.

## Rust application

The `l3ms` crate builds one binary and keeps the implementation in a small
module tree:

- `src/cli.rs` owns argument parsing, model/script pickers, repository
  discovery, and headless command execution.
- `src/llama_swap.rs` owns the authenticated llama-swap HTTP boundary.
- `src/chat.rs` owns the authenticated Chat client, endpoint probing, bounded
  server-sent-event parsing, and cancellable streaming chat requests.
- `src/commands.rs` owns typed command metadata, contextual help, and palette
  search.
- `src/config_store.rs` owns typed download configuration, normalization,
  validation, atomic writes, and configuration snapshots.
- `src/download_editor.rs` owns download-editor selection, validation, dirty
  state, and snapshot save/reload/restore operations independently of the UI.
- `src/download_ui.rs` owns Download-view focus, text drafts, runtime speed
  controls, version selection, and derived dirty state without depending on
  Ratatui.
- `src/downloader_command.rs` owns portable, shell-free Python interpreter and
  downloader-script argv construction.
- `src/download_preflight.rs` owns strict estimator JSON validation, bounded
  cancellable estimator execution, and platform-aware disk-space probing.
- `src/gguf.rs` owns bounded GGUF v2/v3 metadata parsing and safe directory
  inventory.
- `src/job_history.rs` owns the persisted job lifecycle and safe reconstruction
  of retryable script jobs.
- `src/chat_history.rs` adapts the TUI chat transcript to the compatible
  JSON/Markdown session store.
- `src/script_store.rs` owns safe repository-contained script access,
  command construction, atomic writes, mode preservation, and snapshots.
- `src/script_editor.rs` owns reusable bench and maintenance editor state,
  including dirty tracking and snapshot save/reload/restore operations.
- `src/state_store.rs` owns the bounded legacy-compatible job and chat-session
  file formats.
- `src/telemetry.rs` owns process-tree CPU/RAM and optional NVIDIA memory
  sampling plus disk-free and network byte counters.
- `src/bench_results.rs` owns bounded JSONL/Markdown result discovery,
  deterministic sorting, and metric comparison.
- `src/settings.rs` owns persisted operator settings and validated portable
  profile import/export bundles.
- `src/prompt_store.rs` owns bounded local Chat system-prompt files.
- `src/script_lint.rs` and `src/serve_bench.rs` own optional shellcheck
  diagnostics and read-only serving/bench flag audits.
- `src/text_buffer.rs` owns Unicode-safe editing and cursor movement shared by
  the inline terminal editors.
- `src/app.rs` owns the Ratatui event loop, Workbench and Chat model state,
  request-ID guarded background operations, supervised child processes, and
  the seven top-level views.

The TUI keeps network requests and subprocess output off the rendering thread.
Background delivery is bounded, process lines are size-capped, and each tick
handles a bounded number of events so continuous output cannot starve keyboard
input. On Unix, scripts run in their own process group with stdin detached;
shutdown sends `TERM`, waits briefly, then escalates to `KILL` if required.

## Runtime boundaries

llama-swap remains the serving daemon. The Rust client reads:

- `LLAMA_SWAP_URL`, defaulting to `http://localhost:8080`
- `LLAMA_SWAP_API_KEY`, when bearer authentication is enabled

The client uses `/v1/models`, `/models/load`, and `/models/unload`. Non-success
HTTP responses are errors; they are not recorded as successful model loads.

The existing Python downloader remains a compatibility child process. Rust
constructs argv without a shell and chooses its interpreter from a non-empty
`L3MS_DOWNLOADER_PYTHON`, then the repository's `.venv/bin/python3` (or
`.venv/Scripts/python.exe` on Windows) when it is a file, and finally `python3`
from `PATH`. The next argv element is
`model_downloader/download_hf_model.py`; the environment override is one
executable path or command name, not a shell fragment with flags. The Python
boundary still owns Hugging Face downloads while Rust supervises the child.
Runtime worker precedence is global override, then per-model value, then the
optional slow preset.

Download output is supervised as a bounded stream. The Rust Download view
parses common Hugging Face progress lines (`downloaded/total` and `rate/s`) to
derive an advisory ETA; when a backend does not emit those lines, the strict
preflight total and disk-space report remain available without pretending an
ETA is known.

Chat detection is read-only until an explicit `K` action. A selected PID must
still be present in a fresh `pgrep -fa llama-server` result and its command
basename must be `llama-server` before Rust sends a termination signal.

Before starting a Download job, Rust clones the exact launch argv and appends
`--estimate-json` only to the preflight copy. The Python boundary uses the same
revision and allow/ignore filters with Hugging Face `dry_run` metadata, then
emits one bounded schema-versioned JSON document containing total, cached, and
remaining bytes. Rust validates aggregate counts and byte totals, probes the
target filesystem off the rendering thread, and reports advisory size-versus-
free-space feedback. Estimator failure does not change the legacy launch
behavior: it is logged and the immutable original command can still run. A
pending preflight is single-flight, request-ID guarded, cancellable with `Esc`,
and reaps its child process group on Unix.

## Persistence and safety

Download configuration snapshots remain under
`.toolkit/download_config_versions/`. Script snapshots remain under
`.toolkit/script_versions/`. Context-aware store APIs take the runtime
repository and version roots explicitly, which keeps installed binaries and
alternate checkouts from writing snapshots into the build checkout.

Writes use a temporary file plus rename, snapshot names are unique even during
rapid saves, restore names cannot traverse out of their version directory, and
script paths must resolve inside the selected repository. New Download history
uses a sanitized config-path key plus a stable path hash so distinct config
paths cannot collide; listing and restore also read the former sanitized-only
namespace for compatibility.

Download snapshots are strictly parsed before replacement. A valid restore of
an existing config must first create an undo snapshot containing the exact
displaced bytes; if that snapshot cannot be written, the live config remains
untouched. The same parsed source bytes update the editor after the atomic
restore, avoiding a fallible post-write reload. Reload and restore require a
second activation before discarding dirty persisted or unapplied fields.
Snapshot discovery is secondary: a listing failure is surfaced as a warning
without reclassifying an otherwise successful load, save, or restore. Existing
Unix script permissions are preserved.

The Rust TUI uses bounded, atomic `jobs.json` persistence and compatible
JSON/Markdown chat sessions under `L3MS_DATA_DIR` (defaulting to `~/.l3ms`).
Job startup reconciles stale running entries, the UI keeps the newest 200 rows,
and retry reconstruction is limited to contained bench or maintenance scripts.
Malformed top-level job state is never overwritten until an explicit clear.

## Migration boundary

The Rust binary currently provides the headless llama-swap run/list workflow,
benchmark execution, typed stores, and a functional
seven-view TUI. The TUI
now includes streamed chat with request parameters, an executable searchable
command palette, context-derived help, supervised processes, and live
CPU/RAM/GPU telemetry. Jobs and chat sessions persist through their compatible
legacy formats. The GGUF browser now uses the bounded metadata scanner and
supports recursive inventory, filtering, deterministic sorting, file details,
and per-file parse warnings.

The Rust TUI now wires safe inline editors for bench and maintenance scripts and
an expanding typed Download configuration surface, including model CRUD,
strict load/save/restore, dirty guards, speed controls, disk feedback, and
dedicated process output. Download launches now include bounded asynchronous
cache-aware size/disk preflight, responsive wide/compact/focused-pane layouts,
and verified selected/enabled execution through the real Hugging Face boundary.
Rust now owns the Chat endpoint draft/commit workflow, authenticated
server detection/connect, independent model selection, request-ID guarded
streaming, cancellation, and persistence guards. The Python TUI remains
available as a fallback for any remaining live-only operational gaps and should
only be retired after live llama-swap smoke verification passes. `CAR-97` is
still in progress; this is not a claim of a fully green release matrix.

The headless CLI also exposes a bounded benchmark result browser (`--list
results` and `--compare-results`) and a prompt library listing. The Chat tab's
`l` picker loads Markdown prompts from `L3MS_DATA_DIR/prompts`, while settings
and profile bundles use atomic writes and existing snapshot-aware stores.

## Rust vs. Textual readiness

The legacy Textual application is `l3ms.py` and `l3ms/app.py` on `main`. The
Rust application is the `l3ms` binary on the `CAR-97` branch. The comparison
below records the current replacement decision rather than maintaining a
second implementation backlog.

| Area | Rust | Textual | Current assessment |
| --- | --- | --- | --- |
| Workbench and model load/unload | Authenticated typed client, TUI, and headless CLI | Mature TUI workflow | Rust has the stronger boundary |
| CLI | `--run`, `--bench`, `--list`, `--quickstart`, filters, and extra args | Similar interactive modes | Rust has the stronger packaging |
| Chat | Endpoint draft/commit, authenticated Connect/Detect, independent model selection, streaming, cancellation, parameters, sessions, token feedback | Detect, Connect, endpoint editing, model selection, streaming, cancellation, sessions | Rust now matches the declared feature set; Textual remains the live fallback until smoke coverage is complete |
| Downloads | CRUD, validation, snapshots, restore, preflight, disk feedback, shell-free execution | Broadly complete workflow | Rust has the stronger safety model |
| Bench and Maintenance | Run/stop, UTF-8 editing, reload/restore, snapshots, supervised processes | Run/stop, editing, and snapshots | Rust is the stronger long-term base |
| Jobs | Bounded persistence, stale-run reconciliation, stop/retry, safe reconstruction | Persistence, stop/retry, and history | Rust is ahead on recovery safety |
| GGUF browser | Bounded v2/v3 parsing, recursive inventory, metadata, symlink and budget protection | Similar browsing features | Rust is ahead on malformed-input safety |
| Keyboard and navigation | Command palette, contextual help, keyboard-first focus | Better mouse/widget ergonomics | Rust matches the project direction |
| Automated confidence | Locked toolchain, CI, format/lint gates, and bounded Rust unit coverage | No comparable Textual TUI test suite in `main` | Rust is ahead |
| Runtime boundary | Compiled binary plus the intentional Python downloader boundary | Python/Textual runtime | Rust is lighter after build |

The Rust branch is the preferred application for Workbench, model operations,
Downloads, Bench, Maintenance, Jobs, GGUF browsing, and Chat. Keep Textual as
a fallback until the Rust branch has completed live authenticated Chat smoke
coverage on KPC and the remaining release checklist is closed.

Rust is the better long-term foundation because its process supervision,
persistence, input bounds, cancellation ownership, and automated verification
are stronger and better fit the keyboard-first operating model. Textual
remains the lower-friction fallback during the live-verification period.

Linear issue `CAR-97` contains the authoritative ordered implementation
checklist; this document records the implemented boundary rather than
maintaining a second backlog.
