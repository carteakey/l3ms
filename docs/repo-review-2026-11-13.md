# Repository review, 2026-11-13

General review of the repo outside the TUI. The workbench and media notes are in
[workbench review](workbench-review-2026-11-12.md). Observations and suggestions
only, nothing here is implemented.

## How this was reviewed

Read the root layout, `AGENTS.md`, `TODO.md`, the docs index, CI config and the
tracked file list. Ran `cargo clippy --all-targets` and `cargo test` in a sandbox
with no router, GPU or model files.

- Clippy: clean.
- Rust tests: 585 passed, 2 failed, 4 ignored. The two failures
  (`download_ui::tests::successful_save_...` and `successful_restore_...
  _when_post_write_history_listing_fails`) set a directory to mode `0o300` and
  expect a listing to fail. The sandbox runs as root, which ignores that mode, so
  these are an artifact of the environment, not a regression. They would be more
  robust if they skipped when the effective uid is 0.
- Python suites were not run here.

## Summary

The knowledge in this repo is excellent. Its weakness is where that knowledge
lives: too much is prose that a person or agent has to reread, and too little is
data or code that enforces it.

## What is good

- **Measurement discipline.** Warm-up discard, paired controls, the governor
  finding that invalidated earlier rows, and "measured but not served" for
  `STRATA_ROUTE_PRIOR=1.5`. Numbers come with their conditions.
- **Safety habits.** Snapshots before writes, confirm before evicting a model,
  refusal on dirty vendor trees, incident postmortems.
- **CI and tests.** `rust.yml` runs format, clippy with `-D warnings`, Rust tests,
  `--list stack`, and four Python suites, all offline.

## What I would change

### 1. `AGENTS.md` is doing a database's job

353 lines, with tier specs, sweep results and dated measurements packed into long
paragraphs. It has already drifted once (the `STRATA_GR_NORM_FAST` correction).

Suggestion: move per-tier facts (engine, quant, flags, `env`, measured rates with
date and conditions) into a structured file next to `llama-swap.yaml`. Generate
the tier prose from it, and let the sidebar and dashboard read the same file. Keep
`AGENTS.md` for rules and gotchas, with links out for history.

### 2. Root clutter

- `thread.md`: a 4,840-line pasted agent transcript.
- `mtp_bench.py`, `mtp_bench_gemma.py`, `mtp_bench_qwen.py`, `mtpmeter.py`,
  `sortling.py`, `srv_start.sh`, `preflight-check.sh`: belong in `bench-models/`
  or `maintenance/`.
- `configuration.md`: belongs in `docs/`.
- `windows/`: seven files including a tracked `run.log`.

Suggestion: remove `thread.md`, move the scripts, and either archive `windows/` or
state in the README that it is unmaintained.

### 3. The legacy Python TUI

`l3ms/` and `l3ms.py` are kept "until parity" while the crate is at 1.0.0.
Suggestion: write down what parity is still missing, or delete them. An open-ended
retention rule means they never go.

### 4. Version story

Already filed as [#29](https://github.com/carteakey/l3ms/issues/29): `Cargo.toml`
says 1.0.0, the changelog stops at 0.7.0, and there are no tags. Cheap to fix, and
it matters as soon as anyone else looks at the repo.

### 5. One-host assumptions

232 tracked files mention `kchauhan` paths (many are logs and docs, but configs
and scripts are among them). Fine for a personal stack; it is the main thing
between this and something another person could clone and run.

Suggestion: one `L3MS_MODELS_ROOT`-style setting, used by `llama-swap.yaml`
macros, the downloader config and the media scripts. Leave logs and docs alone.

### 6. Bench sprawl

Of 287 tracked files under `bench-models/`, 208 are logs, and there are 66
per-model scripts that are mostly the same shape.

Suggestion: one parameterised runner plus a small config file per model (model
path, engine, flags, ladder). That also makes the UI's bench view simple to build,
since it would list configs and results instead of script files. Decide separately
whether raw logs belong in git or only the `.jsonl` result rows.

### 7. Docs volume

128 tracked files in `docs/`, including two superseded implementation plans, a
checkpoint and the migration history. The index helps.

Suggestion: split into current truth and `docs/archive/` (how we got here), so a
reader or agent can tell which statements still hold.

## Biggest risk

The repo's centre of gravity is drifting toward Strata tuning notes and away from
the tool. That is fine for a lab notebook. If the goal is a product other people
use, the README should lead with install, pick a model, run, and the lab material
should sit one level down.

## Suggested order

1. Root tidy and `thread.md` removal (no behaviour change).
2. Version, changelog and tag reconciliation (#29).
3. `docs/archive/` split.
4. Decide the fate of the Python TUI.
5. Structured tier data, with `AGENTS.md` trimmed to rules.
6. Parameterised bench runner.
7. Models-root setting.
