# L3MS Codebase Review — 2026-09-28

**Date**: 2026-09-28
**Crate**: `l3ms` (`v1.0.0`, per `Cargo.toml:3`)
**Baseline commit**: `68fd7bc` (branch `main`, the deployed checkout)
**Scope**: Full repository review — Rust crate, CI, tracked files, docs, and operations surface.
**Companion**: `docs/monorepo-consolidation-plan.md`; migration state is tracked in `docs/migration/status.md` on `feature/l3ms-monorepo`.

## Scope and status — read this first

**This review targets `main` @ `68fd7bc` — the deployed checkout. It does not describe the migration branch.**

The consolidation is already underway on a second worktree: `feature/l3ms-monorepo` @ `5e405af` in `/home/kchauhan/repos/l3ms-monorepo`, 9 commits ahead of `main`. Its `docs/migration/status.md` records Phases 0–3 in progress and Phase 4 in development, with Linear tracking CAR-215 → CAR-221.

Three consequences:

1. **Phase 0 protections already exist and are verified.** `l3ms-history.bundle`, `l3ds-history.bundle`, and `l2m2-selected-private.bundle` (all `git bundle verify` clean), plus `l3ms-working.patch` and `l3ms-untracked.tar.gz`, live outside Git under `/home/kchauhan/repos/l3ms-migration-backup.Z0zIPT/` and `/home/kchauhan/repos/l2m2-migration-backup-20260928/`. The mutable-state ownership table required by Phase 0 step 6 is already written. **Do not re-run Phase 0 and do not add a second baseline record here** — the plan states "Re-record status at execution time; earlier counts are not migration inputs," so a duplicate on `main` would be worse than none. Phase 0's gate is satisfied; the remaining work is the *pending live checks* listed in that status file (real L3DS inference, L3MS router cutover, representative model generation).

2. **The repo-hygiene items in this document must not be applied to the migration branch.** H2, H6, H7, H8 and the H1 log untracking touch root files, and the Phase 1 gate — already passed at `ee27a64`/`4deece4` — requires that "the existing L3MS root diff is limited to navigation/instructions and the two imported trees." Re-touching root files on `feature/l3ms-monorepo` reopens a closed gate and pollutes a reviewed import. These belong on a **separate branch off `main`**, or after Phase 10 retires the old paths. The "Suggested execution order" table below predates the branch being cut and is superseded by this section.

3. **`main` is not CI-green today.** `cargo fmt --all -- --check` fails there on real diffs in `config_store.rs:7,1188` and `fs_util.rs:111`. The migration branch formats clean. This is pre-existing drift in the deployed checkout, reported here rather than fixed. The "CI is real" entry under *What is already good* describes the workflow's configuration and is not a claim that `main` currently passes.

### What the in-flight Phase 4 work does and does not cover

Verified against `feature/l3ms-monorepo` (uncommitted worktree state):

| This review | Status on the migration branch |
| --- | --- |
| Phase 4 read-only three-lane catalog | **In progress** — `src/control/catalog.rs` (784 lines) plus `mod.rs` exist but are untracked. `runtime.rs`, `inference.rs`, `jobs.rs`, `config.rs` do not exist yet. |
| B3 (blocking fsync on render thread), R2 (3 snapshot pipelines), R6 (2 validators) | **Partially addressed** — uncommitted work in `state_store.rs` (+359), `settings.rs` (+383), and `job_history.rs` (+401) is Phase 4 steps 2–3 (layered settings, cross-toolkit job metadata). Not yet the `control/jobs.rs` seam or a snapshot-pipeline consolidation. |
| B1 (9 no-op Param Builder commands) | **Still present** — `app.rs:2309` on the branch. |
| B5 (`swap_yaml` + `serve_bench` unreachable) | **Still present** — still only in `lib.rs:24,27`, no `app.rs` caller. |
| R1 (5 `atomic_write` clones) | **Still present** — all five survive on the branch. Note `state_store.rs` and `settings.rs` have shifted to `:969` and `:575` respectively there, so R1's line references below are `main`-only. |

R4 (`app.rs` is 7,824 lines) and the B2 event-queue finding should be re-checked against the branch before acting on the `main`-based line numbers.

---

## Monorepo framing
> **Superseded in part.** The Phase 1 integration branch is already cut and its gate has passed,
> so point 1 below no longer describes an available window, and the "before Phase 1" rows of the
> ordering table are void. Points 2 and 3 still hold. Retained because they explain *why* the
> hygiene items must stay off the migration branch.


Three consequences of the in-progress consolidation change *when* and *where* these fixes belong.

1. **Do the repository-hygiene work on `main`, before the Phase 1 integration branch is cut.** The plan's Phase 1 gate requires that "the existing L3MS root diff is limited to navigation/instructions and the two imported trees" (`monorepo-consolidation-plan.md:360-361`). Deleting `thread.md`, `configuration.md`, or `docs/vibe_configuration.md` from the integration branch would break that gate. Phase 0 is the window: commit hygiene to `main`, then cut the branch from a clean root.

2. **Hygiene fixes get more expensive after Phase 1.** Phase 1 rewrites the sibling histories with `git filter-repo --to-subdirectory-filter` and merges with unrelated histories. L3MS's own history is not rewritten, so the 2 MB of tracked raw bench logs, the 273 KB transcript, and the `sk-` literal in `configuration.md` stay in history regardless. Phase 1 step 2 explicitly requires auditing imported history for "secrets and bulk artifacts" and warns that "checking only the final tree cannot detect deleted historical content" (`:348-349`). Fixing the tracked files on `main` now gives the audit a clean final tree; it does not remove the historical blobs, which should be a separate, explicit decision rather than a side effect of the import.

3. **Several findings are already stated acceptance criteria in the plan and currently fail.** They are not new scope — they are unmet gates:
   - Phase 2 step 4 requires validating "ignore behavior from the monorepo root" (`:382-384`). Finding H1 is an ignore-behavior bug that would be carried to the new root.
   - Phase 2 step 2 requires auditing "path assumptions in scripts, docs, tests, and setup instructions" (`:375-377`). Finding H4 has already done 44 files of that audit.
   - Phase 4 step 4 says "extract shared Rust code only when it has multiple callers" (`:461-462`), and the architecture section says "do not create a second implementation next to the existing one" (`:116-118`). Findings R1, R2, and R3 are existing violations of a rule the plan has already adopted.
   - Phase 10 step 2 requires inventorying "remaining users of sibling paths: wrappers, aliases, service units, timers, skills, documentation, and external callers" (`:615-617`). Finding H5 means the legacy launch path being retired is already non-functional, which makes that inventory wrong on contact.
   - Phase 8's gate requires verifying "the existing `l3ms`, `l2m2`, and `l3ds` commands" (`:564`). Finding R3 means `--list` and the TUI currently label the same script differently.

Corrected execution order (revised after finding the migration branch):

| Where | Findings | Why |
| --- | --- | --- |
| **Already done — do not repeat** | Phase 0 baseline, bundles, mutable-state table | Verified in `docs/migration/status.md` and `/home/kchauhan/repos/l3ms-migration-backup.*`. Re-recording would contradict plan `:301-303` |
| **Pending live checks, not code work** | Real L3DS inference, L3MS router cutover, representative model generation | Already logged as pending in `docs/migration/status.md`; these close Phases 3–4 gates and need host access, not patches |
| Separate branch off `main` (never the migration branch) | H1, H2, H3, H4, H6, H7, H8, H10 | Root-file deletions and the CI shell gate. The Phase 1 gate already passed, so these must not land on `feature/l3ms-monorepo` |
| `main`, as its own commit | B6 | Version-ledger honesty. Cheap, and Phase 9’s documentation pass should not inherit the inconsistency |
| With the Phase 4 extraction (in flight) | B2, B3, R1, R2, R3, R5, R6 | The `control/jobs.rs`, `control/config.rs`, `control/inference.rs` seams. Plan `:116-118` forbids a second implementation beside the existing one, so collapse during extraction |
| With the Phase 5 adapters | B4, B5 | `swap_yaml` + `serve_bench` + `llama_swap` are the llama-swap adapter’s existing ancestors. B5 must be decided before they are extracted, or dead code moves into `control/` |
| Phase 8 / 9, or explicitly reject | R4, H5, H10 (dashboard part) | `app.rs` splitting is what the plan’s `ui/`/`lanes/` split does anyway; the dashboard repoint is gated by `dashboard.yml` |

Decisions still owed, unchanged: the five in [Open decisions](#open-decisions-for-the-operator).

---

## What is already good — preserve

Stated first so cleanup work does not regress it.

- **Test suite**: 259 inline `#[test]` functions, 1,139 assertions, no integration-test directory. Best-covered modules are `gguf.rs` (22 tests / 843 prod lines), `state_store.rs` (22 / 856), `config_store.rs` (17 / 692).
- **CI is real**: `.github/workflows/rust.yml:21-25` runs `cargo fmt --all -- --check`, `cargo clippy --all-targets --locked -- -D warnings`, and `cargo test --all-targets --locked`. `--locked` is enforced. `permissions: contents: read`. `docs/dashboard.yml:22` gates generated-artifact drift with `git diff --exit-code`.
- **No bare `.unwrap()` in production**: 869 total call sites, of which only 6 are outside `#[cfg(test)]` — all invariant `expect()`s with justification comments (`app.rs:1046`, `download_editor.rs:244`, `download_preflight.rs:487,491,513`, `download_ui.rs:678`).
- **No shell interpolation**: 20 `Command::new` sites, all argv-based. The single `sh -c` (`app.rs:7716`) is a test literal with no interpolation.
- **Script execution is contained**: `script_store.rs:198-216` canonicalizes, enforces a repo-root prefix and an extension allow-list, with tests for both `../outside.sh` and symlink escape (`:473`, `:489-491`).
- **PID safety**: `pid > 1` guard plus a fresh `pgrep` re-verification before every signal (`chat.rs:245-253`, `swap_yaml.rs:278`).
- **No secrets in the tracked config**: `llama-swap.yaml:12-13` uses `${env.LLAMA_SWAP_API_KEY}`; every read is `env::var`; `HeaderValue::from_str` (`llama_swap.rs:263-273`) blocks CRLF header injection, with a test.
- **No TLS verification disabled**: `rustls-tls` with defaults (`Cargo.toml:17`); zero hits for `danger_accept_invalid*`. `normalize_base_url` (`llama_swap.rs:231-253`) is strict and tested.
- **No blocking-on-tokio violation**: `reqwest::blocking` is only ever constructed inside `thread::spawn` (`app.rs:614,639,676,730`); the async path builds its own async client.
- `model_downloader/` is the correct pattern for a Python boundary: dependency-injected interpreter (`downloader_command.rs:16-35`), three test files, and the only Python with CI coverage.
- `codex-skills/.../audit_models.py` is read-only (no `unlink`/`shutil`/`subprocess`) — which is why the CHANGELOG can record "7 orphan files deleted" as a deliberate human decision.
- `.toolkit/` is correctly gitignored runtime state.

---

## P0 — Bugs

### B1. Nine Param Builder commands are advertised but silently do nothing

`commands.rs:480-552` declares all nine `ParamBuilder*` commands plus `OpenParamBuilder` with `implemented: true`, and `CommandSpec::is_visible()` (`commands.rs:407-409`) is `implemented && enabled` — but `enabled` is hard-wired to `implemented` by the `command!` macro (`commands.rs:425`), so all 96 commands are always visible. `app.rs:2301-2309` then matches those nine ids to `=> {}`.

Ctrl+P therefore offers "Open Param Builder tab" and eight ParamBuilder actions, and selecting any of them does nothing, with no error and no log line. `CommandContext::TABS` (`commands.rs:26-34`) has 8 entries including `ParamBuilder`, while `enum Tab` (`app.rs:98-106`) has 7 and `TAB_NAMES` (`app.rs:71-79`) has 7 — the mismatch is locked in by a passing test at `commands.rs:1474-1481`.

Fix: set `implemented: false` for the nine entries (the catalog already supports it) or wire the tab. Whichever is chosen, delete the `enabled` field or pass a runtime predicate to `visible_commands`, which the module doc at `commands.rs:4-5` already describes as the design.

### B2. A full event queue aborts a live chat stream

`app.rs:422` creates `mpsc::sync_channel(512)`; `drain_background_events` (`app.rs:759-760`) drains at most 128 per 100 ms tick, a ~1,280 events/s ceiling. The chat delta closure is `sender.send(BackgroundEvent::ChatDelta { … }).is_ok()` (`app.rs:3368-3375`), and `chat.rs:502-504` treats a `false` from `on_delta` as `anyhow::bail!("chat stream cancelled")`.

So under load — precisely the memory-pressured conditions this tool is built for — a full queue kills an in-flight response mid-answer and reports it as a user-initiated cancellation. The same bounded channel makes `stream_reader` (`app.rs:5537-5544`) block, which back-pressures the child's stdout pipe.

Fix: `try_send` for `ChatDelta` and drop deltas rather than aborting; drain until empty or until a time budget, not a fixed count.

### B3. Blocking subprocess and fsync'd writes on the render thread

`app.rs:1900` runs `lint_shell_script` (`script_lint.rs:46-97`) — `Command::new("shellcheck").output()`, blocking, no timeout, no progress — directly from the save path. `persist_jobs` (`app.rs:3895-3899`) performs a synchronous fsync'd write and is called from `app.rs:3936` inside `drain_background_events`, i.e. on the render thread, on **every child process exit**. Further blocking sites: `job_history.rs:315-330` (`Command::new("date")` per job start), and two fsync'd config saves at `app.rs:2779`/`2813`.

There are 15 bare `Command::new(..).output()/.status()` sites with no timeout, versus exactly one careful implementation: `download_preflight::run_capped_command` (`download_preflight.rs:452-562`) with a poll loop, cancel flag, output cap, and process-group kill.

Related: 16 `thread::spawn` sites with no watchdog. The `*_pending` flags (`loading_models`, `browser_scanning`, `chat_connection_pending`, `telemetry_pending`, `chat_session_pending`, `chat_pending`) are cleared only in the matching event arm, so a panicking worker leaves the TUI in a "…" state with no recovery path for five of the six.

**Monorepo mapping**: this is the `control/jobs.rs` seam. Plan line 133 — "The existing job supervisor is the shared seam for process lifecycle" — and Phase 4 step 3, "Reuse the existing job supervisor and persistent history." Fix the blocking persistence when extracting, not after.

### B4. Unbounded HTTP response bodies in the llama-swap client

`llama_swap.rs:152`, `:173` (60 s load timeout), and `:220` all call `response.text()` with no size cap. `chat.rs:158-167` does the equivalent correctly with `take(MAX_MODEL_BODY_BYTES + 1)` plus `ensure!`. A hostile or misconfigured `LLAMA_SWAP_URL` can exhaust memory on any of the three llama-swap paths.

Fix: share one `read_bounded(response, max)` helper across both clients. This is also the natural place to land B5's endpoint-unification.

### B5. 2,576 lines (15% of the crate) are unreachable from any entry point

`param_builder.rs` (1,113), `param_registry.rs` (783), `swap_yaml.rs` (557), `serve_bench.rs` (123), and `boot_anim.rs`'s `play` are referenced only by `lib.rs` and their own tests. `app.rs:20-49` imports none of them.

Consequences worth stating plainly:

- `SwapModel.disabled` and `SwapModel.size_bytes` (`llama_swap.rs:88-91`) are hard-coded `false`/`None` at `llama_swap.rs:303-304` and only ever set by `swap_yaml::merge_into:99-107`, which is itself test-only. **The model-enable/disable-via-yaml plus SIGHUP flow documented in `AGENTS.md:28-30` is not reachable from the TUI.** `draw_models_table:4578-4615` renders only `Model` and `State`.
- `serve_bench.rs` is the flag-drift audit. The invariant in `AGENTS.md:16-18` — "When serving and benching flags drift, update them together" — has **no executable enforcement path**, and `maintenance/check-serve-bench-drift.sh` is never run in CI (see H3).
- 36 of 259 tests assert on code no user can reach.
- `gguf::group_shard_sets` (`gguf.rs:581`) has no production caller — only its own 5 tests. The shard-set aggregation the 2026-09-03 review documented as a Model Browser feature is no longer reachable from the TUI, and `ShardSet` appears zero times in `app.rs`. Either the feature was dropped or it moved somewhere unreviewed; worth confirming which, because "incomplete shard sets" was a reported capability.

Fix: wire the Param Builder tab, or gate the four modules behind a feature. Note the plan's constraint at `:116-118` — "do not create a second implementation next to the existing one" — applies to the decision either way.

**Monorepo mapping**: Phase 5 implements "adapters for llama-swap" (`:475-477`) with status/health/activation. `swap_yaml` + `serve_bench` are that adapter's existing ancestors. Deciding wire-vs-drop *before* Phase 4 avoids extracting dead code into `control/`. Record the unverified-invariant status in the Phase 0 evidence table, since Phase 0 step 4 is where baseline checks are captured.

### B6. Version ledger is inconsistent

`Cargo.toml:3` says `1.0.0`; `AGENTS.md:164` says the Rust port "starts at version `0.7.0`"; `CHANGELOG.md` has no `## [0.8.0]` and no `## [1.0.0]` section (`CHANGELOG.md:118` describes the Winamp skin as 0.8.0 while still inside `[Unreleased]`); there are zero git tags across 149 commits. `CHANGELOG.md` also has 5 duplicated `section :: heading` pairs at lines 12/120, 19/131, 328/457, 336/374/555, and 543/560, which Keep-a-Changelog parsers will concatenate.

**Monorepo mapping**: Phase 0 step 1 records "the three current commit IDs, branch names, status output." A correct version baseline is part of that. Phase 9's documentation pass will rewrite README and site descriptions; fixing the ledger now avoids a second pass.

---

## P1 — Repository hygiene

### H1 (HIGH). 82 raw `llama-server` logs tracked, contradicting the stated policy

`.gitignore:13-16` says bench raw logs are gitignored and that structured results are tracked, then negates itself:

```
bench-models/logs/*
!bench-models/logs/results/
!bench-models/logs/results/**      # <-- re-includes every raw log under results/
```

`bench-models/logs/.gitignore:1-4` repeats the same contradiction independently. Result: 82 tracked `.log` files under `bench-models/logs/results/**` (raw server stdout, ~2.0 MB total), plus `windows/run.log` — a mojibake'd PowerShell capture (`file` reports Hewlett-Packard Graphics Language; `â†’` for `→`) committed 2026-08-15 and never touched.

The pollution is active, not historical. `git status` is currently dirty because a live bench run overwrote four tracked logs: `fit-test-16k-fit512.log`, `fit-test-16k-ncmoe44.log`, `fit-test-16k-ncmoe45.log`, `fit-test-16k-ncmoe46.log`.

There are also 56 untracked root-level `*.log` files, which the root-scoped `/*.log` rule was written for; the pattern recurs because the rule was never generalized.

Fix: ignore `bench-models/logs/results/**` and re-include only `*.jsonl` / `*.json` / `*.tsv`; `git rm --cached` the 82 logs and `windows/run.log`; delete the redundant nested `.gitignore`.

**Monorepo mapping**: this is exactly what Phase 2 step 4 asks to be validated (`:382-384`). Fix it before that check runs, or the check fails on the new root.

### H2 (HIGH). `thread.md` is a 273 KB pasted agent transcript

`thread.md` is a Zed agent conversation export, not documentation: `thread.md:1-3` are `## User` / `## Assistant` turn headers with a `zed:///agent/thread/…` link; it contains 33 turn pairs, 41 `**Tool Call:` blocks with embedded JSON, verbatim `ls`/glob tool output (`thread.md:24-27`, `:213-225`), an entire deleted shell script inlined as `#` comments (`:426-497`), and a 4 KB single-line JSON blob containing a superseded run script pointing at the dead `/mnt/lab/models` path (`:243`).

It is actively misleading. It documents the `run-models/` architecture that `llama-swap.yaml` replaced (`docs/llama-swap-runbook.md:4`), and `l3ms.py:14` still globs that same removed directory (see H5). Added 2026-03-18, never touched. The only reference in the repo is `CHANGELOG.md:128`, which explicitly labels it "historical record."

Fix: `git rm --cached thread.md` and move it to the transcript archive. Per the plan's Phase 2 step 2 guidance, leave historical records *labeled* as historical rather than rewriting their paths.

### H3 (MED). 81 tracked shell scripts, zero CI lint

`git ls-files '*.sh' | wc -l` → 81 (61 in `bench-models/`, 20+ in `maintenance/`). None are linted, syntax-checked, or tested. `shellcheck` is not installed on the host, and `.github/` contains no shellcheck, bats, or matrix step. `maintenance/mem-latency-probe.c` and `maintenance/gguf_tensor_types.py` are likewise never built or compiled.

This matters because `AGENTS.md:16-18` makes `bench-models/*.sh` load-bearing — "They are editable text, not hard-parameterized UI forms" — and because `maintenance/check-serve-bench-drift.sh` exists specifically to catch serving/benching drift, yet nothing runs it. B5 shows the in-TUI drift check is also unreachable, so nothing enforces the invariant today.

Fix: add a CI step running `shellcheck -S warning -e SC1091` and `bash -n` over `git ls-files '*.sh'`, invoke `check-serve-bench-drift.sh` as a non-zero-exit gate, and add `gcc -fsyntax-only` / `python3 -m py_compile` for the two unbuilt helpers.

**Monorepo mapping**: Phase 2 step 3 requires "child-focused CI jobs or a matrix with explicit working directories" and Phase 0 step 4 requires L3MS's existing checks to pass. Adding the shell gate now means the monorepo inherits a repo where the bench scripts are actually verified.

### H4 (MED). 44 tracked files reference the dead `/mnt/lab` mount

`ls -d /mnt/lab/models` → does not exist. `CHANGELOG.md:122` documents that this exact path "fell back from the dead `/mnt/lab/models` mount to `/home/kchauhan/models`" — but only in the audit script. Still referencing it: 31 files in `bench-models/` (env-overridable `MODEL="${MODEL:-/mnt/lab/models/…}"`), 4 in `.claude/commands/`, 2 in `docs/`, plus `thread.md`, `preflight-check.sh`, `mtp_bench_qwen.py`, `model_downloader/`, `maintenance/`, `CHANGELOG.md`, and — contradicting the changelog — `codex-skills/model-inventory-manager/scripts/audit_models.py`.

Fix: `${L3MS_MODELS:-/home/kchauhan/models}` throughout the 31 bench scripts and the 4 command files; re-verify `audit_models.py`.

**Monorepo mapping**: this is Phase 2 step 2's audit, already half-done. Complete it as part of that step rather than separately.

### H5 (MED). The legacy Python TUI is 4,647 duplicated lines and is currently broken

`l3ms/` + `l3ms.py` duplicate `src/config_store.rs`, `src/script_store.rs`, and `src/llama_swap.rs` almost function-for-function — with regressions:

- `l3ms.py:14` sets `RUN_SCRIPT_GLOB = "run-models/run-llama-cpp-*.sh"`. `run-models/` no longer exists, so the Run panel globs a missing directory.
- `l3ms/llama_swap.py:69-76` POSTs to `/models/load`; `src/llama_swap.rs:166` uses `GET /upstream/{model}/load`.
- `l3ms/llama_swap.py:33-39` returns the raw status string; `src/llama_swap.rs:320-345` normalizes 18 aliases.
- `l3ms/llama_swap.py:89-99` `probe()` sends no auth and swallows all errors.

Zero CI coverage (CI's Python steps target only `model_downloader`). `README.md:149-158` presents it as a co-equal install path, while the feature list at `README.md:164-292` describes Rust-only surface.

`AGENTS.md:165` states the retention rule: "Retain the legacy Python TUI and downloader until their Rust replacement or compatibility boundary has explicit parity coverage." `model_downloader/` meets that bar. `l3ms/` does not, and has not for some time.

Fix: either delete `l3ms/` + `l3ms.py` and drop the README/AGENTS/ARCHITECTURE mentions, or mark `l3ms.py:14` explicitly deprecated and add `pip install -r requirements-tui.txt` to CI so it stops rotting silently. Note that `requirements-tui.txt:2` has `httpx>=0.27` with no upper bound, so a major bump can break it with no signal.

**Monorepo mapping**: Phase 10 step 2 inventories "remaining users of sibling paths: wrappers, aliases, service units, timers, skills, documentation, and external callers." Decide the disposition of `l3ms/` *before* that inventory, or the inventory is wrong on contact. Also note plan `:72-74`: `l3ms.py` is a root file, and Phase 1's gate restricts root diffs to navigation and the two imported trees — so this decision also belongs on `main`, before the branch is cut.

### H6 (MED). `configuration.md` is a stale third-party copy that the runbook defers to

`configuration.md:1` is literally `# config.yaml` — upstream llama-swap’s `config.example.yaml` pasted in whole, which `configuration.md:89` itself admits is superseded upstream. Its defaults diverge from the tracked config in the two places that matter operationally: `globalTTL: 0` (`:172`) vs `600` (`llama-swap.yaml:24`), and `logToStdout: "proxy"` (`:136`) vs `both` (`llama-swap.yaml:17`). A reader following the in-repo copy would conclude idle models are never unloaded — the opposite of the deployed behaviour documented in `AGENTS.md:28-30`. (`startPort` agrees: `configuration.md:153` and `llama-swap.yaml:20` are both `10001`.)

Meanwhile `llama-swap.yaml:6` points readers at it as the in-repo schema reference. It also embeds two `sk-`-prefixed literals (`:209-211`), one 64 characters of high entropy, which will hard-fail any `gitleaks`/`trufflehog` pre-commit hook or GitHub push-protection.

Fix: delete it and point `llama-swap.yaml:6` and `docs/llama-swap-runbook.md` at the pinned `$schema` URL already at `llama-swap.yaml:1`.

**Monorepo mapping**: the `sk-` literal matters beyond this file's usefulness — Phase 1 step 2 requires a secrets audit of imported history (`:348-349`). Removing it now clears the final tree; the historical copy is a separate, explicit decision.

### H7 (MED). `docs/vibe_configuration.md` documents a different product

`docs/vibe_configuration.md:2-4` carries `id: vibe_configuration` / `slug: vibe_configuration` frontmatter; `:9` describes `./.vibe/config.toml` and `~/.vibe/config.toml`; `:26-30` links to the Mistral console for a Vibe/Devstral API key. It uses Docusaurus `<Tabs>`/`<TabItem>` MDX components while the site publishes via Jekyll (`docs/.nojekyll`, `docs/CNAME` → `l3ms.carteakey.dev`). Grep for L3MS-domain terms in it yields one incidental hit. Added 2026-03-18, never touched.

Fix: `git rm docs/vibe_configuration.md`.

### H8 (MED). 888 lines plan a document that is gitignored and was never written

`docs/implementation_plan_v1.md:6` and `v2.md:3` both target `docs/local-llm-optimization.md`, which `.gitignore:30` excludes and which does not exist. `v1` is superseded by `v2` in its own title, and `v2` was never executed. Together they are 888 lines. `docs/tui-color-theme-draft.patch` is a stale `git diff` against `src/app.rs` for theme work already merged as `src/theme.rs` — a rebase hazard.

Fix: delete both plans (or fold the ideas into `TODO.md` per `AGENTS.md:175`) and delete the stale patch.

### H9 (LOW). 62 `*.bak*` snapshots, 1.9 MB, no retention policy

All gitignored via `.gitignore:48`, so not a repo-size problem — an operational one. 39 at root (28 `llama-swap.yaml.bak-*`, 9 `CHANGELOG.md.bak-*`, 2 `AGENTS.md.bak-*`), 12 in `docs/` (9 `qwen38-flash-next-internal.md.bak-*`, up to 57 KB each), 11 in `bench-models/`. `AGENTS.md:24` mandates snapshots with no retention window.

Fix: add `maintenance/prune-snapshots.sh` keeping the N most recent per target, and document the window in `AGENTS.md`. Phase 0 step 3 inventories ignored state that must survive; snapshot sprawl belongs in that table.

### H10 (LOW). Dead scripts, duplicated skill artifact, and a `sudo`-through-`eval` path

- `srv_start.sh` (20 lines) hardcodes a build tree (`srv_start.sh:2-3`) no longer referenced by `llama-swap.yaml`, expands `$PORT`/`$TAG`/`$LL`/`$MODEL` unquoted, and `kill -9`s whatever `pgrep -f` matches with no PID-1 guard. It also writes `srv_${TAG}.log` into the repo root, which is the reason `/*.log` exists in `.gitignore` at all. Superseded by `llama-swap.yaml`; compare `src/chat.rs:245-253`, which does all four guards.
- `preflight-check.sh:20-26` implements `apply_fix` as `eval "$cmd"`, and the call at `:218` interpolates a `$GPU_PCI` value derived from `nvidia-smi` output into `sudo tee $RUNTIME_PM`. All other call sites pass literals. `:212` also silently falls back to a hardcoded `0000:01:00.0` if the PCI parse yields nothing, which would target the wrong device. It is referenced from three docs and overlaps heavily with `bench-models/bench-env.sh`.
- `mtp_bench.py` and `mtpmeter.py` (8 lines) are fully orphaned — zero references. `mtp_bench_gemma.py` and `mtp_bench_qwen.py` are referenced as benchmark commands by the *published* dashboard (`docs/dashboard-meta.json:17,25,33`, `docs/generated-models.js:23,64,79`) and hardcode the dead `/mnt/lab/models` path (`mtp_bench_qwen.py:9`).
- `codex-skills/model-inventory-manager.skill` is a Zip archive committed alongside the byte-identical `codex-skills/model-inventory-manager/` directory it was built from, and the zip is the stale copy. `SKILL.md:15-30` still reconciles against `model_downloader/models_config.json`, superseded by `disabled: true` in `llama-swap.yaml` per `CHANGELOG.md:116`.
- Two tracked `.ps1` files have CRLF endings, and there is no `.gitattributes` or `.editorconfig`. `windows/run.log` is mojibake with no encoding declaration.

**Monorepo mapping**: the dashboard benchmark-command pointers are read by `docs/generate_dashboard_data.py` and gated by `dashboard.yml`, so repointing them to `bench-models/*.sh` is a Phase 9 concern (`:583`) with a CI drift gate — do it there, and delete the four `.py` files in the same change so the generator cannot regress.

### H11 (LOW). `AGENTS.md` documents 7 of 31 `src/` modules and has a stale CLI list

`AGENTS.md:138-148` lists `app.rs`, `cli.rs`, `param_registry.rs`, `param_builder.rs`, `llama_swap.rs`, `config_store.rs`, `script_store.rs`. Twenty-four modules are undocumented, including `commands.rs` (the 96-command dispatch table), `state_store.rs`, `settings.rs`, `telemetry.rs`, `gguf.rs`, `fs_util.rs`, and `theme.rs`. `AGENTS.md:141` lists CLI flags as `--run`, `--bench`, `--list`, but `cli.rs:27` also accepts `quickstart`, `settings`, `export_profile`, `import_profile`, and `compare_results`. `AGENTS.md:164` says 0.7.0 (see B6).

Fix: expand the Structure table, correct the subcommand list and version. This is part of the Phase 9 documentation pass but is small enough to do now.

---

## P2 — Duplication

### R1 (HIGH). Five `atomic_write` implementations

| Location | Lines | Temp filename | Cleans up on error | Retry loop |
| --- | --- | --- | --- | --- |
| `fs_util.rs:24-81` (canonical) | 58 | `.{name}.l3ms-tmp-{pid}-{nonce}-{ctr}-{try}` | yes | 100 attempts |
| `state_store.rs:726-790` | 65 | **identical string** | yes | yes |
| `swap_yaml.rs:196-243` | 48 | close variant | yes | 100 |
| `settings.rs:311-332` | 22 | `.{name}.{ctr}.tmp` | **no — leaks the temp file** | no |
| `prompt_store.rs:117-131` | 15 | `.prompt-{ctr}.tmp` | **no — leaks the temp file** | no |

`state_store.rs:726` is a line-for-line clone of `fs_util.rs:24` including the identical temp filename, so the two can collide in the same directory. `settings.rs:311` and `prompt_store.rs:117` propagate `?` out of the write/rename without `remove_file`, littering the data dir on any I/O error.

`swap_yaml.rs` also reimplements `fs_util` wholesale: `format_utc_timestamp:245-265` is the same Hinnant civil-from-days algorithm as `format_utc_seconds:144-164` with different output, and `snapshot:168-193` duplicates `write_unique_snapshot:93-129` with a different naming scheme. It also triplicates its own entry-scan loop (`entry_span:55-65`, `parse_entries:78-94`, `set_disabled:122-135`).

Fix: delete the four clones, route every writer through `fs_util::atomic_write`, and extract one `entry_bounds(lines, id) -> Option<(usize, usize)>`.

### R2 (HIGH). Three snapshot/restore pipelines, five containment idioms

`config_store.rs` and `script_store.rs` each implement resolve+containment, version directories, version listing, note sanitization, and snapshot-before-write, with *divergent* semantics: `config_store.rs:366-406` filters `.json` in `list_versions_in` while `script_store.rs:230-263` does not; `sanitize_note` has a different signature in each (`config_store.rs:670` vs `script_store.rs:376`); `config_store.rs:335` snapshots only `if target.is_file()` while `script_store.rs:305-322` always snapshots, including an empty one for a new file. `state_store.rs:620-624` uses a fourth containment idiom (`canonicalize` both sides, requires existence) and `job_history.rs:290` a fifth.

Fix: one `versioned_save` / `versioned_restore` in `fs_util`, parameterized by the containment resolver.

### R3 (HIGH). `cli.rs` and `app.rs` reimplement five helpers and disagree

| Function | `cli.rs` | `app.rs` | Divergence |
| --- | --- | --- | --- |
| `is_repository_root` | `:564-568` | `:5467-5469` | CLI requires `Cargo.toml.is_file()`; the TUI does not. A checkout without `Cargo.toml` is a valid TUI root and an invalid CLI root. |
| `repository_root` | `:530-555` (exe → cwd → manifest) | `:5432-5452` (cwd → exe → manifest) | opposite search order, different error text |
| `pretty_name` | `:331-339` (strips `bench-ik-llama-cpp-`, `bench-llama-cpp-`, `bench-`) | `script_store.rs:141-157` (also strips `run-llama-cpp-`) | `run-llama-cpp-*.sh` gets **different labels** in `--list` vs the TUI |
| script collection | `:292-294` → `script_store` | `:5471-5487` own `read_dir` impl | two implementations |
| `parse_extra_args` | `:493-498` | `script_store.rs:98-104` | **verbatim duplicate**, both `shell_words::split` |
| `command_for_script` | `:500-508` (hardcodes `bash`) | `script_store.rs:75-93` (`.sh`/`.ps1`/`.bat`/`.cmd`) | the CLI cannot run `.ps1` |

`CommandSpec` is also the name of two unrelated types (`cli.rs:94-98` program+args; `commands.rs:377-395` id/label/shortcut/context).

Fix: delete the `cli.rs` copies and import from `script_store`; rename one `CommandSpec`.

**Monorepo mapping**: this is a direct Phase 8 gate risk (`:564`, "Verify the existing `l3ms`, `l2m2`, and `l3ds` commands"). Standalone CLI parity means one script-name → one label, one repo-root rule.

### R4 (MED). `app.rs` is 7,824 lines

One `impl App` block spans `app.rs:415-5367` — **4,953 lines, 151 methods** — plus 49 free functions and a 96-field struct that mixes UI state, rendering, process supervision, and telemetry. Largest functions: `drain_background_events:758-1111` (355), `execute_command:2076-2330` (256), `draw_download:4846-5077` (233).

Near-duplicate detection found **103 repeated 6-line production blocks**, concentrated in: 7 copies of one download error handler (`app.rs:2710,2743,2767,2793,2878,3594,3600`); 5 copies of `self.status = …; self.push_log(…)` (`app.rs:842,909,940,957,974`); 3 near-identical copies of the chat connect/detect/refresh prologue (`app.rs:622-663,665-693,716-747`, ~85% identical); 11 hand-rolled `if !self.download_config_is_usable() { return; }` guards. `draw_download` also carries two parallel renderer trees (compact at `:5732-5986`, full at `:5924-6061`, ~250 lines).

Fix: one `download_action(label, f)` wrapper handling the guard and Ok/Err formatting; one `chat_connect_task` helper; `app/{mod,draw,keys,process}.rs`; `ProcessSupervisor` extracted from `App`.

**Monorepo mapping**: the plan already prescribes this split at `:99-108` (`ui/`, `control/`, `lanes/`). Doing it now would be throwaway work; the repository-hygiene (H) and duplication (R1–R3, R5) items are the parts that survive the move.

### R5 (LOW). Two HTTP clients, two SSE parsers, two tokenizers, four formatters

- **Two blocking clients.** `SwapClient` (`llama_swap.rs:103-224`) and `ChatClient` (`chat.rs:100-197`) duplicate `from_env`/`new`/`base_url`/`auth_headers`/`normalize_base_url`/`parse_models`, and `endpoint_url` is duplicated verbatim at `llama_swap.rs:255-261` and `chat.rs:618-622`. They differ where it matters: `list_models` is bounded in `ChatClient` and not in `SwapClient` (B4). `ChatClient` also builds a *second* `reqwest::Client` inside the async path (`chat.rs:387-393`), so the async stream never reuses the blocking connection pool.
- **SSE parsing written twice.** `consume_sse_with_cancellation:543-605` (63 lines, blocking) and `stream_completion_async:376-506` (91 lines, async) both implement the line cap, `data:` stripping, `[DONE]`, per-line JSON, and completion-token extraction. Six duplicated literals confirm it: `"chat stream cancelled"` ×6, `"/v1/chat/completions"` ×2, `MAX_SSE_LINE_BYTES` message ×2, `"/choices/0/delta/content"` ×2. `chat.rs:535-541` even keeps a `#[cfg(test)]`-only shim so only the *blocking* path has test coverage.
- **Two shell tokenizers.** `param_builder.rs:692-722` hand-rolls a quote-preserving splitter; `script_store.rs:102` and `cli.rs:497` use `shell_words`. They have different quoting semantics for the same input, so `-m` path extraction and `--extra` parsing can disagree.
- **Four byte formatters, four tilde expanders, six clock formatters**: `app.rs:6211` vs six inline percent computations (`app.rs:4098-4100,4128-4134,4296-4306,4326-4341,4352-4362`) vs `telemetry.rs:79` vs `swap_yaml.rs:104`; `app.rs:6348-6358` (`HOME`) vs `download_ui.rs:698-712` (`HOME` or `USERPROFILE`); `fs_util.rs:144`, `swap_yaml.rs:245`, `job_history.rs:304,333`, `app.rs:6138,6259`.

Fix: one `read_bounded`, one `endpoint_url`, one `parse_sse_line`; `shell_words::split` in `parse_cmd_block`; a shared `fmt` module and `fs_util::expand_tilde`.

### R6 (LOW). Two download-config validators, 36 unreferenced `pub fn`s

`config_store.rs:171-186` (`validate_config`, typed) runs on save; `config_store.rs:190-245` (`validate_json_config`, raw JSON) runs on load. They share duplicated message strings, and the typed path never reports the pattern-array errors the raw path checks.

Separately, 36 `pub fn`s are never referenced outside their own module and tests — they escape `dead_code` because `lib.rs` re-exports every module publicly. `job_history.rs:341 persist_with_context` is the clearest: `app.rs:3895` does the same thing inline. `script_store.rs:106,162,166,170,174,178,188` are seven env-defaulting wrappers whose `*_in` variants are the real API.

---

## Test coverage gaps

259 tests is a genuine strength, but the distribution is uneven and the untested areas are the risky ones.

| Logic | Location | Status |
| --- | --- | --- |
| Process supervision | `app.rs:3653-3782` (131), `:3784-3832` (50) | **Zero tests.** Process-group setup, the 100 ms poll loop, the `try_wait` error path, and the TERM→250 ms→KILL escalation are unverified. |
| Event-loop draining | `app.rs:758-1111` (355, 12 event variants) | No direct test. Stale-event rejection is spot-checked by 3 tests. |
| `shellcheck` subprocess | `script_lint.rs:46-97` | **Untested** — the 1 test in the module covers `LintReport::summary` only. |
| llama-swap HTTP calls | `llama_swap.rs:161,179,184` | **Untested** — only `parse_models`, `normalize_base_url`, `auth_headers`, and `response_summary` are covered. |
| Draw functions | 5 of 7 tabs | Render tests exist for `Download` and `Workbench` only. The two largest untested draw functions, `draw_browser:4725` (121 lines) and `draw_chat:4619` (106), are the two with the most conditional layout math. |
| `stream_reader` | `app.rs:5506-5563` | 16 KB line cap and UTF-8 tail handling untested; the nearby tests exercise the helpers, not the reader. |
| Full-channel degradation | `app.rs:422,5537` | No test that a saturated queue degrades gracefully — this is B2. |
| `gguf.rs` invariant | `gguf.rs:677` `unreachable!` | Depends on `is_capture_key:705-710` and `capture_value`'s arms `658-677` agreeing. They do, but no test asserts it, and the two lists are maintained separately. |
| Main loop | `app.rs:578-593` | Poll budget, `KeyEventKind::Press` filter, `should_quit` unverified. |

Coverage is genuinely good in `gguf.rs` (22), `state_store.rs` (22), `config_store.rs` (17), `download_ui.rs` (20), `download_preflight.rs` (19), `text_buffer.rs` (12), `script_editor.rs` (12), `download_editor.rs` (11), `script_store.rs` (10), `swap_yaml.rs` (11) — though 36 of the 259 tests, in `param_builder`/`param_registry`/`swap_yaml`/`serve_bench`/`boot_anim`, cover unreachable code per B5.

Thinnest relative to size: `commands.rs` (1,429 prod lines / 6 tests, 868 of which are the static table) and `script_lint.rs` (120 / 1).

---

## Prioritized summary

| # | Sev | Finding | Monorepo phase |
| --- | --- | --- | --- |
| B1 | HIGH | 9 Param Builder commands no-op but advertised in the palette | Now (pre-Phase 8 parity) |
| B2 | HIGH | Saturated event queue aborts live chat as "cancelled" | Phase 4 / `control/inference.rs` |
| B3 | HIGH | `shellcheck` + fsync'd writes on the render thread; 6 flags with no watchdog | Phase 4 / `control/jobs.rs` |
| B4 | HIGH | Unbounded `response.text()` ×3 in the llama-swap client | Phase 5 adapter |
| B5 | HIGH | 2,576 lines unreachable; `swap_yaml` drift audit has no executable path | Decide before Phase 4 |
| B6 | MED | Version ledger inconsistent; 0 tags; 5 duplicate CHANGELOG headings | Now, then Phase 9 docs pass |
| H1 | HIGH | 82 raw logs tracked; `.gitignore` negates its own rule | Now, before Phase 2 step 4 |
| H2 | HIGH | `thread.md` = 273 KB pasted agent transcript, stale, orphaned | Now, before Phase 1 |
| H7 | MED | `docs/vibe_configuration.md` = another product's docs | Now, before Phase 1 |
| H5 | MED | Legacy Python TUI broken (`run-models/` gone) + regressed API | Decide before Phase 10 inventory |
| H6 | MED | `configuration.md` stale upstream copy + `sk-` literal; runbook defers to it | Now, before Phase 1 secret audit |
| H3 | MED | 81 shell scripts, zero CI lint; drift check never run | Now, before Phase 2 CI |
| H4 | MED | 44 files reference dead `/mnt/lab` | Now, as part of Phase 2 step 2 |
| R1 | HIGH | 5 `atomic_write` clones; 2 leak temp files | With Phase 4 extraction |
| R3 | HIGH | `cli.rs`/`app.rs` helpers diverge → CLI/TUI label mismatch | Before Phase 8 parity gate |
| R2 | HIGH | 3 snapshot pipelines, 5 containment idioms, divergent semantics | With Phase 4 extraction |
| R4 | MED | `app.rs` 7,824 lines, one 4,953-line `impl`, 96 fields | Defer to Phase 8 `ui/` split |
| R5 | MED | 2 HTTP clients, 2 SSE parsers, 2 tokenizers | Phase 4 / `crates/stack-control` |
| R6 | LOW | 2 validators; 36 dead `pub fn`s | Phase 4 |
| H8–H11 | LOW | Superseded plans, snapshot sprawl, dead scripts, stale `AGENTS.md` | Now / Phase 9 |

## Open decisions for the operator

These are not code changes; they need an answer before the corresponding phase.

1. **Param Builder: wire or delete?** 1,896 lines across `param_builder.rs` + `param_registry.rs` are reachable only from tests. AGENTS.md lists the Param Builder as the curated `llama-server` flag table and the CHANGELOG records it as shipped. But the tab is not implemented, so either the catalog is `implemented: false` (B1) or the tab gets built.
2. **`swap_yaml` + `serve_bench`: wire or drop?** B5 means the documented SIGHUP toggle and drift audit are not executable. The plan's Phase 5 builds a llama-swap adapter with status/health/activation — that adapter is where this logic belongs if it is kept.
3. **`l3ms/` Python TUI: delete or maintain?** It is broken today and has no CI. `AGENTS.md:165` conditions retention on "explicit parity coverage," which it does not have, whereas `model_downloader/` does.
4. **Snapshot retention window** for `*.bak*` (H9) — `AGENTS.md:24` mandates creation but not pruning.
5. **Historical blobs** — removing `configuration.md`'s `sk-` literal and the tracked logs from the final tree does not remove them from history. Decide separately whether Phase 1's import needs a history-rewrite pass beyond `toolkits/`, given the plan's own warning at `:348-349`.

## Related documents

- `docs/monorepo-consolidation-plan.md` — the consolidation phases this review is sequenced against.
- `docs/code-review-2026-09-03.md` — the previous review. Four of its five findings are confirmed fixed: `swap_yaml::snapshot` now uses a pure-Rust timestamp (`swap_yaml.rs:174,245`) instead of spawning `/bin/date`; `reload_router` has the `pid > 1` filter (`swap_yaml.rs:278`); `print_quickstart` numbering is now sequential 1–7 (`cli.rs:177`); and `swap_yaml::atomic_write` exists (`swap_yaml.rs:196`) for the `llama-swap.yaml` write path. The fifth — double `group_shard_sets` cloning in `draw_browser` — is moot, because `visible_browser_sets` and `ShardSet` are no longer referenced in `app.rs` at all (see B5). That refactor is itself worth confirming, since it silently changed a documented Model Browser capability.
- `ARCHITECTURE.md`, `AGENTS.md`, `docs/bench-runbook.md`.
