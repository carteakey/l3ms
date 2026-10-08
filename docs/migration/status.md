# L3MS monorepo migration status

Working branch: `feature/l3ms-monorepo` in the isolated checkout
`/home/kchauhan/repos/l3ms-monorepo`.

Plan: [monorepo-consolidation-plan.md](../monorepo-consolidation-plan.md).
This record tracks implementation evidence. A phase is complete only after its
gate in the plan is met; a passed offline test does not stand in for a live
runtime check.

## Current resumed checkpoint — 2026-10-07

Original source HEAD moved from `5792522` to `ed7538a`; the migration branch
remains separate, with discovery/forms/supervisor/source-sync committed at
`2bec24c` and the subsequent embedding/health/drain slice verified. Original
dirty files are preserved. Committed serving YAML is synchronized
without a live reload. See [source reconciliation](serving-baseline.md) for
imported benchmark provenance and missing harnesses. Dashboard regeneration,
nine Python data-contract tests and JavaScript DOM fixtures pass; browser/live
acceptance and publication remain gated. A stale coordinator can authorize only
exact-generation stop-only cleanup, not reuse or inference admission. This
checkpoint is not a full-port completion claim.

The single-supervisor model-free live probe passed on 2026-10-07: named FD
handoff, actual helper/backend namespace equality and host separation,
coordinator-only Unix transport, one native proof GET, exact-generation stop,
process birth-identity disappearance and empty/removed cgroup. Existing live
services and GPU state were unchanged. See [measured launch gate](private-engine-probe.md).
Real native health, model/TTS execution and coordinated GPU switching remain open.

Combined verification after the embedding/health/drain slice:
**688 passed, 1 ignored**, strict all-target Clippy, formatting and whitespace
checks pass. Typed embedding request/vector/usage contracts, one-shot native
adapter, transactional paged results and keyboard forms now pass fixtures;
archived profiles and general managed dispatch remain disabled. Admission close
tokens are core-instance/generation-bound and targeted drain reservations cannot
select another runtime's pending work. Interrupted untracked history remains a
conservative drain blocker. Owned-engine health checks bind fixed native routes
to the exact supervisor peer and owner, not GPU readiness or release.

The owner approved isolated Nagi dependencies/weights and bounded live service
maintenance/restoration on 2026-10-07. Pinned dependencies are installed. The
first staged installation exposed stale activation/console-wrapper paths after
rename; supported relocatable venv creation and offline exact-pin reinstallation
repaired the installed environment. Direct Python, activation and `hf --help`
now work. Minimal revision/checksum-pinned model acquisition completed; six
offline acquisition safety tests pass. Real offline CPU Nagi execution and
the bounded ten-iteration native benchmark passed (P50 1.61 s, P95 1.64 s,
zero option-order flips), without claiming routing accuracy or GPU performance.
The real CPU foreground-coordinator probe also passed: one submission, matched
caller ID, native typed output, a unique durable successful terminal row and
verified daemon/process-group cleanup. Eight offline harness regressions pass.
See [Nagi evidence](nagi-acceptance.md). Explicit owned-engine stack-root
environment handoff and its actual-child fixtures also pass. No
existing service was stopped in this continuation; nothing was pushed/published.
The [managed foreground integration contract](managed-session-contract.md)
records the concrete remaining orchestration and ingress gates.

Linear tracking: [CAR-215](https://linear.app/carteakey/issue/CAR-215/consolidate-l3ms-l2m2-and-l3ds-into-the-local-3-mode-model-stack)
is the parent; [CAR-216](https://linear.app/carteakey/issue/CAR-216/import-l2m2-and-l3ds-into-the-l3ms-monorepo-with-verified-standalone)
covers Phases 0–3, CAR-217 Phases 4–5, CAR-218 Phases 6–7, CAR-219 Phase 8,
CAR-220 Phase 9, and CAR-221 Phase 10.

| Phase | State | Evidence / remaining gate |
| --- | --- | --- |
| 0. Baseline and protection | In progress | Original L3MS `68fd7bc48f803aecb5ae5d25a3f30967b46da3eb`; L2M2 `39e4d40a4e5ed993e99aa4ed70c9d183873a90ba`; L3DS `4716ddf3508d224645416f3ba70d5e34c138cedd`. Source inventories and backups in progress. |
| 1. Folder/history import | In progress | L3DS five-commit and L2M2 three-commit histories imported under `toolkits/`; source/WIP audit, private recovery, and README wordmark done. Final tree/secret/public-path review pending. |
| 2. Independent operation | In progress | Imported child builds, tests, profile lists, and explicit old-root lists pass; CI added locally, pre-push validation pending. |
| 3. Runtime cutover | In progress | Bounded L2M2 GPU-mode pilot, real branch L3MS language request, and a Qwen Image 2.1 branch-script generation passed; original ComfyUI service was restored after language use. L3DS now fails closed instead of synthesizing answers, but real decision weights and full deployed-path reconciliation remain pending. |
| 4. Shared contracts | In development | Read-only catalog and offline `l3ms --list stack`, layered settings, existing-store job metadata, and typed lane request/result validation are on the branch. Adapter contracts and full compatibility/live release gates remain; release stays gated by Phase 3. |
| 5. Runtime adapters | In development | Concrete bounded language execution, ComfyUI job handling, CPU-local decision supervision, Breeze speech execution/artifact validation, and inert generic media recipes pass fixtures. Private language/Comfy Unix HTTP transport also passes native fixtures. General lifecycle/admission, GPU release, real TTS/decision pilots, and user-facing switching remain pending. |
| 6. Local coordinator | In development | Durable queue, ownership locks, v1 control/v2 native result pages, one-shot CLI Submit and opt-in foreground CPU-local decision dispatch pass offline fixtures. Current CUDA profiles remain gated. General managed-engine/media dispatch, installation, server-pushed streaming and real cross-client acceptance remain pending. |
| 7. HTTP ingress | Pending | Gateway to the same coordinator. |
| 8. Unified workbench | In development | Three-lane Stack Catalog shares settings and has opt-in strict-file submit/observe/cancel with retained IDs and private native output. Full task-form parity and live managed routing remain pending. |
| 9. Public site | In development | Local static site now has separate language, media/TTS, and decision views generated from native manifests; six generator/site checks pass. Visual/browser review and all publication gates remain pending. |
| 10. Final acceptance | Pending | Live gates and old-path retirement review. |

## Current safeguards

- The original `~/repos/l3ms` checkout remains on `main` with its pre-existing
  tracked and untracked changes. The migration branch lives in a separate Git
  worktree; bounded pilots have been restored, and no deployed runtime path or
  permanent service has been switched.
- L2M2 and L3DS originals remain in place. Their source and ignored runtime
  state are being classified before history import.
- The source tree and plan are being reviewed for paths, secrets, and large
  generated files before the first push. The current
  [publication audit](publication-audit.md) is **NO-GO** for a public push:
  retained imported history still exposes a Funnel endpoint and host-specific
  paths. Keep this branch local until the exact outgoing history is sanitized
  and re-audited.

## Mutable runtime state (not Git migration inputs)

| State | Current location / writer | Pilot strategy and rollback |
| --- | --- | --- |
| L3MS vendor, weights, `.toolkit`, user state | Original `~/repos/l3ms` and external model roots; no active llama-swap service at baseline | Keep root deployment paths unchanged; reconcile dirty config before any deployment. Restore the saved config/service revision if the pilot fails. |
| ComfyUI vendor, user data, outputs, and model symlinks | Original `~/repos/l2m2/vendor/ComfyUI` and `.toolkit`; the active `l2m2-comfyui.service` is the writer | Snapshot mutable user/workflow state, verify symlink targets, and reconnect the imported child only in an exclusive pilot window. Stop old writer before starting new; preserve pilot writes and return the unit to the original checkout on rollback. |
| L2M2 Python environments | Original `.venv-comfy/` (6.5 GiB) and `.venv-comfy-mcp/` (228 MiB); no Git owner | Rebuild or reconnect by a verified path at the new child; do not copy venvs blindly because interpreter paths are absolute. Keep originals until acceptance. |
| L3DS Python environment and model cache | Original `.venv/` (5.1 GiB) and shared host Hugging Face cache (36 MiB, no snapshots) | Rebuild only after resolving dependency conflicts; leave the original venv/cache intact. No local model inference acceptance until a real model is available. |

## Evidence log

| Date | Check | Result |
| --- | --- | --- |
| 2026-09-28 | Created isolated L3MS worktree from `68fd7bc` | `feature/l3ms-monorepo`; deployed checkout untouched. |
| 2026-09-28 | Copied the user's revised plan into the branch | SHA-256 matched the source plan before the README-art amendment. |
| 2026-09-28 | L3MS baseline `cargo test --all-targets --locked` in original checkout | Passed: 258 tests, 1 ignored. |
| 2026-09-28 | L3MS baseline lint/Python checks | `cargo clippy --all-targets --locked -- -D warnings` passed; 14 downloader unit tests passed. `cargo fmt --all -- --check` failed on pre-existing formatting drift in clean L3MS source files. |
| 2026-09-28 | Branch formatting recovery | Formatted five L3MS source files (18 insertions, 17 deletions); root `cargo fmt --all -- --check` now passes on the branch. Original checkout remains unchanged. |
| 2026-09-28 | Read-only live host baseline | `l2m2-comfyui.service` active from the original L2M2 checkout in GPU mode; `http://127.0.0.1:8188/system_stats` responds. `llama-swap.service` inactive; ports 8080/8001 do not respond. RTX 4070 reports 12,282 MiB total and 362 MiB used. No service was switched. |
| 2026-09-28 | L3MS recovery snapshot | Verified Git history bundle, tracked WIP patch, and archive of 12 untracked files under `/home/kchauhan/repos/l3ms-migration-backup.Z0zIPT/`. Ignored runtimes and weights remain in the original checkout. |
| 2026-09-28 | Dashboard data baseline repair | Three generator unit tests pass. Regeneration initially differed from the checked-in `docs/generated-models.js` in four command examples' reasoning-effort settings; this was pre-existing source/payload drift, not caused by the monorepo import. The branch now commits the deterministic regenerated payload to keep existing dashboard CI green. |
| 2026-09-28 | L3DS import | Audited disposable five-commit history merged as `ee27a64`; 15 tracked paths and blob IDs match source under `toolkits/l3ds/`. Full-history high-signal pattern and size audit found no hits; dedicated secret scanner remains a pre-push safeguard. |
| 2026-09-28 | L3DS recovery bundle | Original complete history bundled and verified at `/home/kchauhan/repos/l3ms-migration-backup.Z0zIPT/l3ds-history.bundle`; source checkout remains clean. |
| 2026-09-28 | L3DS child operation | `cargo build --locked`, `cargo test --locked` (0 tests), `l3ds --list` (6 profiles), and `l3ds --root /home/kchauhan/repos/l3ds --list` passed from the imported child; child target output is ignored. No decision inference was claimed. |
| 2026-09-28 | Root regression after first child import | `cargo test --all-targets --locked` passed: 258 passed, 1 ignored. |
| 2026-09-28 | Root lint after first child import | `cargo clippy --all-targets --locked -- -D warnings` passed. |
| 2026-09-28 | L3DS child path follow-through | Documented child-directory commands and compile-time default root; added two offline tests for the default root and `--root` override. Child fmt, offline test (2 passed), Clippy, build, and `--list` pass. |
| 2026-09-28 | L2M2 import | Audited public-safe three-commit history merged as `4deece4`; 87 source/evidence files under `toolkits/l2m2/`. A third-party transcript and a host-specific Funnel note were excluded from every imported commit and retained in the verified private WIP backup at `/home/kchauhan/repos/l2m2-migration-backup-20260928/l2m2-selected-private.bundle`. No weights/bulk binaries in imported history. |
| 2026-09-28 | L2M2 child operation | `cargo fmt --check`, Clippy, Rust tests (4 passed), release build, 7-profile list, Python syntax, 14 downloader tests, 9 media-model tests, shell syntax, and explicit old-root list passed. No inference or runtime switch was attempted. |
| 2026-09-28 | Model ignore safeguard | Imported child `.gitignore` now ignores GGUF/safetensors and other `models/` contents while allowing only direct `*-files.txt` inventory files; verified with `git check-ignore --no-index`. |
| 2026-09-28 | L2M2 live pilot and rollback | After user approval, confirmed ComfyUI queue empty and llama-swap inactive; stopped original transient unit, archived 81 `.toolkit` entries to a verified mode-600 backup at `/home/kchauhan/repos/l2m2-migration-backup-20260928/l2m2-toolkits-prepilot.tar.gz`, and started a separate `l2m2-comfyui-monorepo-pilot` unit from the branch with ignored symlinks to original vendor, venv, and `.toolkit`. ComfyUI 0.36.0 served `127.0.0.1:8188` in GPU mode, exposed 11 saved workflows, and completed a 64×64 `EmptyImage` → `SaveImage` job (`65b9ccfe-6ce0-4ac5-acdb-6f8f497eeaaa`), producing `l3ms-monorepo-pilot_00001_.png` in the normal media output directory. This verifies service-path wiring, not a heavyweight model workflow. Pilot queue was empty before stop. Original `l2m2-comfyui.service` was restarted and is active/healthy with its original user-directory path; pilot unit is inactive. |
| 2026-09-28 | Isolated llama-swap probe | A loopback-only router with the branch config parsed and listened on port 18080, but its declared preload failed because the branch worktree has no `vendor/llama.cpp` (the deployed root still owns ignored vendor state). The probe was shut down; ComfyUI remained active. Branch L3MS `--quickstart` and `--list bench` passed; `--list run` still requires a live router. No language inference is claimed. |
| 2026-09-28 | Phase 4 offline development | Read-only three-lane manifest catalog, layered stack settings in the existing settings store, and cross-toolkit metadata in the existing job-history store are being implemented/tested. No new inference or service action is exposed by these contracts. |
| 2026-09-29 | Phase 4 offline catalog and CLI | `l3ms --list stack` lists 21 lane-qualified profiles from the three native manifests without contacting llama-swap; each is explicitly `manifest-only`, not installed/healthy. It works from another working directory with `L3MS_ROOT` set. Root `cargo test --all-targets --locked` passed 284 tests (1 ignored); root fmt, Clippy, and dashboard generator checks passed. |
| 2026-09-29 | Phase 4 settings and jobs | Existing settings store now has versioned optional stack preferences, declared path anchoring, built-in → host → user → environment → CLI precedence, snapshots on writes, and portable profile exports that omit machine roots/policy. Existing job history now supports queued cross-toolkit metadata, explicit pre-dispatch persistence, transitions/timestamps/artifacts, and no-replay interruption on restart while preserving legacy row shape. Targeted tests pass; final full-suite check pending after integration. |
| 2026-09-29 | Phase 9 site development | Added native-manifest-derived media/TTS and decision catalog payloads, separate site tabs, Breeze's four TTS modes, hosted/local decision labels, and explicit absence of measured results. Existing language ranking remains separate. Six Python tests, JavaScript syntax, and whitespace checks pass. No browser surface was available for visual review; no site was published. |
| 2026-09-29 | Branch language runtime pilot and restoration | With the approved maintenance window and an empty ComfyUI queue, stopped the original transient ComfyUI unit; linked the branch's ignored `vendor` path to the original verified builds; started branch-configured llama-swap on loopback `127.0.0.1:18080`. The preloaded Gemma 4 QAT+MTP model passed health, authenticated `/v1/models` listed eight profiles, and a bounded `/v1/chat/completions` request returned `READY` (89 total tokens). Stopped the temporary router and restarted the original ComfyUI transient unit via its original `maintenance/comfyui-service.sh start-gpu` launcher. Verified active original `ExecStart`, ComfyUI 0.36.0 GPU health, empty queue, and closed pilot port. No permanent service path changed. |
| 2026-09-29 | Branch media model execution | With the restored original ComfyUI GPU service, the branch's Qwen Image 2.1 script passed `--check` and submitted a bounded 512×512, four-step inference job (`d5c28456-a97e-4d5f-87ab-e40cdb277c11`). It wrote a verified 512×512 PNG at `/home/kchauhan/media-output/l3ms-monorepo-qwen-pilot-20260929.png`; visual inspection showed the requested blue circle. The ComfyUI queue returned empty and the service stayed active. This proves branch script/model invocation against the original service, in addition to the earlier branch-hosted service-path pilot; it is not an end-to-end branch-hosted heavyweight model run. |
| 2026-09-29 | Phase 4 lane request/result shapes | Added in-memory chat, media workflow/native script/Breeze speech, and decision request/result types. Validation checks lane-qualified profile ownership, native task declarations, required cloning/voice-design inputs, decision answer alignment, and a 1 MiB request payload budget; it cannot start engines or claim readiness beyond manifest presence. Five new tests pass. Root full suite: 289 passed, 1 ignored; root Clippy and formatting pass. Adapter execution contracts remain Phase 4 work. |
| 2026-09-29 | L3DS fail-closed acceptance safeguard | Removed silent synthetic fallback from normal Nagi, Laya, SemIf, and hosted decision requests. `--simulate` is an explicit demo path with synthetic engine labels and no model/hosted API loading; SemIf has no native forward pass yet. Missing native inference now exits nonzero, and the Rust benchmark fails if any profile lacks a valid result rather than counting raw/fabricated output as success. Five offline Python safety tests pass; Rust child tests, fmt, and Clippy pass. A normal SemIf one-shot and benchmark both exited 1; explicit simulation exited 0 with a `semif-simulated` label. This does not satisfy the real L3DS inference gate. |
| 2026-09-29 | Phase 5 offline lifecycle contract | Defined explicit lifecycle ownership, GPU resource claims, runtime state, admission/drain/health/release adapter methods, and a fail-closed in-process exclusive transition sequence. Six mock tests cover order, drain and admission failures, target-health rollback, unknown release, external ownership, and pre-cancellation. No concrete adapter or live/manual switch command is connected. Root full suite: 295 passed, 1 ignored; formatting and Clippy pass. |
| 2026-09-29 | Structured transition failures | The offline exclusive-GPU switch now returns a typed disposition (`NoTransition`, `PreviousRestored`, or `OwnershipUncertain`) instead of requiring a future coordinator to parse error text. Existing mock tests assert the disposition for failed admission/drain, restored target-health failure, unknown release, and external ownership. Root full suite remains 295 passed, 1 ignored; Clippy and formatting pass. |
| 2026-09-29 | Phase 9 keyboard semantics | Added tablist/tabpanel relationships, selected-state updates, roving tab focus, and Arrow/Home/End navigation to the three-lane static site while preserving its existing language views. Six site/generator tests and inline JavaScript syntax pass. Browser visual review remains unavailable in this session, and the site remains unpublished. |
| 2026-09-29 | Breeze-TTS-2 offline route/readiness work | The imported L2M2 wrapper now has a read-only `--check` for the local audio.cpp executable, parsed `breeze_tts` loader key, model, and model spec; cloning checks a nonempty reference WAV. Standard TTS, voice-design alias, native clone task, and streaming chunk output have fixture tests. Local preflight passed with installed files but did not load the model or allocate the GPU. L2M2: 16 media tests, 14 downloader tests, 4 Rust tests, fmt, Clippy, and shell syntax pass. Shared request validation now accepts an optional clone transcript, matching the native audio.cpp contract. Root: 295 passed, 1 ignored; fmt and Clippy pass. A real TTS smoke remains pending. |
| 2026-09-29 | External serving-mode change observed | At approximately 12:49 local time, the original ComfyUI transient unit stopped and the original `llama-swap.service` became active on port 8080; this was not a branch operation. At inspection, the original ComfyUI unit was absent/inactive and `/health` on the language router returned `OK`. No service was toggled for TTS; preserve current mode and recheck direct-ingress/admission before any GPU pilot. |
| 2026-09-29 | Nagi-Smol environment preparation | The L3DS child has a pinned Nagi-only Python/CUDA dependency input and a separate dry-run-by-default staged preparer that refuses to overwrite an existing `.venv`; the original `maintenance/setup-env.sh` is byte-identical to the imported source for standalone compatibility. The preparer does not acquire weights or run inference. Its no-write preview, shell syntax, eight Python safety/setup tests, two Rust child tests, fmt, and Clippy pass. Local uv resolves CPython 3.12.13, and the offline resolver found the 51 non-Git pins against the original environment; a full offline sync cannot resolve the uncached pinned Nagi Git source. The imported child still has no `.venv`, no Nagi-Smol snapshot, and no native decision result. |
| 2026-09-29 | Publication re-audit | A private NO-GO audit is pinned to commit `2287d9d`: the imported reachable history still discloses a Funnel endpoint and host-specific records. The owner confirmed an unchanged baseline high-entropy key-like value was only a documentation example; a public export should still replace it with a clear placeholder. No push or history rewrite occurred. Later commits and the settled worktree need an exact outgoing-ref re-scan before publication. |
| 2026-09-29 | Phase 4 native route metadata | A read-only resolver now preserves media script paths and decision entrypoint/model/type/tier from the native manifests, keeps profile/runtime identities separate, and uses a symbolic llama-swap endpoint reference without guessing URL or auth. Declared relative paths are bounded and lexically anchored under each canonical toolkit root; a future dispatcher must revalidate symlink containment before opening them. It does not check file existence, contact endpoints, or start engines. Root tests passed 299/1 before the TUI change and 301/1 after integration; fmt and Clippy pass. |
| 2026-09-29 | Phase 8 read-only TUI catalog | Added F8/Alt+8 Stack Catalog navigation, palette open/refresh/selection commands, declared task/runtime/device/source details, explicit manifest-only readiness, and missing-manifest errors. CLI and TUI now share settings/environment resolution; navigation exits previous editor input safely. Tests exercise navigation, configured roots, failed refresh, and wide/compact rendering. Combined verification: root 301 passed/1 ignored, fmt/Clippy; L2M2 4 Rust, 16 media Python, 14 downloader; L3DS 2 Rust, 8 Python; six site/generator tests and deterministic payload. The three localhost HTTP fixtures required the approved unsandboxed test run. No coordinated run action was enabled. |
| 2026-09-29 | Native decision payload preservation | Shared decision requests and answers now retain the native named question/answer objects, including choice criteria/options and score payloads, instead of flattening them to text. Validation bounds the complete state/question JSON and verifies answer IDs and declared choices. Six request-contract tests pass; no backend invocation is enabled. |
| 2026-09-29 | Bounded runtime operations | Every potentially blocking lifecycle adapter operation receives its deadline/cancellation context; successful admission opening is the commit point. Cleanup and restoration share one independently bounded recovery budget with caller cancellation masked. Twelve runtime tests cover cancellation during startup/health, slow preflight, and recovery exhaustion, in addition to existing transition failures. Combined root suite after queue integration: 314 passed, 1 ignored; fmt and Clippy pass. The three localhost fixtures required the approved unsandboxed run. |
| 2026-09-29 | Phase 6 offline coordinator core | Added a bounded queue using the existing job store: acceptance is persisted before dispatch, retained request IDs refuse replay, restart interrupts rather than replays accepted work, conflicting GPU work is FIFO while hosted work can bypass it, and cancellation/drop retains active reservations until acknowledged completion. Quarantine also blocks already-reserved dispatch. Queue/cold-start/execution timings are separate; prompts and native payloads are not persisted. Six queue tests pass. Request completion does not establish resident-model GPU release; concrete adapters must still prove it. |
| 2026-09-29 | Offline host ownership guard | Added lifetime advisory locks in an explicitly shared, private, local host lock directory, including a coordinator lock and sorted GPU locks. Existing lock inodes are never unlinked or replaced; symlink/shared/nonregular lock files fail closed. Four tests include a competing independent process and release/reacquisition. This guard is not installed, does not fence legacy direct clients, and cannot replace admission/reconciliation. |
| 2026-09-29 | Phase 6 IPC contract and reconnect | Added bounded length-prefixed v1 submit/cancel/status envelopes, explicit version/unknown-field rejection, lane-native inputs, and bounded serialization. Request-ID validation reuses the existing history rules. Status lookup survives restart using retained caller IDs rather than rehydrated numeric job IDs; terminal records cannot be relabelled cancelled. Twelve IPC and seven queue tests pass. Combined root suite: 331 passed, 1 ignored; fmt, Clippy, and whitespace checks pass. Transport deadlines, peer authentication, daemon installation, concrete dispatch, results, and streaming remain unimplemented. No engine/service action or public command was enabled. |
| 2026-09-30 | Concrete language execution | The branch has a caller-configured, bounded HTTP language adapter that preserves native chat turns, separates reasoning/content, requires terminal SSE completion, checks cancellation/deadlines while waiting for headers/body, and never retries uncertain dispatch. Errors do not expose credentials, endpoints, prompts, or response bodies. Eight adapter tests include isolated HTTP fixtures; shared chat results bound combined output and permit reasoning-only completion. This is not connected to CLI/UI inference or service switching. |
| 2026-09-30 | Native path/recipe verification | Catalog refresh preserves canonical configured roots even for empty lanes. Six dispatch-time path tests reject missing/empty/nonregular files, symlink escapes, and changed runtime identity. Twelve native recipe tests cover actual Breeze mode flags, clone-file checks/path anchoring, explicit decision devices, bounded/redacted stdin and argv, missing interpreters, and unsupported workflow/simulation dispatch. These preparations start no engines. |
| 2026-09-30 | Decision stdin compatibility | Added optional `--input-stdin` with a bounded UTF-8 `{state,questions}` JSON envelope; malformed input is rejected before engine construction. Legacy inline/benchmark flags remain supported. Eleven child Python safety/setup tests pass; no environment or model was installed. |
| 2026-09-30 | Supervised local decision execution | Added a Linux child-group runner with bounded nonblocking stdin/output, distinct stdout/stderr under one shared cap, deadline/cancellation checks, and independently bounded recovery. The leader stays unreaped until the final owned-group signal; after reaping only a non-signalling absence probe is used. Nine runner tests pass. The concrete local decision adapter prepares/executes/decodes harmless fixtures with offline model flags, validates native engine labels/answers, and preserves native latency/usage; five decoder/execution tests pass. Escaped/daemonized descendants remain outside the process-group proof; local cleanup cannot establish remote Comfy cancellation or GPU release. Hosted execution remains gated by SDK retry/auth verification; standalone hosted CLI is unaffected. |
| 2026-09-30 | Combined adapter checkpoint | Root full suite: **373 passed, 1 ignored**, plus fmt, Clippy, and whitespace checks. Changes are committed locally on `feature/l3ms-monorepo`; no migration service switch, real model pilot, endpoint cutover, push, or publication occurred. Pi delegation was attempted with a read-only task and timed out; further Pi use is deferred at the user's request. Implementation assumptions and release gates are documented in [execution-contracts.md](execution-contracts.md). |

## Open decisions

Latest offline adapter/transport/control/safety checkpoint: **429 root tests passed, 1
ignored**, with fmt, Clippy, and whitespace checks passing. A deterministic
fork-inheritance regression reproduced an ownership guard release defect;
explicit unlock now releases both complete and partially acquired guards even
while a child retains inherited descriptors. A second regression rejects child
success with truncated stdin. Backend IDs also have a restart-retention test:
they can be persisted before dispatch and are retained without automatic replay.
The concrete ComfyUI adapter passes nine tests: unchanged graph/UUID submission,
adapter-local single-use ownership, native terminal status, bounded artifact
references, scoped cancellation with no unsafe fallback, unknown/no-op states,
uncertain submission without replay, manifest runtime drift, response limits,
and deadline/cancellation waits. It requires an explicit installed-server
compatibility assertion; durable token recovery and lifecycle/GPU-release proof
are not implemented. Luna workers provided native protocol and focused safety
reviews; the root implemented the adapter after narrowing an overlong worker
task. No live service was changed. These are offline checks, not completion of
the lifecycle/live gates.

The Linux local transport passes 14 tests, including two independent process
clients, private directory/socket permissions, peer UID denial, retained host
locks, protected existing/replaced socket paths, frame limits, request/reply
ordering and ID matching, slow/stalled IO, cancellation/deadlines, and full-backlog
connect behavior. A targeted native-socket probe confirmed that EAGAIN with
zero SO_ERROR can still mean unconnected; the reviewer withdrew an initial
same-descriptor polling suggestion after that evidence. Client frames are never
replayed. The daemon/event loop, installation, coordinator dispatch and result
streams remain pending; no live service was changed or started.
After the connector initially blocked the export, the owner approved a brief
progress summary to CAR-218. It was posted successfully on 2026-10-01 UTC
(comment `9b7a3e80-9b90-40ca-a953-931aa04f57e8`).

2026-10-01 control-RPC checkpoint: fifteen queue/RPC tests pass, including seven
new regressions for typed admission/duplicate/full/invalid/unbound rejection,
persist-before-acknowledgment, active cancellation reservation retention, failed
acceptance/cancellation persistence, restart status without replay, incomplete
retained state, and lost socket acknowledgment followed by status/cancel/status.
The existing Rust submit interface wraps the same typed submission path; persistence
failures now quarantine admission. A focused Luna review found no blocker. No
daemon, engine, or live service was started; runtime reconciliation/dispatch and
peer-loop ownership remain pending. This checkpoint is recorded locally; it does
not imply any further Linear export beyond the separately approved earlier summary.

2026-10-01 control-loop checkpoint: nine daemon fixtures pass, covering ownership
before history access, startup failure cleanup, closed submission admission,
restart interruption without replay, malformed/stalled peer isolation, outer
cancellation/deadlines, private data paths, GPU-lock coverage and release on
exit. The consuming serial loop bounds peer time and connection attempts and
has no dispatch or admission-opening interface. No live service was touched;
installation, lifecycle reconciliation and result delivery remain pending.

2026-10-01 decision-route correction: the native Nagi Smol/Big bridge now passes
the declared repository ID to its loader, including CPU fallback; previously it
selected only a tier and silently loaded the upstream default repository. Three
new offline mock regressions cover repository forwarding and preserve Huge's
existing call shape. L3DS Python: fourteen tests pass. No weights or environment
were installed. Public-metadata-only acquisition preflight is recorded in
[l3ds-prep.md](l3ds-prep.md); installation approval remains pending.

2026-10-03 CLI/Breeze checkpoint: five new CLI regressions cover parser bounds,
private-socket status/cancel exchanges, wrong-operation replies, bounded JSON,
and explicit unavailable errors without fallback. Six Breeze fixtures cover all
four modes, explicit fresh private outputs, sorted chunks, missing/unsafe/bounded
artifacts, redacted failures, cancellation/deadline cleanup, and PCM container
validation rather than a bare WAVE marker. Root: 426 passed, 1 ignored; no real
model or live service was used. Dispatch, lifecycle reconciliation and full UI
integration remain incomplete.

2026-10-03 shared-control-settings checkpoint: optional socket/artifact paths
share existing host/user/environment/explicit precedence and declaring-directory
anchoring. Defaults remain absent, resolution creates no files, and portable
exports strip both paths without changing the operator's store. Three new
settings regressions include an isolated-process check of environment-variable
collection. Root: 429 passed, 1 ignored; no deployment or admission was enabled.

2026-10-03 common-client/NVIDIA checkpoint: Linux CLI status/cancel now uses the
shared one-shot client and opt-in socket configuration, with explicit flag
precedence and no inferred default or automatic startup. Client fixtures cover
submit/status/cancel reply shapes, correlation, fixed errors and no fallback;
CLI fixtures preserve parser, path-selection and bounded JSON behavior. Ten
NVIDIA fixtures cover explicit identity, bounded queries, malformed/unsupported
readings, changing capacity, claimants, memory limits and cancellation/deadlines.
Root: 444 passed, 1 ignored; Clippy, formatting and whitespace checks pass. All
GPU tests use fake executables. No live service, admission or inference dispatch
was enabled. The managed-service ownership primitive is still in development.

2026-10-03 managed-ownership/history-isolation checkpoint: ten controller
fixtures verify single-use pre-dispatch intents, bounded restored records,
uncertain-launch retention, transient generation/cgroup checks, rejection of
preexisting/non-transient units and refusal to stop changed generations. Stop
acknowledgment is not cgroup/GPU release; same-user replacement races remain an
explicit trust constraint. Twelve daemon fixtures now include canonical
directory aliases, symlink/hard-link history rejection and absent standalone
paths without creation. Root: 457 passed, 1 ignored; Clippy, formatting and
whitespace checks pass. No actual service command or GPU query was run. Durable
intent storage, owned cgroup cleanup and adapter/admission wiring remain pending.

2026-10-03 durable-service-owner checkpoint: eight store fixtures validate
strict bounded restored records, immutable duplicates, private inode/type checks,
FIFO rejection, corruption/unsafe entries, capacity and no automatic creation.
Seven owner fixtures compose the actual store/controller under retained host
ownership, asserting record-before-launch, lost-launch-result observation without
replay, persistence refusal, deadline refusal, post-persistence cancellation,
nonmember-handle rejection and corrupt-store stop refusal. Mutable owner methods
serialize one instance; shared-directory/controller identity remains a caller
precondition. A focused Luna review found no blocker under that contract.
Twelve controller fixtures now include explicit working directories in identity
and argv; environment expansion is disabled. Root: 474 passed, 1 ignored;
Clippy/all-targets, formatting and whitespace checks pass. No live manager/GPU
operation occurred; only installed `systemd-run --help` was read. Records remain
retained, with no retirement/release/admission proof or daemon dispatch wiring.

2026-10-05 owned-stop/dispatch checkpoint: generation-matched stop now returns
an opaque receipt; ten cgroup fixtures verify bounded fixed-root traversal,
empty/removed observations and fail-closed populated, unsafe, malformed,
missing-event and cancelled/expired cases. Four new owner fixtures compose
stop and observation while retaining durable intents and transferable receipts
after observation failure, without repeating stop. The dispatch gate snapshots
the accepted payload, refreshes native routes and rechecks binding, history,
cancellation and actual deadline before persisting running. Tampered/stale
tickets quarantine without dropping reservations; start is not replayable.
Root: 499 passed, 1 ignored; locked/offline all-targets tests, Clippy with denied
warnings, formatting and whitespace checks pass. No live manager/GPU operation
occurred. Result retention/delivery, concrete lifecycle/admission adapters,
intent retirement and end-to-end dispatch remain incomplete.

2026-10-05 dashboard behavior checkpoint: six Python generator/data-contract
tests pass. The shipped generated payload and inline dashboard JavaScript also
pass a Node DOM fixture for native lane card rendering, click selection,
Left/Right wrapping, Home/End, roving focus/ARIA selection and HTML escaping.
Dashboard CI now runs both checks; no remote CI or publication was triggered.
The browser inventory is empty and the in-app browser is unavailable, so the
loopback preview was stopped without claiming visual/mobile/browser acceptance.
The fixture deliberately does not emulate layout or an accessibility tree.

2026-10-05 results/runtime-binding checkpoint: twelve result-buffer fixtures
verify native lane separation, bounded transactional retention, exact pagination,
UTF-8/escaping, explicit terminal eviction and strict decoded-page validation.
V2-only result queries preserve v1 behavior, correlate ID/version/cursor/budget,
and keep durable status independent from output-cache loss. Real private-socket
fixtures cover queued/not-found lookup, lost-response retry and restart with
retained succeeded history but unavailable output. CLI results uses shared
settings/explicit socket precedence, compact bounded JSON and no runtime startup.
Runtime-bound service intents retain lane/name in strict schema 2; unbound
schema-1 records remain compatible. Manager description/recipe checks preserve
the association through dispatch/restart, without replay or process-name guesses.
Review corrected explicit-null deserialization and mixed-lane page acceptance.
Root: 530 passed, 1 ignored; locked/offline all-targets tests, denied-warning
Clippy, formatting and whitespace checks pass. No live manager/GPU or inference
operation occurred. Native dispatch/event production, result retention policy,
intent retirement, concrete lifecycle/admission and deployment remain incomplete.

Repeat-run follow-up: the all-targets suite subsequently exposed intermittent
fake-manager `SpawnFailed` errors. Captured errno 26 (`ETXTBSY`) identified
parallel fixture writes overlapping fork/exec, not a runtime-binding or deadline
failure. Test-only executable-write/spawn serialization corrected the race;
100 parallel managed-service suite repetitions passed. A new four-thread fresh
executable regression reproduced the failure with its write guard removed.
With the guard restored, the full suite passed 531 tests, 1 ignored.
Production spawn behavior and failure disposition are unchanged.

### 2026-10-05 — owned intent retirement, isolated transport and CPU worker

Root verification: 556 passed, 1 ignored; locked/offline all-targets tests,
strict denied-warning Clippy, formatting and whitespace checks pass without
lint suppression. Explicit intent retirement freshly checks stopped generation
and cgroup, validates exact private records, then unlinks/syncs with retained
metadata on uncertainty. Runtime identity and declared native device are now
part of accepted-route checks, including both recipe refreshes. Uncertain
completion retains its active reservation until separate verified-cleanup
acknowledgment; interrupted outcomes cannot be reclassified by ordinary completion.

The isolated byte transport pins coordinator PID/UID/start, pidfd and retained
nsfs identity. Real child-namespace fixtures verify the FD handoff, private Unix
peer, isolated loopback backend, early response/backpressure and uncertain partial
writes. Captured EACCES for cross-user-namespace `/proc` namespace lookup was
resolved by authoritative FD handoff, not a boolean bypass. Namespace loopback
setup is fixture-only; no real manager units, inference engines or GPU were used.
Production manager launch/property validation and native admission/drain remain.

A one-job CPU-only local-decision worker now checks accepted routes, disables
CUDA discovery and downloads, supports cancellation and returns typed native
outputs/cleanup categories over bounded channels. Its success fixture initially
returned an undeclared answer, correctly rejected by native validation; the
fixture's accepted choices were corrected and full verification rerun. Restricted
CPU startup refuses GPU/hosted bindings and unresolved retained history. Default
profiles remain CUDA/remote, not silently converted. Output quotas reserve each
job's maximum before dispatch rather than competing with other active output.
Daemon execution/event-loop wiring, public submit/UI/HTTP, general media mapping,
live acceptance and publication remain incomplete. No goal-completion claim.

- Choose the host admission mode for direct llama-swap/ComfyUI requests before
  claiming automatic cross-engine switching. The plan allows coordination
  only where the runtime can block and drain new direct work, or where callers
  use exclusive coordinated ingress.
- Live GPU and external-route cutovers require host access and recorded
  before/after state. They are not implied by a successful build.
- The opt-in fresh Nagi environment and multi-gigabyte model acquisition still
  await approval. No dependency environment or weights were acquired in this
  continuation; real local decision acceptance remains unsatisfied.

### 2026-10-05 — foreground CPU dispatch and native client path

Locked/offline all-targets verification passes 594 tests, with 1 ignored. The
opt-in CPU loop has 21 focused daemon tests covering durable submit/lost replies,
responsive status/cancel, result pages, timeout and active shutdown cleanup,
output-quota preflight, GPU refusal and restart without replay. Explicit foreground
startup reads a strict bounded private configuration and handles SIGINT/SIGTERM
cooperatively. The normal control-only loop and standalone defaults are unchanged.
See [foreground-coordinator.md](foreground-coordinator.md).

One-shot CLI Submit uses a shared bounded nofollow reader and caller-owned IDs.
Private language SSE and Comfy submission fixtures verify native HTTP over the
selected Unix socket. Shared transactional result reassembly and multi-page
private IPC reading preserve native fields and reject incomplete groups, stale
cursors, wrong lanes and unavailable output. Output is not persisted or replayed.

Strict denied-warning Clippy passes after narrowing managed-launch error storage,
without lint suppressions. Managed engine integration is still unfinished.
Keyboard-first strict-file submit/observe/cancel and the explicit reassembled-output
CLI now pass fixtures; full native task forms remain pending. No
model-serving manager unit, real engine/model, dependency environment or GPU was changed. A new
browser availability check still returned no enabled apps or browsers, so actual
site visual/mobile acceptance remains open. Full-port acceptance and publication
are not complete.

The subsequent [model-free manager namespace probe](private-engine-probe.md)
disproved the proposed two-unit namespace-join topology on this installed user
manager: declared join properties still left the bridge in the host namespace.
Removing the join produced two genuinely isolated namespaces; a startup delay
and manager-opened namespace descriptor did not correct joining. All probe units
were generation-checked, stopped and collected. Managed launch is being reworked
to one isolated service owning backend and bridge, with kernel checks retained.
This is a discovered deployment gate, not real inference or GPU acceptance.

### 2026-10-05 — resumed live-state read-only check

The original ComfyUI unit is currently inactive and llama-swap is active; the
RTX 4070 reports 44 MiB of 12,282 MiB in use. These are point-in-time observations,
not ingress/drain or GPU ownership proof. Breeze `--check` still succeeds against
the imported toolkit's existing audio.cpp executable, compiled loader, model and
model spec. No synthesis, output creation, service stop/start, environment install
or model download was performed. Recheck live state before any approved pilot;
do not reuse the earlier Comfy-active baseline.

Offline compatibility checks in this continuation also pass: 16 media Python
tests, 14 decision safety Python tests, 14 imported downloader tests, six site
data tests and shipped-JavaScript DOM fixtures. These do not load models or
replace real browser visual/mobile/accessibility acceptance.

### 2026-10-05 — authenticated loopback ingress checkpoint

The explicit `--coordinator-http-gateway` action reads a strict private 0600
configuration, binds only loopback, and forwards one native v1/v2 request per
connection to private IPC. Fourteen gateway fixtures cover authentication,
framing, bounded deadlines/attempts, native correlation and lost Submit replies
without retry. The all-target suite passes 617 tests with one ignored; strict
denied-warning Clippy passes. CPU startup and gateway sessions share scoped
SIGINT/SIGTERM ownership without changing normal startup. See
[http-gateway.md](http-gateway.md).

The internal single-service supervisor entrypoint also compiles and passes five
argument/descriptor-contract tests. Its process-cleanup regressions, typed manager
role replacement, actual service/namespace/readiness and native engine health are
still being implemented or awaiting acceptance. This checkpoint does not enable
general managed GPU/media dispatch or certify full-port completion.
