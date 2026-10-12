# L3MS: a small wrapper, not another inference platform

User-approved direction: simplicity and fewer lines of code, 2026-10-09.
This supersedes the hardened-platform delivery requirements in the older
consolidation plan. It does not waive preservation, safe process cleanup or
real-model acceptance. Linear remains the committed-work tracking authority.

## Product stays; implementation gets smaller

One monorepo and keyboard-first workbench for language, media (including TTS)
and decisions. Shared discovery, settings, jobs, logs and sensible GPU switching.
Keep the existing engines, scripts, model directories and standalone commands.
Keep the distinctive visual direction and expanded L3MS branding.

The default path is an operator-launched application with in-process dispatch,
not a new always-on daemon. CLI and TUI call the same operational functions.
An optional API or daemon is not required to complete this wrapper.

```text
CLI / TUI
    |
catalog + settings + jobs + one GPU queue
    |
    +-- language --> existing llama-swap API
    +-- media ----> existing ComfyUI API / native scripts
    +-- TTS ------> existing audio.cpp / speech scripts
    `-- decisions > existing local runner / hosted API
```

Use concrete adapters; do not invent a universal engine framework. Expose only
the operations callers actually need. Reuse existing process/job supervision
and native output conventions rather than maintaining parallel implementations.

## Keep versus retire

| Keep and consolidate | Candidate for retirement after replacement passes |
| --- | --- |
| Catalog, declared routes, shared settings and task-specific inputs | Separate strict managed-daemon configuration and launch paths |
| Existing native command recipes and trusted engine APIs | Private network namespaces, supervisor helpers and custom transport |
| One bounded process/job supervisor and one GPU queue | Owned-job packet stores and duplicate service/job supervisors |
| Timeouts, cancellation, bounded output and redacted diagnostics | Generation-capability handoffs and durable automatic crash reconciliation |
| Simple history with interrupted jobs on restart; no inference replay | Custom v1/v2/v3 IPC and HTTP gateway as mandatory execution infrastructure |
| Validated local output paths, native artifacts and normal file access | Versioned hex artifact transfer and a pinned-FD artifact registry |

These are removal candidates, not assertions that modules are unused. Trace
callers and replace real behavior before deletion. Do not delete a safeguard
while retaining the workflow that depends on it.

## Safety proportional to an operator-owned homelab

- Only control services/processes explicitly configured as managed. External
  engines are never killed or adopted. Preserve authentication and public routes.
- Serialize conflicting GPU jobs. Do not silently preempt active work or claim
  to fence unrelated direct engine clients with an advisory lock.
- Switch using native stop/start operations, bounded waits and readiness/device
  checks. If cleanup is uncertain, stop scheduling and show a recovery action.
- Kill/reap owned child process groups on cancellation or timeout. Trusted native
  tools must not daemonize outside their documented containment.
- No automatic retry of inference after an ambiguous submission or crash.
  Restore known service state where possible and report failed restoration.
- Restart marks unfinished jobs interrupted. Automatic resumption and recovery
  from every crash window are explicitly out of scope for this wrapper.
- No implicit downloads, original-checkout removal, permanent service cutover,
  public deployment or credential exposure.

## Replacement and deletion sequence

1. Trace current CLI/TUI execution and native launchers. Choose one existing job
   supervisor as the owner; avoid adding a second wrapper implementation beside
   the managed platform. Record baseline LOC and behavior before changes.
2. Connect native media/TTS/decision execution to that owner, sharing validated
   recipes and task forms. Keep native language streaming and results. Verify
   success, cancellation, timeout and artifact access through CLI and TUI.
3. Add the smallest GPU switch path through existing native service controls.
   Verify a real language -> image/TTS -> language cycle and exact restoration
   within the approved maintenance window. Unknown GPU claimants block switching.
4. Move default execution off managed IPC/namespace/packet machinery. Then remove
   the displaced code, flags, tests and contradictory documentation in cohesive
   changes. Preserve checkpoint commits and originals as rollback, not a second
   indefinitely maintained runtime in the final default build.
5. Verify standalone toolkit behavior, shared UI/config/jobs, real local decisions,
   TTS modes and site/branding against the revised wrapper scope. No GPUI rewrite
   or new server framework is part of this simplification.

## Evidence and size target

Baseline at `d2a1c17`: `src/control/*.rs` contains **60,729 lines**, including
tests, comments and blank lines. This is not a production-only LOC count.
Measure the same file scope and changed module placement after each deletion;
moving code, compressing formatting or deleting useful tests is not simplification.
Aim to remove at least half of this control implementation over the replacement,
but do not trade away required behavior to meet a number. Prefer fewer concepts,
state stores, protocols and launch paths, then fewer lines.

Acceptance is a smaller, usable integrated stack with real end-to-end evidence,
not a growing infrastructure test count. Tests should cover the user-facing
workflow and meaningful failure cases. The Stop journal and hardened crash
recovery contract are deferred; do not implement them as the next slice.

## Execution plan — parallel audit resolved, 2026-10-09

Three Luna Max agents traced the engine, UI and deletion dependencies at
`6110daa`. Their recommendation is one direct inference operation shared by CLI
and TUI, not directly embedding `CoordinatorCore` into the UI or wrapping it in
another daemon. No new framework, service installation or GPUI pivot.

### Chosen seams and reuse

- Keep `SwapClient` for existing llama-swap discovery/load/unload and existing
  `ChatClient` streaming. llama-swap remains the language serving/swap engine.
  Reuse the existing embeddings adapter; consolidate duplicated language HTTP
  logic as callers move, not by maintaining two permanent chat implementations.
- Reuse `catalog`, `requests`, `routes`, `native_commands`, `decision`,
  `media_execution` and `process::execute`. Native Comfy workflows retain native
  submit/status/cancel semantics; scripts remain scripts.
- Turn task-form validation into a typed request builder independent of IPC.
  Use one concrete direct-run dispatch function, a small foreground inference
  job worker and existing `JobHistory`. No engine trait hierarchy or serialized
  events. CLI runs the same operation; TUI receives bounded in-memory updates.
- Do not route sensitive inference through `app::start_process_from_parts`
  unchanged: it logs argv and supplies null stdin. Its argv is retained in memory
  but omitted from persisted job rows. Use the existing bounded process runner
  and stdin payload contract; persist only safe job metadata.
- Keep ordinary bench/download workflows working. Consolidate inference ownership
  first rather than rewriting every non-inference background task at once.

### Delivery order and gates

| Slice | Change | Evidence before proceeding |
| --- | --- | --- |
| 1. Direct run | De-wire task-form validation; direct prepared CPU-local Nagi/Laya execution via the existing runner and one job owner. Add one native-input CLI run action and wire the Stack task form to that same operation. | Real decision result, exact declared CPU profile, stdin handling, cancel/timeout, no payload logs, interrupted-on-restart/no replay, CLI/TUI tests. No CUDA-to-CPU fallback. |
| 2. Native tasks | Add Breeze modes and media scripts, then Comfy workflows. Preserve native language chat/embeddings and consolidate their shared invocation/results. | Task inputs, native output paths/results, partial language streaming, accurate failures, cancellation and artifacts through the workbench. |
| 3. GPU switching | One conflicting-GPU queue; existing explicitly managed service controls, bounded release/readiness checks and visible recovery. | Approved real language -> image/TTS -> language cycle, unknown-claimant refusal and exact prior service restoration. No preemption or direct-client fencing claim. |
| 4. Delete platform | Remove displaced CLI/UI coordinator actions and their implementation closure; strip managed-only branches from mixed native adapters. | New paths cover former supported workflows; old native commands/history remain readable; compile, focused regressions and actual LOC reduction. |
| 5. Finish experience | Clean Run/Jobs/Results UI, align the native palette with Signal/03, and reconcile site/docs/settings with the smaller product. | Keyboard workflow, readable states/focus, real language/media/TTS/decision acceptance, standalone compatibility and no conflicting default paths. |

Do not ship a second default execution mode while leaving both stacks maintained.
Temporary overlap is limited to replacing actual callers; each slice records the
next deletion it unlocks. Existing private-coordinator flags are branch-added
surfaces to retire explicitly, not permanent compatibility requirements for the
original standalone tools. Do not claim unsupported tasks are complete merely
because their toolkit remains available separately.

### Parallel implementation ownership

Use all three available child-agent slots with disjoint writes once the shared
direct request/result interface is fixed:

1. Engine agent: direct dispatch and native adapters; no `app.rs` or CLI edits.
2. UI agent: `app.rs` task run/cancel/results and keyboard tests; no runner edits.
3. CLI/deletion agent: direct CLI input/output and dependency pruning after caller
   replacement; coordinate every deletion with the root and engine agent.

Root owns shared interface decisions, module exports, history compatibility,
integration, docs, Cargo verification, commits and any approved live test. Agents
must not independently run Cargo, mutate services, change configs/download weights
or commit. Reassign completed slots to focused workflow review, not more design
documents or speculative infrastructure. Prefer one small dispatch match over
generic lifecycle abstractions.

### Deletion map and reporting

The broad retirement candidate is **41,161 gross existing lines in 30 files**:
17,087 in coordinator/transport/result delivery and 24,074 in managed ownership,
helpers, namespace transport and related probes. This is not immediately dead
code or a promised net saving. Some utilities must be retained or simplified;
`comfy.rs` and `inference.rs` also contain mixed native/managed logic outside that
count. No public module was found obviously unused by production callers.

Delete from the callers inward: UI/CLI -> coordinator/gateway/worker -> managed
adapter branches -> ownership/helpers/packet stores -> protocol/results machinery.
Untie forms and native results from IPC first. Preserve legacy optional history
fields as historical data only; never reconstruct execution authority from them.
Keep necessary GPU checks even if their current module appears in the candidate.

Report each implementation checkpoint as: user workflow gained, concepts/modules
removed, net LOC change across the whole `src/` tree (so relocation cannot count),
verification, and next concrete slice. Test count alone is not a progress metric.

### First direct replacement checkpoint — 2026-10-10

CPU-local Nagi/Laya decisions now share a typed direct operation across the CLI
and Stack form. Real offline Nagi CLI execution passed with the branch's explicit
CPU profile; see [acceptance evidence](direct-decision-acceptance.md). This is a
partial caller replacement, not full slice acceptance or a LOC reduction yet.
No platform module was removed in this checkpoint.

The next slice must generalize this foreground inference owner for native
speech/media, not grow `managed_session`, owned-job helpers or IPC. Reuse native
recipe/process execution and normal validated output files. Breeze currently
declares CUDA and its native wrapper hardcodes the CUDA backend: retain that
profile unchanged. A CPU TTS option requires an explicitly declared profile,
route-derived backend and a verified CPU-capable audio.cpp build; never silently
reinterpret a CUDA profile. Otherwise GPU TTS must wait for the small native
switching path and its live acceptance. Consolidate one result/job owner as lanes
move, then remove displaced caller dependencies inward.

### Shared CPU speech replacement checkpoint — 2026-10-10

Standard CPU Breeze now shares the direct decision foreground owner, typed
builder and runner. Real cached-weight CLI execution and file delivery pass;
see [speech acceptance](direct-speech-acceptance.md). CUDA defaults remain intact.
No coordinator IPC or second result store was added for this lane.

Still temporary overlap: this slice adds 1,233 net root source lines, and no
platform module is retired yet. Next replace remaining coordinator form callers
with native media/language workflows and delete the displaced caller dependency
closure. Do not extend the managed daemon, owned-job helpers or recovery platform.

### First caller-first deletion checkpoint — 2026-10-10

Retired generic envelope-file Submit from CLI/Stack, its parser and file-specific
workbench preparation. Typed forms and observe/cancel/results stay; all native
task implementations and original standalone commands remain. Net whole-root
source reduction: 714 lines, including tests; cumulative change since `702b4cd`
is still +2,425, so the broader reduction target is not met.

Next reuse the existing Chat tab/client for exact Stack language model selection,
not another chat execution owner. Native embeddings/media still need replacement
before removing the full coordinator closure. Historical probe entry points must
fail before launching retired infrastructure, not silently run a broken test.

### Original Chat owner replacement checkpoint — 2026-10-10

Stack chat-only profiles now hand off exact model selection to the existing
Chat tab/client, with no automatic connect/send/load. The raw-turn coordinator
form and builder branch are removed; embedding-based fixtures retain typed
submission/observation/cancellation coverage. Current committed connection,
catalog drift and active-work checks keep selection explicit and side-effect free.

879 tests pass (1 ignored), strict Clippy/format/build pass. This slice adds
306 net root source lines including tests, so cumulative source change remains
+2,731 against `702b4cd`, not the promised reduction. Next native embedding
batches through the existing language adapter, then native media and inward
coordinator deletion. Keep HTTP cancellation uncertainty separate from local
child cleanup; do not add another Chat owner or expand managed infrastructure.

### Native embedding implementation checkpoint — 2026-10-12

Embedding batches now share the direct native owner and existing language
adapter across CLI/UI. HTTP uncertainty is distinct from child cleanup; UI
preserves the committed connection credentials and refuses overlapping Chat
sends after unconfirmed completion. No coordinator envelope or result store
was added for the replacement path.

Verification remains gated: build/lint/fmt and focused non-transport tests pass,
but the restricted session blocks local socket operations (772 full-suite passes,
121 transport failures, 1 ignored). Native loopback and real-model acceptance
must pass before claiming this slice accepted or pruning dependent working paths.
Net source change is +1,260; cumulative change remains +3,991 against `702b4cd`.
Next verify embeddings, replace remaining media/GPU callers, then delete the
coordinator closure. Do not confuse temporary replacement overlap with LOC reduction.
