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
