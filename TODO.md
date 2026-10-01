# TODO

All intentionally committed TODO items were reconciled into the single Linear
`L3MS` project and its `Operations backlog completion — 2026-08-10` milestone
on 2026-08-10. See CAR-140 through CAR-146 for the seven grouped issues and
their acceptance criteria. Linear is the source of truth for committed work;
this file is reserved for ideas and research that have not been intentionally
committed, except for the requested next-work index below, which summarizes
existing Linear migration issues rather than replacing their acceptance criteria.

## Monorepo: next work

Latest offline checkpoint: bounded control-daemon ownership on `feature/l3ms-monorepo`;
415 root tests passed,
1 ignored. This is offline implementation evidence, not live cutover acceptance.
See the [migration status](docs/migration/status.md),
[plan](docs/monorepo-consolidation-plan.md), and
[execution contracts](docs/migration/execution-contracts.md).

### Safety decisions and runtime completion — CAR-217

- [ ] Choose direct-client admission policy: exclusive coordinated ingress, or
  verified blocking/draining of direct requests. Refuse unsafe swaps meanwhile.
- [ ] Implement concrete lifecycle adapters and reconcile actual runtime
  ownership; verify stop, GPU release, startup/health, and bounded rollback.
- [ ] Connect bounded media/TTS command recipes to execution and artifact
  discovery. Verify hosted decision SDK retry/auth behavior before enabling it.
- [ ] Define durable ComfyUI ownership recovery: fresh IDs persisted before
  submission, no replay, and reconciliation of unknown/cancelled backend jobs.

### Coordinator and clients — CAR-218 / CAR-219

- [x] Build and fixture-test Linux private socket transport, peer UID checks,
  retained host/device locks, bounded IO, and single-exchange request IDs.
  This is transport only; no installed daemon or inference dispatch yet.
- [x] Connect IPC submit/status/cancel to the durable queue with fixed rejection
  codes, persistence-before-acknowledgment, lost-ack reconnect tests, and active
  cancellation that retains reservations until cleanup is acknowledged.
- [x] Add a listener/history-owning serial control loop with bounded peer time
  and connection attempts, closed startup admission, and safe ownership cleanup.
- [ ] Implement the local IPC daemon with private socket/peer authentication,
  bounded transport, singleton ownership, startup reconciliation, and concrete
  dispatch. Deliver results, artifact references, and streams without persisting
  sensitive payloads.
  Next implementation slice: connect concrete lifecycle/admission reconciliation
  and supervised dispatch; keep dispatch closed until those checks are integrated.
- [ ] Integrate CLI and keyboard-first UI run/observe/cancel with the same
  coordinator, settings, routing, and request IDs; preserve standalone commands.
- [ ] Add HTTP ingress to that coordinator only after local operation is proven;
  verify authentication and cancellation/reconnect behavior across clients.

### Live acceptance and publication — CAR-216 / CAR-220 / CAR-221

- [ ] Obtain approval for the fresh Nagi environment and multi-GB weights;
  complete real local-decision inference without synthetic fallback.
- [ ] Run real Breeze standard/design/clone/streaming TTS acceptance and a
  coordinated language → media → language pilot in an agreed maintenance window.
  Record before/after service/GPU state, artifacts, cancellation, and restoration.
- [ ] Finish import/standalone compatibility and CI checks; exercise crash,
  restart, timeout, contention, and rollback paths before enabling public actions.
- [ ] Complete visual/keyboard/mobile review of the three-lane site and verify
  generated data against native manifests; do not publish before the audit gate.
- [x] Replace the confirmed documentation-only key-like example in the current
  tree with a clear placeholder; historical blobs remain unaudited/unchanged.
- [ ] Sanitize the exact outgoing history/export and re-audit endpoints, host paths, secrets, and large
  files. Publication remains NO-GO; no push/history rewrite without authorization.
- [ ] Reconcile final deployed paths and document rollback/retirement criteria.
  Keep original repos and runtime state until acceptance; retire nothing early.

Pi delegation remains deferred at the owner's request.

## Speculative / research

There are no uncommitted ideas at this time. Add future possibilities here
only until they are explicitly chosen and promoted to Linear.

## ✅ Done

- ~~Generated public profile dashboard~~ — served profiles and portable
  `llama-server` commands now derive from `llama-swap.yaml`; benchmark metadata
  stays separate, rankings are deterministic, and mobile uses readable cards.
- ~~TUI: teach Model Ops about llama-swap~~ — Run mode now reads from
  `/v1/models`; Start/Stop call `/models/load` and `/models/unload`;
  editor shows model state + curl snippets. Bench mode unchanged.
- ~~Chat tab default port~~ — already `8080` (checked post-migration,
  TODO was stale)
- ~~Retire `gemma-vision.service`~~ — moved unit + helper to
  `maintenance/systemd/archive/` and `maintenance/archive/`; llama-swap
  preloads `gemma-4-26b-a4b-vision` instead
- ~~Fix `preserve_existing` field loss~~ — removed from schema, no longer needed
- ~~Add `action_quit` method~~ — implemented with graceful subprocess + task cleanup
- ~~Add missing Maintenance tab output capture~~ — `run_script` no longer awaits
  its own task; output streams live
- ~~Add Jobs tab stop/retry functionality~~ — `■ Stop` / `↺ Retry` buttons,
  `script_path` + `mode` persisted in history, `s` / `r` shortcuts
- ~~Add proper task cleanup~~ — `action_quit` terminates processes and cancels
  resource/chat/run async tasks
- ~~Add command palette~~ — `Ctrl+P` `CommandPaletteScreen` with fuzzy filter
  over all app actions
- ~~Chat history persistence~~ — `save_chat` writes `.md` + `.json`;
  `Sessions` / `Load` open `ChatHistoryScreen` to restore sessions
- ~~Zellij-style footer~~ — only `F1–F6`, `q`, `?`, `Ctrl+P` visible; all
  `Ctrl+*` / `Alt+*` shortcuts in `?` help overlay grouped by context
