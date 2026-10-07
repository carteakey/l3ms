# TODO

All intentionally committed TODO items were reconciled into the single Linear
`L3MS` project and its `Operations backlog completion — 2026-08-10` milestone
on 2026-08-10. See CAR-140 through CAR-146 for the seven grouped issues and
their acceptance criteria. Linear is the source of truth for committed work;
this file is reserved for ideas and research that have not been intentionally
committed, except for the requested next-work index below, which summarizes
existing Linear migration issues rather than replacing their acceptance criteria.

## Monorepo: next work

Latest offline checkpoint: opt-in CPU daemon dispatch and foreground launch,
one-shot Submit, private native HTTP and shared result reassembly on
`feature/l3ms-monorepo`; 617 root tests passed,
1 ignored. This is offline implementation evidence, not live cutover acceptance.
Strict denied-warning Clippy, formatting and whitespace checks pass. Managed
engine supervisor/lifecycle integration is still unfinished.
The repeat-run fixture `ETXTBSY` race is corrected with test-only fork/write
serialization; production launch failures remain fail-closed without retries.
See the [migration status](docs/migration/status.md),
[plan](docs/monorepo-consolidation-plan.md), and
[execution contracts](docs/migration/execution-contracts.md).

### Safety decisions and runtime completion — CAR-217

- [ ] Choose direct-client admission policy: exclusive coordinated ingress, or
  verified blocking/draining of direct requests. Refuse unsafe swaps meanwhile.
- [ ] Implement concrete lifecycle adapters and reconcile actual runtime
  ownership; verify stop, GPU release, startup/health, and bounded rollback.
  The fixture-tested NVIDIA probe provides point-in-time idle evidence only;
  owned runtime/cgroup cleanup and ingress exclusion remain separate gates.
  Transient-service launch now persists its intent before dispatch under host
  ownership. Add proven cgroup cleanup and explicit retirement of retained
  intents before opening admission; adapter/daemon wiring is still pending.
  A generation-bound stop receipt and fixed-root cgroup-v2 observer are now
  implemented on the branch; empty/removed observations alone do not prove GPU
  release or fence direct clients. Owner composition retains receipts after
  observation failure so reconciliation does not repeat the stop operation.
  Runtime-bound intents now persist lane/name and match that binding during
  restart observation; never infer it from ports or process names.
  Proof-based intent retirement and an isolated-engine Unix bridge with retained
  coordinator identity/namespace descriptors now pass fixtures, including a real
  isolated loopback relay. Managed launch and lifecycle wiring remain pending;
  a successful `unshare` command alone is not admission/GPU acceptance.
- [x] Connect Breeze speech recipes to bounded execution and per-job WAV/chunk
  artifact validation. Real-model tests and coordinator dispatch remain gated.
- [ ] Add explicit artifact contracts for other native media scripts; do not
  guess from logs or scan shared output directories. Verify hosted decision
  SDK retry/auth behavior before enabling it.
- [ ] Define durable ComfyUI ownership recovery: fresh IDs persisted before
  submission, no replay, and reconciliation of unknown/cancelled backend jobs.

### Coordinator and clients — CAR-218 / CAR-219

- [x] Build and fixture-test Linux private socket transport, peer UID checks,
  retained host/device locks, bounded IO, and single-exchange request IDs.
  No daemon is installed; CPU-local dispatch requires explicit foreground opt-in.
- [x] Connect IPC submit/status/cancel to the durable queue with fixed rejection
  codes, persistence-before-acknowledgment, lost-ack reconnect tests, and active
  cancellation that retains reservations until cleanup is acknowledged.
- [x] Add a listener/history-owning serial control loop with bounded peer time
  and connection attempts, closed startup admission, and safe ownership cleanup.
- [x] Resolve optional coordinator socket/artifact paths through shared settings;
  keep defaults absent and omit host-local paths from portable exports.
- [ ] Implement the local IPC daemon with private socket/peer authentication,
  bounded transport, singleton ownership, startup reconciliation, and concrete
  dispatch. Deliver results, artifact references, and streams without persisting
  sensitive payloads.
  Next implementation slice: connect concrete lifecycle/admission reconciliation
  and supervised dispatch; keep dispatch closed until those checks are integrated.
  Startup now rejects history aliases; installation must supply the actual
  standalone store path. Verify concurrent workflows without lost history.
  The pre-dispatch gate now refreshes the native catalog and verifies the
  accepted route, resource binding, ticket association, and durable running
  transition. Native executors must use this gate, not legacy `mark_running`.
  Bounded native result retention and v2 page queries are integrated with the
  internal daemon and shared client; reconnect/restart fixtures pass. Native
  event production and terminal retention are connected for CPU decisions;
  general managed-engine/media dispatch remains pending.
  CPU-only worker and restricted core admission are now connected to an opt-in
  foreground daemon with responsive private status/cancel/result queries,
  quota-before-acknowledgment, cancellation and bounded shutdown fixtures.
  See [foreground setup](docs/migration/foreground-coordinator.md). Require an
  explicit CPU profile and no-GPU binding; current CUDA profiles stay gated.
  Never reinterpret a configured CPU fallback as CPU admission. General managed
  engine/media dispatch and real-model CPU acceptance remain outstanding.
- [ ] Integrate CLI and keyboard-first UI run/observe/cancel with the same
  coordinator, settings, routing, and request IDs; preserve standalone commands.
  Linux CLI request status/cancel/results and one-shot Submit are implemented.
  Shared bounded output reassembly, explicit CLI output and keyboard-first
  strict-file submit/observe/cancel UI pass fixtures. Native bounded task forms
  for chat, media workflows/scripts, speech modes and decisions are implemented;
  managed engine/media routing and live acceptance remain unfinished.
  Preserve the full plan's language scope: add manifest-derived embedding task
  discovery (implemented from active command flags) and typed embedding request/result/adapter support (the shared
  language contract currently supports chat only). Connect incremental language
  stream events, task-aware artifact access and visible runtime recovery/status;
  native envelope forwarding and JSON output panels alone are not full parity.
- [ ] Add HTTP ingress to that coordinator only after local operation is proven;
  verify authentication and cancellation/reconnect behavior across clients.
  Explicit bounded loopback gateway and strict private foreground configuration
  pass native v1/v2, authentication, framing, deadlines and lost-ack/no-retry
  fixtures. Installed/live multi-client and managed-runtime acceptance remain.

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
  Nine data-contract tests and shipped-JavaScript DOM fixtures pass locally;
  CI now runs both. No browser is available in this session, so real rendering,
  mobile layout and browser accessibility/keyboard acceptance remain pending.
  Committed serving source ed7538a is synchronized; imported metrics are not
  monorepo live acceptance. New source-referenced benchmark harnesses were not
  committed and remain an explicit import/validation gap.
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
