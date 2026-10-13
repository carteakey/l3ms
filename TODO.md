# TODO

All intentionally committed TODO items were reconciled into the single Linear
`L3MS` project and its `Operations backlog completion — 2026-08-10` milestone
on 2026-08-10. See CAR-140 through CAR-146 for the seven grouped issues and
their acceptance criteria. Linear is the source of truth for committed work;
this file is reserved for ideas and research that have not been intentionally
committed, except for the requested next-work index below, which summarizes
existing Linear migration issues rather than replacing their acceptance criteria.

## Monorepo: next work

This is a forward index, not a second implementation log. See the dated
[migration status](docs/migration/status.md) for evidence and checkpoint history;
the [simplification plan](docs/migration/simplification-plan.md) defines scope,
replacement order and acceptance gates. Linear remains authoritative for
committed work.

- [ ] Complete the full Blender contract: headless Python rendering and the
  distinct interactive external GUI/MCP workflow. The wrapper's `--prompt`
  currently prints MCP guidance only. The opt-in single-action headless wrapper
  and four argv/preflight fixtures are implemented; root dispatch and real
  rendering remain pending. Do not invent inference, output artifacts, or a
  CPU/CUDA device declaration.
- [ ] Add the small operator-owned native service handoff/restoration path.
  Reuse native controls, perform bounded health/device checks, refuse unknown
  GPU claimants, and restore the exact prior service/runtime state. Use only an
  approved maintenance window; an advisory lock does not fence outside clients.
- [ ] Continue caller-first managed-platform deletion only after each real
  replacement passes. Count net source removal rather than relocation, preserve
  safety needed by surviving callers, and work toward the plan's half-control-
  source reduction target without adding a daemon or engine hierarchy. Keep local
  SemIf unsupported while its native forward pass is absent; do not fake it.
- [ ] Complete live acceptance across all lanes and supported tasks, including
  local/hosted decisions, language/chat/embeddings, media, TTS modes, Comfy and
  GPU workflows; verify cancellation, results, artifacts and exact handoff
  restoration. Finish keyboard/visual/mobile review in the Signal / 03 direction,
  site-manifest compatibility, standalone/import compatibility and CI checks.
- [ ] Run `cargo test --all-targets --locked` where local TCP/Unix sockets are
  permitted, then run a bounded real embedding batch with an explicitly prepared
  profile, installed weights and configured endpoint. Do not silently enable or
  download models.
- [ ] Keep original repositories, configs, runtime state and history until
  acceptance and rollback criteria are documented. Publication remains **NO-GO**
  until the exact outgoing history/export, endpoints, host paths, secrets and large
  files are audited; no push/history rewrite without authorization. No service
  change, download or provider call without explicit approval.

## Previous Chat replacement checkpoint

Current replacement: Stack chat-only language profiles now open the existing
Chat tab on the exact enabled advertised model, using the committed connection.
No connect, send, load or default-model fallback occurs; drafts/conversations
stay intact. The duplicate raw-turn chat form and builder branch are retired.
Embedding/media forms remain, and their typed submit/observe/cancel coverage is
retained. **879 root tests passed, 1 ignored**; strict Clippy, formatting and
build pass. This slice adds **306 net root source lines** including tests;
cumulative source change against `702b4cd` is **+2,731**. No platform module
was removed here, and the larger LOC reduction is not complete.

The native embedding implementation is now integrated above; its transport and
real-model acceptance gates remain open before broader dependency deletion.

## Previous deletion checkpoint

Latest simplification cut: retire branch-added generic `--request-submit` and
Stack `s` file submission, remove `submission_file` and its workbench adapter.
Typed `f` forms, observe/cancel/results, direct CPU tasks and original standalone
commands remain. This removes **714 net root source lines**, including tests;
the parser/duplicate entry point is gone, not relocated. The typed keyboard
integration test retains submit/observe/cancel/results coverage. Current root
suite: **877 passed, 1 ignored**; 17 decision-maintenance Python tests pass.
Cumulative `src/` change against `702b4cd` remains **+2,425 lines**.

The planned Stack-to-Chat handoff is now implemented in the current checkpoint
above. Native embedding/media callers still block broad coordinator deletion.

**Revised direction:** follow the [simplification plan](docs/migration/simplification-plan.md).
Next finish native TTS/media wrapper workflows, then delete displaced
infrastructure. Do not build the Stop journal
next. Older infrastructure entries below are historical inventory, not the
current implementation order or required hardened-platform acceptance scope.

Parallel audit is complete. The first CPU-local decision implementation now
shares a direct typed operation across CLI and the Stack native form, using
existing recipes/process supervision and safe TUI job metadata. Real offline
CLI Nagi execution passed, including the explicit CPU profile; this is not
full interactive, media or GPU acceptance. See
[direct decision evidence](docs/migration/direct-decision-acceptance.md).
Standard CPU Breeze speech now shares that foreground owner and returns validated
WAV paths in CLI/TUI. Real offline CPU execution passed; see
[speech evidence](docs/migration/direct-speech-acceptance.md).
Next other native media, GPU switching, platform deletion and UI polish in the
[execution plan](docs/migration/simplification-plan.md#execution-plan--parallel-audit-resolved-2026-10-09).

Previous speech replacement checkpoint on `feature/l3ms-monorepo`: **884 root tests
passed, 1 ignored**; six media-toolkit Rust tests, 14 downloader and 17 media
Python tests pass, along with strict all-target Clippy, formatting and shell
syntax checks. Real CPU speech and UI lifecycle fixtures passed. This slice adds
**1,233 net root source lines**; cumulative change against `702b4cd` is **+3,139**,
including tests/comments/blanks. No platform modules removed yet. Next replace
remaining coordinator form callers with native workflows, then retire their
dependency closure. Keep language routing in llama-swap; do not expand owned-job
helpers or the managed daemon.

Earlier retained platform checkpoint (historical inventory, not next work): 859 root tests passed,
1 ignored, strict all-target Clippy, formatting and whitespace checks pass.
Opt-in CPU dispatch, managed-language launch, resumable native results and
owned per-job packet/service handoff are implemented. Subsequent branch work
composes allowlisted local jobs with managed dispatch and retry-safe finalization;
full mixed-worker tests, crash recovery, full Comfy acceptance, native decision
TUI/mixed-runtime live acceptance, full-file artifact consumption and real managed acceptance remain
unfinished. See migration status for the current slice and its explicit gates.
This is offline implementation evidence, not live cutover acceptance.
The repeat-run fixture `ETXTBSY` race is corrected with test-only fork/write
serialization; production launch failures remain fail-closed without retries.
See the [migration status](docs/migration/status.md),
[plan](docs/monorepo-consolidation-plan.md), and
[execution contracts](docs/migration/execution-contracts.md).

Immediate integration work: managed Comfy candidate/generation persistence,
peer-verified native session, opt-in worker/configuration and typed backend-ID
acknowledgment accounting are implemented. Fresh pinned output validation yields
Backend references now have bounded v3 chunk access through private IPC,
authenticated HTTP and an explicit one-read CLI command. Durable success and
retained result state gate exact pinned references; no history restore or
arbitrary path access. Full joined worker/lifecycle fixtures, full-file/TUI
artifact consumption and crash-safe mixed-runtime recovery remain. Verified venv
interpreter handoff preserves Python environment semantics. Real CPU native
metadata delivery is verified; GPU switching is not.
Recovery batch preflight is complete. The
[crash recovery contract](docs/migration/crash-recovery-contract.md) is deferred
under the revised wrapper scope; its existing managed path stays gated rather
than treating unresolved records as implicitly safe to retry.

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
  The opt-in concrete managed-language session now composes the listener-owned
  host lock, service WAL, isolated router, one-shot activation, read-only selected
  model readiness, fresh accepted-route dispatch and bounded native results.
  Language integration fixtures pass; full daemon/native-engine composition,
  real model acceptance and multi-runtime switching remain required. For Comfy,
  candidate UUID persistence and exact-workflow move-only permits now pass
  joined native HTTP fixtures before POST, separately from a verified
  `backend_id`; attach that ID only after exact acknowledgment or a positive
  read-only native observation. Owned-generation recovery and the actual
  worker/coordinator handoff still need integration. Neither 404 nor cancel dispatch proves
  non-submission/completion. Per-job Breeze/GPU-decision ownership must use the
  exact-service cgroup path, not process-group cleanup alone.
  Breeze now explicitly declares `device: cuda`; typed physical UUID selection
  is implemented for resident engines and accepted Breeze/local-decision native
  execution, with model-free child fixtures. Per-job service ownership and
  coordinator dispatch still need integration; visibility is not GPU isolation.
  Owned per-job packet/claim and service-launch handoff are now implemented:
  pinned private directories, full accepted route, exact device/parent identity,
  payload-free schema-4 WAL, and parked one-shot native wrappers. Wire the
  coordinator's dispatch/completion path to these handles, stop-only startup
  recovery, cgroup/GPU release and fresh artifact publication before enabling
  cross-runtime jobs. Do not treat packet readiness as job completion.
  CPU-only worker and restricted core admission are now connected to an opt-in
  foreground daemon with responsive private status/cancel/result queries,
  quota-before-acknowledgment, cancellation and bounded shutdown fixtures.
  See [foreground setup](docs/migration/foreground-coordinator.md). Require an
  explicit CPU profile and no-GPU binding; current CUDA profiles stay gated.
  Never reinterpret a configured CPU fallback as CPU admission. General managed
  engine/media dispatch remains outstanding. Real-model CPU coordinator
  acceptance passed with one submission, native output, durable terminal history
  and verified process cleanup; see the Nagi evidence below.
- [ ] Integrate CLI and keyboard-first UI run/observe/cancel with the same
  coordinator, settings, routing, and request IDs; preserve standalone commands.
  Linux CLI request status/cancel/results and one-shot Submit are implemented.
  Shared bounded output reassembly, explicit CLI output and keyboard-first
  strict-file submit/observe/cancel UI pass fixtures. Native bounded task forms
  for chat, media workflows/scripts, speech modes and decisions are implemented;
  managed engine/media routing and live acceptance remain unfinished.
  Preserve the full plan's language scope: add manifest-derived embedding task
  discovery (implemented from active command flags) and typed embedding request/result/adapter support (the shared
  language contract now preserves typed vectors/native usage and bounded pages;
  keyboard batch forms and adapter fixtures pass, production dispatch pending).
  Incremental native chat deltas/usage/stream completion and cursor-resuming
  explicit observations pass fixtures without resubmission or automatic polling.
  Connect them to managed dispatch, task-aware artifact access and visible runtime recovery/status;
  native envelope forwarding and JSON output panels alone are not full parity.
- Historical HTTP ingress pilot: retired on 2026-10-12 under the simplification
  plan. Its gateway/launcher were removed rather than requiring another proxy;
  native endpoints and retained private coordinator actions remain unchanged.

### Live acceptance and publication — CAR-216 / CAR-220 / CAR-221

- [ ] Complete real local-decision inference without synthetic fallback.
  Isolated environment and required weights were approved on 2026-10-07;
  pinned dependencies and checksum-verified minimal models are installed. Real
  offline CPU native execution, bounded benchmark and private foreground
  coordinator acceptance passed. GPU inference and routing-quality validation
  remain; see [Nagi evidence](docs/migration/nagi-acceptance.md).
- [ ] Run real Breeze standard/design/clone/streaming TTS acceptance and a
  coordinated language → media → language pilot in an agreed maintenance window.
  Record before/after service/GPU state, artifacts, cancellation, and restoration.
  Bounded maintenance/restoration was approved on 2026-10-07; permanent cutover
  remains outside that approval. Native harness preflights safely refused before
  any service stop: the absent Comfy unit is now handled, but NVIDIA lists the
  existing KDE compositor as a compute claimant. Explicit background-compositor
  allowance is awaiting the owner's choice; no TTS inference pass is claimed.
  See [Breeze evidence](docs/migration/breeze-acceptance.md). Follow the
  [managed session contract](docs/migration/managed-session-contract.md).
- [ ] Finish import/standalone compatibility and CI checks; exercise crash,
  restart, timeout, contention, and rollback paths before enabling public actions.
- [ ] Complete visual/keyboard/mobile review of the three-lane site and verify
  generated data against native manifests; do not publish before the audit gate.
  Ten data/contrast tests and shipped-JavaScript DOM fixtures pass locally;
  CI runs both. The opt-in real Firefox probe passes desktop/mobile lane layout,
  pointer, keyboard and focus checks, console capture and disposable-profile
  cleanup. Signal / 03 styling is implemented and visually checked on the site.
  Comprehensive accessibility and public-link/export review remain pending.
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

- Native Rust/GPUI frontend, separate from the coordinator and optional for CLI
  builds. Carry the Signal / 03 design tokens into a bounded read-only prototype
  after full-port acceptance; measure GPU footprint, then establish client/IME/
  keyboard parity before replacing any existing UI. See
  [interface direction](docs/interface-direction.md). This recommendation is not
  an implemented native app or a new full-port completion requirement.

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
