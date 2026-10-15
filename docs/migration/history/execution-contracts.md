# Native execution contracts

These are branch-local implementation contracts, not a deployed coordinator or
live acceptance record. See [status.md](../status.md) for phase gates and evidence.

## Direct CPU operation

CPU-local decisions and Breeze speech use `--run-profile` or the Stack typed form
with the shared native runner and foreground owner. No listener, daemon,
installation or download is required. A CUDA profile is never silently converted
to CPU. Cancellation/timeout perform bounded owned-child cleanup; uncertainty
closes foreground admission. See [native CPU workflows](../../cpu-native-workflows.md).
The branch-added CPU daemon launcher is retired; its
[historical contract](foreground-coordinator.md) is retained only as evidence.

Generic envelope-file submission and its shared parser have been retired.
Remaining Stack task forms build typed native requests, validate the refreshed
catalog and retain the caller ID before dispatch. The daemon remains the durable
acceptance authority for this older path. A lost acknowledgment never authorizes
client replay; CPU direct decisions/speech do not use this path.

`ResultConsumer` reconstructs from cursor zero transactionally under the 8 MiB
serialized-event cap, with contiguous request-correlated cursors, selected-lane
checks and ordered cross-page decision/media groups. Chat content/reasoning
stay separate; decision JSON must be a canonical object; artifact references
remain relative. A rejected page leaves state unchanged. The shared native
reader performs at most 128 read-only exchanges under one absolute deadline,
stopping at currently available output instead of automatically polling,
retrying, submitting or starting engines. A running snapshot is not completion;
unknown/expired output is not empty success.

Language and Comfy adapters support explicitly selected private Unix HTTP.
Their shared internal socket module pins the private directory and socket inode,
owner and permissions, revalidates before client construction and disables
proxy/TCP fallback. It does not prove the helper's engine namespace, lifecycle,
admission or GPU release; the owning managed launch must establish those facts.
Ordinary HTTP constructors and standalone endpoints are unchanged.

## Ownership and dispatch

Shared stack settings now resolve optional `coordinator_socket_dir` and
`artifact_root` paths through host/user settings, `L3MS_COORDINATOR_SOCKET_DIR`
and `L3MS_ARTIFACT_ROOT`, then explicit overrides. Neither has a built-in
default. Stored relative paths anchor to their declaring settings directory;
environment/CLI paths anchor to the invocation directory. Resolution is
read-only and does not create a socket, output directory or service. Portable
profile exports omit these machine-local paths. Runtime consumers must still
verify private ownership and admission before use.

The future executor must acquire the shared host/device ownership guard, reconcile
actual runtimes and GPU use, prove admission, and reserve accepted work through
the coordinator before invoking an adapter. A configured resource claim is not
evidence that the GPU is free. Direct llama-swap/ComfyUI callers are not fenced by
an advisory lock; switching remains disabled without the selected host ingress
mode and a real admission/drain guarantee.

Native manifests remain authoritative. Catalog refresh retains each original
canonical root, including empty media/decision lanes. Script dispatch refreshes
the route and checks that the nonempty regular entrypoint resolves inside its
toolkit root. A changed runtime identity requires resource rebinding, not reuse
of an earlier reservation. Internal symlinks are allowed; escaping ones are not.
Path verification assumes trusted toolkit directories and is not a sandbox or
a defense against concurrent malicious filesystem replacement.

The final `begin_dispatch` gate privately retains the accepted request, refreshes
the catalog, compares the complete payload and route, verifies its runtime
resource binding and queued history association, then rechecks cancellation and
the actual deadline before durably marking running. Tampered/stale tickets or
persistence failures close admission while retaining their reservation. The
accepted route includes the declared native device, including an absent device;
a device-only manifest change invalidates the ticket rather than reusing its
resource assumptions. Native command preparation checks that device at both
catalog refreshes, alongside runtime identity. This rejects stale reservations;
it does not itself prove that a CPU/GPU binding or running backend is safe. Native
executors must use this gate rather than the compatibility `mark_running` method.

An uncertain completion durably records `interrupted` and closes admission while
retaining its active reservation. Ordinary completion cannot reclassify that
outcome. Only `acknowledge_reconciled_cleanup` removes the reservation after the
host executor independently verifies cleanup/resource ownership; it preserves
the interrupted outcome and does not reopen admission. The method is a caller
contract, not a native cleanup probe or permission to discard unknown ownership.

## Managed transient-service ownership

`ManagedRuntimeOwner::retire_stopped` explicitly retires a retained launch intent
only after fresh Empty/Removed cgroup observation and a fresh absent-or-exactly-
inactive manager generation check. The stop receipt retains its original intent.
The store validates all records and the exact bound record, rechecks descriptor/
inode/owner/mode/link constraints, unlinks only that filename, then syncs its
directory. Pre-unlink failures preserve the record; sync failure after unlink
returns `DurabilityUncertain` with original metadata rather than retrying. Success
does not release a GPU claim or establish admission. The held host lock, trusted
stable store and no concurrent same-user unit replacement remain assumptions.

The private `EngineTransport` helper accepts opaque native bytes only from its
captured coordinator PID/UID/start identity, retaining a pidfd and authoritative
network-namespace file descriptor. It verifies nsfs/device/inode and requires its
own actual namespace to differ. Reading the coordinator namespace through `/proc`
inside a user namespace is not a viable fallback on this host (EACCES); the
namespace descriptor must be handed over before isolation. Source identity,
private 0700 directory/0600 Unix socket, bounded nonblocking relay and byte limits
fail closed. Caps are per connection/direction, not per HTTP request; half-close
and byte receipts are not native completion, async-job drain or GPU proof.
The trusted coordinator must retain its original network namespace.

Real isolated-child fixtures verify retained-FD handoff, loopback backend relay,
early response, backpressure and post-dispatch uncertainty. No live engines were
launched. Subsequent bounded model-free user-manager probes verified manager
`OpenFile` handoff but disproved two-unit namespace joining on this host:
declared join properties left the bridge in the host namespace. All probe units
were generation-checked and stopped. The replacement is one isolated supervisor
owning backend and bridge; its actual kernel isolation, descriptor names and
lifecycle still require acceptance. See the
[private-engine probe record](private-engine-probe.md) and the
[systemd 262 service documentation](https://raw.githubusercontent.com/systemd/systemd/v262/man/systemd.service.xml).

The manager's generation-matched stop acknowledgment now produces an opaque,
non-cloneable `StoppedService` receipt. A read-only `CgroupReleaseProbe` can
observe only that receipt's original cgroup, rooted at the fixed, filesystem-
verified `/sys/fs/cgroup`. Descriptor-relative no-symlink traversal and bounded
`cgroup.events` reads require hierarchical `populated 0`; populated, malformed,
missing events, unsafe paths, cancellation, and timeout fail closed. An absent
owned directory produces only a `Removed` observation after acknowledged stop.
The privately constructed observation matches its receipt's unit, generation,
and cgroup; it cannot be applied to another receipt accidentally.

Neither observation fences new ingress or proves GPU release. It assumes a
trusted executable tree that cannot escape its cgroup and no concurrent
same-user operator replacing/restarting the unique transient unit. The host
guard excludes cooperating coordinators, not those operators. Retained intents
remain intact: these observations do not authorize retirement or dispatch.

The internal Linux controller prepares a unique transient-unit intent before
dispatch; callers must persist that record under the retained coordinator lock.
Dispatch consumes its token once. Fixed, root-owned systemd client binaries use
explicit argv, `KillMode=control-group`, `Restart=no` and a finite stop timeout.
Recipes require an existing canonical working directory, included in their
identity. `--expand-environment=no` preserves literal command arguments rather
than manager-side variable expansion ([systemd-run documentation](https://raw.githubusercontent.com/systemd/systemd/main/man/systemd-run.xml)); unsupported versions fail rather than dropping the flag.
Existing loaded units or installed unit files are refused before launch.
Post-spawn failures preserve the intent as ownership-uncertain; killing the
local launcher cannot prove that the manager rejected its start.

Validated restored records are data, not stop capabilities. Reconciliation
issues a private handle only for a matching active transient unit, intent marker,
InvocationID and cgroup. Ownership uses the documented `Transient` property,
not an assumption that `FragmentPath` is empty ([systemd property documentation](https://raw.githubusercontent.com/systemd/systemd/main/man/org.freedesktop.systemd1.xml)).
State/stop recheck that generation. A disappeared or unverifiable cgroup is
unknown, not released. Stop is name-targeted and cannot atomically compare an
InvocationID: the host lock must serialize coordinator callers and trusted
same-user operators must not concurrently replace/restart that unique unit.
Manager stop acknowledgment is not cgroup emptiness, health, ingress exclusion
or GPU-release proof. This controller is not yet a runtime adapter or wired
to admission/dispatch; fixtures start only private fake executables.

The composed `ManagedRuntimeOwner` now borrows a retained host guard and records
each intent before dispatch. The dedicated existing absolute mode-0700 store
uses descriptor-relative, no-symlink access; intent files are mode 0600,
current-UID, single-link regular files. Schema/fields are strict, each record is
capped at 4 KiB and the store at 128 entries. Create-new writes synchronize the
file and parent before acknowledgment; duplicate, corrupt, unexpected or unsafe
entries fail closed. FIFO/special-file entries are rejected before reading.
Failure can retain a partial record: no delete, repair or automatic retry occurs.
Only bounded ownership metadata is stored, never argv, environment or payloads.
Generic unbound records remain schema 1. Managed runtime starts use schema 2,
adding only the runtime lane/name; reconciliation compares that binding against
the manager description. Explicit nulls, partial bindings, invalid lane/name,
unknown fields and incompatible schemas fail closed. Restored unbound intents
must not be assigned a runtime by guessing from ports, commands or process names.

Restart observation composes the stored records with live controller checks;
it does not recreate recipes or replay launches. Stop first verifies the retained
record and then the live handle, retaining the record even on success. There is
also a composed `stop_and_observe` operation under the owner's retained guard:
success privately bundles the receipt and its matching cgroup observation;
failure after stop preserves a transferable receipt for read-only re-observation
without repeating the manager stop. Stop failure produces no receipt. There is
no intent retirement operation yet; reaching the cap refuses new launches.
Caller lifecycle logic must reconcile unresolved records and prove admission and
resource authority before starting an engine. The wrapper is not wired to the
daemon and neither its host lock nor successful observation opens admission.
Filesystem synchronization is synchronous/non-preemptible on the trusted local
filesystem; contexts are checked before and after it. Mutable methods serialize
operations through one owner; callers must create only one owner for the same
store/controller under the shared host guard. Seven fake-manager fixtures
exercise the actual store/controller composition without live services.

## NVIDIA idle observation

The internal Linux probe uses the fixed `/usr/bin/nvidia-smi` executable and an
explicit GPU UUID, expected index and maximum idle memory allowance. It reads
identity/memory, compute claimants, then identity/memory again, with bounded
child supervision, a shared operation deadline and 64 KiB per-query output
limits. Malformed/unsupported values, changed device capacity, foreign devices,
compute claimants and excessive memory usage fail closed with fixed errors.
There is no automatically learned idle baseline or retained claimant PID.

A positive result is only a point-in-time observation. It does not cover
graphics/desktop ownership, prevent new direct requests, establish a runtime
lease, or prove cgroup cleanup. Concrete switching must separately prove owned
runtime termination and ingress exclusion; this probe is not wired to admission.
Tests use fake executables only, not the actual GPU or running services.

## Language

The concrete language adapter translates the native model key and ordered
system/user/assistant turns using the existing chat payload builder. It does not
trim system turns, add another system prompt, or inject a thinking command.
Assistant content and `reasoning_content` remain separate; a bounded reasoning-only
completion is valid. Output is capped at 1 MiB combined, individual SSE lines at
1 MiB, and total wire bytes at 8 MiB. A terminal `[DONE]` event is required.

Cancellation/deadline checks cover header waiting and idle body reads. Call the
synchronous adapter from a blocking executor thread; callbacks must return
promptly. Redirects are disabled. Failures expose fixed categories rather than
endpoint, credential, prompt, or response-body text. Once HTTP dispatch begins,
any failure is conservatively uncertain and is never retried automatically.
Closing a stream does not acknowledge server cancellation or VRAM release.

## Decision input and output

The additive native runner flag `--input-stdin` accepts a UTF-8 JSON envelope
containing exactly `state` and `questions`, capped at 1 MiB. It rejects malformed
input before engine construction and cannot be combined with the legacy inline
flags or benchmark mode. Existing standalone flags remain supported.

Captured native decision JSON is bounded and validated against the original
named questions and declared choices. Current accepted engine labels are
`nagi-native`, `laya-native`, and `systemone-official-api`, matching the native
runner. Synthetic labels and SemIf's unimplemented forward pass fail closed.
Native latency is preserved separately from queue/cold-start/execution timing.
Fixture labels demonstrate decoder behavior, not real inference acceptance.

The concrete local decision adapter now combines command preparation, bounded
child supervision, and decoding. It forces the Hugging Face/Transformers offline
flags in the child; it does not install models or environments. Hosted execution
through this new adapter remains gated until native SDK retry/auth behavior is
verified; the existing standalone hosted CLI is unchanged.

## Native commands and process supervision

Media script and Breeze speech recipes use argv items, never a shell command
string. They check their interpreter/entrypoint and keep payloads out of Debug
and job metadata. Each argv item is capped at 64 KiB and aggregate argv at 256 KiB;
the shared 1 MiB request budget does not guarantee that a native CLI can accept a
1 MiB single argument. Clone references must be nonempty regular files; relative
paths anchor to the toolkit root, while explicit absolute external samples are
allowed. Generic ComfyUI workflow graphs use the separate bounded jobs adapter
described below, not a script command recipe.

## Breeze speech execution

The internal executor composes the native Breeze recipe and bounded child runner
for standard speech, design, cloning and streaming. The caller supplies a fresh,
existing absolute mode-0700 current-UID job directory. Existing result/chunk
targets are rejected before spawning; explicit `--output` and `--out-dir`
prevent inherited default output settings from redirecting artifact discovery.
Captured logs are discarded, never parsed for output paths or persisted.

Success requires child-group cleanup plus a regular, single-link in-root WAV,
with a bounded RIFF/RIFX PCM/float format and nonempty aligned sample data.
This validates container structure, not audible quality; RF64/BW64, compressed
formats and real synthesis acceptance require separate contracts/evidence.
Streaming also requires 1–63 safe-named, sorted WAV chunks, for at most 64
artifact references and 512 MiB total. Symlinks, hard links, directories, missing
outputs, malformed/empty WAVs and oversized artifacts fail without success.
Relative references remain separate from the private in-memory output root.

Cancellation/deadlines are checked through execution and artifact inspection;
post-child errors retain the reaped disposition and never authorize replay.
The caller must keep the toolkit/job directory and ancestors trusted and stable;
filesystem operations themselves cannot be preempted. This is not a filesystem
sandbox or GPU-release proof. Generic media scripts have no inferred artifact
contract and remain unwired to coordinator dispatch.

## ComfyUI job safety

The installed ComfyUI implementation supports caller-supplied canonical UUID
prompt IDs and an atomic, job-scoped `/api/jobs/{id}/cancel` operation. This is
an explicit compatibility requirement, not a guarantee about every ComfyUI
release. The older global `/interrupt` operation is not a safe fallback.
Its scoped-looking variant checks the running job before setting a global
interrupt flag; the installed jobs cancellation instead checks and signals
under the queue mutex.

The concrete branch adapter accepts a caller-configured HTTP endpoint/auth and
that explicit compatibility assertion. Preparation validates the refreshed
native profile and preserves the graph unchanged; submission refreshes again
and refuses changed runtime identity. Its private, non-Clone token exposes the
UUID before dispatch, is tied to its adapter instance, and permits submission
at most once. It remains used after rejected or uncertain dispatch. Status and
cancellation cannot target an arbitrary UUID or an unattempted token. Durable
token recovery after restart is not implemented; retained backend IDs are
inputs to future startup reconciliation, not authorization to blindly cancel.

HTTP headers and body reads check cancellation/deadlines at 25 ms intervals.
Responses are capped at 2 MiB, artifact counts at 128, and relative descriptors
reject traversal, absolute paths, control characters and URL metacharacters.
No redirect or automatic replay is allowed. History workflow/error payloads
are discarded; snapshots and failures do not expose prompts or artifact names
through Debug. Terminal state requires matching native history status flags,
not just the jobs normalizer's default `completed` label. Other ComfyUI
versions may fail this contract and must not trigger a global interrupt fallback.

The native `/prompt` handler does not reject duplicate client IDs. Therefore a
future executor must allocate a fresh ID, persist it before dispatch, submit at
most once, and reconcile that same ID after a lost response. A request timeout
cannot prove that submission failed. Local token ownership also cannot fence
another direct client that deliberately reuses that UUID.

A cancellation response acknowledges only that a cancellation was dispatched,
not completion or VRAM release. A no-op response or a missing job is not proof
of cleanup: dequeued pending work can disappear without a terminal history
entry, and history can be cleared. Unknown state retains the reservation until
reconciliation establishes safe ownership. Artifact descriptors must remain
backend-relative references; they are not trusted local filesystem paths.

The Linux process runner starts a fresh process group, keeps stdout and stderr
distinct under one combined byte budget, and bounds stdin, reads, waits, and
recovery. It retains the unreaped leader until the final group signal to avoid
PID reuse hazards. After reaping, only a non-signalling group-absence probe is
permitted; closed pipes alone do not prove cleanup. The executable must be
trusted not to escape its group/daemonize. Group cleanup does not establish
remote backend-job cancellation or GPU release. Child output remains sensitive
and must not be copied wholesale into persistent job history.

Successful supervision also requires the entire stdin payload to have been
written; early stdin closure or leader exit with unwritten bytes fails even
when the child exits zero. Pipe writes assume the normal Rust runtime's ignored
SIGPIPE disposition. Callers must not reset it to the default disposition, or
provide blocking `pre_exec` hooks: synchronous OS process creation is not
preemptible by the polling deadline. These constraints apply to trusted native
commands, not arbitrary executable sandboxing.

## Local IPC transport

The Linux transport now binds a pathname Unix socket in a pre-existing,
current-effective-UID-owned private directory and restricts the socket to 0600.
It retains the shared host/device ownership guard for the listener lifetime;
the future daemon must create this listener before loading/writing history.
Both client and accepted connections verify `SO_PEERCRED` against that UID.
This authenticates local user identity, not individual applications of that
user, and assumes trusted directory ancestors on a local filesystem.

Existing paths, including stale sockets and symlinks, are refused rather than
silently removed. Normal drop removes only a matching socket inode; replacement
entries are preserved. A crash or unverifiable post-bind setup failure can leave
a stale endpoint; explicit safe recovery remains daemon integration work.

Accept/connect and nonblocking frame reads/writes obey cancellation and absolute
operation deadlines with at most 25 ms polling waits. Whole-frame limits reuse
IPC v1; trickling bytes do not reset deadlines. A tested Linux AF_UNIX full
backlog returns EAGAIN without a pending connection, even when SO_ERROR is zero;
only connection establishment is retried, never a request frame. The connection
enforces one request/reply and poisons itself after partial/failed IO. Explicit
reply IDs must match; generic v1 errors may omit an ID on that single exchange.
Version 2 errors require the queried ID, and response versions must match their
request. V1 retains submit/status/cancel; v2 permits only results queries.
Failures expose fixed categories, not socket paths or payloads.

The caller supplies operation contexts and must bound active peers and handler
work. This transport is not a daemon/event loop, result stream, installation,
coordinator dispatch, or proof of runtime admission. Two independent test clients
exercise socket/status framing only; they do not satisfy the live inference gate.

## Bounded native result retrieval

The internal daemon now owns an in-memory `ResultBuffer` alongside durable
history. Protocol v2 fetches a page by request ID, an exclusive `after_cursor`
and a 1–128 KiB serialized-page budget. Chat content/reasoning remain separate;
decision outputs are explicit canonical-JSON parts, and media outputs are typed
local-job/backend relative artifact references. Bodies, paths and answers are
redacted in Debug and are never persisted in job history. This is authenticated
single-exchange pagination, not server-pushed streaming or artifact-file transfer.

Append is transactional, capped at 8 MiB serialized events per job and 64 MiB
total, with 64 payload slots. Native chat combines content/reasoning under its
1 MiB raw cap; decision JSON is also bounded to 1 MiB. Only explicit terminal
eviction removes a payload; active jobs are never silently evicted. A bounded
256-entry tombstone window reports expired results, while durable acceptance
retains its separate replay-prevention responsibility. The dispatch integration
must establish an explicit retention policy and reserve result capacity before
native execution; the cache itself neither admits nor replays requests.
`can_begin_reserved` is a read-only single-owner acceptance check, followed by
`begin_reserved` after durable acceptance. It reserves the full 8 MiB allowance
against the shared 64 MiB event budget before dispatch. Appends consume reserved
quota transactionally; terminal acknowledgment frees only unused quota, while
retained output remains charged. Other jobs cannot consume that reservation,
and capacity exhaustion must reject intake rather than silently evict results.

The `CpuDecisionWorker` owns one serial thread, a bounded ticket channel and one
completion channel. Its in-flight slot is freed only when completion is consumed;
refused handoffs return their owned ticket. Only accepted CPU local Nagi/Laya
routes with `ResourceClaim::None` execute; the final prepared route is checked
against both refreshed catalogs, with offline flags and CUDA visibility disabled.
The core's opt-in CPU startup path excludes GPU/hosted bindings and interrupted
or foreign retained history, and enforces CPU-only admission even when CPU/CUDA
profiles share a runtime. It is not resident-engine reconciliation. Default
catalog profiles are unchanged. Worker shutdown requests cancellation; detached
or missing completions are not cleanup proof. Native daemon event-loop wiring
and public submit actions are still in progress, not accepted as functional.

The page carries authoritative retained-history status separately from cache
availability. Restart can therefore report `succeeded` with `Unknown` cached
output, not turn a retained request into `not_found`. Reads do not consume data
or advance a stored cursor. A too-small first-page budget returns a fixed error
without skipping its event; retry with a larger budget, not another inference.
Clients validate request/version/cursor/budget correlation, contiguous cursors,
bounded native parts, relative paths and single-lane pages before consuming them.
Paging uses exact incremental serialized lengths plus a final check, rather
than repeatedly serializing the growing page.

Production native dispatch is still absent: tests seed private daemon fixtures
to exercise completed output, lost-reply reconnect and restart loss. Delivering
real executor events, cancellation cleanup and result retention across clients
remains a separate integration gate.

## Coordinator control RPC

Internal IPC submit/status/cancel now uses the durable queue's single authoritative
submission path and typed rejection categories. The older Rust `submit` interface
remains available as an anyhow-returning wrapper; it no longer exposes free-form
validation or OS details. Missing runtime bindings or unreconciled admission
return unavailable; retained IDs refuse replay even at full queue capacity.
Opaque caller IDs reuse the history validator and are never filesystem paths.

Submission is acknowledged only after candidate history is persisted and the
request enters memory. Queued cancellation is persisted before removal; active
cancellation acknowledges signalling and retains the reservation. A reserved
request may still report queued until execution is marked running; cancellation
acceptance is not terminal cleanup. Persistence failure closes admission rather
than acknowledging a change whose durable outcome may be uncertain.

The Linux single-connection handler checks its I/O context before control
mutation and never reserves or starts inference. Local filesystem persistence is
synchronous and not preemptible. If reply delivery fails after acceptance, work
stays accepted and a fresh connection queries the retained ID; it is not replayed.
Restart status reports interruption, not automatic execution. A retained ID
without a valid lifecycle state is an internal error, not not-found.

Control handling does not perform lifecycle reconciliation, inference dispatch,
or result delivery.

## Bounded control-daemon owner

The daemon's explicit data directory must remain separate from the standalone
UI/bench history directory. Both use the compatible `jobs.json` format, but the
legacy client does not acquire coordinator ownership locks. Those locks alone
therefore cannot prevent competing writes to one store. Binding now requires an
explicit absolute standalone data path and rejects directory aliases (including
symlinks/inode aliases) and nonregular or multiply linked coordinator history
files before binding/loading. A missing standalone directory is not created.
Installation must supply the actual standalone client path; this gate cannot
discover arbitrary other writers, and trusted ancestors must remain stable.
Coordinated clients must use IPC for writes, never load and persist the daemon's
history themselves. A combined read-only view is distinct from merging stores.

The internal Linux `ControlDaemon` acquires the listener and host/device locks
before loading or normalizing history. Its core is private and remains
unreconciled: status/cancel can be served, but submissions are unavailable and
there is no dispatch or admission-opening interface. Every runtime GPU binding
must be covered by a requested device lock. Data, socket and lock directories
must already exist, be private, and have the current effective owner; trusted
local directories/ancestors must not be concurrently replaced.

A consuming `run` session handles one peer at a time, one exchange per peer,
with 1–4096 connection attempts and a positive peer deadline of at most 30
seconds. Peer deadlines are capped by the caller's overall deadline and forward
its cancellation. Malformed, disconnected, denied or timed-out peers do not
replay or stop subsequent peers; listener errors stop the loop. Overall
cancellation/deadline stops intake. Filesystem persistence remains synchronous
and non-preemptible. Counts are returned without IDs, paths or payloads.

All return paths drop history ownership, unlink only the owned socket inode,
and release the retained locks. Startup failure after binding also cleans up.
Existing sockets are never reclaimed automatically. Same-user flooding can
still deny service; serial serving bounds memory/work, not fairness. This is
fixture-tested foreground infrastructure, not a CLI command, installed service,
runtime lifecycle owner or enabled inference scheduler.

## Release gates

The Linux CLI can query/cancel or fetch one results page through an explicitly supplied
private socket directory or an opt-in shared socket setting, with a single
three-second exchange. The explicit flag wins; absent configuration is an error,
not an inferred socket or automatic startup. Settings are read only for control
actions without a directory flag. A common Linux client module validates both
request IDs and reply operation, prints bounded JSON, and returns nonzero for a
typed error. It never starts the daemon, submits inference, retries a frame or
falls back to an engine. A cancelled active request retains its lease until
cleanup is acknowledged; the CLI does not claim otherwise. This is control
integration, not full run/stream or keyboard-workbench parity. `--request-results
ID` uses v2 with optional `--after-cursor` (default 0) and `--result-page-bytes`
(default 131072; allowed 1024–131072). Results print compact JSON bounded to the
page limit plus 4 KiB envelope; status/cancel retain their 2 KiB ceiling. Cursor
and page-budget flags without the results action are rejected. Fetch never
reruns inference, retries a frame, interprets native JSON parts or writes files.

No adapter is connected to a public run/switch command yet. IPC currently defines
messages, framing, authenticated bounded Linux transport, control handling and
a bounded internal control loop, not an installed dispatching daemon or
native result production or server-pushed streams. Read-only v2 page retrieval
is implemented internally, but concrete lifecycle/admission adapters, backend ID and
artifact propagation, coordinated client integration, and real TTS/local decision acceptance
must all be verified before enabling coordinated inference. The public export
remains blocked by the separate publication audit; no endpoint cutover is implied.
