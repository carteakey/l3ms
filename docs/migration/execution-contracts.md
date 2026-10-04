# Native execution contracts

These are branch-local implementation contracts, not a deployed coordinator or
live acceptance record. See [status.md](status.md) for phase gates and evidence.

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
Failures expose fixed categories, not socket paths or payloads.

The caller supplies operation contexts and must bound active peers and handler
work. This transport is not a daemon/event loop, result stream, installation,
coordinator dispatch, or proof of runtime admission. Two independent test clients
exercise socket/status framing only; they do not satisfy the live inference gate.

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

The Linux CLI can query/cancel one retained request through an explicitly supplied
private socket directory with a single three-second exchange. It validates both
request IDs and reply operation, prints bounded JSON, and returns nonzero for a
typed error. It never starts the daemon, submits inference, retries a frame or
falls back to an engine. A cancelled active request retains its lease until
cleanup is acknowledged; the CLI does not claim otherwise. This is control
integration, not full run/result/stream or keyboard-workbench parity.

No adapter is connected to a public run/switch command yet. IPC currently defines
messages, framing, authenticated bounded Linux transport, control handling and
a bounded internal control loop, not an installed dispatching daemon or
result/stream delivery. Concrete lifecycle/admission adapters, backend ID and
artifact propagation, coordinated client integration, and real TTS/local decision acceptance
must all be verified before enabling coordinated inference. The public export
remains blocked by the separate publication audit; no endpoint cutover is implied.
