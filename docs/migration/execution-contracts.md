# Native execution contracts

These are branch-local implementation contracts, not a deployed coordinator or
live acceptance record. See [status.md](status.md) for phase gates and evidence.

## Ownership and dispatch

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
allowed. Generic ComfyUI workflow graphs require a separate HTTP adapter.

## ComfyUI job safety

The installed ComfyUI implementation supports caller-supplied canonical UUID
prompt IDs and an atomic, job-scoped `/api/jobs/{id}/cancel` operation. This is
an explicit compatibility requirement, not a guarantee about every ComfyUI
release. The older global `/interrupt` operation is not a safe fallback.
Its scoped-looking variant checks the running job before setting a global
interrupt flag; the installed jobs cancellation instead checks and signals
under the queue mutex.

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

## Release gates

No adapter is connected to a public run/switch command yet. IPC currently defines
messages and framing, not a listening daemon, authenticated peer transport, or
result/stream delivery. Concrete lifecycle/admission adapters, backend ID and
artifact propagation, coordinated client integration, and real TTS/local decision acceptance
must all be verified before enabling coordinated inference. The public export
remains blocked by the separate publication audit; no endpoint cutover is implied.
