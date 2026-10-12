# Managed foreground session: remaining integration contract

This records the managed integration contract, not deployed functionality or live
acceptance. The branch has a concrete managed-language executor with opt-in
allowlisted local-job composition under verification, alongside the existing
CPU-decision executor. Original repositories and services
are preserved; migration work stays on
`feature/l3ms-monorepo`.

## Ownership and ordering

One explicit foreground owner must retain the existing shared host/device lock
through intake, execution, transition, recovery and shutdown. The listener and
`ManagedRuntimeOwner` must share that same guard; acquiring separate locks is
not a substitute. Native engines run behind the accepted single-supervisor
private-network transport, not an adopted legacy manager unit.

The transition order is:

```text
close selected runtime's coordinator intake
    → dispatch/drain its previously accepted work only
    → verify every native job terminal, retaining uncertain reservations
    → stop the exact owned supervisor generation
    → verify its cgroup empty or removed
    → observe the configured physical GPU idle
    → activate the exact target, with durable intent before dispatch
    → verify owner + native endpoint/model readiness
    → reopen intake as the commit point
```

`RuntimeAdmissionClose` is instance- and generation-bound. Scoped reservations
must not run another runtime's pending work during drain. Queue counts alone do
not prove native completion or GPU release. Interrupted history without a durable
original runtime binding conservatively blocks drain for every bound runtime.
Uncertain start, inference, cancellation or cleanup never permits automatic replay.

The existing `RuntimeAdapter` lifecycle protocol and concrete service owner,
transport, GPU probe and native execution adapters should be composed rather than
replaced by another general service manager. Recovery gets an independent bounded
deadline and does not inherit caller cancellation. Failed restoration quarantines
conflicting work and retains ownership records.

## Native contracts remain different

### Managed Comfy worker composition

Comfy configuration is an explicit optional declaration; absence keeps the
language/local-job configuration unchanged. Only allowlisted manifest-bound
CUDA image workflows use this path, not native media scripts or arbitrary HTTP
engines. Each dispatch releases the resident language engine first, allocates
fresh private user/output directories without reuse, and starts a new isolated
Comfy generation on the configured physical GPU. `/system_stats` is bounded
read-only startup liveness, not proof that a workflow's model is loaded.

The existing strict private managed-session JSON may include a `comfy` object
with `profiles` (`media:qwen-image-2.1` and/or `media:krea-2`),
`python_executable` (the toolkit's exact `.venv-comfy/bin/python` path),
`comfy_checkout` (the installed native checkout),
`private_user_directory_root`, `private_output_directory_root`, and `port`.
`extra_model_paths_config` is optional; when supplied it is a pinned private
0600 JSON subset of native YAML: one `comfyui` object, a canonical absolute
`base_path`, optional boolean `is_default`, and allowlisted category values as
newline-separated relative strings. Arrays, duplicate/secret fields, traversal
and environment/user expansion are rejected. The user/output parents must
already exist with mode 0700 and must not overlap native/model directories,
transport, history, WAL or local-job roots. No environment, directory parent,
checkout or model is installed by configuration validation.

This declaration does not fence original/direct engine clients. Live startup
and switching still require their verified exclusion and the background-GPU
claimant policy; a configured UUID or momentarily idle GPU is not that proof.

The worker prepares a fresh UUID and sends the full transient ticket and
owner-issued generation to the history-owning main thread. That thread checks
the active slot and persists the candidate/generation atomically before
returning the move-only permit. A missing, late or lost reply never retries the
message or POST. The session revalidates generation and exact request/route
after receiving the permit. Positive submit/status evidence crosses a separate
typed exchange before `backend_id` is persisted; 404, unknown status and scoped
cancel acknowledgments cannot produce that evidence.

Terminal output is not publication authority. Independent bounded cleanup
requires exact Stop, cgroup release, fresh configured-GPU idle and intent
retirement before descriptors are checked against the output root pinned before
launch. Only explicit `output` descriptors mapping to fresh regular nonempty,
single-link, bounded files are accepted. Symlinks, traversal, duplicate paths,
special files, root substitution and input/temp storage are rejected. Result
references are `Backend` paths relative to the configured private output parent,
including the unique job directory; they are not invented `LocalJob` paths.
These references do not deliver bytes or prove content quality. Unknown native
attempts remain interrupted even when physical cleanup is subsequently proven.

This branch composition still needs full injected daemon/owner/native-engine
acceptance, durable crash recovery and real GPU switching. A retained Comfy or
local-job intent on restart remains fail-closed; no engine is adopted or replayed.

| Runtime | Activation/readiness | Drain |
| --- | --- | --- |
| llama-swap | Router reachability is not model readiness. Load the selected model once, then reconcile its exact state after uncertain replies. Direct llama-server `/health` must not be mistaken for router readiness. | Track coordinator work through native completion; connection loss is not server cancellation. |
| ComfyUI | Launch its direct Python entrypoint with explicit existing private output/user directories. `/system_stats` is inspection/liveness, not loaded-model readiness. | Track each submitted native UUID; cancellation dispatch is not terminal acknowledgment. |
| Breeze and local decisions | Per-request native executable/Python work, not implicitly a resident HTTP runtime. | Process-group cleanup is not proof that escaping descendants or the GPU are released. |

For Comfy, the supplied UUID is initially a **dispatch candidate**, not a known
native job. Persist a separate dispatch intent before the one POST. The branch
now stores `comfy_dispatch_candidate` separately and mints a move-only permit
after the final dispatch gate and a durable history write. The permit binds the
live coordinator, job, request, full accepted route and exact workflow graph.
The existing history `backend_id` records a verified native ID;
do not populate it from the candidate alone. Attach it only after the exact
acknowledgment or a positive read-only observation of that UUID. Missing history
or a 404 is `Unknown`, not evidence of non-submission. Joined HTTP fixtures check
the history write before POST, exact/malformed native acknowledgments,
read-only observation and no POST replay; another UUID or graph cannot reuse the
permit. Unused permits expire with the coordinator, reservation, quarantine,
cancellation or request deadline; they cannot keep their own grant alive.
Retained histories with malformed/duplicate-key/duplicate-candidate or
truncated rows fail coordinator startup without being rewritten. Standalone
history normalization remains compatible.

Legacy candidate-only rows retain their old shape. The managed path now
persists an additive versioned `comfy_service_generation` reference with the
candidate in the same cloned-history update and one atomic save. It retains
safe immutable intent/runtime/launch association, supervisor coordinator birth,
UID/network namespace identity and exact manager invocation/cgroup/PID; no
workflow, command, argv or environment enters that reference. Nested tuple,
unknown-field, duplicate-key and malformed association forms are rejected.
Deserializing this comparison data never creates an owned handle or permit.

`OwnedComfySession::prepare_dispatch` mints one fresh canonical UUID, prepares
the exact accepted workflow and records the managed candidate before returning
a distinct `ManagedComfyDispatchPermit`. The managed job wrapper cannot enter
the legacy reqwest adapter. Submit consumes that permit, checks its candidate,
request/route and exact ephemeral generation, then uses only a peer-verified
owned connection. Native operation paths are fixed: POST `/prompt`, GET
`/api/jobs/{uuid}`, and POST `/api/jobs/{uuid}/cancel`. No TCP/auth/global-interrupt
fallback or POST retry exists. Cancel is attempted once; its acknowledgment is
not terminal or cleanup evidence. Owner generation is checked before connection,
after peer verification/before request bytes, and after the response. Comfy
responses are bounded to 2 MiB; existing health probes retain their 512 KiB cap.

Concrete daemon/worker configuration, backend-ID acknowledgment accounting,
artifact mapping/retrieval and crash-safe recovery still remain before managed
Comfy dispatch is enabled. The full accepted route is transient, not a durable
replay recipe. Restart never reconstructs a submission permit or resubmits the
workflow. A candidate/reference is not evidence of native acceptance, readiness
or GPU release. Fixture evidence for these seams is not a real mixed-worker or
systemd/GPU acceptance claim.

The manifest currently permits llama-swap model startup up to 500 seconds while
the old standalone load client allows 60. The managed path needs one explicit
bounded activation horizon and reconciliation, not repeated load requests.

Read-only readiness must use the selected model's exact `ready` process state
from `/running`, plus router plaintext `/health`. Installed llama-swap v247
(`40027d62fdc0cf08131973c02db2d4a7de172df1`) exposes that state in
[its running-model handler](https://github.com/mostlygeek/llama-swap/blob/40027d62fdc0cf08131973c02db2d4a7de172df1/internal/server/api.go).
`/v1/models` collapses starting/ready processes into `loaded` and cannot prove
readiness. Forwarding `/upstream/{id}/health` can itself load/switch, so it is
not a read-only reconciliation probe. A failed/lost load acknowledgment must
never cause a second activation request merely to check status.

There is no dedicated router `/upstream/{id}/load` API in installed v247:
generic upstream routes forward the remaining path to the native backend.
For an explicitly declared llama-server-compatible backend, the branch's
activation request is one side-effectful `/upstream/{encoded-id}/health` GET,
which can start/swap that model before forwarding native health. It must not
be used as a read-only probe. Other backend health contracts stay gated until
declared and verified; the old standalone load client is not the managed
activation implementation.

## Concrete language-session entrypoint

`l3ms --coordinator-managed-session /absolute/path/private-session.json` uses its
own schema (the separate CPU daemon launcher is retired). It does not adopt the running
legacy service or enable ComfyUI or arbitrary native engines. Existing configs
remain language-only; local TTS/decisions require the explicit opt-in below.
Do not run it against an occupied GPU as a substitute for a maintenance plan.

Its schema version is `1`; `backend_contract` must be
`"llama_swap_v247_llama_server"`. Configuration must be an existing private
owner-controlled file (0600) and specify these fields explicitly:

- Paths: `repository_root`, `socket_directory`, `lock_directory`,
  `data_directory`, `standalone_data_directory`, `service_intent_directory`,
  `transport_directory`, `private_router_config`, `router_executable` and
  `router_working_directory`. Use existing absolute paths and separate private
  coordinator state from standalone history. Private directories must satisfy
  the existing ownership/mode checks; no original checkout is removed.
- Route: `coordinator_id`, `startup_profile` (a real `language:<profile-id>`),
  `supported_model_tasks` (model IDs mapped to explicit `chat` and/or
  `embeddings` task arrays), and `router_port`. Supported IDs must exist in the
  dedicated router configuration; the startup profile must resolve in the
  current native catalog with its declared tasks.
- Physical resource: `gpu_uuid`, `gpu_index` and `gpu_idle_mib`. Idle memory
  allowance is not permission to ignore an unknown compute claimant.
- Bounds: `queue_capacity` (1–128), `peer_timeout_ms` (1–30000),
  `max_connections` (1–4096), `session_timeout_seconds` and
  `request_timeout_seconds` (1–3600 each), `activation_timeout_seconds`
  (1–500), and `stop_timeout_seconds` / `recovery_timeout_seconds` (1–300 each).

Optional local-job fields must be declared together: a nonempty
`local_job_profiles` array (at most 128 unique full profile keys) and an existing
absolute `job_directory_root`. Omitting both keeps language-only behavior.
The root must be current-user owned, 0700, nonsymlinked and separate from
history/WAL/transport roots; its captured device/inode must remain unchanged.
Only `media:breeze-tts-2` with explicit CUDA/audio.cpp and local Nagi/Laya
decision routes with explicit CPU or CUDA are supported. Profiles sharing one
runtime cannot declare conflicting resource claims. Native catalog/request
validation remains authoritative; an allowlist is not permission to substitute
routes or fall back to CPU/hosted inference.

CUDA local work stops and releases the resident router before a new owned
one-shot service starts. CPU local work preserves the resident router. Native
results stay unpublished until exact Stop, fresh cgroup/device release checks,
fresh artifact validation and service-intent retirement. Stop is sent once;
bounded retries retain its generation receipt and only reobserve. A fresh
language launch can follow proven cleanup, not an uncertain native attempt.
Retained job intents still block startup rather than being adopted/replayed;
durable mixed-runtime crash recovery remains a prerequisite for deployment.

Breeze result pages and durable history carry relative `LocalJob` references
prefixed by the private job-directory name, not private absolute paths.
The HTTP gateway and `/v1/coordinator` route were retired on 2026-10-12; v3
artifact envelopes remain available only through the retained private IPC path.
Managed Comfy/Breeze publication
moves pinned file handles to the main thread after release, and appends their
references transactionally; a failed append drops the new handles. Reads require
durable `Succeeded` status and retained result availability. Exact request/kind/
relative-reference lookup selects an existing capability, never a client path
or a global `artifact_root`. Pinned root/file identities are checked before and
after each 1–65536-byte read. Nonzero offsets require the opaque version returned
by the initial chunk. Changed files or tokens return no bytes; unknown/expired
results likewise do not grant access. The registry is memory-only, capped at
128 requests, 128 artifacts per batch and 256 retained file/root descriptors.
Failed jobs and queried expired results release their handles; restart never
reconstructs handles from history. Trusted private roots are still required;
this is not a hostile same-user filesystem sandbox or native audio streaming.
Do not expose the private job directory through a generic static file server.
Full-file saving/TUI consumption and mixed-worker/live acceptance remain open.
Optional native decision latency/usage
now reach result pages as additive `DecisionMetadataJsonPart` events: exact
bounded integer nanoseconds and unchanged provider-specific usage fields.
Answers and metadata publish transactionally. Legacy answer-only pages stay
unchanged, but metadata pages require upgraded strict clients. The consumer
exposes typed metadata separately. Explicit CLI output includes an optional
`native_decision_metadata` sibling to the unchanged answer output, omitted when
absent. The private TUI result panel shows it only after explicit observation;
request resets/unavailable observations clear it with the output. Native timing
is not queue, cold-start or coordinator timing; usage is not chat-token usage.

The dedicated router file is strict duplicate-free JSON, which is also a YAML
subset accepted by the router. It must be separate from the original
`llama-swap.yaml`, contain an object-valued `models` map, contain no `apiKeys`
at any depth, and use no ambient `${env.*}` expansion except explicit
`${env.L3MS_ROOT}`. Its device/inode and exact bytes are retained and checked
before activation and inference; config drift closes the session, not a silent
hot reload. Operator-provided native commands remain trusted configuration, not
proof that arbitrary engines implement the declared backend contract.

The listener owns history, result retention and reservations. A scoped worker
borrows that same host lock for owner/transport/native HTTP operations. Accepted
routes are refreshed before activation and again before inference. Native chat
deltas, usage and stream completion reach the existing bounded result interface;
embeddings retain typed vectors/usage. Readiness does not authorize unsupported
models or tasks. On uncertainty, retain/quarantine until exact-service stop,
cgroup release and physical GPU idle are proven. Read-only cleanup observation
retries retain the original stop receipt and never issue a second Stop. An
uncertain managed-session CLI report exits nonzero.

Offline fixtures and a successful build do not prove real managed language
execution, cross-client acceptance or cross-runtime switching. Those remain
separate gates before deployment.

## Explicit configuration, no ambient assumptions

Keep the CPU launcher schema unchanged. A separate strict private configuration
must specify canonical repository and private socket/history/WAL/transport paths,
standalone history exclusion, existing artifact directories, physical GPU UUID,
index and idle-memory allowance, exact engine entrypoints/ports and operation
limits. Missing environment, models or directories must fail before activation.

Pass canonical `L3MS_ROOT` through a bounded non-secret launch environment; do not
assume the user's shell environment reaches the systemd user manager. Preserve
literal argv and native manifest authority. Credentials must not enter argv,
history or logs. Installed services require separately verified deployment paths.

The initial private-router implementation requires an explicitly prepared,
existing dedicated configuration without `apiKeys`; it does not strip authentication
from the deployed configuration or inherit the shell/user manager's secrets.
The original authenticated public/loopback router remains unchanged. This private
configuration is valid only with verified coordinator-only Unix ingress and the
owned network namespace; it must never be reused for a public router listener.

Exclusive coordinated ingress is valid only while engines and their children
remain inside the owned private namespace and access is restricted to the captured
coordinator. The running legacy service is not fenced by a coordinator lock;
legacy direct clients cannot coexist with a claim of automatic exclusive GPU
switching. The coordinator HTTP gateway is retired; no replacement proxy or
automatic ingress fencing is provided by the direct native workflows.

## Acceptance authority

On 2026-10-07 the owner approved creating the isolated Nagi environment and
downloading its required weights, and a bounded maintenance test that stops the
current llama-swap service and restores its prior state. This authorizes testing,
not permanent cutover, original-repository retirement, publication or history
rewriting. Capture fresh service/GPU state before that window and verify restoration
after every success/failure path. Real acceptance still requires actual native
outputs; fixtures, model discovery and successful downloads are not substitutes.
