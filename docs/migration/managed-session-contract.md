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

The candidate row retains job/profile/task/request identifiers, but not a
service-generation identity or a persisted full-route snapshot. Those owned
generation/recovery checks and the actual worker/coordinator handoff remain
required before managed Comfy dispatch is enabled. Restart never reconstructs
a submission permit or resubmits the workflow from this field. A candidate is
not evidence of native acceptance, readiness or GPU release.

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

`l3ms --coordinator-managed-session /absolute/path/private-session.json` uses a
separate schema from `--coordinator-cpu-daemon`. It does not adopt the running
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
The current HTTP gateway forwards coordinator envelopes and references only;
it does **not** serve artifact bytes. `artifact_root` settings alone do not
establish a retrieval route or a matching pinned root. Byte retrieval still
needs request/job-bound reference authorization, no-follow root/file access,
bounded reads and drift rejection. Do not expose the private job directory
through a generic static file server. Optional native decision latency/usage
also remain absent from client result pages, despite retention in job packets.

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
switching. The coordinator HTTP gateway forwards native coordinator envelopes,
not arbitrary inference requests.

## Acceptance authority

On 2026-10-07 the owner approved creating the isolated Nagi environment and
downloading its required weights, and a bounded maintenance test that stops the
current llama-swap service and restores its prior state. This authorizes testing,
not permanent cutover, original-repository retirement, publication or history
rewriting. Capture fresh service/GPU state before that window and verify restoration
after every success/failure path. Real acceptance still requires actual native
outputs; fixtures, model discovery and successful downloads are not substitutes.
