# Managed foreground session: remaining integration contract

This records the next implementation slice, not deployed functionality or live
acceptance. The current foreground executor remains CPU-decision-only. Original
repositories and services are preserved; migration work stays on
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

The manifest currently permits llama-swap model startup up to 500 seconds while
the old standalone load client allows 60. The managed path needs one explicit
bounded activation horizon and reconciliation, not repeated load requests.

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
