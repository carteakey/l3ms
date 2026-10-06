# Managed engine isolation: measured launch gate

On this host's systemd 262 user manager, the proposed two-service topology fails
the actual network-namespace identity gate. Both units report `PrivateNetwork=yes`
and the bridge reports the exact `JoinsNamespaceOf` target, but the bridge process
is in the coordinator/host namespace rather than the engine namespace.

The model-free, explicitly opt-in probe is:

```sh
python3 maintenance/probe-private-engine-netns.py --run
```

It starts only uniquely named `/usr/bin/sleep` fixtures, each bounded by a
ten-second manager lifetime. Cleanup stops bridge before engine only after
matching captured InvocationIDs; unknown ownership fails rather than stopping
an arbitrary unit. No model, GPU, dependency environment or existing service is
involved. This diagnostic is not part of CI or an installation step.

Measured on 2026-10-05:

| Probe | Result |
| --- | --- |
| Joined private engine/bridge | Fails: bridge equals host namespace. |
| Same with manager-opened engine namespace FD | Same failure; FD opening itself succeeds. |
| Same with two-second startup delay | Same failure. |
| `--no-join` control | Passes: each process has its own private namespace, both different from host. |

This identifies namespace joining as the load-bearing configuration difference;
it does not prove the exact upstream implementation cause. The documented
private-network behavior and user-manager conditions are insufficient as live
proof. See the [systemd 262 private-network contract](https://raw.githubusercontent.com/systemd/systemd/v262/man/systemd.exec.xml)
and [user-manager namespace conditions](https://raw.githubusercontent.com/systemd/systemd/v262/man/system-or-user-ns.xml).

The selected replacement is one isolated, owned service containing a typed
supervisor, its backend child and the private byte bridge. They inherit the same
actual namespace; the helper must still verify it differs from the coordinator
before binding/readiness. One captured service generation/cgroup owns the whole
pair. A generic one-shot process runner is not a long-lived supervisor.

That replacement is being implemented, not accepted yet. It still needs helper
lifetime, descriptor handoff, readiness, cancellation, exact cgroup cleanup and
real engine/GPU acceptance. The existing helper's kernel namespace refusal is
retained; no boolean bypass or host-TCP fallback is permitted.
