# Managed engine isolation: measured launch gate

Historical investigation, not the current runtime design. The upstream-first
port retires the branch-added managed supervisor and its two opt-in probe
scripts. The old commands below are evidence from that investigation, not
instructions supported by the current checkout. Source and probe history remain
recoverable from Git; no live service, stored job or original checkout is removed.
Use native llama-swap serving and existing native scripts instead.

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

## Single-supervisor probe — 2026-10-07

The bounded model-free replacement probe passed on this host:

```sh
cargo build --locked --offline
python3 maintenance/probe-isolated-supervisor.py --self-test
python3 maintenance/probe-isolated-supervisor.py --run
```

It starts one unique transient `Type=notify`, `PrivateNetwork=yes` service with
the actual branch helper and a Python standard-library HTTP backend. A named
manager-opened coordinator namespace descriptor is passed to the helper. The
probe's own process is the authorized Unix client; no model, GPU, installed
service, dependency environment or host TCP listener is used. The unit has a
20-second runtime ceiling, five-second start and two-second stop timeouts.

Measured evidence: host namespace `(5, 4026531833)`; helper and backend both
`(5, 4026535525)`. The private Unix socket's peer PID matched the helper, the
backend parent matched that helper, and both belonged to the captured service
cgroup. One bounded native proof GET succeeded (78 request / 247 response
bytes). Generation, MainPID and isolation properties were checked again before
stopping. Captured helper/backend PID birth identities were absent afterward,
and the owned cgroup was empty or removed; only then was the probe's private
temporary directory removed. No probe units remained loaded. Existing
llama-swap stayed running at PID 138426, ComfyUI stayed inactive, and GPU memory
remained 44 MiB. The parser self-test passes nine framing cases without a unit.

This accepts the model-free descriptor/namespace/byte-transport/cleanup topology,
not full runtime lifecycle or inference admission. Native engine health, native
asynchronous draining, real model/TTS execution, coordinated GPU release and
crash/restart acceptance still remain. `READY=1` means helper transport setup,
not backend health. The kernel namespace refusal remains; no boolean bypass or
host-TCP fallback is permitted. Unknown start/generation or incomplete cleanup
retains the private probe directory and relies on the bounded unit lifetime;
it never authorizes stopping an unverified unit.
