# Explicit idle-GPU native work

CUDA-local Nagi/Laya decisions, `media:breeze-tts-2` speech and HeartMuLa music use the same
foreground runner as CPU work. This is an **operator-managed idle window**, not
automatic switching of a live router. No service is adopted, stopped, installed
or restored automatically. Original standalone commands remain available.

Before running, the operator must quiesce other clients, make the selected GPU
idle, and keep those clients quiesced through execution and release checks.
Do not pause a service while it is doing active work. A router's `loaded` model
status is not evidence that its requests are idle.

## One explicit configuration for CLI and TUI

Set `L3MS_NATIVE_GPU` before starting L3MS. Its value is strict JSON:

```json
{
  "uuid": "GPU-<full-lowercase-physical-uuid>",
  "index": 0,
  "idle_mib": 512,
  "lock_dir": "/absolute/shared/private/l3ms-locks",
  "clients_quiesced": true
}
```

Replace the illustrative UUID and index with the physical device's actual
identity. `idle_mib` is an explicit allowance for idle memory, not an automatically
learned baseline; choose it for your machine. The lock directory must already
exist, be owned by the current user and be private. All cooperating L3MS
processes/checkouts must use the **same directory**. Do not derive a separate
lock directory per job or delete/replace its persistent lock inodes.

`clients_quiesced: true` is the operator's assertion, not an engine admission
fence. Missing/invalid configuration refuses CUDA work. CPU, hosted decisions,
existing language HTTP and Comfy endpoints do not silently acquire this device
or fall back to another execution mode.

Use the existing native CLI shapes, selecting the declared CUDA profile:

```sh
l3ms --run-profile decisions:nagi-smol --task decision --input input.json
l3ms --run-profile media:breeze-tts-2 --task tts --input speech.json --output-dir /absolute/private/fresh-output
```

The Stack Catalog typed form (`f`, then `Ctrl+Enter`) uses the same operation.
Speech supports standard, design, cloning with a reference transcript, and
completed streaming chunks; see [CPU/native input shapes](cpu-native-workflows.md).
Completed chunks are not live playback. The selected runtime and weights must
already be installed; there are no implicit downloads or simulated decisions.
SemIf/OpenJev and unrelated media scripts are not admitted by this path.

### HeartMuLa music

```sh
l3ms --run-profile media:heartmula-music --task music --input music.json --output-dir /absolute/private/fresh-output
```

`music.json` contains a native argv array, not a shell command:

```json
{"args":["--prompt","short ambient instrumental music","--instrumental","--duration","5","--seed","42"]}
```

The wrapper supports the script's prompt/text/file, tags, lyrics/file,
instrumental, duration, steps, seed, temperature, top-k and guidance options.
Incomplete/duplicate/conflicting options, unknown flags, help, backend and all
output overrides are refused. Numeric interpretation remains native; original
standalone commands retain their behavior. The shared runner supplies both
output file and directory, so environment defaults cannot create an unrelated
output directory. The final WAV is validated using the existing PCM/float audio
checks, returned under `artifacts`, and shown by path in the UI. Music has no
speech-mode field. Prompts, lyrics and argv stay out of status/log/history.

This integration has model-free native-script fixtures, not real HeartMuLa
generation or music-quality acceptance. GPU inference/release, installed native
WAV compatibility and interactive acceptance still need a live test.

## What the wrapper checks

1. Acquire the existing advisory host/device locks. A competitor is refused,
   not preempted or silently queued across processes.
2. Observe exact GPU UUID/index, memory allowance and no compute claimants with
   the existing bounded NVIDIA probe. Unknown/occupied states refuse dispatch.
3. Create one payload-free `native-gpu-recovery-required` marker and run the
   existing native command once, bound to the selected physical UUID.
4. After child cleanup, observe idle again with a fresh bounded recovery context,
   including after cancellation/timeout. Only then clear the unchanged marker
   and return native results. Uncertain child cleanup or release leaves it intact.

The probe is point-in-time evidence, not a lease or external-client fence. The
operator quiescence requirement is essential. Desktop/graphics processes are
not enumerated as compute claimants; their memory must fit the explicit allowance.

## Recovery and current limits

An interrupted job or uncertain release leaves the recovery marker; later
cooperating CUDA invocations refuse it even after the original CLI has exited.
No job is replayed, no guardian daemon is installed, and no payload is stored in
the marker. If it exists, inspect Jobs and native processes, establish that owned
work has actually stopped and the selected GPU is idle, then remove the marker
manually from the configured lock directory. Do not clear it merely to retry.
This is not a power-loss-safe journal or automatic crash recovery.

Native dispatch, configuration/serialization, cancellation/release and form
fixtures do not establish real CUDA inference acceptance. Live GPU inference,
automatic language→media→language handoff and verified restoration remain open.
