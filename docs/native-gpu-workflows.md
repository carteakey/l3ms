# Explicit idle-GPU native work

> **CUDA Breeze speech and images no longer use this window.** `audiocpp_auto` and
> `comfyui_auto` are declared in `llama-swap.yaml`, so llama-swap swaps them in and
> out like any model, and `l3ms --run-profile media:breeze-tts-2` (and the TUI form)
> call llama-swap's `/audiocpp/` route directly; see
> [media through llama-swap](migration/media-through-llama-swap.md). The
> pause/restore handoff that used to stop `llama-swap.service` was removed. This
> page now covers only direct scripts that bypass the router.

HeartMuLa and H3 audio also use the router on this branch; they no longer need
`L3MS_NATIVE_GPU`. Their live acceptance remains pending; see [TODO](../TODO.md).

CUDA-local Nagi/Laya decisions, H3 video, LTX and YuE2 use the same
foreground runner as CPU work. This is an **operator-managed idle window**: no
service is stopped or restored, and other clients are not fenced. Original
standalone commands remain available.
Blender headless work shares this admission/release guard but keeps its native
scene-selected backend: its manifest does not declare CPU or CUDA, and no CUDA
connection or visibility override is fabricated for it.

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

On the supported KDE desktop, an explicit `"kde_desktop_baseline": true` member
may be added to the same JSON. It permits only the captured compositor child
from the fixed `plasma-kwin_wayland.service`, with its root-owned packaged unit,
wrapper command, invocation/main-process identity and exact cgroup member set
verified before and after each GPU observation. The launcher's actual executable
must match the installed root-owned, non-writable wrapper by path and inode.
The selected child must have that captured launcher as its direct parent, the
expected argv[0], UID, process birth and cgroup. Its protected `/proc/exe` is not
attested: argv[0] is a selector, not executable proof. This trusts the local user
manager and launcher environment (including executable lookup); it does not
protect against hostile same-UID injection or a spoofed launch environment.
Changed generations/membership, custom drop-ins, unsupported unit
layouts and all other GPU claimants still refuse. This is not an arbitrary PID
allowlist, an exemption for all graphics programs, or permission to stop KDE.
The existing `idle_mib` limit still bounds total memory before dispatch and after
release; desktop memory is not subtracted or automatically learned. Default is
off. This remains point-in-time evidence, not a fence against new GPU clients.

Read-only host probes are explicitly opt-in (they are skipped by ordinary CI):

```sh
cargo test --lib real_kde_desktop_baseline_capture -- --ignored
# Only while the device has no compute job beyond the desktop baseline:
L3MS_LIVE_GPU_UUID=GPU-your-physical-uuid cargo test --lib real_kde_baseline_accepts_idle_desktop -- --ignored
# Only while an unrelated compute claimant is known to be present:
L3MS_LIVE_GPU_UUID=GPU-your-physical-uuid cargo test --lib real_kde_baseline_refuses_foreign_compute -- --ignored
```

These inspect manager/process/GPU state only; neither starts inference nor
changes a service. A rejection test is not a successful switching test.

Use the existing native CLI shape, selecting the declared CUDA profile:

```sh
l3ms --run-profile decisions:nagi-smol --task decision --input input.json
```

The Stack Catalog typed form (`f`, then `Ctrl+Enter`) uses the same operation.
The selected runtime and weights must already be installed; there are no implicit
downloads or simulated decisions. SemIf/OpenJev and unrelated media scripts are
not admitted by this path.

### Breeze speech (routed, no window)

```sh
export LLAMA_SWAP_URL=http://127.0.0.1:8080 LLAMA_SWAP_API_KEY=...
l3ms --run-profile media:breeze-tts-2 --task tts --input speech.json --output-dir /absolute/private/fresh-output
```

Standard, design, cloning (reference WAV plus transcript) and streaming take the
same JSON shapes as the CPU profile ([input shapes](cpu-native-workflows.md)). The
request goes to the router, which evicts the resident model, starts audio.cpp and
restores nothing on its own: the next request for the language model reloads it.
Run it when evicting that model is acceptable. Streaming returns after completion
with `result.wav.chunks/chunk_NNNN.wav`; it is not live playback. In the TUI, connect
Chat first; the form reuses that confirmed endpoint and key.

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

### MiniMax H3 audio and video

Use `media:minimax-h3` with `--task music`, `tts` or `video`. Each accepts the same
`{"args":["--prompt","native H3 prompt","--seed","42"]}` shape, not a Breeze
speech object or Comfy graph. The UI offers a declared task selector and native
argv field. The runner supplies `--audio-only` or `--video` from the selected task;
caller mode/output/backend/help overrides are refused.

Prompt/text/file, steps, seed, width, height and frames remain native. Direct
`--request-option` supports only the wrapper-evidenced layerwise keys:
`text_layerwise`, `text_layerwise_batch`, `dit_layerwise`, `dit_layerwise_batch`,
`dit_mlp_chunk_tokens`. Unknown or duplicate keys and `return_video`/dimension/
output overrides are refused; the original standalone script still permits its
original options.

Outputs stay in the private root: `result.wav`, video mode's retained
`minimax_h3_video_rgb24.json`, and optional `result.mp4`. Native mux temporaries
also use that root. All fixed targets must be absent before dispatch. The raw
JSON validator checks positive dimensions/frame count/fps, duplicate fields,
hex shape and exact RGB24 byte count with bounded, cancellation-aware reading;
additional native metadata fields are allowed. WAV validation is shared; MP4
checking is header/container sanity, not codec/playback/quality verification.
The existing aggregate 512 MiB artifact cap applies.

If muxing tools are unavailable, the native script's raw-only success is retained;
no tools are installed and no MP4 is invented. Native failure produces no result
references or inference retry, even if partial private files remain. Fixtures
cover audio, raw-only and raw+MP4 results, but real H3 output/decoder/GPU behavior
and interactive acceptance still require the original runtime and a live window.

### LTX-2.5 synchronized audio/video

Use `media:ltx-2.5` with its declared `video` or `music` task. Both invoke the
existing distilled pipeline and return synchronized audio/video in `result.mp4`;
the music label does not select a fabricated audio-only pipeline.

```json
{"args":["--prompt","a paper boat crossing a puddle","--frames","121","--seed","42"]}
```

Native prompt/file, frames, seed, transformer, quantization and offload options
remain available, including repeated `--image PATH FRAME STRENGTH` conditioning.
The wrapper refuses conflicting/incomplete/unknown flags and caller output
overrides, and supplies a fresh private MP4 target. Numeric interpretation stays
with the native pipeline. This is not audio-to-video or video-to-video support.

The LTX checkout, five model files and its `.venv/bin/python` must already be
prepared. Workbench execution sets `L3MS_LTX_PREPARED_ONLY=1`: the native script
uses that interpreter directly with Hugging Face/Transformers offline settings,
without `uv` resolution, syncing or interpreter downloads. Normal standalone
execution keeps its original `uv run` behavior. Missing assets fail; no fallback
or automatic setup runs. MP4 checks are bounded header sanity, not proof of
playable synchronized tracks or generation quality. Real model/GPU/codec and
interactive acceptance remain open.

### Blender headless Python and rendering

Use `media:blender-3d` with `3d`, `render` or `image`, plus a private output
directory and the same explicit GPU-window configuration. `3d` accepts exactly
one `--code PYTHON` or `--script FILE`; `render`/`image` accept `--render FILE`.
For example, the input file for `3d` is:

```json
{"args":["--code","import bpy; bpy.ops.mesh.primitive_monkey_add()"]}
```

These are trusted local commands with arbitrary user-level filesystem/network
effects, not a sandbox or automatic prompt-to-code generator. Code/script success
reports completed execution with no invented artifact (`artifacts: []`). Render
returns fixed private `result_0001.png`, with regular-file, bounded PNG/IHDR
header checks; these are not decoding, visual-quality or real-render acceptance.

Only the native headless script mode runs. Caller output/help/prompt/GUI/check/
port controls and mismatched actions/tasks refuse. Native Python exceptions exit
nonzero, no logs/code are persisted, and inference is never retried. GPU admission
is conservative even when the scene ultimately renders on CPU; the backend is
not changed or labelled CUDA. An unknown claimant still refuses the shared window.
For GPU-rendered scenes, configure the GPU the scene actually uses. This guard
observes that one device; it does not constrain Blender's device settings or
establish ownership/release of a multi-GPU scene.

The existing GUI/MCP commands remain standalone and externally owned. They are
not adopted, stopped, or counted as completed headless jobs; `--prompt` still only
prints guidance. See [Blender workflow](../toolkits/l2m2/docs/blender-mcp-setup.md).
Standalone and root CLI CPU headless rendering and a real software-rendered
GUI/MCP scene round trip have passed, but the GUI crashed during shutdown.
Manual Stack rendering, clean interactive lifecycle and other GPU/device
qualification are separate; see [Blender evidence](migration/blender-acceptance.md).

## What the wrapper checks

1. Acquire the existing advisory host/device locks. A competitor is refused,
   not preempted or silently queued across processes.
2. Observe exact GPU UUID/index, memory allowance and no compute claimants with
   the existing bounded NVIDIA probe. Unknown/occupied states refuse dispatch.
3. Create one payload-free `native-gpu-recovery-required` marker and run the
   existing native command once. Declared CUDA profiles are bound to the selected
   physical UUID; Blender retains its native backend/device configuration.
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
