# Direct CPU speech acceptance — 2026-10-10

Partial replacement evidence under the [simplification plan](simplification-plan.md).
This does not complete GPU switching, other media tasks or the full port.

## Real execution

The final branch debug CLI ran:

```bash
target/debug/l3ms --run-profile media:breeze-tts-2-cpu --task tts \
  --input /tmp/l3ms-nagi-accept-tts-lbxHPo/input.json \
  --output-dir /tmp/l3ms-nagi-accept-tts-lbxHPo/speech-final \
  --task-timeout-ms 120000
```

Input: `{"text":"Hello."}`. Seed: 0; CPU threads: 2; native mode: standard.
`L3MS_ROOT` pointed at this branch, `L3MS_AUDIO_CPP_ROOT` at the installed
`/home/kchauhan/repos/audio.cpp`, and `L3MS_BREEZE_TTS_THREADS=2`.
The existing bounded offline harness supplied empty CUDA visibility, isolated
private data/cache settings and a 150-second outer deadline. The recipe passed
`--backend cpu`; no CUDA fallback, service mutation, inference retry, installation
or download was used. The existing CUDA-named build supports both CPU and CUDA;
its directory name alone is not evidence of the chosen backend.

Cached model: `/home/kchauhan/models/media/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf`
(5,079,668,352 bytes). Installed specification:
`/home/kchauhan/repos/audio.cpp/model_specs/breeze_tts.json`.

Final wall time: **4.809005044 seconds**, including CLI startup and cleanup.
Artifact: `/tmp/l3ms-nagi-accept-tts-lbxHPo/speech-final/result.wav`.
It is a 19,244-byte PCM16 mono WAV, 24,000 Hz, 9,600 frames (0.4 seconds),
with nonzero samples (peak absolute PCM16 amplitude 9,295).
The harness confirmed all observed child process birth identities were released.
An earlier deliberately separate CPU run also succeeded; its wall time was not
recorded. No voice-quality judgment or latency guarantee is inferred.

The mode-0700 local fixture retains `probe.py`, `input.json`,
`executed-breeze-wrapper.sh`, `executed-model-spec.json`, `receipt.json` and
the WAVs. These temporary artifacts are outside Git and may be removed by normal
temporary-directory cleanup. The receipt is the bounded acceptance log;
native payload-bearing diagnostics are intentionally not persisted.

## Shared CLI and UI behavior

The same typed request builder and direct native runner handle decisions and
speech. Standard CPU speech rejects instruction/reference/mode extensions.
Validated files, not audio bytes or coordinator references, form the result.
CLI output directories are existing private owned absolute directories; the
TUI creates fresh directories without overwriting prior jobs. CPU route drift,
unsafe files, missing weights and invalid backends fail closed.

UI fixtures cover success, metadata-only history, path/profile gating,
cancellation and preserving earlier artifacts when job identifiers repeat.
Native fixtures cover backend visibility and output validation. Manual TUI
acceptance, cloning/design/streaming, GPU orchestration and deletion of old
platform callers remain separate work. No platform module was removed.

## CPU voice design follow-up — 2026-10-12

The shared CLI/form now also accepts `--task design` with strict text/instruction
JSON. Final-binary acceptance used `input.json` containing
`{"text":"Hello.","instruction":"Speak gently with a calm low voice."}` and:

```sh
target/debug/l3ms --run-profile media:breeze-tts-2-cpu --task design --input /tmp/l3ms-cpu-design-UdXVk0/input.json --output-dir /tmp/l3ms-cpu-design-UdXVk0/output-final --task-timeout-ms 120000
```

The private fixture retains `probe.py`, input, native wrapper/spec copies,
`receipt.json` and WAVs. Two deliberately separate native verification runs
passed (6.93 and final **6.09 seconds wall time**), seed 0, two CPU threads,
cached Q8_0 weights, empty CUDA visibility and offline settings. The bounded
harness confirmed process identities released. Final WAV: PCM16 mono 24 kHz,
15,360 frames (0.64 seconds), peak 7,488. No service was stopped or model installed.
This proves the instruction/CPU execution and artifact path, not voice quality.

An initial harness attempt omitted HOME and used the wrong model-root variable;
read-only `--check` reproduced a shell preflight error before native inference.
That environment was corrected, not an ambiguous inference replay. The failed
preflight receipt is retained. The final implementation shares the route predicate
and existing runner/WAV contract; the subsequent CPU modes are recorded below.
GPU switching remains open.

## CPU cloning and completed streaming chunks — 2026-10-12

Reusable runner: `toolkits/l2m2/maintenance/probe-breeze-cpu-modes.py`; it uses the
existing bounded `run_cli` process-identity probe, refuses a reused case directory
and never retries. Fixture: `/tmp/l3ms-cpu-modes-PBc4zg` (current UID, 0700).
Before execution, MemAvailable was 48,452,824 KiB and no audiocpp_cli was running.
The same installed audio.cpp binary, 5,079,668,352-byte Q8_0 model and local spec
were reused. No download, environment installation, service or GPU change.

Exact invocations, seed 0, two CPU threads, CPU backend, empty CUDA visibility,
offline environment, native script/spec/probe copies, inputs, logs, receipts and
WAVs are retained in the fixture:

```sh
python3 toolkits/l2m2/maintenance/probe-breeze-cpu-modes.py --mode clon --fixture /tmp/l3ms-cpu-modes-PBc4zg --audio-cpp-root /home/kchauhan/repos/audio.cpp --media-root /home/kchauhan/models/media --voice-reference /tmp/l3ms-cpu-design-UdXVk0/output-final/result.wav --reference-text 'Hello.' > /tmp/l3ms-cpu-modes-PBc4zg/clone.log 2>&1
python3 toolkits/l2m2/maintenance/probe-breeze-cpu-modes.py --mode streaming --fixture /tmp/l3ms-cpu-modes-PBc4zg --audio-cpp-root /home/kchauhan/repos/audio.cpp --media-root /home/kchauhan/models/media > /tmp/l3ms-cpu-modes-PBc4zg/streaming.log 2>&1
```

Both used public synthetic text, `Hello from the local model stack.` Cloning
used only the earlier synthetic design WAV, not a person's voice. Whole-CLI
wall time: **14.197 seconds** clone, **11.515 seconds** streaming. All artifacts
were nonempty 24 kHz mono PCM16; owned process identities were confirmed released.
Clone final: 61,440 frames, peak 16,987. Streaming final: 46,080 frames, peak 17,890;
two native chunks: 7,680 and 38,400 frames. Copies are returned only after native
completion. This proves execution, bounded artifact delivery and process cleanup,
not speaker similarity, voice quality, live playback or GPU lifecycle acceptance.
