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
