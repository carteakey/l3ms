# CPU Breeze cloning and completed chunks

2026-10-12 cached-weight smoke tests reused the installed CPU-capable audio.cpp
runtime and Breeze Q8_0 model/spec, seed 0 and two CPU threads. No services changed
or weights downloaded; CPU execution was selected explicitly. The reference was earlier synthetic
speech, not a person's voice.

| Operation | Whole-CLI wall time | Validated 24 kHz mono PCM16 output |
| --- | --- | --- |
| Clone | 14.20 seconds | 61,440-frame final WAV |
| Streaming TTS | 11.52 seconds | 46,080-frame final WAV; 7,680/38,400-frame chunks |

Owned process cleanup was confirmed. These are execution/artifact observations,
not speaker-similarity, voice-quality, latency guarantees or live-playback tests.
Chunk files are retrieved after completion; the UI does not play incremental audio.

The reusable `maintenance/probe-breeze-cpu-modes.py` runner requires an existing
private fixture directory, explicit audio.cpp/model roots, and reference audio
plus transcript for cloning. It refuses repeat dispatch into a case directory,
never retries, and retains exact invocation/environment, input, script/spec copies,
logs, receipt and artifacts. See the monorepo's
[exact private acceptance record](../../../../docs/migration/direct-speech-acceptance.md)
and [portable CPU guide](../../../../docs/cpu-native-workflows.md).
