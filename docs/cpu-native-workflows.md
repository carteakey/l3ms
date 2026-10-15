# Explicit CPU workflows

These Linux profiles use installed native runtimes and cached weights. They
do not install models, start services, silently substitute CPU for a CUDA
profile, or retry inference. The original CUDA profiles remain separate.
See the [decision setup](../README.md) and
[audio.cpp setup](../docs/audio-cpp-runtimes.md).

## Nagi-Smol CPU decisions

```sh
l3ms --run-profile decisions:nagi-smol-cpu --task decision --input input.json --task-timeout-ms 120000
```

Input is native JSON with exactly `state` and `questions`; output contains
native answers and optional latency/usage. Stack Catalog (`F8`) also exposes
the native decision form with `f`, run with `Ctrl+Enter`, cancel with `c`.
History stores metadata, not decision payloads.

An offline cached-weight acceptance smoke run returned native choice output
and approximately **1.61 seconds native latency**, with child process cleanup
confirmed. This is one execution observation, not an accuracy, deadline,
stability or performance guarantee. The manifest's expected CPU latency remains
a declaration, distinct from a benchmark distribution.

## Breeze CPU speech

```sh
mkdir -m 700 "$PWD/speech-output"
l3ms --run-profile media:breeze-tts-2-cpu --task tts --input speech.json --output-dir "$PWD/speech-output" --task-timeout-ms 120000
```

Standard input is `{"text":"Hello."}`. The output directory must be fresh,
private and owned by the caller. The recipe explicitly selects the CPU backend;
the native script's default remains CUDA. This wrapper profile currently
supports standard text-only speech, offline voice design, reference-audio cloning and chunked
TTS output. For design, use `--task design` with
`{"text":"Hello.","instruction":"Speak gently with a calm low voice."}`.
An instruction is required for design; standard TTS does not accept one.
The CLI returns validated WAV paths; the TUI allocates a fresh private output
directory and keeps only metadata/relative artifact references in history.

For cloning, use `--task clon` with
`{"text":"Hello.","voice_reference":"/absolute/reference.wav","reference_text":"The reference transcript."}`.
The reference must be an existing nonempty audio file and its transcript is
required by the native encoder. Use audio you have permission to clone.
For streaming chunks, use `--task tts` with
`{"text":"Hello.","mode":"streaming"}` (`offline` is the default).
The final WAV and validated chunk paths are returned after completion; neither
CLI nor TUI plays live audio during synthesis. The TUI exposes task-specific
fields and keeps hidden instructions/references out of unrelated submissions.

An offline cached-weight smoke run produced a valid **24 kHz mono PCM16 WAV**
(9,600 frames, 0.4 seconds) in approximately **4.81 seconds whole-CLI wall time**.
Child process cleanup was confirmed and no GPU service was stopped. This verifies
execution and artifact delivery only: the brief synthetic sample was not a
voice-quality evaluation or a throughput benchmark. Native decision latency and
speech whole-CLI wall time are different metrics and should not be ranked together.

A separate instructed CPU design smoke run produced 15,360 frames (0.64 seconds)
of valid 24 kHz mono PCM16 audio in approximately **6.09 seconds whole-CLI wall time**.
Process release was confirmed; no service was stopped. This verifies the design
execution/artifact path, not whether the voice perceptually matches the instruction.

Separate cached-weight CPU smoke runs also passed: cloning produced 61,440 frames
in **14.20 seconds whole-CLI wall time** using an earlier synthetic WAV; chunked
TTS produced a final 46,080-frame WAV and two chunks in **11.52 seconds**.
All outputs were 24 kHz mono PCM16, with process release confirmed. No GPU service
changed or model was downloaded. These are execution observations, not cloning
similarity, latency guarantees, quality or live-playback evaluations.

These profiles are declared capabilities, not proof that a visitor's runtime or
weights are installed. Multi-engine GPU switching and full-port acceptance remain
unfinished. This portable summary contains no host endpoints, local artifact
locations or credentials; publication of the branch remains separately gated.
