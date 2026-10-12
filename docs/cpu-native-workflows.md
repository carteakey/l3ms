# Explicit CPU workflows

These Linux profiles use installed native runtimes and cached weights. They
do not install models, start services, silently substitute CPU for a CUDA
profile, or retry inference. The original CUDA profiles remain separate.
See the [decision setup](../toolkits/l3ds/README.md) and
[audio.cpp setup](../toolkits/l2m2/docs/audio-cpp-runtimes.md).

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

## Breeze standard CPU speech

```sh
mkdir -m 700 "$PWD/speech-output"
l3ms --run-profile media:breeze-tts-2-cpu --task tts --input speech.json --output-dir "$PWD/speech-output" --task-timeout-ms 120000
```

Input is exactly `{"text":"Hello."}`. The output directory must be fresh,
private and owned by the caller. The recipe explicitly selects the CPU backend;
the native script's default remains CUDA. This wrapper profile currently
supports **standard text-only speech**, not cloning, design or streaming.
The CLI returns validated WAV paths; the TUI allocates a fresh private output
directory and keeps only metadata/relative artifact references in history.

An offline cached-weight smoke run produced a valid **24 kHz mono PCM16 WAV**
(9,600 frames, 0.4 seconds) in approximately **4.81 seconds whole-CLI wall time**.
Child process cleanup was confirmed and no GPU service was stopped. This verifies
execution and artifact delivery only: the brief synthetic sample was not a
voice-quality evaluation or a throughput benchmark. Native decision latency and
speech whole-CLI wall time are different metrics and should not be ranked together.

These profiles are declared capabilities, not proof that a visitor's runtime or
weights are installed. Multi-engine GPU switching and full-port acceptance remain
unfinished. This portable summary contains no host endpoints, local artifact
locations or credentials; publication of the branch remains separately gated.
