# Direct CPU Breeze acceptance

The shared L3MS wrapper's standard CPU speech run passed with cached weights.
Exact command, settings, measured duration, retained script/log/artifact paths
and limits are recorded in
[the acceptance record](../../../../docs/migration/direct-speech-acceptance.md).
This is execution evidence, not a quality benchmark. The original CUDA default
is unchanged; select `breeze-tts-2-cpu` or explicitly pass `--backend cpu`.

The 2026-10-12 offline voice-design follow-up also passed through the shared CLI,
with a required instruction, validated WAV and confirmed process release. Exact
command, settings, failed harness preflight and final receipt are appended to
the same acceptance record. This does not verify perceptual voice quality.
