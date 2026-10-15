# Direct CPU speech acceptance — 2026-10-15

Four direct CLI speech modes passed after retirement of the branch-added managed
execution platform. Each used an explicit CPU-only audio.cpp build at source pin
`955c8725c611d511774e6be132aff6609163b2d2`, existing weights, and verified
owned process-identity cleanup. No model downloads, quality evaluation, live
playback, service changes or GPU use occurred.

| Mode | Wall seconds | Final frames | Peak PCM16 |
| --- | ---: | ---: | ---: |
| Standard | 41.651 | 46,080 | 17,890 |
| Design | 90.567 | 53,760 | 16,469 |
| Clone | 31.506 | 59,520 | 15,691 |
| Streaming | 20.272 | 57,600 | 13,107 |

Streaming also returned two completed chunks (7,680 and 49,920 frames); this is
chunked artifact output, not live playback. Cloning used a synthetic reference,
not a person's voice. The private ignored receipts and outputs are under
`.toolkit/speech-post-retirement-wdSeFa/{standard,design,clon,streaming}/`.
These are execution/artifact checks, not voice-quality, similarity or latency
guarantees.

An earlier separate 80×24 keyboard UI run accepted standard CPU speech, recorded
metadata-only history and successfully cancelled a running job before reopening
admission after cleanup. That UI run predates the current retirement checkpoint;
the four post-retirement receipts above are CLI runs, not a new UI acceptance.
GPU orchestration/restoration and complete artifact navigation remain open.

The audio.cpp server route is not a replacement for the tested CLI behavior:
the pinned server has no inference-cancel endpoint, its busy timeout only bounds
waiting, and incremental speech uses PCM SSE rather than the CLI's WAV/chunk
artifacts. Keep the current direct path unless an alternative demonstrates the
required modes, output contract and honest cancellation semantics. See the
[upstream-first audit](upstream-first-audit.md) for the v265 route constraints.
