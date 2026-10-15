# Live port acceptance — 2026-10-15

Operator authorization: Pi had finished, permitting the previously approved
bounded maintenance tests. Code under test is the release build from `b039e6e`,
with documentation checkpoint `1084bcf`; binary SHA-256:
`49e7757ced067f97ea2eb813df9fe2f616552deb781ddcdb6c1c14e4df4c11da`.
No inference retry, model download or permanent serving-config change was made.

## Actual language → CUDA speech → language continuation

The shared live v265 router's sole ready model was
`qwen38-flash-next-plat-abliterated-vision`, served by Strata. Its direct backend
reported loaded weights and zero requests in flight. Original unit/config and
drop-in hashes, ready model identity and operational state were captured before
the test. A temporary runtime-only `KillMode=control-group` override enabled the
existing native resident guard; persistent unit/config files were unchanged.

An actual chat request produced a `synthesize_speech` tool call. The small
external test client retained its messages/tool-call ID outside the model
process. The branch release CLI then ran `media:breeze-tts-2`, using the pinned
audio.cpp CUDA build, the existing weights and the shared GPU guard. It stopped
the resident unit, verified release, executed speech, cleaned up, restarted the
unit and restored the exact captured ready-model ID in a new process generation.
The guard cleared its recovery marker only after restoration checks passed.

The entire guarded call took **115.171 seconds**, including model restoration.
Its real WAV is mono PCM16 at 24 kHz, 38,400 frames (1.6 seconds), with nonzero
samples. Peak PCM16 is 32,767; this is execution/artifact evidence, **not** an
intelligibility, clipping, voice-quality or latency guarantee.

After restoration, the client sent the saved conversation plus the successful
speech result back to the same model. It replied `SPEECH_DONE` with the correct
session token from the original user message. This proves an external tool
client can wait and continue after a model restart. It does not preserve an
open SSE stream, the old KV cache, PIDs or TTL remainder, and does not add an
automatic tool executor to the Chat UI.

Private evidence:

- `.toolkit/gpu-handoff-20261015-1084bcf.8cuaJX/`: separate original/accepted
  baselines, exact invocation, stderr, service transitions, native result and WAV.
- `.toolkit/live-tool-loop-b4SWMoEZ/`: the two actual native requests/responses,
  externally retained conversation and continuation receipt. No API key was
  persisted in those records.

## Remaining local checks in the same maintenance window

| Check | Result | Private evidence |
| --- | --- | --- |
| Clef image | Corrected native image request passed through isolated CPU-only v265 `/v1/systemone`; expected `green`, confidence 0.97679, native output tokens 0. Owned listeners closed. | `.toolkit/native-runtime-acceptance-20261015/run-bbb5eb0a/` |
| Root Blender CLI | Successful guarded headless render of the unchanged factory CPU scene. PNG fully decoded as 32×32; private output, exit 0, no recovery marker. | `.toolkit/native-runtime-acceptance-20261015/blender-cpu.wJ1i8gms/` |
| Comfy scoped cancellation | Authenticated prefixed artifact copy passed. After its exact job reached `in_progress`, SIGINT to the owned CLI caused scoped cancellation. The backend confirmed `cancelled` plus `execution_interrupted`; client output remained empty, CLI reported cancellation, and owned listeners closed. | `.toolkit/comfy-native-cancel-acceptance-20261015/run-0a3a871c/` |
| Learned GPU image | Root `media:krea-2` CLI completed the existing Krea/LoRA graph through authenticated v265 Comfy passthrough. CUDA backend, 256×256, four steps, seed 42; native and downloaded PNGs matched and fully decoded. Owned listeners closed and GPU returned to the desktop baseline. | `.toolkit/krea-native-acceptance-20261015/run-825e0b6c/` |

The cancellation fixture uses bounded chained resizes up to 4096² instead of one
8192² operation. The product's five-second cleanup budget was not extended and
its uncertain-completion behavior was not weakened. Earlier unconfirmed results
remain historical evidence, not retroactively relabeled successes.

These are bounded sanity/contract cases, not quality or accuracy evaluation.
The Blender scene selects Cycles CPU; checking the idle GPU does not make its
render a CUDA test. Its separate software-Wayland GUI shutdown problem remains
recorded in [Blender acceptance](blender-acceptance.md).

## Window closure

The window is **closed**. After all isolated tasks released their processes and
listeners, GPU usage was 44 MiB with only the known KDE compositor. The temporary
runtime drop-in was removed; its pre-existing empty parent directory was kept.
The user unit was reloaded and started with its original `KillMode=process` and
the same two original persistent drop-ins. Original fragment, drop-in and router
config hashes match. One routed health activation restored the sole original
ready model, whose backend reports loaded weights and zero in-flight requests.
The recovery marker is absent. The original checkouts, models and history were
not removed or migrated, and no persistent deployment/cutover was performed.

Final restoration receipt:
`.toolkit/gpu-handoff-20261015-1084bcf.8cuaJX/restoration-receipt.json`.
The final model activation/readiness check took 77.597 seconds. Initial credential
inspection briefly encountered an exec-time `/proc/environ` permission race;
bounded read-only inspection resumed before the single model activation. No
inference or activation was replayed.

The installed unit still has its original `KillMode=process`. Future use of the
optional resident handoff needs an explicitly configured compatible unit; this
test does not silently make that option production-enabled. Native routing and
standalone commands remain the default product shape.

Final verification on the unchanged runtime source: root Rust **645 passed,
4 ignored**, strict all-target Clippy and formatting pass; standalone Rust suites
**6 + 2** pass. Retained earlier CPU speech, embedding, source-export and browser
receipts remain scoped evidence, not a claim that every model/backend/quality
combination has been qualified.
