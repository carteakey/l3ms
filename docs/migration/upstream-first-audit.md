# llama-swap-native replacement audit

Updated 2026-10-15 for the `feature/l3ms-monorepo` source tree. The source/caller
audit is against tagged llama-swap v265. A separate read-only live check verified
the installed/running build as v265 `d1e743f` through `/api/version`,
`/v1/models` and `/running`; no model was dispatched. This does not establish
end-to-end task, GPU or restoration acceptance. The later approved window is recorded in [live acceptance](live-port-acceptance.md).

## Current decisions

| Area | Current source choice | Evidence and remaining limit |
| --- | --- | --- |
| d3/Clef decisions | Root and standalone callers POST once to `/v1/systemone` with the pinned model, state and questions; this integration requires llama-swap v262+. Keep auth, URL-prefix handling, typed answers and uncertain post-dispatch behavior; no retry or passthrough fallback. | In isolated v265 CPU runs, d3 and Clef text sanity cases passed (`.toolkit/native-decision-acceptance-CgGCvU/run-eb534949`). The d3 image case passed in `run-4ebbef8a`; Clef returned HTTP 400 because that attempt's question was invalid. The corrected Clef image case later returned `green` (`run-bbb5eb0a`). No quality or live-router claim. |
| Managed execution platform | Retire the branch-added managed-session creator, daemon/worker/server/private-engine vertical and duplicate language chat facade. Do not implement a replacement coordinator. | Explicit older-server clients for retained-ID observe/cancel/results and history/artifact reads remain; no on-disk records were removed or migrated. Actual Chat uses `ChatClient`. Direct cancellation/deadline/recovery and GPU safeguards remain with their callers. |
| Discovery and logs | Reuse `/v1/models` and its `architecture.output_modalities`. Do not add a log bridge. | Explicit non-chat outputs (decision, media or embeddings without text) are excluded from Chat selection but retained in the complete inventory for native task forms. Text-output vision models and unknown-only capabilities remain selectable. No root Rust `/api/events` consumer remains. Check external clients before deployment. |
| Comfy | Keep the native scoped submit/status/cancel/artifact contract; the opt-in router recipe is not an implicit config change. | A bounded CPU `EmptyImage` → `SaveImage` job returned an artifact through v265 (`run-06530039`); its scoped cancellation completed. A later run (`run-8b6bdda7`) returned an artifact, but cancellation was still `in_progress` under load. A quiet fixture later confirmed scoped cancellation, and a Krea CUDA graph completed ([live acceptance](live-port-acceptance.md)). |
| Speech | Keep the existing four-mode audio.cpp CLI workflow; do not add a parallel server-backed path on route availability alone. | All four real CPU modes and process-identity cleanup passed after platform retirement; see [speech acceptance](direct-speech-acceptance.md). This establishes execution/artifacts, not voice quality or GPU acceptance. The pinned server still lacks an inference-cancel endpoint and has different streaming output semantics. |
| LLM tool continuation | Keep conversation/tool-result history outside the model process. | No evidence currently justifies a second scheduler or supervisor. Prove a concrete gap after native routing and task acceptance before adding glue. |

## Comfy route constraints

Tagged v265 [Comfy passthrough](https://github.com/mostlygeek/llama-swap/blob/v265/internal/server/comfyui.go)
uses reserved model `comfyui_auto` and strips the prefix. Only matching ignored
static/websocket/job-list GET paths refuse cold starts with 409; specific-job
status and `/view` are not passive exclusions. Auth still applies. The tracked
serving YAML does not declare this reserved model; the [recipe](comfy-router-serving.md)
is opt-in and has not been deployed against the user's live service.

The native adapter uses caller-owned IDs with `/prompt`,
`/api/jobs/{id}`, `/api/jobs/{id}/cancel` and `/view`. Preserve this backend
contract rather than assuming passthrough supplies completion, scoped
cancellation or outputs. Prefix/auth HTTP fixtures pass. The latest isolated
router run's cancellation observation remained `in_progress`; repeat only in a
quiet, bounded fixture before claiming reliable proxy cancellation. Do not add a
global-interrupt fallback.

## Speech route constraints

Tagged v265 [audio passthrough](https://github.com/mostlygeek/llama-swap/blob/v265/internal/server/audiocpp.go)
uses reserved model `audiocpp_auto`. The separate model-routed task endpoint is
POST `/audioapi/v1/tasks/run`, forwarding `/v1/tasks/run` upstream. Neither
reserved model is declared in the branch serving YAML.

Offline inspection of audio.cpp pin
`955c8725c611d511774e6be132aff6609163b2d2` found synthesis support but no
inference-cancel endpoint: busy timeout bounds waiting, not running inference.
Its offline route returns WAV bytes, while incremental speech emits PCM SSE
events rather than the CLI's WAV/chunk files. Keep current CLI behavior until a
replacement demonstrates output and honest cancellation semantics.

## Remaining checks

1. Rerun the corrected Clef image case in the isolated CPU setup; do not interpret
   the prior HTTP 400 as model output.
2. Recheck Comfy cancellation without the observed system load if safe, and test
   any learned-model/GPU workflow only in an approved quiet window. Do not change
   serving config or live service state as part of this audit.
3. Verify end-to-end live task and restoration behavior only in an approved
   window after Pi's GPU work is clear. No automatic inference retries,
   stop/start experiments or permanent cutover are authorized here.
4. Reassess tool-loop glue only if an accepted native workflow demonstrates a
   specific missing conversation behavior.

The retired source is an actual deletion, not code relocation. Keep historical
records readable; do not revive managed-platform tasks from older status entries.
