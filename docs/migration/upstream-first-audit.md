# llama-swap-native replacement audit

2026-10-15, branch `feature/l3ms-monorepo`, starting at `ffdf1a7`.
Scope: tagged v265 source and local callers, not live endpoint acceptance.
The user is updating llama-swap independently; no installed version is assumed
verified here. Do not probe or control the router during that update or Pi's
parked GPU window. Linear sync currently requires reauthentication.

## Decisions and first deletion

| Workflow | Native replacement / retained behavior | Acceptance and deletion |
| --- | --- | --- |
| d3/Clef decisions | POST `/v1/systemone`, with `model`, `state`, `questions`; require v262+ for this newly added integration | Root and standalone callers use one request, retaining auth, URL prefix, typed results and uncertainty. Delete the per-model passthrough URL construction; no retry or dual routing mode. |
| Generic cross-engine switching | No production caller or concrete implementation of the old `RuntimeAdapter` protocol | Remove the orphan protocol and simulated switching implementation from `runtime.rs`. Keep real cancellation/deadline/recovery primitives and direct-workflow safety. |
| Live discovery | Existing `/v1/models` client remains | Upstream capabilities do not yet feed our `SwapModel` parser. Prefer consuming native capabilities over new discovery infrastructure; local recipes remain necessary for offline setup and scripts. |
| Logs | No root Rust `/api/events` consumer was found | No log transport migration needed in current root callers. Check external clients before any deployment claim; don't build a new event bridge. |
| Comfy | Native `/comfyui/` is a possible serving path, not a replacement job protocol | Retain scoped submit/status/cancel and artifact checks pending tests through the proxy. No global-interrupt fallback. |
| Speech | Keep existing four-mode audio.cpp CLI recipes until server equivalence is demonstrated | Native audio routes alone do not replace owned child cleanup, output receipts, cloning inputs or streaming-mode behavior. |
| LLM tool continuation | Conversation/tool-result history must live outside the model process | Prove native swapping and task completion first. Add only missing conversation glue, not another scheduler or supervisor. |

The main managed platform still has callers: `--coordinator-managed-session`
launches the session owner, while retained-ID observe/cancel/results and artifact
reads use private IPC. Removing that entire cluster without a deliberate caller
transition would break advertised commands. It remains a removal candidate, not
an excuse to expand it. The isolated unused switching protocol is the immediate
safe deletion; it is not the live GPU guard or resident-unit implementation.

## Media details that affect the next replacement

Tagged v265 [Comfy passthrough](https://github.com/mostlygeek/llama-swap/blob/v265/internal/server/comfyui.go)
uses reserved `comfyui_auto`, strips the prefix and prevents GET cold-starts
(409); POST can start the backend. Auth middleware still applies. Our serving
YAML does not declare that reserved model. A UI URL alone cannot enable it.

Our `ComfyAdapter` uses caller-owned IDs with `/prompt`, `/api/jobs/{id}`,
`/api/jobs/{id}/cancel` and `/view`. Preserve its actual backend contract rather
than assuming passthrough supplies completion, scoped cancellation or outputs.
Its configured base URL can carry `/comfyui`; verify all paths and credentials
with an isolated fixture before migrating serving ownership.

Tagged v265 [audio passthrough](https://github.com/mostlygeek/llama-swap/blob/v265/internal/server/audiocpp.go)
uses reserved `audiocpp_auto`. The separate model-routed task endpoint is
POST `/audioapi/v1/tasks/run`, forwarding `/v1/tasks/run` upstream—not
`/v1/task/run`. See the [tagged route definitions](https://github.com/mostlygeek/llama-swap/blob/v265/internal/server/server.go).
Neither reserved model is configured in this branch. Do not silently replace
speech CLI execution with an unconfigured server route.

## Ordered follow-up

1. Finish offline native decision route checks and unused protocol deletion.
2. Prepare Comfy's explicit router serving recipe and prefix/auth fixtures,
   reusing its native job client. Keep standalone commands and originals.
3. Check the pinned audio.cpp server against each existing speech mode and
   cancellation/output requirement before choosing a server-backed replacement.
4. After the user's update and an arranged quiet window, verify exact installed
   version/config compatibility and real decision/media requests. No automatic
   inference retries, stop/start experiments or permanent cutover here.
5. Delete each displaced caller cluster, including obsolete flags/configs/docs,
   once its replacement is usable. Keep historical results readable. Reassess
   whether any custom service-stop/restoration path is needed at all.

Source reduction is measured against the same tracked `src/control/*.rs` scope,
including tests/comments/blanks. Retired simulation tests are removed with their
unused implementation; retained runtime primitives keep direct unit coverage.
Live switching, task quality and seamless tool-loop acceptance remain open.
