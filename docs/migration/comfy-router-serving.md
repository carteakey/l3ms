# Opt-in ComfyUI serving through llama-swap v265

Recipe only: this does not enable ComfyUI in the tracked root
`llama-swap.yaml`. Merge the model entry below into the existing canonical
`models:` map after confirming the installed router is v265. The reserved model
ID is `comfyui_auto`; `/comfyui/` is not enabled by adding only a UI URL.

The existing standalone launch reference is
[`run-comfyui.sh`](../../maintenance/run-comfyui.sh#L18): it uses
`toolkits/l2m2/vendor/ComfyUI` and `toolkits/l2m2/.venv-comfy/bin/python`; the
snippet uses its common startup flags but not its standalone CPU/GPU mode
switch. Do not invoke that script as llama-swap's `cmd`: its GPU mode requires
llama-swap to be stopped. `toolkits/l2m2/.toolkit` is a child symlink;
do not write through it. Choose and create explicit absolute Comfy user/output
directories outside that symlink (or use an operator-chosen standalone
checkout). Replace every `/ABSOLUTE/...` path below with a literal path before
use; do not assume `$HOME` or other shell environment expansion. No download,
service call, or config activation was performed for this recipe.
The sample assumes paths without spaces; quote each path inside the shell
command if your chosen paths contain spaces.

v265's model config has no `workingDirectory` field. This command sets cwd with
`cd`, while using absolute executable/script/data paths; `${PORT}` is the
llama-swap macro, not a shell variable. `checkEndpoint` must override v265's
default `/health`; ComfyUI serves `/system_stats`.

```yaml
# Merge under the existing root `models:`; do not replace the config.
  comfyui_auto:
    cmd: >-
      /usr/bin/bash -c 'cd /ABSOLUTE/PATH/toolkits/l2m2/vendor/ComfyUI &&
      exec /ABSOLUTE/PATH/toolkits/l2m2/.venv-comfy/bin/python
      /ABSOLUTE/PATH/toolkits/l2m2/vendor/ComfyUI/main.py --listen 127.0.0.1
      --port ${PORT} --disable-auto-launch --disable-api-nodes
      --lowvram --reserve-vram 1.0
      --user-directory /ABSOLUTE/USER-DATA/comfyui
      --output-directory /ABSOLUTE/MEDIA-OUTPUT/comfyui'
    proxy: http://127.0.0.1:${PORT}
    checkEndpoint: /system_stats
```

`GET /comfyui` redirects to `/comfyui/`; the dedicated route strips that prefix
when proxying. It is behind llama-swap's API-key middleware. If `apiKeys` is
configured in the existing root config, configure L3MS with the same key as
`L3MS_COMFY_API_KEY` (Bearer auth). Point `L2M2_COMFY_URL` at the router prefix,
for example `http://127.0.0.1:8080/comfyui`, using the actual listener port.

## Lifecycle caveats and acceptance

v265's `/comfyui` handler returns 409 only for unloaded-model GETs matching the
built-in static suffix defaults, `^/ws(/|$)`, or the exact `^/api/jobs$` path.
It does not ignore all GETs: `/system_stats`, `/api/jobs/{id}`, and `/view` may
load ComfyUI after another model has swapped it out. Writes, including POSTs to
otherwise ignored paths, may start it. These dedicated-route ignores are
built-in; do not assume `upstream.ignorePaths` changes them. A status poll or
artifact read is therefore not necessarily passive after eviction.

The L3MS adapter is a separate async job protocol, not supplied by llama-swap's
passthrough. It submits `POST /prompt` with `{"prompt_id": <owned UUID>,
"prompt": <graph>}` and requires the returned ID to match. That response is
only an enqueue acknowledgement, not compute completion. L3MS polls
`GET /api/jobs/{id}` and requires the same ID plus `pending`, `in_progress`,
`completed`, `failed`, or `cancelled` state. A 404 is unknown, not terminal
success. Cancellation is `POST /api/jobs/{id}/cancel` with `{}`; a
`{"cancelled": true}` reply is only an acknowledgement, so L3MS continues
polling for a terminal state (the job may finish before cancellation). Terminal states
must include consistent `execution_status`; cancellation also requires an
`execution_interrupted` message. On completed jobs it fetches only `type=output`
artifacts via `GET /view?filename=…&subfolder=…&type=output` and validates
descriptors and bounded file bytes before returning results.

Because `/prompt` returns before GPU work finishes, llama-swap's HTTP in-flight
tracking alone does not protect an active Comfy job from an eviction. Do not
send another external model/client request until the job reaches terminal state
and output downloads finish. Keep the Comfy adapter; its current gate
`L3MS_COMFY_SCOPED_JOBS=1` asserts canonical IDs and atomic scoped cancellation
and should remain unset until the configured Comfy checkout passes that exact
contract through the prefixed, authenticated route. Acceptance must include a
full submit/status/cancel/artifact round trip and a check of the above cold-start
behavior. HTTP success or process exit does not prove GPU memory was restored;
verify process and VRAM state independently in an arranged maintenance window.

Sources: [v265 model config](https://github.com/mostlygeek/llama-swap/blob/v265/internal/config/model_config.go), [v265 config schema](https://github.com/mostlygeek/llama-swap/blob/v265/config-schema.json), [v265 Comfy handler](https://github.com/mostlygeek/llama-swap/blob/v265/internal/server/comfyui.go), [v265 route/auth registration](https://github.com/mostlygeek/llama-swap/blob/v265/internal/server/server.go), and [pinned ComfyUI server routes](https://github.com/Comfy-Org/ComfyUI/blob/6bfaacc67c2103481e5f0c84d75257cd0581d86a/server.py). Local contracts: [`ComfyAdapter`](../../src/control/comfy.rs#L923), [direct job loop and completion](../../src/control/direct_run.rs#L955), and [Comfy env gate](../../src/cli.rs#L1583).
