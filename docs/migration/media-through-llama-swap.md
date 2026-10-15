# Media engines through llama-swap — 2026-10-15

llama-swap (v264+) routes `/audiocpp/` to the reserved model `audiocpp_auto` and
`/comfyui/` to `comfyui_auto`, and swaps them like any other model. The root
`llama-swap.yaml` now declares both, so one router owns language, embeddings,
decisions, speech and images. L3MS sends requests and presents results; it does
not start, stop or fence engines for these lanes.

## What is declared

| Reserved id | Engine | Launch | Health |
| --- | --- | --- | --- |
| `audiocpp_auto` | audio.cpp `955c872` | `vendor/audio.cpp-955c872/build/bin/audiocpp_server --config audiocpp-server.json` | `/health` |
| `comfyui_auto` | ComfyUI (low-VRAM flags) | `${comfy_root}/.venv-comfy` + `${comfy_root}/vendor/ComfyUI` | `/system_stats` |

`comfy_root` is `${env.L3MS_ROOT}` (the repository root: the ComfyUI checkout in `vendor/ComfyUI`, its
Python environment and user data were reflink-cloned there; see
[sibling retirement](sibling-retirement.md)).

## Breeze-TTS-2 through `/audiocpp/v1/audio/speech`

The server config registers three model ids over the same GGUF; with
`max_loaded_models: 1` only one is resident.

| Mode | Request |
| --- | --- |
| Standard | `{"model":"breeze-tts","input":"…","response_format":"wav"}` |
| Voice design | same model plus `"options":{"instruction":"Warm female narrator, calm pacing."}` |
| Voice cloning | `"model":"breeze-clone"` with `"voice_ref":"/path.wav"` and `"reference_text":"…"` |
| Streaming | `"model":"breeze-stream","stream":true,"stream_format":"sse","response_format":"pcm"` |

## Evidence

Branch follow-up: HeartMuLa music and H3 music/tts use
`POST /audiocpp/v1/tasks/run` with the existing `{"args":[...]}` CLI/form input
translated to native request fields. Use the same `LLAMA_SWAP_URL` and
`LLAMA_SWAP_API_KEY`, or a confirmed TUI Chat connection. Output remains
`result.wav`. H3 video was removed from the root workbench (it needed RGB24-to-MP4
post-processing; the standalone script remains). Server model paths, threads and session options now belong in
`audiocpp-server.json`; shell-wrapper environment overrides do not
configure routed jobs. The native `options.duration_seconds` is normalized to
HeartMuLa's `duration_sec` by the pinned engine.

These new routes have fixture coverage, **not real-model acceptance**. On
2026-10-15 the conventional local HeartMuLa/H3/LTX/YuE2 weight directories were
absent, and `/mnt/storage` was only an autofs mount without mounted storage.
No model download, NAS mutation, live config reload or engine eviction was
performed for this follow-up. Live acceptance remains a required TODO gate.

`maintenance/test-audiocpp-via-llama-swap.sh [cpu|cuda]` starts an
isolated router on port 18932 (never the live service), runs the four modes,
and checks that `/unload` stops the audio.cpp process.

- **CPU, passed (2026-10-15):** standard 2.56 s, design 2.48 s, clone 2.32 s,
  streaming 4 SSE events (all 24 kHz mono), and the engine process ended on
  unload. Cold start through the router took about 34 s.
- **CUDA on the live router, passed (2026-10-15, 11:28-11:31, router idle with
  zero clients):** after a SIGHUP reload, the live v265 router swapped engines
  on its own with no service stop and no custom guard.
  chat model (`qwen38-flash-next-plat`, 11.5 GB) → Breeze on CUDA (3.9 GB) →
  ComfyUI (`/comfyui/system_stats` ready in 6.7 s) → the original
  `qwen38-flash-next-plat-abliterated` restored by one chat request (23 s,
  VRAM back to 11.6 GB held by the Strata process only).
  Breeze on CUDA: standard 2.08 s of audio (12.7 s cold swap), design 2.0 s
  (1.2 s warm), clone 3.04 s (6.2 s), streaming 4 SSE events (5.7 s).
- **SIGHUP is not free.** Reloading evicted the resident model and the config's
  startup preload loaded `qwen38-flash-next-plat` (77 s). The router stayed up
  and served throughout; the earlier note that running models keep serving was
  wrong whenever the config preloads a model.

Known upstream limits, unchanged: the pinned audio.cpp server has no
mid-inference cancel (only a busy timeout), and streaming emits PCM SSE events,
not the CLI's WAV chunk files. Comfy cancellation and learned-model CUDA
execution through this route were verified earlier; see
[live acceptance](live-port-acceptance.md).

## L3MS itself uses the route (2026-10-15)

`l3ms --run-profile media:breeze-tts-2` and the TUI form no longer run the shell
script inside a GPU window; they post to `/audiocpp/v1/audio/speech` through
`LanguageAdapter::execute_speech` and publish the result with the same file names
and validation as before. Real CLI runs against the live v265 router on CUDA, no
`L3MS_NATIVE_GPU`, exit 0: standard (2.08 s of audio, 21 s total including evicting
the resident model and cold-starting audio.cpp), design (2.0 s), clone (3.04 s, from
the generated reference) and streaming (2.48 s, two `chunk_NNNN.wav` files). The
resident `qwen38-flash-next-plat-abliterated` returned with one chat request
(14 s); no `audiocpp_server` process remained.

## What this let us delete

The resident pause/restore handoff (the opt-in stop-and-restore of
`llama-swap.service`, about 1,700 lines of service management plus its tests) is
removed.

## Shared workflows and remaining advanced direct work

The idle-GPU window (locks, idle observation, recovery marker, KDE desktop
baseline) remains only for explicit advanced direct work, not as the default
for new models. LTX and YuE2 share `comfyui_auto`; their standalone Python
scripts remain toolkit-only, with no root fallback. See [runtime policy](runtime-policy.md).

| Lane | Engine | Why it is not routed |
| --- | --- | --- |
| `blender-3d` | Blender 5.2 / blender-mcp | An application, not a model engine. Root execution guards the shared GPU because the scene chooses its backend. |

HeartMuLa and H3 audio use `audiocpp_auto` on this branch, subject to the live
acceptance gate above. Blender headless is the only lane left in the window;
CUDA Nagi/Laya decisions and H3 video were retired from the root workbench.
