# Media engines through llama-swap — 2026-10-15

llama-swap (v264+) routes `/audiocpp/` to the reserved model `audiocpp_auto` and
`/comfyui/` to `comfyui_auto`, and swaps them like any other model. The root
`llama-swap.yaml` now declares both, so one router owns language, embeddings,
decisions, speech and images. L3MS sends requests and presents results; it does
not start, stop or fence engines for these lanes.

## What is declared

| Reserved id | Engine | Launch | Health |
| --- | --- | --- | --- |
| `audiocpp_auto` | audio.cpp `955c872` | `vendor/audio.cpp-955c872/build/bin/audiocpp_server --config toolkits/l2m2/audiocpp-server.json` | `/health` |
| `comfyui_auto` | ComfyUI (low-VRAM flags) | `${comfy_root}/.venv-comfy` + `${comfy_root}/vendor/ComfyUI` | `/system_stats` |

`comfy_root` is `${env.L3MS_ROOT}/toolkits/l2m2` (the ComfyUI checkout, its
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

`toolkits/l2m2/maintenance/test-audiocpp-via-llama-swap.sh [cpu|cuda]` starts an
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

## What this let us delete

The resident pause/restore handoff (the opt-in stop-and-restore of
`llama-swap.service`, about 1,700 lines of service management plus its tests) is
removed. The idle-GPU window, the KDE desktop baseline and the recovery marker
stay for the direct scripts that still bypass the router (LTX, YuE2, Blender,
HeartMuLa). Moving those lanes onto routed engines would let the rest of the
window code go too, and is the next simplification.
