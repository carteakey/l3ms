# LTX-2.5 and YuE2 through shared ComfyUI

The root workbench uses the existing shared ComfyUI workflow route for these
profiles. The root CLI requires **`{"graph": {...}}` JSON**: the `graph` value is
an API-format dictionary of node IDs containing `class_type` and `inputs`.
The bundled files already include this wrapper and can be passed directly to
`--input`. A frontend export with a top-level `nodes` list, a bare API graph, or
the standalone generation script's flags is not the root CLI input contract.
For direct ComfyUI API or UI API-format import, extract the file's `.graph`
value; do not send the root wrapper as the node dictionary.

The bundled recipes are:

- `workflows/yue2/audio_yue2_text2music_api.json`: native YuE2 checkpoint loading,
  ABC planning, autoregressive conditioning, diffusion sampling and FLAC saving.
  Edit style in node `2`, lyrics in `3`, maximum seconds in `5`, and checkpoint
  filename in `1`. Seeds in nodes `4`, `5` and `8` are fixed at 42; edit them
  together when changing the run. To supply an ABC score, replace node `5`'s
  `abc` link with a string and remove the unused ABC planner node `4`.
- `workflows/ltx-2.5/video_ltx2_5_t2v_preview_api.json`: a 384 × 256, 49-frame,
  24-fps distilled first-pass **low-resolution preview**, saving synchronized MP4
  plus a separate FLAC. Edit the prompt in `5`; keep video/audio frame counts in
  `8`/`9` and frame rates in `7`/`9`/`19` aligned. Width/height must be multiples
  of 32; frame count uses `8n + 1`. Both profile tasks, `video` and `music`, use
  workflow JSON; the separate audio saver makes the music artifact explicit.
  This preview omits the official template's second sampling pass, latent
  upscaling and optional prompt enhancement. It does not claim full-template
  quality or image conditioning. Use a matching image-to-video workflow for
  still-image input.

These graphs derive from installed official `comfyui_workflow_templates_json`
0.1.96 (`audio_yue2_text2music.json` and `video_ltx2_5_t2v.json`, from the
`comfyui_workflow_templates` 0.11.70 installation). They use the native nodes in
the prepared ComfyUI checkout. The offline tests inspect that checkout's source
schemas without importing ComfyUI, PyTorch or any model.

Model filenames must match files already installed on the selected server:

| ComfyUI model folder | Recipe filename |
| --- | --- |
| `checkpoints` | `yue2_3b_int8_convrot.safetensors` |
| `diffusion_models` | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` |
| `text_encoders` | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` |
| `vae` | `ltx-2.5-video-vae-bf16.safetensors` |
| `vae` | `ltx-2.5-audio-vae-bf16.safetensors` |

The standalone YuE model/VAE directories and LTX Python BF16 components are not
automatically substituted for these Comfy exports. Update loader names only for
compatible installed artifacts. No dependency installation, model download or
server startup is performed by these recipes. Actual weights, successful
generation, memory fit, output quality and cancellation remain pending acceptance.

By default the root CLI targets the router's `/comfyui` endpoint under
`LLAMA_SWAP_URL`; `L2M2_COMFY_URL` explicitly selects another ComfyUI endpoint.
The root workbench still requires `L3MS_COMFY_SCOPED_JOBS=1`, asserting that the
server supports canonical job IDs and atomic scoped cancellation. Set it only
after confirming that contract; it does not assert GPU ownership. Use a prepared
server and an existing, empty output directory owned by you with mode `0700`.
For example, from the monorepo root, after satisfying those requirements:

```sh
L3MS_COMFY_SCOPED_JOBS=1 l3ms --run-profile media:yue2-music --task music \
  --input workflows/yue2/audio_yue2_text2music_api.json \
  --output-dir /path/to/private-empty-directory
```

Choose `media:ltx-2.5`, task `video` or `music`, and its preview graph for LTX.
Output prefixes are relative to that server's ComfyUI output directory; returned
artifacts are collected by the root route. These are generated server filenames,
not a promise of a fixed local `result.wav` or `result.mp4` filename.

The manifest retains `media-models/generate-ltx-2.5.sh` and
`media-models/generate-yue2.sh` for direct standalone use. Their separate Python
runtimes, flags, input conditioning and model layout remain separate from this
shared workflow route.
