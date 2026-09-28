# MiniMax-H3 reference baseline

This baseline follows the quality recipe captured from the Batman post:
ComfyUI's native reference workflow, the official pruned Ref2VA INT8 ConvRot
model, 20 sampling steps, and no Turbo LoRA. It is distinct from L2M2's
text-only audio.cpp Q4_K path.

The official workflow is saved at
`workflows/minimax-h3/video_minimax_h3_r2v-official.json`. The first-quality
variant is `video_minimax_h3_r2v-20step-no-lora.json`: its Turbo LoRA loader is
bypassed and the sampler uses 20 steps. Keep both files so upstream intent and
our controlled change remain reviewable.

Required files and exact upstream paths live in
`models/minimax-h3-comfyui-files.txt`. Run:

```sh
./maintenance/setup-minimax-h3-comfyui.sh
./maintenance/comfyui-service.sh stop
./maintenance/comfyui-service.sh start-gpu
```

Before a real run, supply one or two scene/character images and a clean voice
clip of about 15 seconds. Reference them in the prompt as `<Picture 1>` and
`<Audio 1>`. Start at `ref_image_size=match`, 0.2–0.3 MP, about five seconds,
seed 42, and 20 steps. Request no music. Save all inputs, hashes, workflow,
prompt, output, logs, and timing under the experiment record.

Do not judge reference identity from a text-only run. Do not add acceleration
until the 20-step baseline completes; compare one change at a time afterward.

## Completed Baseline Run — 2026-09-26 17:55

- Runner: `maintenance/test-h3-comfyui.py --steps 20`
- Output: `/home/kchauhan/media-output/h3-comfyui-20260926-175336/`
- Configuration:
  - Resolution: 672×384 (0.25 MP, 16:9), 124 frames (5.17 s @ 24 fps)
  - Sampling: 20 steps, `res_multistep`, `simple` scheduler, seed 42, no Turbo LoRA
  - Models: `minimax_h3_ref2va_pruned_int8_convrot.safetensors` (DiT), `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (Text/Vision), `minimax_h3_video_vae_fp16.safetensors` (Video VAE), `minimax_h3_audio_vae_fp32.safetensors` (Audio VAE)
  - References: `<Picture 1>` (`ref_vashti_chair.png`), `<Picture 2>` (`ref_vashti_standing.png`), `ref_image_size=match`
- Performance & Result:
  - `exit_code=0`, `wall_seconds=126.15` (~2 min 6 s)
  - ComfyUI Prompt ID: `73ab49f5-3726-4e96-b462-4a96091d55b5`
  - Peak VRAM: ~11.2 GB (dynamic loading within 12 GB budget)
- Artifacts:
  - Video: `MiniMax_H3_00002_.mp4` (672×384, 24 fps, H.264 High profile, 269 KB)
  - Audio: Synced 32 kHz stereo AAC
  - Metadata: `probe.json`, `result.txt`, `prompt.txt`, `workflow-api.json`

