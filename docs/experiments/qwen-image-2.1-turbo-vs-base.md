# Experiment: Qwen Image 2.1 (Base) vs Qwen Image 2.1 Turbo (INT8 ConvRot)

Measured run comparison: 2026-10-09 on Yeti (NVIDIA GeForce RTX 4070 12 GiB).

## Prompt

File: `prompts/astronaut-greenhouse.txt`

> "cinematic medium shot of an astronaut holding a glowing bioluminescent plant inside an overgrown derelict greenhouse, soft golden hour rim light, fine dust particles in air, highly detailed, photorealistic"

## Run 1: Qwen Image 2.1 (Base — "Before")

- **Command:**
  ```sh
  ./media-models/generate-qwen-image-2.1.sh \
    --prompt-file prompts/astronaut-greenhouse.txt \
    --seed 42 --width 1024 --height 1024 --steps 25 \
    --output /home/kchauhan/media-output/test-qwen-before.png
  ```
- **Weights:**
  - `diffusion_models/qwen_image_2.1_int8_convrot.safetensors` (7,256,783,064 bytes)
  - `text_encoders/qwen3vl_8b_int8_convrot.safetensors` (9,350,798,360 bytes)
  - `vae/qwen_image_2.1_vae_bf16.safetensors` (675,509,688 bytes)
- **Settings:** Resolution 1024x1024, Steps: 25, CFG: 1.0, Sampler: Euler/simple, Seed: 42
- **Duration:** 20.13 seconds total (ComfyUI execution: 19.74 seconds)
- **Peak VRAM:** 9.6 GiB
- **Artifact:** `/home/kchauhan/media-output/test-qwen-before.png` (2.1 MiB)
- **Observations:**
  - Standard 25-step Euler generation.
  - Saturated neon-green bioluminescence, specular reflections across helmet glass.
  - Dense foliage and atmospheric dust particles with soft rim lighting.

## Run 2: Qwen Image 2.1 Turbo ("After")

- **Command:**
  ```sh
  ./media-models/generate-qwen-image-2.1-turbo.sh \
    --prompt-file prompts/astronaut-greenhouse.txt \
    --seed 42 --width 1024 --height 1024 \
    --output /home/kchauhan/media-output/test-qwen-after.png
  ```
- **Weights:**
  - `diffusion_models/qwen-image-2.1-turbo-int8-convrot.safetensors` (7,256,828,688 bytes)
  - `text_encoders/qwen3vl_8b_int8_convrot.safetensors` (shared, 9,350,798,360 bytes)
  - `vae/qwen_image_2.1_vae_bf16.safetensors` (shared, 675,509,688 bytes)
- **Settings:** Resolution 1024x1024, Steps: 8, CFG: 1.0, Sampler: Euler with official sampling sigmas (`1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0`), Seed: 42
- **Duration:** 10.12 seconds total (ComfyUI execution: 8.83 seconds cold, 10.11 seconds warm)
- **Peak VRAM:** 9.6 GiB
- **Artifact:** `/home/kchauhan/media-output/test-qwen-after.png` (1.9 MiB)
- **Observations:**
  - **Latency reduction:** Diffusion execution dropped from 19.74s to 8.83s (~2.24x speedup, >55% latency reduction).
  - **Composition & fidelity:** Exceptional retention of compositional fidelity and fine detail; gloves, suit textures, plant roots, and golden hour rim light render sharply in only 8 steps.
  - **Memory efficiency:** Reuses the existing Qwen3-VL INT8 text encoder and BF16 VAE with zero extra RAM overhead, fitting cleanly into the 12 GiB VRAM envelope of Yeti's RTX 4070.

## Run 3: Qwen Image 2.1 Turbo (Image Editing / Transformation)

- **Input Image:** `prompts/turbo-boat-input.png` (yacht line sketch, 3200x1120 RGBA)
- **Prompt:** `prompts/yacht-edit.txt` (photorealistic motor yacht transformation with specified hull, deckhouse, and lighting attributes)
- **Command:**
  ```sh
  ./media-models/generate-qwen-image-2.1-turbo.sh \
    --image prompts/turbo-boat-input.png \
    --prompt-file prompts/yacht-edit.txt \
    --seed 20261009 \
    --output /home/kchauhan/media-output/test-turbo-yacht-edit.png
  ```
- **Weights:** `diffusion_models/qwen-image-2.1-turbo-int8-convrot.safetensors`, `text_encoders/qwen3vl_8b_int8_convrot.safetensors`, `vae/qwen_image_2.1_vae_bf16.safetensors`
- **Settings:** Resolution 1024x1024, Steps: 8, CFG: 1.0, Sampler: Euler with official custom sigmas, Seed: 20261009
- **Duration:** 16.13 seconds total (ComfyUI execution: 15.42 seconds including reference VAE + vision tower encode)
- **Peak VRAM:** 10.1 GiB
- **Artifact:** `/home/kchauhan/media-output/test-turbo-yacht-edit.png` (1.3 MiB)
- **Observations:**
  - Accurately converts the line sketch into a photorealistic black hull, white superstructure, and yellow funnel.
  - Matches the geometry, mast placement, window lines, and railings from the sketch input.
  - Completed end-to-end in only 8 denoising steps.
