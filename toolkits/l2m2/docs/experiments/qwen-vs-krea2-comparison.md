# Experiment: Qwen Image 2.1 vs Krea 2 Raw INT8 + Turbo LoRA

Measured run comparison: 2026-09-24 on Yeti (NVIDIA GeForce RTX 4070 12 GiB).

## Prompt

File: `prompts/astronaut-greenhouse.txt`

> "cinematic medium shot of an astronaut holding a glowing bioluminescent plant inside an overgrown derelict greenhouse, soft golden hour rim light, fine dust particles in air, highly detailed, photorealistic"

## Run 1: Qwen Image 2.1

- **Command:**
  ```sh
  ./media-models/generate-qwen-image-2.1.sh \
    --prompt-file prompts/astronaut-greenhouse.txt \
    --seed 42 --output /home/kchauhan/media-output/compare-qwen.png
  ```
- **Weights:**
  - `diffusion_models/qwen_image_2.1_int8_convrot.safetensors`
  - `text_encoders/qwen3vl_8b_int8_convrot.safetensors`
  - `vae/qwen_image_2.1_vae_bf16.safetensors`
- **Settings:** Resolution 1024x1024, Steps: 25, CFG: 1.0, Sampler: Euler/simple, Seed: 42
- **Duration:** 23.2 seconds
- **Peak VRAM:** 9.6 GiB
- **Artifact:** `/home/kchauhan/media-output/compare-qwen.png` (2.1 MiB)
- **Observations:**
  - Dynamic frontal low-angle composition.
  - Excellent prompt adherence: seedling glows vibrant neon green, and its reflection is mirrored in the astronaut's tinted helmet visor.
  - Atmospheric volumetric light beams through broken roof panes.
  - Astronaut glove grip on the root clump is anatomically coherent.

## Run 2: Krea 2 Raw INT8 + Turbo LoRA (r128)

- **Command:**
  ```sh
  ./media-models/generate-krea-2.sh \
    --prompt-file prompts/astronaut-greenhouse.txt \
    --seed 42 --steps 12 --cfg 1.5 --lora-strength 0.60 \
    --output /home/kchauhan/media-output/compare-krea.png
  ```
- **Weights:**
  - `diffusion_models/krea2_raw_int8_convrot.safetensors`
  - `loras/krea2_turbo_distill_r128.safetensors` (strength 0.60)
  - `text_encoders/qwen3vl_4b_fp8_scaled.safetensors`
  - `vae/qwen_image_vae.safetensors`
- **Settings:** Resolution 1024x1024, Steps: 12, CFG: 1.5, Sampler: Euler/simple, Seed: 42
- **Duration:** 36.1 seconds
- **Peak VRAM:** 10.4 GiB
- **Artifact:** `/home/kchauhan/media-output/compare-krea.png` (2.2 MiB)
- **Observations:**
  - Profile/3-quarters medium framing showing NASA/US-style spacesuit markings, chest electronics pack, and ribbed air hoses.
  - Striking fiber-optic/dandelion style bioluminescent spores emanating from a lush moss ball.
  - Extremely tactile fabric weave texture, realistic wrinkles, and dirt/wear without plastic over-smoothing.
  - Warm directional rim lighting from the side wall with soft bokeh on distant greenhouse vegetation.
