# Qwen Image 2.1 Turbo on Yeti

Verified inputs: 2026-10-12. This guide documents the accelerated
**Qwen Image 2.1 Turbo** checkpoint in L2M2, utilizing INT8 ConvRot quantization
for Yeti's NVIDIA GeForce RTX 4070 (12 GiB VRAM) with its calibrated 8-step denoising schedule.

---

## 1. Turbo vs. Normal (Base): Key Differences

| Attribute | Qwen Image 2.1 (Normal / Base) | Qwen Image 2.1 Turbo |
| :--- | :--- | :--- |
| **Denoising Steps** | **25 steps** (default) | **8 steps** (68% fewer steps) |
| **Diffusion Latency (Yeti RTX 4070)** | **19.74 s** | **8.83 s** (~2.24× faster) |
| **Total Job Time (1024×1024)** | **20.13 s** | **10.12 s** (>50% latency reduction) |
| **Image Edit Latency** | ~35–40 s | **15.42 s** |
| **Sampling Schedule** | Standard `euler` + `simple` scheduler | Calibrated 8-step non-linear sigmas |
| **Guidance (CFG)** | `CFG = 1.0` (tolerates 1.0–2.5) | `CFG = 1.0` fixed default (distilled single trajectory) |
| **Model Type** | Full foundation diffusion DiT | Accelerated / trajectory-distilled checkpoint |
| **Peak VRAM** | 9.6 GiB (INT8 ConvRot) | 9.6 GiB (INT8 ConvRot) |
| **Text Encoder & VAE** | Qwen3-VL 8B INT8 + Wan VAE | **Byte-identical, shared** (zero duplicate disk/RAM) |
| **CLI Runner** | `generate-qwen-image-2.1.sh` | `generate-qwen-image-2.1-turbo.sh` |

### Why Turbo?
- **Speed**: Cuts diffusion step runtime on Yeti from ~20s down to ~8.8s without noticeable quality degradation.
- **Resource Footprint**: The Turbo INT8 ConvRot DiT weight is only 7.25 GB and reuses the existing Qwen3-VL 8B text encoder (9.35 GB) and Wan/Qwen BF16 VAE (0.67 GB). Total storage is ~16.1 GiB, and inference runs completely inside Yeti's 12 GiB VRAM budget via ComfyUI dynamic low-VRAM staging.
- **Native Image Editing**: Reuses the vision encoder to accept an `--image` input sketch/photo and conditionally transform it in 8 steps.

---

## 2. Sampling Schedule & CFG Mechanics

### Non-Linear Sigma Schedule
Turbo cannot use ordinary named schedulers (like `simple` or `karras`). It requires the non-linear 8-step schedule calibrated during trajectory distillation:
```text
1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0
```
Notice the steep trajectory compression at the final two steps (`0.704 → 0.414 → 0.0`) to complete the denoise. In ComfyUI, this is wired via:
- `ManualSigmas` with the explicit string above.
- `KSamplerSelect` with `sampler_name: "euler"`.
- `SamplerCustom` combining model, conditioning, sigmas, and sampler.

### Guidance Policy
- **Recommended CFG: 1.0** (default).
- Turbo weights were distilled along a CFG=1 trajectory. Increasing CFG above 1.5–2.0 leads to contrast blowout, over-saturation, and high-frequency edge artifacts.

---

## 3. Pinned Inputs & Artifacts

- **Diffusion Transformer:** `diffusion_models/qwen-image-2.1-turbo-int8-convrot.safetensors`
  (7,256,828,688 bytes, SHA-256 `bc77cb2f0220311a158a10e77677f0912fac72d53a66faf78a67d7946b81f5c1`)
  from repository `NidAll/Turbo-INT8-ConvRot-for-Qwen-Image-2.1` @ `f1cff4611af588f67e482ce2a920230a8be84f87`.
- **Shared Text Encoder:** `text_encoders/qwen3vl_8b_int8_convrot.safetensors` (9,350,798,360 bytes).
- **Shared VAE:** `vae/qwen_image_2.1_vae_bf16.safetensors` (675,509,688 bytes).
- **Manifest:** Recorded in [`models/qwen-image-2.1-turbo-comfyui-files.txt`](../models/qwen-image-2.1-turbo-comfyui-files.txt).

---

## 4. Installation

Run from the L2M2 root:

```sh
./maintenance/setup-comfyui.sh
./maintenance/setup-qwen-image-2.1-turbo-comfyui.sh
```

The installer verifies SHA-256 checksums, downloads missing weights via `download_hf_model.py` to `$HOME/models/media/Qwen-Image-2.1/diffusion_models/`, and creates symlinks under `vendor/ComfyUI/models/diffusion_models/`.

---

## 5. Generation Workflows

Exclusive GPU access is required. Stop `llama-swap` before launching ComfyUI:

```sh
systemctl --user stop llama-swap.service
./maintenance/comfyui-service.sh start-gpu
./media-models/generate-qwen-image-2.1-turbo.sh --check
```

### A. Text-to-Image Generation

```sh
# Using the wrapper script:
./media-models/generate-qwen-image-2.1-turbo.sh \
  --prompt "cinematic medium shot of an astronaut holding a glowing bioluminescent plant inside an overgrown derelict greenhouse, soft golden hour rim light" \
  --seed 42

# Or run the wrapper directly:
./media-models/generate-qwen-image-2.1-turbo.sh --prompt "a lighthouse at dusk" --seed 42
```

### B. Image-to-Image / Transformation Workflows

Supply a reference image with `--image`. The runner copies the file to ComfyUI's input directory, splices it into `TextEncodeQwenImage21` with VAE reference encoding, and initializes the latent dimensions to match the reference:

```sh
./media-models/generate-qwen-image-2.1-turbo.sh \
  --image prompts/turbo-boat-input.png \
  --prompt-file prompts/yacht-edit.txt \
  --seed 20261009 \
  --output ~/media-output/yacht-edit.png
```

When finished, shut down ComfyUI and restore the LLM router:

```sh
./maintenance/comfyui-service.sh stop
systemctl --user start llama-swap.service
```

---

## 6. Measured Benchmark History

Detailed side-by-side run records are maintained in [`docs/experiments/qwen-image-2.1-turbo-vs-base.md`](experiments/qwen-image-2.1-turbo-vs-base.md):

- **T2I Astronaut Greenhouse (1024×1024, Seed 42):**
  - Base: 25 steps, 19.74s ComfyUI time, 20.13s wall-clock ([test-qwen-before.png](file:///home/kchauhan/media-output/test-qwen-before.png)).
  - Turbo: 8 steps, 8.83s ComfyUI time, 10.12s wall-clock ([test-qwen-after.png](file:///home/kchauhan/media-output/test-qwen-after.png)).
- **I2I Yacht Line Sketch Transformation (1024×1024, Seed 20261009):**
  - Turbo: 8 steps, 15.42s ComfyUI time, 16.13s wall-clock ([test-turbo-yacht-edit.png](file:///home/kchauhan/media-output/test-turbo-yacht-edit.png)).
