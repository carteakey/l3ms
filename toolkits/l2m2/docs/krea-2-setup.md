# Krea 2 on Yeti

Verified inputs: 2026-09-24. This runbook installs the official Comfy-Org
Krea 2 raw INT8 weights, the Qwen3-VL 4B text encoder, VAE, and the r128 Turbo
distillation LoRA, registering an L2M2 text-to-image profile. The model files
are about 18.6 GiB total and are tuned for Yeti's RTX 4070 12 GiB GPU when
ComfyUI has an exclusive GPU window.

## Pinned inputs

- ComfyUI commit `6bfaacc67c2103481e5f0c84d75257cd0581d86a`, which includes
  native single-stream DiT support for Krea 2 and `type: "krea2"` in `CLIPLoader`.
- Base model repository `Comfy-Org/Krea-2`, revision
  `eb1eddd3983a54678545a9b2c178c5853b30f7be`.
- LoRA repository `TheDivergentAI/krea2-turbo-distill-lora`, revision
  `eac192a732e1d128c1d0029d79941857fbb321de`.
- Workflow templates revision `a7acaf8cee9ccccdaf4b26ce04a9fced4c4d13de`.
- The manifest in `models/krea-2-comfyui-files.txt` records exact sizes and
  SHA-256 values for:
  - `diffusion_models/krea2_raw_int8_convrot.safetensors` (13,492,686,496 bytes)
  - `text_encoders/qwen3vl_4b_fp8_scaled.safetensors` (5,242,467,968 bytes)
  - `vae/qwen_image_vae.safetensors` (253,806,246 bytes)
  - `loras/krea2_turbo_distill_r128.safetensors` (936,944,890 bytes)

Review the [Krea-2 model repository](https://huggingface.co/Comfy-Org/Krea-2)
and the Krea Community License before using or redistributing the weights.

## Install

Run from the L2M2 repository:

```sh
./maintenance/setup-comfyui.sh
./maintenance/setup-krea-2-comfyui.sh
```

The model installer downloads to `$HOME/models/media/Krea-2` by default,
writes a checksum file, and symlinks the files into `vendor/ComfyUI/models`.
Set `L2M2_KREA_2_ROOT` to choose another filesystem with at least the same
free space. Existing mismatched files and workflow paths are preserved and
cause the installer to stop rather than be overwritten.

The official workflows are saved at:

- `workflows/krea-2/image_krea2_turbo_t2i.json`
- `workflows/krea-2/image_krea2_turbo_t2i_int8.json`
- `workflows/krea-2/image_krea2_turbo_int8_image_style_reference.json`

## Generate

Generation must have exclusive access to the GPU. The service wrapper refuses
to start GPU ComfyUI while llama-swap is active:

```sh
systemctl --user stop llama-swap.service
./maintenance/comfyui-service.sh start-gpu
./media-models/generate-krea-2.sh --check
./media-models/generate-krea-2.sh --prompt "cinematic portrait of an astronaut exploring an ancient overgrown temple"
./maintenance/comfyui-service.sh stop
systemctl --user start llama-swap.service
```

The L2M2 CLI profile forwards quoted arguments to the same script:

```sh
./target/release/l2m2 --media krea-2 --extra '--prompt "cinematic portrait of an astronaut exploring an ancient overgrown temple"'
```

### Recommended settings for 12 GiB VRAM

- **Model:** Krea 2 raw INT8 ConvRot with r128 Turbo distill LoRA.
- **LoRA Strength:** `0.60` (controlled via `--lora-strength 0.60`).
- **Steps:** `12` steps (Euler sampler, simple scheduler).
- **CFG Scale:** `1.5` (controlled via `--cfg 1.5`).
- **Resolution:** `1024x1024` or portrait/landscape combinations up to `1024x1536` / `1536x1024` (max 1,572,864 pixels, multiples of 32).
- **Seed:** `42` default (`--seed`).

Outputs go to `~/media-output` unless `--output` or `L3MS_MEDIA_OUTPUT_DIR` is set.
The ComfyUI listener is loopback-only; use an SSH port forward if accessing ComfyUI
from another machine.

Import `workflows/krea-2/image_krea2_turbo_t2i_int8.json` for native ComfyUI GUI
generation and prompt experimentation.

## Verification

```sh
python3 -m unittest discover -s media-models -p 'test_*.py'
for script in media-models/*.sh maintenance/*.sh; do bash -n "$script"; done
```

These checks are offline. `--check` additionally queries the running ComfyUI
catalog and confirms the required Krea 2 nodes are registered.
