# Qwen Image 2.1 on Yeti

Verified inputs: 2026-09-22. This runbook installs the official Comfy-Org
Qwen Image 2.1 quantized files and registers an L2M2 text-to-image profile.
The model files are about 16.1 GiB and are suited to Yeti's RTX 4070 12 GiB
GPU when ComfyUI has an exclusive GPU window.

## Pinned inputs

- ComfyUI commit `6bfaacc67c2103481e5f0c84d75257cd0581d86a`, the official Qwen
  Image 2.1 support revision.
- Model repository `Comfy-Org/Qwen-Image-2.1`, revision
  `ace0edeb3791a594ddfa36ed5f41a178a394e921`.
- Workflow templates revision
  `371a7b7171bbd11e9cc92ef615ba5ad223d7e5b4`.
- The manifest in `models/qwen-image-2.1-comfyui-files.txt` records exact
  sizes and SHA-256 values for the INT8 ConvRot diffusion model and Qwen3-VL
  encoder plus the BF16 VAE.

Review the [Qwen Image 2.1 model repository](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)
and its Qwen Research License before using or redistributing the weights.

> [!TIP]
> For the 8-step accelerated version (~2.24× faster at 8.8s vs 19.7s on RTX 4070), see [Qwen Image 2.1 Turbo Setup](file:///home/kchauhan/repos/l2m2/docs/qwen-image-2.1-turbo-setup.md).

## Install

Run from the L2M2 repository:

```sh
./maintenance/setup-comfyui.sh
./maintenance/setup-qwen-image-2.1-comfyui.sh
```

The model installer downloads to `$HOME/models/media/Qwen-Image-2.1` by default,
writes a checksum file, and symlinks the files into `vendor/ComfyUI/models`.
Set `L2M2_QWEN_IMAGE_21_ROOT` to choose another filesystem with at least the
same free space. Existing mismatched files and workflow paths are preserved
and cause the installer to stop rather than be overwritten.

The official workflows are saved at:

- `workflows/qwen-image-2.1/image_qwen_image_2_1_t2i.json`
- `workflows/qwen-image-2.1/image_qwen_image_2_1_image_edit.json`

## Generate

Generation must have exclusive access to the GPU. The service wrapper refuses
to start GPU ComfyUI while llama-swap is active:

```sh
systemctl --user stop llama-swap.service
./maintenance/comfyui-service.sh start-gpu
./media-models/generate-qwen-image-2.1.sh --check
./media-models/generate-qwen-image-2.1.sh --prompt "a lighthouse at dusk"
./maintenance/comfyui-service.sh stop
systemctl --user start llama-swap.service
```

The L2M2 CLI profile forwards quoted arguments to the same script:

```sh
./media-models/generate-qwen-image-2.1.sh --prompt "a lighthouse at dusk"
```

Defaults are 1024x1024, 25 Euler/simple steps, CFG 1, and seed 42. Keep
dimensions divisible by 32 and start at 1024 on the 12 GiB GPU. Outputs go to
`~/media-output` unless `--output` or `L3MS_MEDIA_OUTPUT_DIR` is set. The
listener is loopback-only; use an SSH port forward if accessing ComfyUI from
another machine.

Import the text-to-image workflow for native ComfyUI editing. The bundled image
edit workflow is available for image-to-image work; the L2M2 wrapper currently
covers text-to-image generation.

## Verification

```sh
python3 -m unittest discover -s media-models -p 'test_*.py'
for script in media-models/*.sh maintenance/*.sh; do bash -n "$script"; done
```

These checks are offline. `--check` additionally queries the running ComfyUI
catalog and confirms the required Qwen nodes are registered.
