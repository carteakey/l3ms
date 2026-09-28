# Z-Image-Turbo on Yeti

Verified inputs: 2026-09-25. This runbook documents the official Comfy-Org
Z-Image-Turbo weights, Qwen 3 4B text encoders (BF16 & FP8 mixed), and VAE,
registering workflows for native ComfyUI inference. Ranked #10 overall (#1 open Apache 2.0 turbo) on the Text-to-Image Arena.

## Pinned inputs

- ComfyUI commit `6bfaacc67c2103481e5f0c84d75257cd0581d86a`.
- Model repository `Comfy-Org/z_image_turbo`, revision `6fc90a3b1b653e935a0d175e260736de25b84df5`.
- Manifest `models/z-image-turbo-comfyui-files.txt` records verified byte sizes and SHA-256 sums:
  - `diffusion_models/z_image_turbo_bf16.safetensors` (12,309,866,400 bytes, `2407613050b809ffdff18a4ac99af83ea6b95443ecebdf80e064a79c825574a6`)
  - `diffusion_models/z_image_turbo_int8_convrot.safetensors` (6,201,001,296 bytes, `be517ebd47c912a5626a588e1aeea43e6be4a43c0cdcd2b48a2a780d9f358635`)
  - `text_encoders/qwen_3_4b.safetensors` (8,044,982,048 bytes, `6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a`)
  - `text_encoders/qwen_3_4b_fp8_mixed.safetensors` (5,631,994,051 bytes, `72450b19758172c5a7273cf7de729d1c17e7f434a104a00167624cba94f68f15`)
  - `vae/ae.safetensors` (335,304,388 bytes, `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38`)

## Install & Location

Installed on the secondary NVMe Games drive to preserve root filesystem space:
- Storage root: `/run/media/kchauhan/Games/models/media/Z-Image-Turbo`
- Symlinks into: `vendor/ComfyUI/models/` (`diffusion_models`, `text_encoders`, `vae`)
- Workflows:
  - `workflows/z-image-turbo/image_z_image_turbo.json`
  - `workflows/z-image-turbo/image_z_image_turbo_int8.json`
- UI Discovery: Symlinked to `.toolkit/comfyui/user/default/workflows/`

Installer script:
```sh
./maintenance/setup-z-image-turbo-comfyui.sh
```
