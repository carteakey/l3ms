# FLUX.2 [Klein] 9B on Yeti

Verified inputs: 2026-09-25. This runbook documents the FLUX.2 Klein 9B models
(distilled 4-step turbo model, base model, Qwen 3 8B FP8 mixed text encoder, and VAE),
registering workflows for native ComfyUI inference. Ranked #11 overall on the Text-to-Image Arena.

## Pinned inputs

- ComfyUI commit `6bfaacc67c2103481e5f0c84d75257cd0581d86a`.
- Model repositories and verified revisions:
  - `titomatus0203/flux-2-klein-9b-fp8` (rev `6bc983e15e4c905dabe2985902ba26edfdbf277e`, bit-for-bit identical to BFL official release)
  - `wissxi/FLUX.2-klein-base-9b-fp8` (rev `4f274be83da93f436ad65b00b3756f20386944da`, bit-for-bit identical to BFL official release)
  - `Comfy-Org/vae-text-encorder-for-flux-klein-9b` (rev `3f62d9d8ae1fec33c6e91453d5c712855b096b55`)
- Manifest `models/flux-2-klein-9b-comfyui-files.txt` records verified byte sizes and SHA-256 sums:
  - `diffusion_models/flux-2-klein-9b-fp8.safetensors` (9,433,061,528 bytes, `865ba09f5b4c3cbd3468a4bd3acb9fcb2f8740c54317482f0bcd4ed1d3655cee`)
  - `diffusion_models/flux-2-klein-base-9b-fp8.safetensors` (9,567,278,472 bytes, `a9f5028c24a7a96f4f45beb883aad287d9bccc246227a6803edc898ddda42cf4`)
  - `text_encoders/qwen_3_8b_fp8mixed.safetensors` (8,664,848,742 bytes, `abad16806e0cbabc54e0325d6565847443fe396d5f0be38bb3cd3fe75a1201d6`)
  - `vae/flux2-vae.safetensors` (336,211,292 bytes, `868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3`)
  - `vae/full_encoder_small_decoder.safetensors` (symlinked alias to `flux2-vae.safetensors`)

## Install & Location

Installed on the secondary NVMe Games drive to preserve root filesystem space:
- Storage root: `/run/media/kchauhan/Games/models/media/FLUX.2-Klein-9B`
- Symlinks into: `vendor/ComfyUI/models/` (`diffusion_models`, `text_encoders`, `vae`)
- Workflows:
  - `workflows/flux-2-klein/image_flux2_text_to_image_9b.json`
  - `workflows/flux-2-klein/image_flux2_klein_text_to_image.json`
- UI Discovery: Symlinked to `.toolkit/comfyui/user/default/workflows/`

Installer script:
```sh
./maintenance/setup-flux-2-klein-9b-comfyui.sh
```
