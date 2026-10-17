#!/usr/bin/env bash
# Install official LTX-2.5 ComfyUI INT8 ConvRot components and VAEs.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY_DIR="$ROOT/vendor/ComfyUI"
MEDIA_ROOT="${L3MS_MEDIA_ROOT:-$HOME/models/media}"
MODEL_DIR="$MEDIA_ROOT/ltx-2.5"
PYTHON="$ROOT/.venv-comfy/bin/python"
HF_CLI="$ROOT/.venv-comfy/bin/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1
REPO="Lightricks/LTX-2.5"

[[ -x "$PYTHON" ]] || {
    echo "ComfyUI environment missing; run maintenance/setup-comfyui.sh first." >&2
    exit 1
}

mkdir -p "$MODEL_DIR"/{diffusion_models,text_encoders,vae,latent_upscale_models}
mkdir -p "$COMFY_DIR"/models/{diffusion_models,text_encoders,vae,latent_upscale_models}

echo "==> 1. Downloading official LTX-2.5 Comfy INT8 and VAE components from Lightricks/LTX-2.5..."
"$HF_CLI" download "$REPO" \
    diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors \
    text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors \
    vae/ltx-2.5-video-vae-conv-bf16.safetensors \
    vae/ltx-2.5-audio-vae-bf16.safetensors \
    latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    --local-dir "$MODEL_DIR" < /dev/null

echo "==> 2. Linking models into ComfyUI..."
ln -sfn "$MODEL_DIR/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" \
    "$COMFY_DIR/models/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
# Compatibility alias for workflows expecting bf16 name
ln -sfn "$MODEL_DIR/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" \
    "$COMFY_DIR/models/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"

ln -sfn "$MODEL_DIR/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors" \
    "$COMFY_DIR/models/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
# Compatibility alias for workflows expecting bf16 name
ln -sfn "$MODEL_DIR/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors" \
    "$COMFY_DIR/models/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"

ln -sfn "$MODEL_DIR/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
    "$COMFY_DIR/models/vae/ltx-2.5-video-vae-conv-bf16.safetensors"
ln -sfn "$MODEL_DIR/vae/ltx-2.5-audio-vae-bf16.safetensors" \
    "$COMFY_DIR/models/vae/ltx-2.5-audio-vae-bf16.safetensors"
ln -sfn "$MODEL_DIR/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
    "$COMFY_DIR/models/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

echo "==> Done. LTX-2.5 Comfy baseline installed at $MODEL_DIR"
