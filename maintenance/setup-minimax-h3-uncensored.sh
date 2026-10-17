#!/usr/bin/env bash
# Install MiniMax-H3 Ref2VA baseline with uncensored NVFP4 text encoder for ComfyUI.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY_DIR="$ROOT/vendor/ComfyUI"
MEDIA_ROOT="${L3MS_MEDIA_ROOT:-$HOME/models/media}"
MODEL_DIR="$MEDIA_ROOT/MiniMax-H3"
PYTHON="$ROOT/.venv-comfy/bin/python"
HF_CLI="$ROOT/.venv-comfy/bin/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1

[[ -x "$PYTHON" ]] || {
    echo "ComfyUI environment missing; run maintenance/setup-comfyui.sh first." >&2
    exit 1
}

mkdir -p "$MODEL_DIR"/{diffusion_models,vae,text_encoders}
mkdir -p "$COMFY_DIR"/models/{diffusion_models,vae,text_encoders}

echo "==> 1. Downloading MiniMax-H3 base components from Comfy-Org/MiniMax-H3..."
"$HF_CLI" download Comfy-Org/MiniMax-H3 \
    diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
    vae/minimax_h3_video_vae_fp16.safetensors \
    vae/minimax_h3_audio_vae_fp32.safetensors \
    --local-dir "$MODEL_DIR" < /dev/null

echo "==> 2. Downloading uncensored Qwen3-VL-32B text encoder from Momoking/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4..."
"$HF_CLI" download Momoking/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4 \
    qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors \
    --local-dir "$MODEL_DIR/text_encoders" < /dev/null

echo "==> 3. Linking models into ComfyUI..."
ln -sfn "$MODEL_DIR/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" \
    "$COMFY_DIR/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
ln -sfn "$MODEL_DIR/vae/minimax_h3_video_vae_fp16.safetensors" \
    "$COMFY_DIR/models/vae/minimax_h3_video_vae_fp16.safetensors"
ln -sfn "$MODEL_DIR/vae/minimax_h3_audio_vae_fp32.safetensors" \
    "$COMFY_DIR/models/vae/minimax_h3_audio_vae_fp32.safetensors"
ln -sfn "$MODEL_DIR/text_encoders/qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors" \
    "$COMFY_DIR/models/text_encoders/qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"
# Provide compatibility alias so default stock workflows route through uncensored encoder
ln -sfn "$MODEL_DIR/text_encoders/qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors" \
    "$COMFY_DIR/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"

echo "==> Done. MiniMax-H3 Uncensored baseline installed at $MODEL_DIR"
