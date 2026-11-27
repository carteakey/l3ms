#!/usr/bin/env bash
# generate-minimax-h3-image.sh — MiniMax-H3-Image Pruned Q4_K_M via ComfyUI v0.46.3 + ComfyUI-GGUF
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMFY="${REPO_DIR}/vendor/ComfyUI"

if [[ "${1:-}" == "--check" ]]; then
  for f in \
    "${COMFY}/models/unet/MiniMax-H3-Image-Pruned-Q4_K_M.gguf" \
    "${COMFY}/models/vae/minimax_h3_image_vae_fp16.safetensors" \
    "${COMFY}/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
    "${COMFY}/custom_nodes/ComfyUI-GGUF/__init__.py"; do
    [[ -e "$f" ]] || { echo "Missing: $f" >&2; exit 1; }
  done
  echo "OK: MiniMax-H3-Image Q4_K_M + VAE + Qwen3-VL-32B + ComfyUI-GGUF present"
  exit 0
fi

echo "MiniMax-H3-Image (Pruned Q4_K_M) runs on ComfyUI (llama-swap /comfyui/)."
echo "  Workflow: workflows/minimax-h3-image/image_minimax_h3_image_t2i_gguf.json"
echo "  Shared encoder: qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
echo "  VAE: minimax_h3_image_vae_fp16.safetensors"
