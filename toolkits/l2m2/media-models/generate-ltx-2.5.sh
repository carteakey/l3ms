#!/usr/bin/env bash
# Generate a short LTX-2.5 synchronized audio/video clip.
#
# The profile uses the official distilled BF16 checkpoint with fp8-cast and
# CPU offload, which is the practical starting point for Yeti's RTX 4070 12 GB.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  l2m2 --media ltx-2.5 --extra '--prompt "a paper boat crossing a puddle"'

Options:
  --prompt TEXT             Text prompt (required)
  --prompt-file PATH        Read the prompt from a UTF-8 text file
  --output PATH             MP4 output path (default: L3MS_MEDIA_OUTPUT_DIR)
  --image PATH FRAME STRENGTH
                            Optional still-image conditioning. Repeatable;
                            FRAME is a zero-based target frame and STRENGTH
                            is typically 0.8-1.0.
  --frames N                Number of frames (default: 121)
  --seed N                  Deterministic seed (default: 42)
  --transformer PATH        Transformer checkpoint override
  --quantization NAME       fp8-cast (default)
  --offload TARGET          cpu or disk (default: cpu)
  --help                    Show this help

Environment:
  L3MS_LTX_ROOT             LTX-2 checkout (default: $HOME/repos/LTX-2)
  L3MS_LTX_MODEL_DIR        LTX-2.5 model directory (default: $HOME/models/media/ltx-2.5)
  L3MS_LTX_TRANSFORMER      Transformer checkpoint override
  L3MS_MEDIA_OUTPUT_DIR     Output directory (default: $HOME/media-output)
EOF
}

ltx_root="${L3MS_LTX_ROOT:-${HOME}/repos/LTX-2}"
model_dir="${L3MS_LTX_MODEL_DIR:-${L3MS_MEDIA_ROOT:-${HOME}/models/media}/ltx-2.5}"
output_dir="${L3MS_MEDIA_OUTPUT_DIR:-${HOME}/media-output}"

if [[ -d "${HOME}/.local/bin" ]]; then
    PATH="${HOME}/.local/bin:${PATH}"
    export PATH
fi

prompt=""
prompt_file=""
output=""
image_args=()
frames=121
seed=42
quantization="fp8-cast"
offload="cpu"
transformer="${L3MS_LTX_TRANSFORMER:-}"

while (($#)); do
    case "$1" in
        --prompt)
            (($# >= 2)) || { echo "Missing value for --prompt" >&2; exit 2; }
            prompt="$2"
            shift 2
            ;;
        --prompt-file)
            (($# >= 2)) || { echo "Missing value for --prompt-file" >&2; exit 2; }
            prompt_file="$2"
            shift 2
            ;;
        --output|--output-path)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            output="$2"
            shift 2
            ;;
        --image|--input-image)
            (($# >= 4)) || {
                echo "Usage for $1: --image PATH FRAME STRENGTH" >&2
                exit 2
            }
            image_args+=("$2" "$3" "$4")
            shift 4
            ;;
        --frames|--num-frames|--seed|--transformer|--quantization|--offload)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --frames|--num-frames) frames="$2" ;;
                --seed) seed="$2" ;;
                --transformer) transformer="$2" ;;
                --quantization) quantization="$2" ;;
                --offload) offload="$2" ;;
            esac
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            ;;
        *)
            echo "Unknown LTX-2.5 option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "$prompt_file" ]]; then
    [[ -f "$prompt_file" ]] || { echo "Prompt file not found: $prompt_file" >&2; exit 1; }
    [[ -z "$prompt" ]] || { echo "--prompt and --prompt-file are mutually exclusive" >&2; exit 2; }
    prompt="$(<"$prompt_file")"
fi
if [[ -z "$prompt" ]]; then
    echo "LTX-2.5 requires --prompt TEXT" >&2
    exit 2
fi
[[ -d "$ltx_root" ]] || {
    echo "LTX-2 checkout not found: $ltx_root" >&2
    echo "Run maintenance/setup-media-runtimes.sh install-ltx" >&2
    exit 1
}

if [[ -z "$transformer" ]]; then
    transformer="$model_dir/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
fi
text_encoder="$model_dir/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
video_vae="$model_dir/vae/ltx-2.5-video-vae-conv-bf16.safetensors"
audio_vae="$model_dir/vae/ltx-2.5-audio-vae-bf16.safetensors"
spatial_upscaler="$model_dir/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
for required in "$transformer" "$text_encoder" "$video_vae" "$audio_vae" "$spatial_upscaler"; do
    [[ -f "$required" ]] || {
        echo "LTX-2.5 file not found: $required" >&2
        echo "Accept the gated model terms and run the download command in docs/media-generation-runbook.md" >&2
        exit 1
    }
done

for ((image_index = 0; image_index < ${#image_args[@]}; image_index += 3)); do
    image_path="${image_args[image_index]}"
    [[ -f "$image_path" ]] || {
        echo "LTX-2.5 conditioning image not found: $image_path" >&2
        exit 1
    }
    case "$image_path" in
        /*) ;;
        *) image_args[image_index]="$PWD/$image_path" ;;
    esac
done

command -v uv >/dev/null 2>&1 || {
    echo "uv is required to run LTX-2.5" >&2
    exit 1
}

mkdir -p "$output_dir"
if [[ -z "$output" ]]; then
    output="$output_dir/ltx-2.5-$(date +%Y%m%d-%H%M%S).mp4"
fi
case "$output" in
    /*) ;;
    *) output="$PWD/$output" ;;
esac

echo "LTX-2.5 distilled: $(basename "$transformer") / ${frames} frames / ${quantization} / ${offload} offload -> $output"
cd "$ltx_root"
args=(
    --transformer-path "$transformer" \
    --text-encoder-path "$text_encoder" \
    --video-vae-path "$video_vae" \
    --audio-vae-path "$audio_vae" \
    --spatial-upsampler-path "$spatial_upscaler" \
    --num-frames "$frames" \
    --seed "$seed" \
    --quantization "$quantization" \
    --offload "$offload" \
    --output-path "$output" \
    --prompt "$prompt"
)
for ((image_index = 0; image_index < ${#image_args[@]}; image_index += 3)); do
    args+=(
        --image
        "${image_args[image_index]}"
        "${image_args[image_index + 1]}"
        "${image_args[image_index + 2]}"
    )
done
exec uv run python -m ltx_pipelines.distilled "${args[@]}"
