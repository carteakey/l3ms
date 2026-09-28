#!/usr/bin/env bash
# Generate MiniMax-H3 audio or short video with audio.cpp.
#
# Defaults are deliberately conservative for Yeti's RTX 4070 12 GB:
# audio-first 32x32 latents, 241 frames, and staged/layerwise Q4_K weights.
# Pass --video for a short 480p RGB24 video artifact.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  l2m2 --media minimax-h3 --extra '--prompt "a warm analog synth loop"'
  l2m2 --media minimax-h3 --extra '--prompt "a rainy neon street" --video'

Options:
  --prompt TEXT             Text prompt (required)
  --prompt-file PATH        Read the prompt from a UTF-8 text file
  --output PATH             Output audio path (default: L3MS_MEDIA_OUTPUT_DIR)
  --out-dir DIR             Output directory for runtime artifacts
  --video-output PATH       Playable MP4 output path (with --video)
  --video                   Also decode an RGB24 video artifact
  --audio-only              Force audio-only output (default)
  --steps N                 Joint DiT denoising steps (default: 20)
  --seed N                  Deterministic seed (default: 42)
  --width N / --height N    Latent canvas dimensions
  --frames N                Target frame count (24 fps for video)
  --request-option K=V      Additional audio.cpp request option
  --help                    Show this help

Environment:
  L3MS_AUDIO_CPP_CLI        audiocpp_cli path override
  L3MS_AUDIO_CPP_ROOT       audio.cpp checkout (default: $HOME/repos/audio.cpp)
  L3MS_MEDIA_ROOT           model root (default: $HOME/models/media)
  L3MS_H3_MODEL_DIR         MiniMax-H3-Q4-GGUF directory override
EOF
}

find_cli() {
    if [[ -n "${L3MS_AUDIO_CPP_CLI:-}" && -x "${L3MS_AUDIO_CPP_CLI}" ]]; then
        printf '%s\n' "$L3MS_AUDIO_CPP_CLI"
        return 0
    fi
    local root="${L3MS_AUDIO_CPP_ROOT:-}"
    local candidate
    local search_roots=()
    if [[ -n "$root" ]]; then
        search_roots+=("$root")
    fi
    search_roots+=(
        "${REPO_DIR:-$(dirname "$SCRIPT_DIR")}/vendor/audio.cpp"
        "${REPO_DIR:-$(dirname "$SCRIPT_DIR")}/vendor/audio.cpp-master"
        "${HOME}/repos/audio.cpp"
    )
    for r in "${search_roots[@]}"; do
        for candidate in \
            "$r/build/linux-cuda-release/bin/audiocpp_cli" \
            "$r/build/bin/audiocpp_cli" \
            "$r/build/debug/bin/audiocpp_cli" \
            "$r/build/release/bin/audiocpp_cli"; do
            if [[ -x "$candidate" ]]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done
    done
    return 1
}

audio_cpp_root="${L3MS_AUDIO_CPP_ROOT:-${HOME}/repos/audio.cpp}"
media_root="${L3MS_MEDIA_ROOT:-${HOME}/models/media}"
model_dir="${L3MS_H3_MODEL_DIR:-${media_root}/MiniMax-H3-Q4-GGUF}"
model_path="${L3MS_H3_MODEL:-${model_dir}/dit.gguf}"
spec_path="${L3MS_H3_MODEL_SPEC:-${audio_cpp_root}/model_specs/minimax_h3.json}"
output_dir="${L3MS_MEDIA_OUTPUT_DIR:-${HOME}/media-output}"
prompt=""
prompt_file=""
output=""
video_output=""
video=false
steps=20
seed=42
width=""
height=""
frames=""
request_options=()

while (($#)); do
    case "$1" in
        --prompt|--text)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            prompt="$2"
            shift 2
            ;;
        --prompt-file)
            (($# >= 2)) || { echo "Missing value for --prompt-file" >&2; exit 2; }
            prompt_file="$2"
            shift 2
            ;;
        --output|--out)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            output="$2"
            shift 2
            ;;
        --out-dir)
            (($# >= 2)) || { echo "Missing value for --out-dir" >&2; exit 2; }
            output_dir="$2"
            shift 2
            ;;
        --video-output|--video-out)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            video_output="$2"
            shift 2
            ;;
        --video)
            video=true
            shift
            ;;
        --audio-only)
            video=false
            shift
            ;;
        --steps|--num-inference-steps)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            steps="$2"
            shift 2
            ;;
        --seed|--width|--height|--frames)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --seed) seed="$2" ;;
                --width) width="$2" ;;
                --height) height="$2" ;;
                --frames) frames="$2" ;;
            esac
            shift 2
            ;;
        --request-option)
            (($# >= 2)) || { echo "Missing value for --request-option" >&2; exit 2; }
            request_options+=("$2")
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
            echo "Unknown MiniMax-H3 option: $1" >&2
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
    echo "MiniMax-H3 requires --prompt TEXT" >&2
    exit 2
fi

if [[ "$video" == true ]]; then
    width="${width:-${L3MS_H3_VIDEO_WIDTH:-832}}"
    height="${height:-${L3MS_H3_VIDEO_HEIGHT:-480}}"
    frames="${frames:-${L3MS_H3_VIDEO_FRAMES:-121}}"
else
    width="${width:-${L3MS_H3_AUDIO_WIDTH:-32}}"
    height="${height:-${L3MS_H3_AUDIO_HEIGHT:-32}}"
    frames="${frames:-${L3MS_H3_AUDIO_FRAMES:-241}}"
fi

cli_path="$(find_cli || true)"
[[ -n "$cli_path" && -x "$cli_path" ]] || {
    echo "audiocpp_cli not found; run maintenance/setup-media-runtimes.sh install-audio-cpp" >&2
    exit 1
}
[[ -f "$model_path" ]] || {
    echo "MiniMax-H3 DiT not found: $model_path" >&2
    echo "Run maintenance/setup-media-runtimes.sh install-audio-cpp" >&2
    exit 1
}
[[ -f "$spec_path" ]] || {
    echo "MiniMax-H3 model spec not found: $spec_path" >&2
    exit 1
}

mkdir -p "$output_dir"
run_stamp="$(date +%Y%m%d-%H%M%S)"
if [[ -z "$output" ]]; then
    output="$output_dir/minimax-h3-$run_stamp.wav"
fi
if [[ "$video" == true && -z "$video_output" ]]; then
    video_output="${output%.*}.mp4"
fi
if [[ "$video" == true ]]; then
    mkdir -p "$(dirname "$video_output")"
fi

args=(
    --task gen
    --family minimax_h3
    --model "$model_path"
    --model-spec-override "$spec_path"
    --backend cuda
    --threads "${L3MS_H3_THREADS:-8}"
    --text "$prompt"
    --seed "$seed"
    --num-inference-steps "$steps"
    --guidance-scale "${L3MS_H3_GUIDANCE_SCALE:-1.0}"
    --request-option "height=$height"
    --request-option "width=$width"
    --request-option "num_frames=$frames"
    --request-option "return_video=$video"
    --request-option "text_layerwise=true"
    --request-option "text_layerwise_batch=1"
    --request-option "dit_layerwise=true"
    --request-option "dit_layerwise_batch=1"
    --request-option "dit_mlp_chunk_tokens=${L3MS_H3_MLP_CHUNK_TOKENS:-1024}"
    --session-option "minimax_h3.weight_context_mb=${L3MS_H3_WEIGHT_CONTEXT_MB:-512}"
    --session-option "minimax_h3.mem_saver=true"
    --out "$output"
    --out-dir "$output_dir"
    --metrics
)
for option in "${request_options[@]}"; do
    args+=(--request-option "$option")
done

echo "MiniMax-H3: $(basename "$model_path") -> $output"
if [[ "$video" == true ]]; then
    echo "Profile: Yeti 12 GB / ${width}x${height} / ${frames} frames / layerwise Q4_K"
else
    echo "Profile: Yeti 12 GB / audio-first / ${frames} frames / layerwise Q4_K"
fi
"$cli_path" "${args[@]}"

if [[ "$video" == true ]]; then
    video_json="$output_dir/minimax_h3_video_rgb24.json"
    if [[ ! -s "$video_json" ]]; then
        echo "MiniMax-H3 returned no RGB24 video artifact: $video_json" >&2
        exit 1
    fi
    if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v jq >/dev/null 2>&1 || ! command -v xxd >/dev/null 2>&1; then
        echo "MiniMax-H3 raw RGB24 artifact: $video_json"
        echo "Install ffmpeg, jq, and xxd to mux a playable MP4" >&2
        exit 0
    fi

    video_width="$(jq -er '.meta.width' "$video_json")"
    video_height="$(jq -er '.meta.height' "$video_json")"
    video_frames="$(jq -er '.meta.frames' "$video_json")"
    video_fps="$(jq -er '.meta.fps' "$video_json")"
    raw_video="$(mktemp "${TMPDIR:-/tmp}/l3ms-minimax-h3-video.XXXXXX.rgb")"
    trap 'rm -f "$raw_video"' EXIT
    jq -er '.payload_hex' "$video_json" | xxd -r -p > "$raw_video"
    ffmpeg_args=(
        -y
        -f rawvideo
        -pixel_format rgb24
        -video_size "${video_width}x${video_height}"
        -framerate "$video_fps"
        -i "$raw_video"
        -i "$output"
        -map 0:v:0
        -map 1:a:0
        -frames:v "$video_frames"
        -c:v libx264
        -preset veryfast
        -pix_fmt yuv420p
        -c:a aac
        -shortest
        "$video_output"
    )
    if ! ffmpeg -loglevel error "${ffmpeg_args[@]}"; then
        echo "MiniMax-H3 failed to mux playable video; raw artifact retained: $video_json" >&2
        exit 1
    fi
    echo "MiniMax-H3 playable video: $video_output"
fi
