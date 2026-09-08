#!/usr/bin/env bash
# Generate a local song with the Q8_0 HeartMuLa package through audio.cpp.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  l2m2 --media heartmula-music --extra '--prompt "dreamy ambient electronica" --instrumental'

Options:
  --prompt TEXT             Music description (required)
  --prompt-file PATH        Read the prompt from a UTF-8 text file
  --tags TEXT               Comma-separated style/control tags (default: prompt)
  --lyrics TEXT             Lyrics to arrange
  --lyrics-file PATH        Read lyrics from a local UTF-8 file
  --instrumental            Add instrumental/no-vocals control tags
  --duration SECONDS        Maximum output duration (default: 30)
  --steps N                 HeartCodec flow solver steps (default: 10)
  --seed N                  Deterministic seed (default: 42)
  --temperature VALUE       Music-token sampling temperature (default: 1.0)
  --top-k N                 Music-token top-k limit (default: 50)
  --guidance-scale VALUE    MuLa classifier-free guidance (default: 1.5)
  --output PATH             WAV output path (default: L3MS_MEDIA_OUTPUT_DIR)
  --help                    Show this help

Environment:
  L3MS_AUDIO_CPP_CLI        audiocpp_cli path override
  L3MS_AUDIO_CPP_ROOT       audio.cpp checkout (default: $HOME/repos/audio.cpp)
  L3MS_MEDIA_ROOT           model root (default: $HOME/models/media)
  L3MS_HEARTMULA_MODEL      Q8_0 GGUF path override
  L3MS_MEDIA_OUTPUT_DIR     Output directory (default: $HOME/media-output)
EOF
}

find_cli() {
    if [[ -n "${L3MS_AUDIO_CPP_CLI:-}" ]]; then
        printf '%s\n' "$L3MS_AUDIO_CPP_CLI"
        return
    fi
    local root="${L3MS_AUDIO_CPP_ROOT:-${HOME}/repos/audio.cpp}"
    local candidate
    for candidate in \
        "$root/build/linux-cuda-release/bin/audiocpp_cli" \
        "$root/build/bin/audiocpp_cli" \
        "$root/build/debug/bin/audiocpp_cli" \
        "$root/build/release/bin/audiocpp_cli"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    return 1
}

audio_cpp_root="${L3MS_AUDIO_CPP_ROOT:-${HOME}/repos/audio.cpp}"
media_root="${L3MS_MEDIA_ROOT:-${HOME}/models/media}"
model_path="${L3MS_HEARTMULA_MODEL:-${media_root}/HeartMuLa-GGUF/heartmula-q8_0.gguf}"
spec_path="${L3MS_HEARTMULA_MODEL_SPEC:-${audio_cpp_root}/model_specs/heartmula.json}"
output_dir="${L3MS_MEDIA_OUTPUT_DIR:-${HOME}/media-output}"

prompt=""
prompt_file=""
tags=""
lyrics=""
lyrics_file=""
instrumental=false
duration=30
steps=10
seed=42
temperature=1.0
top_k=50
guidance_scale=1.5
output=""

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
        --tags)
            (($# >= 2)) || { echo "Missing value for --tags" >&2; exit 2; }
            tags="$2"
            shift 2
            ;;
        --lyrics)
            (($# >= 2)) || { echo "Missing value for --lyrics" >&2; exit 2; }
            lyrics="$2"
            shift 2
            ;;
        --lyrics-file)
            (($# >= 2)) || { echo "Missing value for --lyrics-file" >&2; exit 2; }
            lyrics_file="$2"
            shift 2
            ;;
        --instrumental)
            instrumental=true
            shift
            ;;
        --duration|--duration-seconds|--steps|--num-inference-steps|--seed|--temperature|--top-k|--guidance-scale)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --duration|--duration-seconds) duration="$2" ;;
                --steps|--num-inference-steps) steps="$2" ;;
                --seed) seed="$2" ;;
                --temperature) temperature="$2" ;;
                --top-k) top_k="$2" ;;
                --guidance-scale) guidance_scale="$2" ;;
            esac
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
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            ;;
        *)
            echo "Unknown HeartMuLa option: $1" >&2
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
if [[ -n "$lyrics_file" ]]; then
    [[ -f "$lyrics_file" ]] || { echo "Lyrics file not found: $lyrics_file" >&2; exit 1; }
    [[ -z "$lyrics" ]] || { echo "--lyrics and --lyrics-file are mutually exclusive" >&2; exit 2; }
    lyrics="$(<"$lyrics_file")"
fi
[[ -n "$prompt" ]] || { echo "HeartMuLa requires --prompt TEXT" >&2; exit 2; }
if [[ "$instrumental" == true && -n "$lyrics" ]]; then
    echo "--instrumental and --lyrics are mutually exclusive" >&2
    exit 2
fi
if [[ -z "$tags" ]]; then
    tags="$prompt"
fi
if [[ "$instrumental" == true ]]; then
    tags="${tags},instrumental,no vocals"
fi

cli_path="$(find_cli || true)"
[[ -n "$cli_path" && -x "$cli_path" ]] || {
    echo "audiocpp_cli not found; run maintenance/setup-media-runtimes.sh install-audio-cpp" >&2
    exit 1
}
[[ -f "$model_path" ]] || {
    echo "HeartMuLa Q8_0 GGUF not found: $model_path" >&2
    echo "Run maintenance/setup-media-runtimes.sh install-music" >&2
    exit 1
}
[[ -f "$spec_path" ]] || {
    echo "HeartMuLa model spec not found: $spec_path" >&2
    exit 1
}

mkdir -p "$output_dir"
if [[ -z "$output" ]]; then
    output="$output_dir/heartmula-$(date +%Y%m%d-%H%M%S).wav"
fi

args=(
    --task gen
    --family heartmula
    --model "$model_path"
    --model-spec-override "$spec_path"
    --backend cuda
    --threads "${L3MS_HEARTMULA_THREADS:-8}"
    --text "$prompt"
    --lyrics "$lyrics"
    --duration-seconds "$duration"
    --num-inference-steps "$steps"
    --seed "$seed"
    --temperature "$temperature"
    --top-k "$top_k"
    --guidance-scale "$guidance_scale"
    --request-option "tags=$tags"
    --session-option "heartmula.mem_saver=true"
    --out "$output"
    --metrics
)

echo "HeartMuLa Q8_0 / audio.cpp / ${duration}s max -> $output"
exec "$cli_path" "${args[@]}"
