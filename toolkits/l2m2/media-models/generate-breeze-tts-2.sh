#!/usr/bin/env bash
# Generate local speech or clone voices with BreezeTTS 2 through audio.cpp.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  # Voice design / standard TTS:
  l2m2 --media breeze-tts-2 --extra '--text "Welcome to local TTS." --instruction "A warm female narrator with calm pacing."'

  # Voice cloning with reference audio:
  l2m2 --media breeze-tts-2 --extra '--task clon --text "Read this in my voice." --voice-ref sample.wav --reference-text "Reference transcript."'

Options:
  --text TEXT               Input text to synthesize (required unless --text-file)
  --text-file PATH          Read input text from a UTF-8 text file
  --task TASK               Generation task: tts (default), clon, or design
  --instruction TEXT        Voice design or style prompt (default: "Speak clearly and naturally.")
  --voice-ref PATH          Reference speaker WAV audio (required for --task clon)
  --reference-text TEXT     Transcript for reference speaker audio
  --reference-text-file PATH Read reference transcript from a file
  --mode MODE               Generation mode: offline (default) or streaming
  --stream-frames N         Streaming frames per event (default: 16)
  --stream-lookahead N      Streaming lookahead margin (default: 12)
  --max-tokens N            Maximum generated acoustic frames (default: 1500)
  --guidance-scale VALUE    Classifier-free guidance scale (default: 1.0)
  --temperature VALUE       Backbone sampling temperature (default: 0.9)
  --depth-temperature VALUE Depth decoder sampling temperature (default: 0.9)
  --top-k N                 Sampling top-k limit (default: 50, 0 to disable)
  --top-p VALUE             Sampling top-p limit (default: 1.0)
  --seed N                  Deterministic seed (default: 0)
  --weight-type TYPE        Matmul weight storage type: native, q8_0, q4_k, etc. (default: native)
  --attention MODE          Attention kernel: auto (default), flash, or eager
  --bf16-activations MODE   Activation rounding: auto (default), on, or off
  --output PATH             Output WAV path (default: $L3MS_MEDIA_OUTPUT_DIR/breeze-tts-*.wav)
  --out-dir PATH            Output directory override
  --help, -h                Show this help

Environment:
  L3MS_AUDIO_CPP_CLI        audiocpp_cli path override
  L3MS_AUDIO_CPP_ROOT       audio.cpp checkout (default: vendor/audio.cpp or $HOME/repos/audio.cpp)
  L3MS_MEDIA_ROOT           Model root (default: $HOME/models/media)
  L3MS_BREEZE_TTS_MODEL     Breeze-TTS-2 GGUF path override
  L3MS_BREEZE_TTS_MODEL_SPEC Model spec JSON override
  L3MS_BREEZE_TTS_THREADS   Thread count (default: 8)
  L3MS_MEDIA_OUTPUT_DIR     Output directory (default: $HOME/media-output)
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
        "${REPO_DIR}/vendor/audio.cpp"
        "${REPO_DIR}/vendor/audio.cpp-master"
        "${HOME}/repos/audio.cpp"
    )
    for r in "${search_roots[@]}"; do
        for candidate in \
            "$r/build/linux-cuda-release/bin/audiocpp_cli" \
            "$r/build/bin/audiocpp_cli" \
            "$r/build/release/bin/audiocpp_cli" \
            "$r/build/debug/bin/audiocpp_cli"; do
            if [[ -x "$candidate" ]]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done
    done
    return 1
}

find_audio_root() {
    if [[ -n "${L3MS_AUDIO_CPP_ROOT:-}" && -d "${L3MS_AUDIO_CPP_ROOT}" ]]; then
        printf '%s\n' "$L3MS_AUDIO_CPP_ROOT"
        return 0
    fi
    for r in "${REPO_DIR}/vendor/audio.cpp" "${REPO_DIR}/vendor/audio.cpp-master" "${HOME}/repos/audio.cpp"; do
        if [[ -d "$r" ]]; then
            printf '%s\n' "$r"
            return 0
        fi
    done
    printf '%s\n' "${HOME}/repos/audio.cpp"
}

audio_cpp_root="$(find_audio_root)"
media_root="${L3MS_MEDIA_ROOT:-${HOME}/models/media}"
model_path="${L3MS_BREEZE_TTS_MODEL:-${media_root}/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf}"
spec_path="${L3MS_BREEZE_TTS_MODEL_SPEC:-${audio_cpp_root}/model_specs/breeze_tts.json}"
output_dir="${L3MS_MEDIA_OUTPUT_DIR:-${HOME}/media-output}"

text=""
text_file=""
task="tts"
instruction="Speak clearly and naturally."
voice_ref=""
reference_text=""
reference_text_file=""
mode="offline"
stream_frames=16
stream_lookahead=12
max_tokens=1500
guidance_scale="1.0"
temperature="0.9"
depth_temperature="0.9"
top_k=50
top_p="1.0"
seed=0
weight_type="native"
attention="auto"
bf16_activations="auto"
output=""

while (($#)); do
    case "$1" in
        --text|--prompt)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            text="$2"
            shift 2
            ;;
        --text-file|--prompt-file)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            text_file="$2"
            shift 2
            ;;
        --task)
            (($# >= 2)) || { echo "Missing value for --task" >&2; exit 2; }
            task="$2"
            shift 2
            ;;
        --instruction)
            (($# >= 2)) || { echo "Missing value for --instruction" >&2; exit 2; }
            instruction="$2"
            shift 2
            ;;
        --voice-ref|--ref-audio)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            voice_ref="$2"
            shift 2
            ;;
        --reference-text|--ref-text)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            reference_text="$2"
            shift 2
            ;;
        --reference-text-file|--ref-text-file)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            reference_text_file="$2"
            shift 2
            ;;
        --mode)
            (($# >= 2)) || { echo "Missing value for --mode" >&2; exit 2; }
            mode="$2"
            shift 2
            ;;
        --stream-frames)
            (($# >= 2)) || { echo "Missing value for --stream-frames" >&2; exit 2; }
            stream_frames="$2"
            shift 2
            ;;
        --stream-lookahead)
            (($# >= 2)) || { echo "Missing value for --stream-lookahead" >&2; exit 2; }
            stream_lookahead="$2"
            shift 2
            ;;
        --max-tokens)
            (($# >= 2)) || { echo "Missing value for --max-tokens" >&2; exit 2; }
            max_tokens="$2"
            shift 2
            ;;
        --guidance-scale)
            (($# >= 2)) || { echo "Missing value for --guidance-scale" >&2; exit 2; }
            guidance_scale="$2"
            shift 2
            ;;
        --temperature)
            (($# >= 2)) || { echo "Missing value for --temperature" >&2; exit 2; }
            temperature="$2"
            shift 2
            ;;
        --depth-temperature)
            (($# >= 2)) || { echo "Missing value for --depth-temperature" >&2; exit 2; }
            depth_temperature="$2"
            shift 2
            ;;
        --top-k)
            (($# >= 2)) || { echo "Missing value for --top-k" >&2; exit 2; }
            top_k="$2"
            shift 2
            ;;
        --top-p)
            (($# >= 2)) || { echo "Missing value for --top-p" >&2; exit 2; }
            top_p="$2"
            shift 2
            ;;
        --seed)
            (($# >= 2)) || { echo "Missing value for --seed" >&2; exit 2; }
            seed="$2"
            shift 2
            ;;
        --weight-type)
            (($# >= 2)) || { echo "Missing value for --weight-type" >&2; exit 2; }
            weight_type="$2"
            shift 2
            ;;
        --attention)
            (($# >= 2)) || { echo "Missing value for --attention" >&2; exit 2; }
            attention="$2"
            shift 2
            ;;
        --bf16-activations)
            (($# >= 2)) || { echo "Missing value for --bf16-activations" >&2; exit 2; }
            bf16_activations="$2"
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
            echo "Unknown Breeze-TTS option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "$text_file" ]]; then
    [[ -f "$text_file" ]] || { echo "Text file not found: $text_file" >&2; exit 1; }
    [[ -z "$text" ]] || { echo "--text and --text-file are mutually exclusive" >&2; exit 2; }
    text="$(<"$text_file")"
fi

if [[ -n "$reference_text_file" ]]; then
    [[ -f "$reference_text_file" ]] || { echo "Reference text file not found: $reference_text_file" >&2; exit 1; }
    [[ -z "$reference_text" ]] || { echo "--reference-text and --reference-text-file are mutually exclusive" >&2; exit 2; }
    reference_text="$(<"$reference_text_file")"
fi

[[ -n "$text" ]] || { echo "Breeze-TTS requires --text TEXT or --text-file PATH" >&2; exit 2; }

# Normalize task
case "$task" in
    tts|design)
        cli_task="tts"
        ;;
    clon|clone)
        cli_task="clon"
        [[ -n "$voice_ref" ]] || { echo "Voice cloning requires --voice-ref PATH" >&2; exit 2; }
        [[ -f "$voice_ref" ]] || { echo "Reference audio file not found: $voice_ref" >&2; exit 1; }
        ;;
    *)
        echo "Unsupported task: $task (expected tts, design, or clon)" >&2
        exit 2
        ;;
esac

cli_path="$(find_cli || true)"
[[ -n "$cli_path" && -x "$cli_path" ]] || {
    echo "audiocpp_cli not found; run maintenance/setup-media-runtimes.sh install-breeze or maintenance/refresh-audio-cpp.sh --build" >&2
    exit 1
}

[[ -f "$model_path" ]] || {
    echo "Breeze-TTS-2 Q8_0 GGUF not found: $model_path" >&2
    echo "Run maintenance/setup-media-runtimes.sh install-breeze" >&2
    exit 1
}

[[ -f "$spec_path" ]] || {
    echo "Breeze-TTS model spec not found: $spec_path" >&2
    exit 1
}

mkdir -p "$output_dir"
if [[ -z "$output" ]]; then
    output="$output_dir/breeze-tts-$(date +%Y%m%d-%H%M%S).wav"
fi

args=(
    --task "$cli_task"
    --family breeze_tts
    --model "$model_path"
    --model-spec-override "$spec_path"
    --backend cuda
    --threads "${L3MS_BREEZE_TTS_THREADS:-8}"
    --text "$text"
    --mode "$mode"
    --out "$output"
    --request-option "instruction=$instruction"
    --request-option "max_tokens=$max_tokens"
    --request-option "guidance_scale=$guidance_scale"
    --request-option "temperature=$temperature"
    --request-option "depth_temperature=$depth_temperature"
    --request-option "top_k=$top_k"
    --request-option "top_p=$top_p"
    --request-option "seed=$seed"
    --session-option "weight_type=$weight_type"
    --session-option "breeze_tts.attention=$attention"
    --session-option "breeze_tts.bf16_activations=$bf16_activations"
)

if [[ "$mode" == "offline" ]]; then
    args+=(--metrics)
fi

if [[ "$cli_task" == "clon" ]]; then
    args+=(--voice-ref "$voice_ref")
    if [[ -n "$reference_text" ]]; then
        args+=(--reference-text "$reference_text")
    fi
fi

if [[ "$mode" == "streaming" ]]; then
    args+=(
        --request-option "stream_frames_per_event=$stream_frames"
        --request-option "stream_lookahead_margin=$stream_lookahead"
    )
fi

echo "BreezeTTS 2 / audio.cpp / task=${cli_task} mode=${mode} -> $output"
exec "$cli_path" "${args[@]}"
