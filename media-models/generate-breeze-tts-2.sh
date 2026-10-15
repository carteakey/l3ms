#!/usr/bin/env bash
# Generate local speech or clone voices with BreezeTTS 2 through audio.cpp.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  # Voice design / standard TTS:
  ./media-models/generate-breeze-tts-2.sh --text "Welcome to local TTS." --instruction "A warm female narrator with calm pacing."

  # Voice cloning with reference audio:
  ./media-models/generate-breeze-tts-2.sh --task clon --text "Read this in my voice." --voice-ref sample.wav --reference-text "Reference transcript."

Options:
  --text TEXT               Input text to synthesize (required unless --text-file)
  --text-file PATH          Read input text from a UTF-8 text file
  --task TASK               Generation task: tts (default), clon, or design
  --backend BACKEND         Explicit cpu or cuda backend (default: cuda; no fallback)
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
  --check                   Read-only check of CLI loader, model/spec, and requested clone input
  --output PATH             Output WAV path (default: $L3MS_MEDIA_OUTPUT_DIR/breeze-tts-*.wav)
  --out-dir PATH            Output directory override
  --help, -h                Show this help

--check does not require synthesis text, create output paths, allocate a GPU,
or run TTS. It checks the local executable, compiled Breeze loader, model, and
model spec; it does not validate GPU memory or driver/runtime compatibility.

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
backend="cuda"
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
check_only=0

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
        --backend)
            (($# >= 2)) || { echo "Missing value for --backend" >&2; exit 2; }
            backend="$2"
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
        --check)
            check_only=1
            shift
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

case "$backend" in
    cpu|cuda) ;;
    *) echo "Unsupported backend: $backend (expected cpu or cuda)" >&2; exit 2 ;;
esac

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

if [[ "$check_only" != "1" ]]; then
    [[ -n "$text" ]] || { echo "Breeze-TTS requires --text TEXT or --text-file PATH" >&2; exit 2; }
fi

# Normalize task
case "$task" in
    tts|design)
        cli_task="tts"
        ;;
    clon|clone)
        cli_task="clon"
        [[ -n "$voice_ref" ]] || { echo "Voice cloning requires --voice-ref PATH" >&2; exit 2; }
        [[ -s "$voice_ref" ]] || { echo "Reference audio file missing or empty: $voice_ref" >&2; exit 1; }
        ;;
    *)
        echo "Unsupported task: $task (expected tts, design, or clon)" >&2
        exit 2
        ;;
esac

case "$mode" in
    offline|streaming) ;;
    *) echo "Unsupported mode: $mode (expected offline or streaming)" >&2; exit 2 ;;
esac

check_readiness() {
    local failed=0
    local loader_output=""
    local cli_path=""

    if cli_path="$(find_cli)"; then
        printf 'OK: audio.cpp CLI: %s\n' "$cli_path"
        if loader_output="$("$cli_path" --list-loaders --json 2>&1)"; then
            if printf '%s' "$loader_output" | python3 -c 'import json, sys; sys.exit(0 if "breeze_tts" in json.load(sys.stdin).get("loaders", {}) else 1)' 2>/dev/null; then
                printf 'OK: Breeze-TTS loader is compiled into the CLI\n'
            else
                printf 'MISSING: Breeze-TTS loader is not listed by %s\n' "$cli_path" >&2
                failed=1
            fi
        else
            printf 'MISSING: could not inspect loaders with %s --list-loaders --json\n' "$cli_path" >&2
            failed=1
        fi
    else
        printf 'MISSING: audiocpp_cli; build audio.cpp or set L3MS_AUDIO_CPP_CLI\n' >&2
        failed=1
    fi

    if [[ -s "$model_path" ]]; then
        printf 'OK: Breeze-TTS model: %s\n' "$model_path"
    else
        printf 'MISSING: Breeze-TTS model file: %s\n' "$model_path" >&2
        failed=1
    fi
    if [[ -s "$spec_path" ]]; then
        printf 'OK: Breeze-TTS model spec: %s\n' "$spec_path"
    else
        printf 'MISSING: Breeze-TTS model spec: %s\n' "$spec_path" >&2
        failed=1
    fi
    if [[ "$cli_task" == "clon" ]]; then
        if [[ -s "$voice_ref" ]]; then
            printf 'OK: cloning reference audio: %s\n' "$voice_ref"
        else
            printf 'MISSING: cloning reference WAV: %s\n' "$voice_ref" >&2
            failed=1
        fi
    fi
    printf 'Read-only preflight: no synthesis was run and no output directory was created.\n'
    return "$failed"
}

if [[ "$check_only" == "1" ]]; then
    check_readiness
    exit $?
fi

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
    --backend "$backend"
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
    stream_output_dir="${output}.chunks"
    mkdir -p "$stream_output_dir"
    args+=(
        --out-dir "$stream_output_dir"
        --request-option "stream_frames_per_event=$stream_frames"
        --request-option "stream_lookahead_margin=$stream_lookahead"
    )
fi

echo "BreezeTTS 2 / audio.cpp / task=${cli_task} mode=${mode} -> $output"
exec "$cli_path" "${args[@]}"
