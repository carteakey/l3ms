#!/usr/bin/env bash
# generate-kokoro-82m.sh — Zero-VRAM CPU E-core TTS via audio.cpp v0.9.6 (Kokoro 82M Q8_0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLI="${L3MS_AUDIOCPP_CLI:-${REPO_DIR}/vendor/audio.cpp-v0.9.6/build/bin/audiocpp_cli}"
MODEL="${L3MS_KOKORO_MODEL:-/home/kchauhan/models/media/Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf}"

TEXT=""
VOICE="af_heart"
OUT="kokoro-output.wav"
SPEED="1.0"
THREADS="4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --text|-t)   TEXT="$2"; shift 2 ;;
    --voice|-v)  VOICE="$2"; shift 2 ;;
    --out|-o)    OUT="$2"; shift 2 ;;
    --speed|-s)  SPEED="$2"; shift 2 ;;
    --threads)   THREADS="$2"; shift 2 ;;
    --check)
      [[ -x "$CLI" ]] || { echo "Missing audiocpp_cli: $CLI" >&2; exit 1; }
      [[ -f "$MODEL" ]] || { echo "Missing Kokoro model: $MODEL" >&2; exit 1; }
      "$CLI" --list-loaders | grep -q kokoro_tts || { echo "kokoro_tts loader not built in $CLI" >&2; exit 1; }
      echo "OK: $CLI + $MODEL (CPU E-cores 12-15)"
      exit 0
      ;;
    --help|-h)
      echo "Usage: $0 --text \"...\" [--voice af_heart] [--speed 1.0] [--out out.wav] [--check]"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$TEXT" ]] || { echo "ERROR: --text is required" >&2; exit 1; }
exec env CUDA_VISIBLE_DEVICES= taskset -c 12-15 "$CLI" \
  --task tts --family kokoro_tts \
  --model "$MODEL" \
  --backend cpu --threads "$THREADS" \
  --voice-id "$VOICE" \
  --request-option "speed=${SPEED}" \
  --text "$TEXT" \
  --out "$OUT" \
  --metrics
