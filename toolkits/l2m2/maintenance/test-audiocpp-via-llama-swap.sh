#!/usr/bin/env bash
# Run the four Breeze-TTS-2 modes (standard, design, clone, streaming) through an
# isolated llama-swap router that owns audio.cpp as `audiocpp_auto`.
#   test-audiocpp-via-llama-swap.sh [cpu|cuda]    (default cpu)
# Uses its own router on 127.0.0.1:${PORT:-18932}; never touches the live service.
# `cuda` needs an idle GPU: stop the live router first, restore it afterwards.
set -euo pipefail

backend="${1:-cpu}"
[[ "$backend" == cpu || "$backend" == cuda ]] || { echo "backend must be cpu or cuda" >&2; exit 2; }
root="${L3MS_ROOT:-$HOME/repos/l3ms}"
server="${AUDIOCPP_SERVER:-$root/vendor/audio.cpp-955c872/build/bin/audiocpp_server}"
swap="${LLAMA_SWAP_BIN:-$HOME/bin/llama-swap}"
model="${BREEZE_GGUF:-$HOME/models/media/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf}"
port="${PORT:-18932}"
for f in "$server" "$swap" "$model"; do [[ -e "$f" ]] || { echo "missing: $f" >&2; exit 1; }; done

work="$(mktemp -d)"
pid=""
cleanup() { [[ -n "$pid" ]] && kill "$pid" 2>/dev/null; wait 2>/dev/null || true; rm -rf "$work"; }
trap cleanup EXIT INT TERM

python3 - "$work" "$backend" "$model" <<'PY'
import json, sys
work, backend, model = sys.argv[1:]
entries = [("breeze-tts", "tts", "offline"), ("breeze-clone", "clon", "offline"), ("breeze-stream", "tts", "streaming")]
json.dump({"host": "127.0.0.1", "port": 8080, "backend": backend, "threads": 8, "lazy_load": True,
           "max_loaded_models": 1,
           "models": [{"id": i, "family": "breeze_tts", "path": model, "task": t, "mode": m} for i, t, m in entries]},
          open(f"{work}/server.json", "w"))
PY
cat > "$work/swap.yaml" <<YAML
healthCheckTimeout: 240
models:
  audiocpp_auto:
    cmd: >-
      $server --config $work/server.json --host 127.0.0.1 --port \${PORT} --no-ui
    proxy: http://127.0.0.1:\${PORT}
    checkEndpoint: /health
YAML
"$swap" -config "$work/swap.yaml" -listen "127.0.0.1:$port" >"$work/swap.log" 2>&1 &
pid=$!
for _ in $(seq 1 30); do curl -sf -m 2 "localhost:$port/health" >/dev/null && break; sleep 1; done

url="localhost:$port/audiocpp/v1/audio/speech"
post() { curl -s -m 900 -o "$1" -w '%{http_code}' "$url" -H 'Content-Type: application/json' -d "$2"; }
wavinfo() { python3 -c "import wave,sys; w=wave.open(sys.argv[1]); assert w.getnframes()>0; print(f'{w.getframerate()} Hz {w.getnchannels()} ch {w.getnframes()/w.getframerate():.2f} s')" "$1"; }
fail=0
check() { # name http file
  if [[ "$2" == 200 ]] && info="$(wavinfo "$3" 2>&1)"; then echo "PASS $1: $info"; else echo "FAIL $1: http=$2"; fail=1; fi
}

check standard "$(post "$work/std.wav" '{"model":"breeze-tts","input":"Welcome to local text to speech.","response_format":"wav"}')" "$work/std.wav"
check design   "$(post "$work/design.wav" '{"model":"breeze-tts","input":"A warm narrator is ready.","response_format":"wav","options":{"instruction":"Warm female narrator, calm pacing."}}')" "$work/design.wav"
check clone    "$(post "$work/clone.wav" "{\"model\":\"breeze-clone\",\"input\":\"Read this with the reference speaker tone.\",\"response_format\":\"wav\",\"voice_ref\":\"$work/std.wav\",\"reference_text\":\"Welcome to local text to speech.\"}")" "$work/clone.wav"
code="$(post "$work/stream.sse" '{"model":"breeze-stream","input":"Streaming speech generation test.","stream":true,"stream_format":"sse","response_format":"pcm","options":{"stream_frames_per_event":"16","stream_lookahead_margin":"12"}}')"
events="$(grep -c '^data:' "$work/stream.sse" || true)"
if [[ "$code" == 200 && "$events" -gt 1 ]]; then echo "PASS streaming: $events SSE events"; else echo "FAIL streaming: http=$code events=$events"; fail=1; fi

# llama-swap owns the engine: unloading it must stop the audio.cpp process.
curl -s -m 30 "localhost:$port/unload" >/dev/null || true
sleep 3
if pgrep -f "[a]udiocpp_server --config $work/server.json" >/dev/null; then echo "FAIL unload: audiocpp_server still running"; fail=1; else echo "PASS unload: audiocpp_server stopped"; fi
exit "$fail"
