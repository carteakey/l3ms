#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO/vendor/llama.cpp-master/build/bin/llama-server"
MODEL="/home/kchauhan/models/qwen38-flash-next/AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf"
PORT=8025
LOGDIR="$REPO/bench-models/logs/results"
mkdir -p "$LOGDIR"

run_probe() {
  local label="$1"
  shift
  local log="$LOGDIR/fit-test-$label.log"
  echo "=================================================================="
  echo "Testing: $label ($*)"
  echo "=================================================================="
  
  GGML_CUDA_NO_PINNED=1 GGML_CUDA_GRAPH_OPT=1 nohup taskset -c 0-11 "$BIN" \
    -m "$MODEL" "$@" \
    --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 &
  local srv_pid=$!
  trap 'kill $srv_pid 2>/dev/null || true' EXIT

  local ok=""
  for _ in $(seq 1 120); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
      ok=1
      break
    fi
    if ! kill -0 $srv_pid 2>/dev/null; then
      break
    fi
    sleep 2
  done

  if [ -z "$ok" ]; then
    echo "   LOAD FAILED. Log tail:"
    grep -aE 'CUDA error|out of memory|failed to allocate|cannot allocate|error' "$log" | tail -n 5 || tail -n 10 "$log"
    kill $srv_pid 2>/dev/null || true
    trap - EXIT
    return 1
  fi

  local vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)
  echo "   LOADED! VRAM: $vram"
  
  # Inspect what layers were offloaded from log
  grep -aoE 'offloaded [0-9]+/[0-9]+ layers to CUDA|CUDA0 model buffer size = [^,]+|llm_load_tensors: [a-z0-9_ ]+ = [0-9.]+' "$log" | head -n 5 || true

  # Run quick benchmark: 2 warmups, then 2 measured probes (1 code, 1 story)
  python3 -c "
import urllib.request, json, time
url = 'http://127.0.0.1:$PORT/v1/chat/completions'

def send(prompt, max_tokens):
    body = json.dumps({'messages': [{'role':'user', 'content': prompt}], 'max_tokens': max_tokens, 'temperature': 0, 'seed': 123}).encode()
    req = urllib.request.Request(url, body, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as resp:
        d = json.loads(resp.read().decode())
        t = d.get('timings', {})
        n = t.get('predicted_n', 0)
        ms = max(t.get('predicted_ms', 1), 1)
        return n / ms * 1000.0

# warmup
send('Count from 1 to 20.', 32)
send('Count from 1 to 20.', 32)

t_code = send('Refactor a Python function that reads JSON with open() into a pathlib implementation. Add useful exception handling and return typed data. Answer with code only.', 192)
t_story = send('Write a restrained 180-word scene about a radio operator receiving a transmission from an abandoned polar station.', 192)
print(f'   PROBE RESULTS: code = {t_code:.2f} t/s, story = {t_story:.2f} t/s, mean = {(t_code+t_story)/2:.2f} t/s')
"
  kill $srv_pid 2>/dev/null || true
  wait $srv_pid 2>/dev/null || true
  trap - EXIT
  sleep 3
}

# Test 16k context layer ladder
run_probe "16k-ncmoe46" -ngl 99 -ncmoe 46 -c 16384 -b 2048 -ub 512 -fa on --jinja -ctk q8_0 -ctv q8_0 -t 10 --threads-batch 12 --lazy-mode on --no-warmup || true
run_probe "16k-ncmoe45" -ngl 99 -ncmoe 45 -c 16384 -b 2048 -ub 512 -fa on --jinja -ctk q8_0 -ctv q8_0 -t 10 --threads-batch 12 --lazy-mode on --no-warmup || true
run_probe "16k-ncmoe44" -ngl 99 -ncmoe 44 -c 16384 -b 2048 -ub 512 -fa on --jinja -ctk q8_0 -ctv q8_0 -t 10 --threads-batch 12 --lazy-mode on --no-warmup || true
run_probe "16k-fit512" --fit on --fit-target 512 -c 16384 -b 2048 -ub 512 -fa on --jinja -ctk q8_0 -ctv q8_0 -t 10 --threads-batch 12 --lazy-mode on --no-warmup || true

# Test 64k context (production gold config)
run_probe "64k-fit512" --fit on --fit-target 512 -c 65536 -b 4096 -ub 1024 -fa on --jinja -ctk q8_0 -ctv q8_0 -t 10 --threads-batch 12 --lazy-mode on --no-warmup || true
run_probe "64k-ncmoe46" -ngl 99 -ncmoe 46 -c 65536 -b 4096 -ub 1024 -fa on --jinja -ctk q8_0 -ctv q8_0 -t 10 --threads-batch 12 --lazy-mode on --no-warmup || true

