#!/usr/bin/env bash
# Compare Plain llama.cpp builds on Qwen3.8-Flash-Next
# Fresh 6-prompt unique corpus, residency-controlled protocol
set -euo pipefail

REPO="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="/home/kchauhan/models/qwen38-flash-next/AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf"
PROMPTS="$REPO/bench-models/prompts/qwen38-flash-next-unique.jsonl"
PORT=8022
LOGDIR="$REPO/bench-models/logs/results"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTDIR="$LOGDIR/plain-master-vs-gold-$STAMP"
mkdir -p "$OUTDIR"

if systemctl --user is-active --quiet llama-swap.service; then
  echo "Stopping llama-swap.service..."
  systemctl --user stop llama-swap.service
fi

declare -A BINS=(
  [gold]="$REPO/vendor/llama.cpp-master/build/bin/llama-server-gold-9d817213a"
  [master]="$REPO/vendor/llama.cpp-master/build/bin/llama-server-master-b78a39a2f"
)

CTX="${CTX:-16384}"
NCMOE="${NCMOE:-46}"

COMMON_FLAGS=(
  -m "$MODEL"
  -ngl 99 -ncmoe "$NCMOE"
  -c "$CTX" --parallel 1
  -b 2048 -ub 512
  -fa on --jinja
  -ctk q8_0 -ctv q8_0
  -t 10 --threads-batch 12 --prio 2
  --lazy-mode on
  --no-warmup
  --host 127.0.0.1 --port "$PORT"
)

run_probe_warmup() {
  local port="$1"
  python3 -c "
import urllib.request, json, time
url = 'http://127.0.0.1:$port/v1/chat/completions'
body = json.dumps({'messages': [{'role':'user', 'content':'Count from 1 to 40, one number per line.'}], 'max_tokens':64, 'temperature':0, 'seed':123}).encode()
req = urllib.request.Request(url, body, {'Content-Type':'application/json'})
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        d = json.loads(resp.read().decode())
        t = d.get('timings', {})
        n = t.get('predicted_n', 0)
        ms = max(t.get('predicted_ms', 1), 1)
        print(f'   warmup: {n/ms*1000:.2f} t/s (n={n})')
except Exception as e:
    print(f'   warmup failed: {e}')
"
}

run_fresh_corpus() {
  local port="$1"
  local arm="$2"
  local outfile="$OUTDIR/$arm.jsonl"
  python3 -c "
import urllib.request, json, time, sys

prompts_path = '$PROMPTS'
url = 'http://127.0.0.1:$port/v1/chat/completions'
out_path = '$outfile'

tasks = []
with open(prompts_path) as f:
    for line in f:
        line = line.strip()
        if line:
            tasks.append(json.loads(line))

results = []
total_gen = 0
total_dec_s = 0.0

out_f = open(out_path, 'w')

for t in tasks:
    tid = t.get('name', t.get('id', 'unknown'))
    max_tokens = t.get('max_tokens', 192)
    prompt = t['prompt']
    
    body = json.dumps({
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': 0,
        'seed': 123
    }).encode()
    
    req = urllib.request.Request(url, body, {'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode())
            dur = time.perf_counter() - t0
            timings = data.get('timings', {})
            gen_n = timings.get('predicted_n', 0)
            pred_ms = timings.get('predicted_ms', dur * 1000)
            dec_s = pred_ms / 1000.0
            tps = gen_n / max(dec_s, 0.001)
            prompt_n = timings.get('prompt_n', 0)
            prompt_ms = timings.get('prompt_ms', 1)
            prompt_tps = prompt_n / max(prompt_ms / 1000.0, 0.001)
            
            row = {
                'row_type': 'attempt',
                'task': tid,
                'run': 1,
                'ok': True,
                'stop': 'limit',
                'generated': gen_n,
                'decode_s': round(dec_s, 6),
                'decode_tps': round(tps, 2),
                'prompt_n': prompt_n,
                'prompt_tps': round(prompt_tps, 1)
            }
            total_gen += gen_n
            total_dec_s += dec_s
            print(f'   task {tid:16s}: {tps:6.2f} t/s (pp: {prompt_tps:5.1f} t/s, n={gen_n})')
            out_f.write(json.dumps(row) + '\n')
            out_f.flush()
    except Exception as e:
        print(f'   task {tid:16s}: ERROR {e}')
        row = {'row_type': 'attempt', 'task': tid, 'run': 1, 'ok': False, 'error': str(e)}
        out_f.write(json.dumps(row) + '\n')
        out_f.flush()

agg_tps = total_gen / max(total_dec_s, 0.001)
summary = {
    'row_type': 'summary',
    'attempts': len(tasks),
    'successes': len([r for r in tasks]),
    'failures': 0,
    'generated': total_gen,
    'decode_s': round(total_dec_s, 6),
    'decode_tps': round(agg_tps, 2)
}
out_f.write(json.dumps(summary) + '\n')
out_f.close()
print(f'   --> AGGREGATE DECODE: {agg_tps:6.2f} t/s across {total_gen} tokens ({total_dec_s:.2f}s)')
"
}

run_arm() {
  local arm="$1"
  local bin="${BINS[$arm]}"
  local log="$OUTDIR/$arm.server.log"
  echo "================================================================="
  echo "== Arm: $arm ($bin)"
  echo "================================================================="
  [ -x "$bin" ] || { echo "Binary missing: $bin"; return 1; }

  GGML_CUDA_NO_PINNED=1 GGML_CUDA_GRAPH_OPT=1 nohup taskset -c 0-11 "$bin" \
    "${COMMON_FLAGS[@]}" > "$log" 2>&1 &
  local srv_pid=$!
  trap 'kill $srv_pid 2>/dev/null || true' EXIT

  echo "Waiting for server to become healthy..."
  local ok=""
  for _ in $(seq 1 180); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
      ok=1
      break
    fi
    if ! kill -0 $srv_pid 2>/dev/null; then
      echo "Server process died."
      break
    fi
    sleep 2
  done

  if [ -z "$ok" ]; then
    echo "Server failed to start. Tail of log:"
    tail -n 30 "$log"
    kill $srv_pid 2>/dev/null || true
    trap - EXIT
    return 1
  fi

  echo "Server is UP. VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) | RAM avail: $(free -g | awk '/^Mem:/{print $7}') GiB"
  echo "Running 2 warmup generations..."
  run_probe_warmup "$PORT"
  run_probe_warmup "$PORT"

  echo "Running fresh 6-prompt corpus..."
  run_fresh_corpus "$PORT" "$arm"

  echo "Arm $arm completed. Cleaning up server..."
  kill $srv_pid 2>/dev/null || true
  wait $srv_pid 2>/dev/null || true
  trap - EXIT
  sleep 4
}

for arm in ${ARMS:-gold master}; do
  run_arm "$arm"
done

echo
echo "================================================================="
echo "Results saved in $OUTDIR"
echo "================================================================="
