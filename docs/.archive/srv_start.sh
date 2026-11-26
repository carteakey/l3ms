#!/bin/bash
LL=/home/kchauhan/runtime-builds/llama.cpp-571d0d5/build-cublas/bin/llama-server
MODEL=${MODEL:-/home/kchauhan/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf}
PORT=$1
TAG=$2
shift 2
for p in $(pgrep -f "llama-server.*$PORT"); do kill -9 $p 2>/dev/null; done
sleep 1
rm -f /home/kchauhan/repos/l3ms/srv_${TAG}.log
nohup $LL -m $MODEL --fit on --fit-target 128 --fit-ctx 131072 --ctx-size 131072 -ctk q5_0 -ctv q4_1 -fa 1 --threads 10 --temp 0.0 --parallel 1 -n 0 --port $PORT --host 127.0.0.1 "$@" > /home/kchauhan/repos/l3ms/srv_${TAG}.log 2>&1 < /dev/null &
echo "pid $!"
for i in $(seq 1 40); do
  sleep 2
  if curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok; then
    echo "healthy after $((i*2))s"
    exit 0
  fi
done
echo "NOT HEALTHY"
tail -20 /home/kchauhan/repos/l3ms/srv_${TAG}.log
exit 1