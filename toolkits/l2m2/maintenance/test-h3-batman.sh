#!/usr/bin/env bash
# One bounded, isolated text-to-video experiment; never evicts another workload.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
for tool in nvidia-smi ffmpeg ffprobe jq xxd timeout; do
    command -v "$tool" >/dev/null || { echo "Missing required command: $tool" >&2; exit 2; }
done
if systemctl --user is-active --quiet llama-swap.service; then
    echo "llama-swap is active; stop it before this full-GPU test." >&2; exit 2
fi
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)"
if (( used > 1500 )); then
    echo "GPU busy (${used} MiB used); leave the existing job running and retry later." >&2; exit 2
fi
mkdir -p "${HOME}/media-output"
out="$(mktemp -d "${HOME}/media-output/h3-batman-XXXXXXXX")"
cp "$root/media-models/generate-minimax-h3.sh" "$out/generate-minimax-h3.sh"
cp "${BASH_SOURCE[0]}" "$out/test-h3-batman.sh"
cp "$root/maintenance/run-h3-batman-isolated.sh" "$out/run-h3-batman-isolated.sh"
git -C "$root" rev-parse HEAD > "$out/repo-commit.txt"
git -C "$root" diff > "$out/working-tree.patch"
nvidia-smi > "$out/gpu-before.txt"
free -h > "$out/memory-before.txt"
cp "$root/prompts/h3-batman-plant.txt" "$out/prompt.txt"
steps="${H3_TEST_STEPS:-20}"
args=(bash "$root/media-models/generate-minimax-h3.sh" --video --prompt-file "$out/prompt.txt"
      --width 640 --height 384 --frames 124 --steps "$steps" --seed 42
      --out-dir "$out" --output "$out/audio.wav" --video-output "$out/video.mp4")
printf '%q ' "${args[@]}" > "$out/command.txt"
printf '\n' >> "$out/command.txt"
start=$SECONDS
echo "Output: $out"
set +e
timeout --signal=TERM --kill-after=30s 30m "${args[@]}" > "$out/run.log" 2>&1
status=$?
set -e
printf 'exit_code=%s\nwall_seconds=%s\n' "$status" "$((SECONDS-start))" > "$out/result.txt"
if (( status != 0 )); then tail -n 30 "$out/run.log"; exit "$status"; fi
ffprobe -v error -show_format -show_streams -of json "$out/video.mp4" > "$out/probe.json"
echo "Playable output: $out/video.mp4"
