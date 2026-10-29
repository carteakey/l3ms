#!/usr/bin/env bash
# Install the pinned speech.cpp runtime and its Nemotron Transcribe models.
set -euo pipefail
root="$HOME/.local/share/lytte-nemotron"
version=0.9.1
archive="speech-${version}-linux-x64-vulkan.zip"
model="NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-IQ4_NL.gguf"
model_sha=9688cee08c96ee0f54c0cc1cb07856307a79097fb0088cdb8fb31de6dd26c2be
repo=https://huggingface.co/sakasegawa/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF
mkdir -p "$root/runtime" "$root/models"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fL --retry 3 -o "$tmp/$archive" "https://github.com/nyosegawa/speech.cpp/releases/download/v${version}/${archive}"
curl -fsSL -o "$tmp/SHA256SUMS" "https://github.com/nyosegawa/speech.cpp/releases/download/v${version}/SHA256SUMS"
(cd "$tmp" && grep "  $archive$" SHA256SUMS | sha256sum -c -)
unzip -oq "$tmp/$archive" -d "$root/runtime"
if [[ ! -f "$root/models/$model" ]]; then
  if [[ -f "$root/models/download-ranges.json" ]]; then
    echo "A parallel download is in progress or needs to be resumed; do not resume its sparse file with curl." >&2
    exit 1
  fi
  curl -fL --retry 5 --continue-at - -o "$root/models/$model.part" "$repo/resolve/main/$model"
  printf '%s  %s\n' "$model_sha" "$root/models/$model.part" | sha256sum -c -
  mv "$root/models/$model.part" "$root/models/$model"
else
  printf '%s  %s\n' "$model_sha" "$root/models/$model" | sha256sum -c -
fi
"$root/runtime/speech" pull silero-vad nemotron-3-diarization
"$root/runtime/speech" devices
