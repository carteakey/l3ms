#!/usr/bin/env bash
# generate-ace-step-1.5.sh — Local music generation via acestep.cpp (ACE-Step 1.5 Turbo Q8_0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACE_SERVER="${L3MS_ACESTEP_SERVER:-${REPO_DIR}/vendor/acestep.cpp/build/ace-server}"
MODELS_DIR="${L3MS_ACESTEP_MODELS:-/home/kchauhan/models/media/ACE-Step-1.5}"

if [[ "${1:-}" == "--check" ]]; then
  [[ -x "$ACE_SERVER" ]] || { echo "Missing ace-server: $ACE_SERVER" >&2; exit 1; }
  for f in Qwen3-Embedding-0.6B-Q8_0.gguf acestep-5Hz-lm-1.7B-Q8_0.gguf acestep-v15-turbo-Q8_0.gguf vae-BF16.gguf; do
    [[ -f "${MODELS_DIR}/${f}" ]] || { echo "Missing model: ${MODELS_DIR}/${f}" >&2; exit 1; }
  done
  echo "OK: $ACE_SERVER + $MODELS_DIR (4 GGUFs present)"
  exit 0
fi

echo "ACE-Step 1.5 Turbo is served via llama-swap (model id: ace-step-1.5; aliases: acestep, music)."
echo "  Wake via router: ./maintenance/l3ms-ctl.sh wake ace-step-1.5"
echo "  Web UI on :8089: http://yeti-cachy.llama-alnair.ts.net:8089/"
echo "  Manual server:   $ACE_SERVER --models $MODELS_DIR --host 0.0.0.0 --port 8089"
