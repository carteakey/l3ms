#!/usr/bin/env bash
# Generate a local song with YuE2 through its official Python pipeline.
#
# Tuned for Yeti's RTX 4070 (12 GB VRAM): FP8 AR quantization, memory budgeting,
# and flow matching NAR generation with YuE2-Vae 48 kHz stereo decoding.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  l2m2 --media yue2-music --extra '--prompt "warm acoustic pop" --lyrics-file prompts/lyrics.txt'

Options:
  --prompt, --style TEXT    Music style and arrangement description
  --prompt-file PATH        Read style description from a UTF-8 text file
  --tags TEXT               Comma-separated style tags (alias for --prompt)
  --lyrics TEXT             Song lyrics to sing, structured with tags like [Verse], [Chorus]
  --lyrics-file PATH        Read lyrics from a local UTF-8 file
  --abc-file PATH           Supplied ABC score for exact composition plan
  --request PATH            Path to song request JSON file (e.g. prompts/yue2-city-lights.json)
  --cot MODE                Chain-of-thought planning: full (default), melody, off
  --quantization MODE       Quantization: fp8 (default on 12GB), none
  --budget GIB              Memory budget in GiB (default: 12)
  --offload-ar              Offload AR model to CPU before NAR synthesis runs
  --stage STAGE             Stop after planning (plan) or synthesize full song (audio, default)
  --seed N                  Deterministic RNG seed (default: 42)
  --cfg-scale VALUE         Classifier-free guidance scale
  --model PATH_OR_REPO      Model path or Hugging Face repo (default: local YuE2-3B or m-a-p/YuE2-3B)
  --vae PATH_OR_REPO        VAE path or Hugging Face repo (default: local YuE2-Vae or m-a-p/YuE2-Vae)
  --output PATH             Output directory for artifacts (default: L3MS_MEDIA_OUTPUT_DIR/yue2-<date>)
  --doctor, --check         Run environment readiness diagnostics
  --quiet                   Suppress progress output
  --help                    Show this help

Environment:
  L3MS_YUE_ROOT             YuE repository checkout (default: $HOME/repos/YuE or vendor/YuE)
  L3MS_YUE_MODEL_DIR        YuE2 models root (default: $HOME/models/media/YuE2)
  L3MS_MEDIA_ROOT           Shared models root (default: $HOME/models/media)
  L3MS_MEDIA_OUTPUT_DIR     Output directory (default: $HOME/media-output)
  L2M2_PYTHON               Python interpreter override
  L3MS_YUE_PREPARED_ONLY    Use the existing YuE .venv offline; never prepare or download
EOF
}

for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
        usage
        exit 0
    fi
done

# Prepare environment paths
if [[ -d "${HOME}/.local/bin" ]]; then
    PATH="${HOME}/.local/bin:${PATH}"
    export PATH
fi
if [[ -x /opt/cuda/bin/nvcc ]]; then
    PATH="/opt/cuda/bin:${PATH}"
    export PATH
    LD_LIBRARY_PATH="/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH
fi

if [[ "${L3MS_YUE_PREPARED_ONLY:-0}" == 1 ]]; then
    if [[ -n "${L3MS_YUE_ROOT:-}" ]]; then
        yue_root="$L3MS_YUE_ROOT"
    elif [[ -d "${HOME}/repos/YuE" ]]; then
        yue_root="${HOME}/repos/YuE"
    else
        yue_root="${REPO_DIR}/vendor/YuE"
    fi
    [[ -d "$yue_root" ]] || {
        echo "Prepared YuE repository not found; install it separately" >&2
        exit 1
    }
    yue_root="$(cd -- "$yue_root" && pwd -P)"
    python_bin="${yue_root}/.venv/bin/python"
    [[ -x "$python_bin" ]] || {
        echo "Prepared YuE Python environment not found; install it separately" >&2
        exit 1
    }
    export L3MS_YUE_ROOT="$yue_root"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
else
    python_bin="${L2M2_PYTHON:-}"
    if [[ -z "$python_bin" ]]; then
        yue_root="${L3MS_YUE_ROOT:-${REPO_DIR}/vendor/YuE}"
        if [[ -x "${yue_root}/.venv/bin/python" ]]; then
            python_bin="${yue_root}/.venv/bin/python"
        elif [[ -x "${HOME}/repos/YuE/.venv/bin/python" ]]; then
            python_bin="${HOME}/repos/YuE/.venv/bin/python"
        else
            python_bin="$(command -v python3 || echo "python3")"
        fi
    fi
fi

exec "$python_bin" "${SCRIPT_DIR}/yue2_runner.py" "$@"
