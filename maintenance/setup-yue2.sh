#!/usr/bin/env bash
# Setup and manage YuE2 foundation music model runtime and weights.
#
# Declares environment readiness checks, virtual environment setup via uv,
# and pinned model weight downloads for YuE2-3B and YuE2-Vae.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

YUE_ROOT="${L3MS_YUE_ROOT:-${HOME}/repos/YuE}"
YUE_REF="${L3MS_YUE_REF:-main}"
YUE_URL="https://github.com/multimodal-art-projection/YuE.git"
MEDIA_ROOT="${L3MS_MEDIA_ROOT:-${HOME}/models/media}"
YUE_MODEL_DIR="${L3MS_YUE_MODEL_DIR:-${MEDIA_ROOT}/YuE2}"
MANIFEST="${REPO_DIR}/models/yue2-files.txt"

log() { printf '%s\n' "-> $*"; }
ok() { printf '%s\n' "[OK] $*"; }
warn() { printf '%s\n' "[WARN] $*" >&2; }
die() { printf '%s\n' "[ERROR] $*" >&2; exit 1; }

prepare_user_path() {
    if [[ -d "${HOME}/.local/bin" ]]; then
        PATH="${HOME}/.local/bin:${PATH}"
        export PATH
    fi
}

prepare_cuda_path() {
    if [[ -x /opt/cuda/bin/nvcc ]]; then
        PATH="/opt/cuda/bin:${PATH}"
        export PATH
        LD_LIBRARY_PATH="/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH
    fi
}

usage() {
    cat <<'EOF'
Usage: maintenance/setup-yue2.sh <command>

Commands:
  check             Check YuE2 checkout, Python venv, and model weights readiness
  doctor            Run yue2 doctor diagnostics via yue2.cli
  install-env       Clone/update YuE repo, set up Python 3.12 venv, and install yue2-infer
  download-models   Download pinned YuE2-3B and YuE2-Vae weights to $L3MS_MEDIA_ROOT/YuE2
  install           Set up environment (and download weights if YUE_DOWNLOAD=1)

Environment:
  L3MS_YUE_ROOT       YuE repository checkout (default: ~/repos/YuE)
  L3MS_YUE_REF        Git reference or branch (default: main)
  L3MS_MEDIA_ROOT     Media models directory (default: ~/models/media)
  L3MS_YUE_MODEL_DIR  YuE2 model root (default: $L3MS_MEDIA_ROOT/YuE2)
  YUE_DOWNLOAD=1      Download weights during 'install'
  HF_TOKEN            Hugging Face token for downloading weights
EOF
}

ensure_checkout() {
    command -v git >/dev/null 2>&1 || die "git is required"
    if [[ -d "${YUE_ROOT}/.git" ]]; then
        log "Using existing checkout at ${YUE_ROOT} ($(git -C "${YUE_ROOT}" rev-parse --short HEAD))"
    else
        log "Cloning ${YUE_URL} (${YUE_REF}) into ${YUE_ROOT}"
        mkdir -p "$(dirname "${YUE_ROOT}")"
        git clone --branch "${YUE_REF}" "${YUE_URL}" "${YUE_ROOT}"
    fi

    # Ensure vendor symlink exists
    mkdir -p "${REPO_DIR}/vendor"
    if [[ ! -e "${REPO_DIR}/vendor/YuE" ]]; then
        ln -s "${YUE_ROOT}" "${REPO_DIR}/vendor/YuE"
        ok "Created symlink: vendor/YuE -> ${YUE_ROOT}"
    fi
}

setup_env() {
    command -v uv >/dev/null 2>&1 || die "uv is required (install via ~/.local/bin/uv)"
    ensure_checkout

    local venv="${YUE_ROOT}/.venv"
    if [[ ! -x "${venv}/bin/python" ]]; then
        log "Creating Python 3.12 virtual environment at ${venv}"
        uv venv "${venv}" --python 3.12
    fi

    log "Installing yue2-infer and dependencies into ${venv}"
    VIRTUAL_ENV="${venv}" uv pip install -e "${YUE_ROOT}"
    ok "YuE2 Python environment ready at ${venv}"
}

download_models() {
    [[ -f "${MANIFEST}" ]] || die "Manifest not found: ${MANIFEST}"
    command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"

    mkdir -p "${YUE_MODEL_DIR}"
    local venv="${YUE_ROOT}/.venv"
    local hf_cli=""
    if [[ -x "${venv}/bin/hf" ]]; then
        hf_cli="${venv}/bin/hf"
    elif command -v hf >/dev/null 2>&1; then
        hf_cli="$(command -v hf)"
    else
        # Try uv run with huggingface_hub
        hf_cli="uv run --with huggingface_hub hf"
    fi

    log "Verifying and downloading pinned YuE2 models into ${YUE_MODEL_DIR}"
    while read -r repo revision repo_file relative expected_size expected_sha; do
        [[ -n "${repo}" && "${repo}" != \#* ]] || continue
        local target="${YUE_MODEL_DIR}/${relative}"
        if [[ -f "${target}" ]]; then
            local actual_size
            actual_size="$(stat -c %s "${target}")"
            if [[ "${actual_size}" == "${expected_size}" ]]; then
                local actual_sha
                actual_sha="$(sha256sum "${target}" | awk '{print $1}')"
                if [[ "${actual_sha}" == "${expected_sha}" ]]; then
                    ok "Verified: ${relative}"
                    continue
                fi
            fi
            warn "Existing file corrupted or mismatched; preserving backup: ${target}"
            mv "${target}" "${target}.bak.$(date +%Y%m%d%H%M%S)"
        fi

        log "Downloading ${repo}@${revision} file ${repo_file} -> ${target}"
        mkdir -p "$(dirname "${target}")"
        if ! ${hf_cli} download "${repo}" "${repo_file}" --revision "${revision}" --local-dir "$(dirname "${target}")" >/dev/null; then
            die "Failed to download ${repo_file} from ${repo}"
        fi

        # If downloaded into nested dir, align with target path
        if [[ ! -f "${target}" && -f "$(dirname "${target}")/${repo_file}" ]]; then
            mv "$(dirname "${target}")/${repo_file}" "${target}"
        fi

        local downloaded_size
        downloaded_size="$(stat -c %s "${target}")"
        local downloaded_sha
        downloaded_sha="$(sha256sum "${target}" | awk '{print $1}')"
        if [[ "${downloaded_size}" != "${expected_size}" || "${downloaded_sha}" != "${expected_sha}" ]]; then
            die "Checksum verification failed for ${target}"
        fi
        ok "Downloaded and verified: ${relative}"
    done < "${MANIFEST}"

    ok "All YuE2 model files successfully downloaded and verified."
}

check_runtime() {
    local missing=0

    # 1. Check GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_info=""
        if gpu_info="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null)"; then
            printf '%s\n' "$gpu_info" | sed 's/^/GPU: /'
        else
            warn "nvidia-smi is installed but cannot communicate with the NVIDIA driver"
            missing=1
        fi
    else
        warn "nvidia-smi is unavailable"
        missing=1
    fi

    # 2. Check checkout and venv
    if [[ -d "${YUE_ROOT}/.git" ]]; then
        ok "YuE checkout: ${YUE_ROOT} ($(git -C "${YUE_ROOT}" rev-parse --short HEAD))"
    else
        warn "YuE checkout missing: ${YUE_ROOT}"
        missing=1
    fi

    local python_bin="${YUE_ROOT}/.venv/bin/python"
    if [[ -x "${python_bin}" ]]; then
        ok "YuE Python venv: ${python_bin}"
    else
        warn "YuE Python venv missing at ${python_bin}"
        missing=1
    fi

    # 3. Check models
    local model_path="${YUE_MODEL_DIR}/YuE2-3B/model.safetensors"
    local vae_path="${YUE_MODEL_DIR}/YuE2-Vae/model.safetensors"
    if [[ -f "${model_path}" ]]; then
        ok "YuE2-3B model: ${model_path}"
    else
        warn "YuE2-3B weights missing at ${model_path} (set YUE_DOWNLOAD=1 and run download-models)"
    fi
    if [[ -f "${vae_path}" ]]; then
        ok "YuE2-Vae decoder: ${vae_path}"
    else
        warn "YuE2-Vae weights missing at ${vae_path} (set YUE_DOWNLOAD=1 and run download-models)"
    fi

    return "$missing"
}

run_doctor() {
    prepare_user_path
    prepare_cuda_path
    local python_bin="${YUE_ROOT}/.venv/bin/python"
    if [[ ! -x "${python_bin}" ]]; then
        python_bin="$(command -v python3)"
    fi
    exec "${REPO_DIR}/media-models/generate-yue2.sh" --doctor "$@"
}

command="${1:-}"
prepare_user_path
prepare_cuda_path

case "${command}" in
    check)
        check_runtime
        ;;
    doctor)
        shift
        run_doctor "$@"
        ;;
    install-env)
        setup_env
        ;;
    download-models)
        download_models
        ;;
    install)
        setup_env
        if [[ "${YUE_DOWNLOAD:-0}" == "1" ]]; then
            download_models
        else
            warn "Model weights were not downloaded; run 'maintenance/setup-yue2.sh download-models' to install weights locally."
        fi
        ;;
    -h|--help|"")
        usage
        [[ -n "${command}" ]] || exit 1
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
