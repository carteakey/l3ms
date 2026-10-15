#!/usr/bin/env bash
# refresh-audio-cpp.sh
# --------------------
# Operational script to inspect, refresh, build, and smoke-test audio.cpp
# binaries in L2M2:
#   - Master/Gold tier: vendor/audio.cpp-master (upstream main)
#
# Usage:
#   maintenance/refresh-audio-cpp.sh [OPTIONS]
#
# Options:
#   --update         Fetch and fast-forward vendor/audio.cpp-master to origin/main
#   --build          Build audiocpp_cli and audiocpp_server targets
#   --all            Update and build (default)
#   --models LIST    Comma-separated model targets (default: minimax_h3,heartmula,breeze_tts; or 'full')
#   --cuda-arch SM   CUDA compute architecture (default: auto-detected, 89-real)
#   --dry-run        Check upstream status and PRs without modifying files or building
#   --check-prs      Query GitHub for active upstream audio.cpp pull requests
#   --smoke          Run smoke test verification (--list-loaders check)
#   --help, -h       Show this help message

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Resolve audio.cpp source tree
if [[ -d "${REPO_DIR}/vendor/audio.cpp-master/.git" ]]; then
    MASTER_DIR="${REPO_DIR}/vendor/audio.cpp-master"
elif [[ -d "${REPO_DIR}/vendor/audio.cpp/.git" ]]; then
    MASTER_DIR="${REPO_DIR}/vendor/audio.cpp"
elif [[ -d "${HOME}/repos/audio.cpp/.git" ]]; then
    MASTER_DIR="${HOME}/repos/audio.cpp"
else
    MASTER_DIR="${REPO_DIR}/vendor/audio.cpp-master"
fi

DO_UPDATE=0
DO_BUILD=0
DRY_RUN=0
CHECK_PRS=0
DO_SMOKE=0
MODELS_LIST="minimax_h3,heartmula,breeze_tts"
CUDA_ARCH=""

log() {
    printf '\033[1;34m->\033[0m %s\n' "$*"
}

success() {
    printf '\033[1;32m✓\033[0m %s\n' "$*"
}

warn() {
    printf '\033[1;33mWARN:\033[0m %s\n' "$*" >&2
}

err() {
    printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2
}

detect_cuda_arch() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "89-real"
        return
    fi
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "")
    case "$gpu_name" in
        *"RTX 40"*|*"RTX 4090"*|*"RTX 4080"*|*"RTX 4070"*|*"RTX 4060"*) echo "89-real" ;;
        *"RTX 30"*|*"RTX 3090"*|*"RTX 3080"*|*"RTX 3070"*|*"RTX 3060"*) echo "86-real" ;;
        *"RTX 20"*|*"RTX 2080"*|*"RTX 2070"*|*"RTX 2060"*)               echo "75-real" ;;
        *"A100"*)                                                           echo "80-real" ;;
        *"H100"*)                                                           echo "90-real" ;;
        *)                                                                  echo "89-real" ;;
    esac
}

prepare_cuda_path() {
    if [[ -x /opt/cuda/bin/nvcc ]]; then
        PATH="/opt/cuda/bin:$PATH"
        export PATH
        LD_LIBRARY_PATH="/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH
    fi
}

# Parse options
if [[ $# -eq 0 ]]; then
    DO_UPDATE=1
    DO_BUILD=1
    DO_SMOKE=1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --update)
            DO_UPDATE=1
            shift
            ;;
        --build)
            DO_BUILD=1
            shift
            ;;
        --all)
            DO_UPDATE=1
            DO_BUILD=1
            DO_SMOKE=1
            shift
            ;;
        --models)
            MODELS_LIST="$2"
            shift 2
            ;;
        --cuda-arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --check-prs)
            CHECK_PRS=1
            shift
            ;;
        --smoke)
            DO_SMOKE=1
            shift
            ;;
        --help|-h)
            sed -n '2,20p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            err "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$CUDA_ARCH" ]]; then
    CUDA_ARCH=$(detect_cuda_arch)
fi

prepare_cuda_path

# -----------------------------------------------------------------------------
# PR Audit
# -----------------------------------------------------------------------------
if [[ "${CHECK_PRS}" = "1" ]] || [[ "${DRY_RUN}" = "1" ]]; then
    log "Checking active upstream PRs on 0xShug0/audio.cpp..."
    if command -v gh >/dev/null 2>&1; then
        gh pr list --repo 0xShug0/audio.cpp --limit 30 \
            | grep -iE "breeze|minimax|heartmula|cuda|stream|tts|qwen" || log "No matching PRs found."
    else
        warn "GitHub CLI (gh) not found, skipping PR check."
    fi
    if [[ "${DRY_RUN}" = "1" ]] && [[ "${DO_UPDATE}" = "0" ]] && [[ "${DO_BUILD}" = "0" ]]; then
        exit 0
    fi
fi

# -----------------------------------------------------------------------------
# Inspect and Update
# -----------------------------------------------------------------------------
log "Inspecting audio.cpp tree at: ${MASTER_DIR}"
if [[ ! -d "${MASTER_DIR}/.git" ]]; then
    err "Directory is not a git repo: ${MASTER_DIR}"
    exit 1
fi

current_head=$(git -C "${MASTER_DIR}" rev-parse --short HEAD)
log "Current HEAD: ${current_head}"

log "Fetching origin..."
git -C "${MASTER_DIR}" config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
git -C "${MASTER_DIR}" fetch origin --tags

upstream_branch="main"
if ! git -C "${MASTER_DIR}" rev-parse --verify "origin/${upstream_branch}" >/dev/null 2>&1; then
    upstream_branch="master"
fi

behind_count=$(git -C "${MASTER_DIR}" rev-list --count "HEAD..origin/${upstream_branch}" 2>/dev/null || echo 0)
log "audio.cpp is behind origin/${upstream_branch} by ${behind_count} commit(s)."

if [[ "${behind_count}" -gt 0 ]]; then
    log "New upstream commits:"
    git -C "${MASTER_DIR}" log "HEAD..origin/${upstream_branch}" --oneline -n 10
    if [[ "${DRY_RUN}" = "1" ]]; then
        log "[dry-run] Would fast-forward to origin/${upstream_branch} and rebuild."
        exit 0
    fi
    if [[ "${DO_UPDATE}" = "1" ]]; then
        log "Fast-forwarding ${MASTER_DIR} to origin/${upstream_branch}..."
        git -C "${MASTER_DIR}" checkout "${upstream_branch}"
        git -C "${MASTER_DIR}" merge --ff-only "origin/${upstream_branch}"
        success "Updated to $(git -C "${MASTER_DIR}" rev-parse --short HEAD)"
    fi
else
    success "audio.cpp is up to date with origin/${upstream_branch}."
    if [[ "${DRY_RUN}" = "1" ]]; then
        exit 0
    fi
fi

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------
if [[ "${DO_BUILD}" = "1" ]]; then
    build_dir="${MASTER_DIR}/build/linux-cuda-release"
    log "Configuring audio.cpp in ${build_dir}..."
    log "CUDA Architecture: ${CUDA_ARCH}"
    log "Target Models:    ${MODELS_LIST}"

    CMAKE_EXTRA_ARGS=()
    if [[ "${MODELS_LIST}" == "full" ]]; then
        CMAKE_EXTRA_ARGS+=("-DAUDIOCPP_MODEL_SET=full")
    else
        CMAKE_EXTRA_ARGS+=(
            "-DAUDIOCPP_MODEL_SET=custom"
            "-DAUDIOCPP_MODELS=${MODELS_LIST}"
        )
    fi

    cmake -S "${MASTER_DIR}" -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DENGINE_ENABLE_CUDA=ON \
        -DENGINE_ENABLE_VULKAN=OFF \
        -DENGINE_ENABLE_HIP=OFF \
        -DENGINE_ENABLE_NATIVE_CPU=ON \
        -DENGINE_ENABLE_LLAMAFILE=ON \
        -DENGINE_BUILD_EXAMPLES=OFF \
        -DENGINE_BUILD_TESTS=OFF \
        -DENGINE_BUILD_WARMBENCH=OFF \
        -DAUDIOCPP_DEPLOYMENT_BUILD=ON \
        -DGGML_CUDA_CUB_3DOT2=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        "${CMAKE_EXTRA_ARGS[@]}"

    log "Compiling audiocpp_cli and audiocpp_server..."
    cmake --build "${build_dir}" --parallel "$(nproc 2>/dev/null || echo 8)" \
        --target audiocpp_cli --target audiocpp_server

    success "Build completed: ${build_dir}/bin/audiocpp_cli"
fi

# -----------------------------------------------------------------------------
# Smoke Verification
# -----------------------------------------------------------------------------
if [[ "${DO_SMOKE}" = "1" ]]; then
    cli_bin="${MASTER_DIR}/build/linux-cuda-release/bin/audiocpp_cli"
    if [[ ! -x "${cli_bin}" ]]; then
        err "audiocpp_cli binary not found at ${cli_bin}"
        exit 1
    fi

    log "Running smoke check: inspecting registered loaders..."
    loaders_output=$("${cli_bin}" --list-loaders 2>&1)
    echo "${loaders_output}"

    # Verify key loaders are present
    missing_loaders=()
    IFS=',' read -ra REQ_MODELS <<< "${MODELS_LIST}"
    for model in "${REQ_MODELS[@]}"; do
        if [[ "${model}" != "full" ]] && ! echo "${loaders_output}" | grep -q "${model}"; then
            missing_loaders+=("${model}")
        fi
    done

    if [[ ${#missing_loaders[@]} -gt 0 ]]; then
        err "Missing expected loaders in build: ${missing_loaders[*]}"
        exit 1
    fi

    success "Smoke check passed! All requested loaders present in ${cli_bin}."
fi

log "Done."
