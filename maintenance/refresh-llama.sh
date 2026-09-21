#!/usr/bin/env bash
# refresh-llama.sh
# ----------------
# Operational script to inspect, refresh, build, and smoke-test llama.cpp
# binaries in L3MS:
#   - Gold tier:        vendor/llama.cpp-master (upstream master)
#   - MTP / Exp tier:   vendor/llama.cpp-pr-test-28770-28699-28213 (Unified QSA + MTP)
#
# Usage:
#   maintenance/refresh-llama.sh [OPTIONS]
#
# Options:
#   --gold           Refresh and rebuild only the gold tier (vendor/llama.cpp-master)
#   --mtp            Refresh and rebuild only the MTP tier (vendor/llama.cpp-pr-test-28770-28699-28213)
#   --all            Refresh and rebuild both tiers (default)
#   --dry-run        Check upstream status and PRs without modifying files or building
#   --check-prs      Query GitHub for active upstream Qwen/MTP/QSA pull requests
#   --smoke          Run chat completion smoke tests against llama-swap after build
#   --help, -h       Show this help message

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MASTER_DIR="${ROOT}/vendor/llama.cpp-master"
MTP_DIR="${ROOT}/vendor/llama.cpp-pr-test-28770-28699-28213"

TARGET="all"
DRY_RUN=0
CHECK_PRS=0
DO_SMOKE=0

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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gold)
            TARGET="gold"
            shift
            ;;
        --mtp)
            TARGET="mtp"
            shift
            ;;
        --all)
            TARGET="all"
            shift
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
            sed -n '2,17p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            err "Unknown option: $1"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# PR Audit
# -----------------------------------------------------------------------------
if [ "${CHECK_PRS}" = "1" ] || [ "${DRY_RUN}" = "1" ]; then
    log "Checking active upstream PRs related to Qwen / MTP / QSA / MoE..."
    if command -v gh >/dev/null 2>&1; then
        gh pr list --repo ggerganov/llama.cpp --limit 100 \
            | grep -iE "qwen|qsa|mtp|shexp|sparse" || log "No matching PRs found."
    else
        warn "GitHub CLI (gh) not found, skipping PR check."
    fi
    if [ "${DRY_RUN}" = "1" ] && [ "${TARGET}" != "gold" ] && [ "${TARGET}" != "mtp" ] && [ "${TARGET}" != "all" ]; then
        exit 0
    fi
fi

# -----------------------------------------------------------------------------
# Gold Tier Refresh (vendor/llama.cpp-master)
# -----------------------------------------------------------------------------
refresh_gold() {
    log "Inspecting Gold tier (vendor/llama.cpp-master)..."
    if [ ! -d "${MASTER_DIR}/.git" ]; then
        err "Directory not found: ${MASTER_DIR}"
        return 1
    fi

    local current_head
    current_head=$(git -C "${MASTER_DIR}" rev-parse --short HEAD)
    log "Current Gold HEAD: ${current_head}"

    log "Fetching origin..."
    git -C "${MASTER_DIR}" fetch origin

    local behind_count
    behind_count=$(git -C "${MASTER_DIR}" rev-list --count HEAD..origin/master)
    log "Gold is behind origin/master by ${behind_count} commit(s)."

    if [ "${behind_count}" -gt 0 ]; then
        log "New upstream commits:"
        git -C "${MASTER_DIR}" log HEAD..origin/master --oneline -n 10
        if [ "${DRY_RUN}" = "1" ]; then
            log "[dry-run] Would fast-forward to origin/master and rebuild."
            return 0
        fi

        log "Fast-forwarding vendor/llama.cpp-master..."
        git -C "${MASTER_DIR}" merge --ff-only origin/master
    else
        success "Gold tier is up to date with origin/master."
        if [ "${DRY_RUN}" = "1" ]; then
            return 0
        fi
    fi

    log "Building llama-server and llama-bench on master..."
    cmake --build "${MASTER_DIR}/build" --config Release --target llama-server llama-bench -j"$(nproc)"
    success "Gold tier built successfully: ${MASTER_DIR}/build/bin/llama-server"
}

# -----------------------------------------------------------------------------
# MTP Tier Refresh (vendor/llama.cpp-pr-test-28770-28699-28213)
# -----------------------------------------------------------------------------
refresh_mtp() {
    log "Inspecting MTP / Exp tier (vendor/llama.cpp-pr-test-28770-28699-28213)..."
    if [ ! -d "${MTP_DIR}/.git" ]; then
        err "Directory not found: ${MTP_DIR}"
        return 1
    fi

    local current_head
    current_head=$(git -C "${MTP_DIR}" rev-parse --short HEAD)
    log "Current MTP HEAD: ${current_head}"

    if [ "${DRY_RUN}" = "1" ]; then
        log "[dry-run] Checking recent commits on MTP branch:"
        git -C "${MTP_DIR}" log -n 5 --oneline
        return 0
    fi

    log "Building llama-server and llama-bench on MTP branch..."
    cmake --build "${MTP_DIR}/build" --config Release --target llama-server llama-bench -j"$(nproc)"
    success "MTP tier built successfully: ${MTP_DIR}/build/bin/llama-server"
}

# -----------------------------------------------------------------------------
# Smoke Testing
# -----------------------------------------------------------------------------
smoke_test() {
    local tier_name="$1"
    log "Running smoke test for ${tier_name} via llama-swap..."

    local api_key_file="${HOME}/.config/systemd/user/llama-swap.service.d/api-key.conf"
    local api_key=""
    if [ -f "${api_key_file}" ]; then
        api_key=$(grep LLAMA_SWAP_API_KEY "${api_key_file}" | cut -d'=' -f2)
    elif [ -n "${LLAMA_SWAP_API_KEY:-}" ]; then
        api_key="${LLAMA_SWAP_API_KEY}"
    fi

    if [ -z "${api_key}" ]; then
        warn "Could not resolve LLAMA_SWAP_API_KEY, skipping curl smoke test."
        return 0
    fi

    local payload
    payload=$(curl -s http://127.0.0.1:8080/v1/chat/completions \
      -H "Authorization: Bearer ${api_key}" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"${tier_name}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2? Answer in one word.\"}],
        \"max_tokens\": 128,
        \"temperature\": 0.0
      }")

    if echo "${payload}" | grep -q '"content"'; then
        local fingerprint
        fingerprint=$(echo "${payload}" | grep -o '"system_fingerprint":"[^"]*"' | cut -d':' -f2 | tr -d '"')
        local answer
        answer=$(echo "${payload}" | grep -o '"content":"[^"]*"' | head -n1 | cut -d':' -f2 | tr -d '"')
        success "Smoke test passed for ${tier_name}! Fingerprint: ${fingerprint}, Output: '${answer}'"
    else
        err "Smoke test failed for ${tier_name}. Response: ${payload}"
    fi
}

# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------
case "${TARGET}" in
    gold)
        refresh_gold
        if [ "${DO_SMOKE}" = "1" ] && [ "${DRY_RUN}" = "0" ]; then
            smoke_test "qwen38-flash-next"
        fi
        ;;
    mtp)
        refresh_mtp
        if [ "${DO_SMOKE}" = "1" ] && [ "${DRY_RUN}" = "0" ]; then
            smoke_test "qwen38-flash-next-mtp"
        fi
        ;;
    all)
        refresh_gold
        refresh_mtp
        if [ "${DO_SMOKE}" = "1" ] && [ "${DRY_RUN}" = "0" ]; then
            smoke_test "qwen38-flash-next"
            smoke_test "qwen38-flash-next-mtp"
        fi
        ;;
esac

log "Done."
