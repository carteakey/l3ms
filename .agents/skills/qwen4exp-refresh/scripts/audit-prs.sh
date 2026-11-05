#!/usr/bin/env bash
# audit-prs.sh
# ------------
# Queries active & merged pull requests for Qwen3.8-Flash-Next in llama.cpp
# and inspects Strata repository git delta.

set -euo pipefail

L3MS_ROOT="${L3MS_ROOT:-/home/kchauhan/repos/l3ms}"

echo "==================================================================="
echo " 1. llama.cpp Upstream PR Audit (ggml-org/llama.cpp)"
echo "==================================================================="

echo "-> Recently merged qwen4exp / MTP PRs:"
gh pr list --repo ggml-org/llama.cpp --state merged \
  --search "qwen4exp OR qwen4 OR Qwen3.8 OR nextn in:title" --limit 15 \
  --json number,mergedAt,title -q '.[] | "   #\(.number) (\(.mergedAt[:10])): \(.title)"'

echo ""
echo "-> Active open qwen4exp / MoE / speculative PRs:"
gh pr list --repo ggml-org/llama.cpp --state open \
  --search "qwen4exp OR qwen4 OR Qwen3.8 OR Flash-Next OR nextn in:title" --limit 15 \
  --json number,updatedAt,title -q '.[] | "   #\(.number) (\(.updatedAt[:10])): \(.title)"'

echo ""
echo "==================================================================="
echo " 2. Strata Upstream Delta (vendor/strata-lean)"
echo "==================================================================="
STRATA_DIR="${L3MS_ROOT}/vendor/strata-lean"
if [ ! -d "${STRATA_DIR}" ] && [ -d "${L3MS_ROOT}/vendor/strata" ]; then
    STRATA_DIR="${L3MS_ROOT}/vendor/strata"
fi

if [ -d "${STRATA_DIR}" ]; then
    git -C "${STRATA_DIR}" fetch origin --tags --force -q 2>/dev/null || true
    CURRENT_STRATA=$(git -C "${STRATA_DIR}" describe --tags --always 2>/dev/null || echo "unknown")
    LATEST_TAG=$(git -C "${STRATA_DIR}" tag --sort=-v:refname 2>/dev/null | { head -1 || true; })
    BRANCH=$(git -C "${STRATA_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "detached")
    echo "Current local Strata checkout : ${CURRENT_STRATA} (branch: ${BRANCH})"
    echo "Latest upstream release tag   : ${LATEST_TAG}"
    echo "Recent upstream releases:"
    git -C "${STRATA_DIR}" tag --sort=-v:refname 2>/dev/null | { head -5 || true; } | sed 's/^/   /'
    echo "Top commit on upstream main:"
    git -C "${STRATA_DIR}" log origin/main -1 --oneline 2>/dev/null | sed 's/^/   /' || true
    echo "Local branch commits ahead of upstream tag:"
    git -C "${STRATA_DIR}" log "${LATEST_TAG}..HEAD" --oneline 2>/dev/null | sed 's/^/   /' || true
else
    echo "Strata vendor directory not found at ${STRATA_DIR}"
fi
