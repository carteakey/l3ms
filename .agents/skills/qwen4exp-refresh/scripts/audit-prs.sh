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
echo " 2. Strata Upstream Delta (vendor/strata)"
echo "==================================================================="
if [ -d "${L3MS_ROOT}/vendor/strata" ]; then
    git -C "${L3MS_ROOT}/vendor/strata" fetch origin --tags --force -q
    CURRENT_STRATA=$(git -C "${L3MS_ROOT}/vendor/strata" describe --tags --always 2>/dev/null || echo "unknown")
    LATEST_TAG=$(git -C "${L3MS_ROOT}/vendor/strata" tag --sort=-v:refname 2>/dev/null | { head -1 || true; })
    echo "Current local Strata checkout : ${CURRENT_STRATA}"
    echo "Latest upstream release tag   : ${LATEST_TAG}"
    echo "Recent upstream releases:"
    git -C "${L3MS_ROOT}/vendor/strata" tag --sort=-v:refname 2>/dev/null | { head -5 || true; } | sed 's/^/   /'
    echo "Top commit on upstream main:"
    git -C "${L3MS_ROOT}/vendor/strata" log origin/main -1 --oneline | sed 's/^/   /'
else
    echo "Strata vendor directory not found at ${L3MS_ROOT}/vendor/strata"
fi
