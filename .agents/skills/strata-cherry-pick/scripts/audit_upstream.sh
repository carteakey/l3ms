#!/usr/bin/env bash
# Audit upstream Strata, StrataGP, and open PRs for inference-level changes.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
STRATA_DIR="$REPO_ROOT/vendor/strata-lean"

echo "=== Strata Remotes in $STRATA_DIR ==="
git -C "$STRATA_DIR" remote -v

echo ""
echo "=== Fetching latest upstream tags & branches ==="
git -C "$STRATA_DIR" fetch origin --tags -q || true

LATEST_TAG=$(git -C "$STRATA_DIR" tag --sort=-v:refname | head -n 1)
CURRENT_HEAD=$(git -C "$STRATA_DIR" rev-parse --short HEAD)
echo "Current 4070-main-lite HEAD: $CURRENT_HEAD"
echo "Upstream latest tag:        $LATEST_TAG"

echo ""
echo "=== Upstream inference commits since v0.1.45 (core, cuda, spec) ==="
git -C "$STRATA_DIR" log v0.1.45..origin/main --oneline \
  -- src/core src/kernels/cuda src/spec | head -n 30 || true

echo ""
echo "=== Recent upstream tags ==="
git -C "$STRATA_DIR" tag --sort=-v:refname | head -n 8
