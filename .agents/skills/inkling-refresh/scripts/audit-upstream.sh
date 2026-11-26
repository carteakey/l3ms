#!/usr/bin/env bash
# Audit upstream Inkling 2 Mini engines (PolyStrata, Strata, llama.cpp) and
# scout newly published community engines on GitHub.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")/../../../.." && pwd)"
POLY_DIR="$ROOT/vendor/polystrata"

echo "=== 1. Local PolyStrata HEAD ==="
if [[ -d "$POLY_DIR/.git" ]]; then
  git -C "$POLY_DIR" log -n 5 --oneline --decorate
else
  echo "vendor/polystrata not found"
fi

echo
echo "=== 2. simplexai-labs/PolyStrata: Latest Releases ==="
gh release list -R simplexai-labs/PolyStrata --limit 5 || true

echo
echo "=== 3. simplexai-labs/PolyStrata: Recent Commits ==="
gh api repos/simplexai-labs/PolyStrata/commits?per_page=15 \
  --jq '.[] | "\(.sha[0:8]) \(.commit.committer.date[0:10]) \(.commit.message | split("\n")[0])"' || true

echo
echo "=== 4. simplexai-labs/PolyStrata: Open PRs (Inkling / CUDA / Prefill / KV) ==="
gh pr list -R simplexai-labs/PolyStrata --state open --limit 25 \
  --json number,title,updatedAt \
  --jq '.[] | "#\(.number) [\(.updatedAt[0:10])] \(.title)"' || true

echo
echo "=== 5. Niko1221/Strata: Recent Releases & Inkling Commits ==="
gh release list -R Niko1221/Strata --limit 3 || true
gh api repos/Niko1221/Strata/commits?per_page=10 \
  --jq '.[] | "\(.sha[0:8]) \(.commit.committer.date[0:10]) \(.commit.message | split("\n")[0])"' || true

echo
echo "=== 6. ggml-org/llama.cpp: Open Inkling / MoE-Cache / NVFP4 PRs ==="
gh pr list -R ggml-org/llama.cpp --state open --limit 15 \
  --search "inkling OR moe-cache OR dspark OR nvfp4" \
  --json number,title,updatedAt \
  --jq '.[] | "#\(.number) [\(.updatedAt[0:10])] \(.title)"' || true

echo
echo "=== 7. New Engine Discovery Radar (GitHub repos mentioning Inkling 2) ==="
gh search repos "inkling2 OR \"inkling 2 mini\" OR \"inkling-2-mini\"" \
  --sort updated --limit 12 \
  --json fullName,stargazersCount,updatedAt,description \
  --jq '.[] | "\(.fullName) (★\(.stargazersCount), \(.updatedAt[0:10])) - \(.description // "")"' || true
