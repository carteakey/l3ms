#!/usr/bin/env bash
# Build Strata 4070-main-lite with Ninja targeting Ada Lovelace (sm_89).
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
STRATA_DIR="$REPO_ROOT/vendor/strata-lean"

export PATH="$STRATA_DIR/.venv/bin:/opt/cuda/bin:$PATH"

echo "=== Building Strata 4070-main-lite (sm_89) ==="
cd "$STRATA_DIR"

cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DSTRATA_ENABLE_QWEN35=OFF \
  -DSTRATA_BUILD_TESTS=OFF

cmake --build build --config Release -j"$(nproc)"

BIN="$STRATA_DIR/build/strata"
if [[ -f "$BIN" ]]; then
  echo "=== Build Success: $BIN ($(du -h "$BIN" | cut -f1)) ==="
else
  echo "Build artifact not found at $BIN" >&2
  exit 1
fi
