#!/usr/bin/env bash
# build-all.sh
# ------------
# Builds Gold master, Exp latest (with MoE LRU cache #29887 and radix-select #28671),
# and Strata latest for Qwen3.8-Flash-Next evaluation.

set -euo pipefail

L3MS_ROOT="${L3MS_ROOT:-/home/kchauhan/repos/l3ms}"
NPROC=$(nproc 2>/dev/null || echo 8)

log() { printf '\033[1;34m->\033[0m %s\n' "$*"; }
ok()  { printf '\033[1;32m✓\033[0m %s\n' "$*"; }

# 1. Build Gold Master
if [ -d "${L3MS_ROOT}/vendor/llama.cpp-master" ]; then
    log "Updating and building Gold tier (vendor/llama.cpp-master)..."
    git -C "${L3MS_ROOT}/vendor/llama.cpp-master" fetch origin -q
    git -C "${L3MS_ROOT}/vendor/llama.cpp-master" merge --ff-only origin/master
    cmake --build "${L3MS_ROOT}/vendor/llama.cpp-master/build" --config Release \
      --target llama-server llama-bench -j"${NPROC}"
    TAG=$(git -C "${L3MS_ROOT}/vendor/llama.cpp-master" describe --tags --always)
    cp "${L3MS_ROOT}/vendor/llama.cpp-master/build/bin/llama-server" \
       "${L3MS_ROOT}/vendor/llama.cpp-master/build/bin/llama-server-${TAG}"
    ok "Gold tier built: ${L3MS_ROOT}/vendor/llama.cpp-master/build/bin/llama-server (${TAG})"
fi

# 2. Build Experimental Stack
if [ -d "${L3MS_ROOT}/vendor/llama.cpp-exp-latest" ]; then
    log "Building Experimental tier (vendor/llama.cpp-exp-latest)..."
    cmake --build "${L3MS_ROOT}/vendor/llama.cpp-exp-latest/build" --config Release \
      --target llama-server llama-bench -j"${NPROC}"
    ok "Experimental tier built: ${L3MS_ROOT}/vendor/llama.cpp-exp-latest/build/bin/llama-server"
fi

# 3. Build Strata Latest
if [ -d "${L3MS_ROOT}/vendor/strata-latest" ]; then
    log "Building Strata latest tier (vendor/strata-latest)..."
    if [ ! -e "${L3MS_ROOT}/vendor/strata-latest/.venv" ] && [ -d "${L3MS_ROOT}/vendor/strata/.venv" ]; then
        ln -sf "${L3MS_ROOT}/vendor/strata/.venv" "${L3MS_ROOT}/vendor/strata-latest/.venv"
    fi
    export PATH="${L3MS_ROOT}/vendor/strata/.venv/bin:$PATH"
    cmake --build "${L3MS_ROOT}/vendor/strata-latest/build" --target strata -j"${NPROC}"

    # Ensure runtime configs exist for strata-latest
    if [ ! -f "${L3MS_ROOT}/vendor/strata-latest/strata-iq3_xxs.json" ] && [ -f "${L3MS_ROOT}/vendor/strata/strata-iq3_xxs.json" ]; then
        sed -e 's|vendor/strata/engine/strata|vendor/strata-latest/build/strata|g' \
            -e 's|vendor/strata/data|vendor/strata-latest/data|g' \
            -e 's|vendor/strata|vendor/strata-latest|g' \
            "${L3MS_ROOT}/vendor/strata/strata-iq3_xxs.json" > "${L3MS_ROOT}/vendor/strata-latest/strata-iq3_xxs.json"
    fi
    if [ ! -f "${L3MS_ROOT}/vendor/strata-latest/strata-vision.json" ] && [ -f "${L3MS_ROOT}/vendor/strata/strata-vision.json" ]; then
        sed -e 's|vendor/strata/engine/strata|vendor/strata-latest/build/strata|g' \
            -e 's|vendor/strata/data|vendor/strata-latest/data|g' \
            -e 's|vendor/strata|vendor/strata-latest|g' \
            "${L3MS_ROOT}/vendor/strata/strata-vision.json" > "${L3MS_ROOT}/vendor/strata-latest/strata-vision.json"
    fi
    ok "Strata latest built & configured: ${L3MS_ROOT}/vendor/strata-latest/build/strata"
fi

echo "All targets built successfully."
