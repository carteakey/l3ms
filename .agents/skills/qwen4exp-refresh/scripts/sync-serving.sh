#!/usr/bin/env bash
# sync-serving.sh
# ---------------
# Snapshots llama-swap.yaml, reloads llama-swap with SIGHUP,
# and performs a smoke test against the active tiers.

set -euo pipefail

L3MS_ROOT="${L3MS_ROOT:-/home/kchauhan/repos/l3ms}"
YAML="${L3MS_ROOT}/llama-swap.yaml"

log() { printf '\033[1;34m->\033[0m %s\n' "$*"; }
ok()  { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m✗\033[0m %s\n' "$*"; }

# 1. Snapshot config
STAMP=$(date +%Y%m%d-%H%M%S)
BAK="${YAML}.bak-${STAMP}"
cp "${YAML}" "${BAK}"
ok "Config snapshotted to $(basename "${BAK}")"

# 2. Hot reload router
PID=$(pgrep '[l]lama-swap' || true)
if [ -z "${PID}" ]; then
    log "llama-swap not running. Starting user service..."
    systemctl --user start llama-swap.service
    sleep 2
    PID=$(pgrep '[l]lama-swap' || true)
fi

if [ -n "${PID}" ]; then
    log "Sending SIGHUP to llama-swap (PID ${PID})..."
    kill -HUP "${PID}"
    sleep 2
    ok "Router reloaded."
else
    err "Failed to reach or start llama-swap process."
    exit 1
fi

# 3. Smoke test verification
API_KEY="${LLAMA_SWAP_API_KEY:-}"
TARGET_MODEL="${1:-qwen38-flash-next-plat}"
log "Smoke testing '${TARGET_MODEL}' via http://127.0.0.1:8080/v1/chat/completions..."

RESP=$(curl -s -m 180 \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  http://127.0.0.1:8080/v1/chat/completions \
  -d "{\"model\": \"${TARGET_MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"Return 2+2 in python: print(2+2)\"}], \"max_tokens\": 64}")

if echo "${RESP}" | grep -q '"content"\|"reasoning_content"'; then
    ok "Smoke test passed for ${TARGET_MODEL}:"
    echo "${RESP}" | jq -c '{model: .model, content: .choices[0].message.content, reasoning: .choices[0].message.reasoning_content, timings: .timings}' 2>/dev/null || echo "${RESP}"
else
    err "Smoke test failed or returned unexpected response:"
    echo "${RESP}"
    exit 1
fi
