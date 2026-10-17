#!/usr/bin/env bash
# Sequential downloader and verifier for YuE2, MiniMax-H3 Uncensored, and LTX-2.5.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

log "1/3: Checking/Downloading YuE2 foundation music model..."
# If already running in background, wait for it
if pgrep -f "setup-yue2\.sh.*download-models" >/dev/null; then
    echo "YuE2 download is currently running in background. Waiting for it to finish..."
    while pgrep -f "setup-yue2\.sh.*download-models" >/dev/null; do
        sleep 5
    done
fi
bash "$ROOT/maintenance/setup-yue2.sh" download-models

log "2/3: Downloading MiniMax-H3 Uncensored (Base Ref2VA + Uncensored NVFP4 Text Encoder)..."
bash "$ROOT/maintenance/setup-minimax-h3-uncensored.sh"

log "3/3: Downloading LTX-2.5 synchronized video model (INT8 ConvRot DiT, Gemma 4, VAEs)..."
bash "$ROOT/maintenance/setup-ltx-2.5.sh"

log "All downloads completed! Generating storage and audit report..."
df -h /home /mnt/dev /run/media/kchauhan/Games 2>/dev/null || true
python3 "$ROOT/.agents/skills/model-inventory-manager/scripts/audit_models.py"

log "Media stack download and verification complete."
