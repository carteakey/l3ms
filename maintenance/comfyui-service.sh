#!/usr/bin/env bash
# Explicit user-session lifecycle; no boot-time autostart.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-status}" in
 start-cpu|start-gpu)
   mode="${1#start-}"
   systemd-run --user --collect --unit=l2m2-comfyui "$root/maintenance/run-comfyui.sh" "$mode" ;;
 stop) systemctl --user stop l2m2-comfyui.service ;;
 status) systemctl --user status l2m2-comfyui.service --no-pager ;;
 logs) journalctl --user -u l2m2-comfyui.service -n 80 --no-pager ;;
 *) echo 'Usage: comfyui-service.sh start-cpu|start-gpu|stop|status|logs' >&2; exit 2 ;;
esac
