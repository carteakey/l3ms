#!/usr/bin/env bash
# Operational helper for llama-swap.service on KPC.
# Reads LLAMA_SWAP_API_KEY in-process from the user systemd unit (or .env) so
# secrets are never echoed or passed inline on agent command lines.
set -euo pipefail

ROUTER_URL="${L3MS_ROUTER_URL:-http://127.0.0.1:8080}"
UI_PORT="${L3MS_UI_PORT:-8089}"
TAILSCALE_HOST="${L3MS_TAILSCALE_HOST:-yeti-cachy.llama-alnair.ts.net}"
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"

load_api_key() {
  local key=""
  key="$(systemctl --user show llama-swap.service -p Environment --value 2>/dev/null \
    | tr ' ' '\n' | sed -n 's/^LLAMA_SWAP_API_KEY=//p' | head -n 1 || true)"
  if [[ -z "$key" && -f "$ROOT/.env" ]]; then
    key="$(sed -n 's/^LLAMA_SWAP_API_KEY=//p' "$ROOT/.env" | head -n 1 || true)"
  fi
  if [[ -z "$key" ]]; then
    echo "ERROR: could not read LLAMA_SWAP_API_KEY from llama-swap.service or $ROOT/.env" >&2
    exit 1
  fi
  printf '%s' "$key"
}

usage() {
  cat <<'EOF'
Usage: maintenance/l3ms-ctl.sh <command> [args]

Commands:
  health             Check router /health
  models             List model IDs from /v1/models
  wake <model>       Wake/load a model via /upstream/<model>/ and report elapsed time
  smoke <model>      Run a 64-token chat completion smoke test against <model>
  ui-check           Verify 0.0.0.0:8089 bind and Tailscale DNS HTTP status
  restart            Restart llama-swap.service cleanly (preferred over SIGHUP on fixed port 8089)
EOF
}

cmd="${1:-}"
case "$cmd" in
  health)
    curl -fsS "$ROUTER_URL/health"
    echo
    ;;
  models)
    key="$(load_api_key)"
    curl -fsS -H "Authorization: Bearer $key" "$ROUTER_URL/v1/models" \
      | python3 -c 'import json,sys; print("\n".join(d["id"] for d in json.load(sys.stdin).get("data", [])))'
    ;;
  wake)
    model="${2:?Usage: l3ms-ctl.sh wake <model>}"
    key="$(load_api_key)"
    curl -sS --max-time 240 -o /dev/null \
      -w "wake ${model}: HTTP %{http_code} (%{time_total}s)\n" \
      -H "Authorization: Bearer $key" \
      "$ROUTER_URL/upstream/${model}/"
    ;;
  smoke)
    model="${2:?Usage: l3ms-ctl.sh smoke <model>}"
    key="$(load_api_key)"
    payload="$(python3 -c 'import json,sys; print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": "Reply with OK"}], "max_tokens": 64}))' "$model")"
    curl -fsS --max-time 180 \
      -H "Authorization: Bearer $key" \
      -H "Content-Type: application/json" \
      -d "$payload" \
      "$ROUTER_URL/v1/chat/completions" \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print(json.dumps({"model": d.get("model"), "content": m.get("content"), "has_reasoning": bool(m.get("reasoning_content"))}))'
    ;;
  ui-check)
    ss -tlnp 2>/dev/null | grep ":${UI_PORT}" || echo "NOT LISTENING on :${UI_PORT}"
    curl -s -o /dev/null -w "Tailscale DNS (${TAILSCALE_HOST}:${UI_PORT}): HTTP %{http_code}\n" \
      "http://${TAILSCALE_HOST}:${UI_PORT}/" || true
    ;;
  restart)
    systemctl --user restart llama-swap.service
    systemctl --user is-active llama-swap.service
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
