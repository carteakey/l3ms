#!/usr/bin/env bash
# Multi-account wrapper for google-colab-cli (verified against v0.8.0).
#
# The CLI keeps ONE OAuth token in the OS keyring (service "google-colab-cli"),
# so two Google accounts (Ultra 5x + Pro) would clobber each other. This wrapper
# gives each account its own authorized_user credential file and session file
# and drives the CLI with `--auth adc`, so both accounts can run concurrently
# without touching the keyring.
#
#   maintenance/colab/colab-acct.sh <ultra|pro> login             # print sign-in URL
#   maintenance/colab/colab-acct.sh <ultra|pro> login-code <code> # finish sign-in
#   maintenance/colab/colab-acct.sh <ultra|pro> whoami    # which Google account + balance
#   maintenance/colab/colab-acct.sh <ultra|pro> <colab args...>
#
# Credentials live outside the repo at ~/.config/colab-cli/accounts/<profile>.json
# (mode 0600) and are never printed. Account emails are not stored in the repo.
set -euo pipefail

PROFILES=(ultra pro)
ACCT_DIR="${L3MS_COLAB_ACCT_DIR:-$HOME/.config/colab-cli/accounts}"
STATE_DIR="$HOME/.config/colab-cli"

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

[[ $# -ge 1 ]] || usage 1
case "${1:-}" in -h|--help) usage 0 ;; esac

profile="$1"; shift
if [[ ! " ${PROFILES[*]} " =~ " ${profile} " ]]; then
  echo "ERROR: unknown profile '$profile' (expected: ${PROFILES[*]})" >&2
  exit 2
fi

command -v colab >/dev/null || { echo "ERROR: colab CLI missing: uv tool install google-colab-cli" >&2; exit 1; }
TOOL_PY="$(uv tool dir)/google-colab-cli/bin/python"
[[ -x "$TOOL_PY" ]] || { echo "ERROR: cannot find google-colab-cli tool python at $TOOL_PY" >&2; exit 1; }

cred="$ACCT_DIR/$profile.json"
sessions="$STATE_DIR/sessions-$profile.json"

# Two-step, non-interactive sign-in (works from scripts and agent `!` shells):
#   login            -> prints the Google URL, saves PKCE state to <profile>.pending.json
#   login-code CODE  -> exchanges the code Google shows, writes <profile>.json
# Mirrors colab_cli.auth._run_remote_flow (bundled public OAuth client and the
# sdk.cloud.google.com copy/paste redirect), split at the input() prompt.
pending="$ACCT_DIR/$profile.pending.json"

cmd_login() {
  mkdir -p "$ACCT_DIR"; chmod 700 "$ACCT_DIR"
  umask 077
  "$TOOL_PY" -I -c '
import json, sys
from importlib import resources
from google_auth_oauthlib.flow import InstalledAppFlow
from colab_cli import auth
cfg = json.loads(resources.files("colab_cli").joinpath("oauth_config.json").read_text())
flow = InstalledAppFlow.from_client_config(cfg, auth.PUBLIC_SCOPES, autogenerate_code_verifier=True)
flow.redirect_uri = auth.REMOTE_REDIRECT_URI
url, state = flow.authorization_url(prompt="consent", token_usage="remote", access_type="offline")
with open(sys.argv[1], "w") as f:
    json.dump({"state": state, "code_verifier": flow.code_verifier}, f)
print(url)
' "$pending"
  echo
  echo "Open the URL above, sign in with the account that holds the '$profile' plan, then run:"
  echo "  $0 $profile login-code '<code Google shows>'"
}

cmd_login_code() {
  [[ $# -eq 1 && -n "$1" ]] || { echo "usage: $0 $profile login-code <code>" >&2; exit 2; }
  [[ -f "$pending" ]] || { echo "ERROR: no pending sign-in; run: $0 $profile login" >&2; exit 1; }
  umask 077
  # Code is passed via env, not argv of the python process.
  COLAB_AUTH_CODE="$1" "$TOOL_PY" -I -c '
import json, os, sys
from importlib import resources
from google_auth_oauthlib.flow import InstalledAppFlow
from colab_cli import auth
pending, out = sys.argv[1], sys.argv[2]
st = json.load(open(pending))
cfg = json.loads(resources.files("colab_cli").joinpath("oauth_config.json").read_text())
flow = InstalledAppFlow.from_client_config(cfg, auth.PUBLIC_SCOPES, state=st["state"], code_verifier=st["code_verifier"])
flow.redirect_uri = auth.REMOTE_REDIRECT_URI
flow.fetch_token(code=os.environ["COLAB_AUTH_CODE"].strip())
info = json.loads(flow.credentials.to_json())
info["type"] = "authorized_user"  # ADC file format
with open(out, "w") as f:
    json.dump(info, f)
os.remove(pending)
print("Saved credentials to", out)
' "$pending" "$cred"
  chmod 600 "$cred"
  cmd_whoami
}

cmd_whoami() {
  [[ -f "$cred" ]] || { echo "ERROR: '$profile' not signed in; run: $0 $profile login" >&2; exit 1; }
  GOOGLE_APPLICATION_CREDENTIALS="$cred" "$TOOL_PY" -I - <<'PY'
import os, warnings, google.auth
warnings.filterwarnings("ignore", message="Your application has authenticated")
from google.auth.transport.requests import AuthorizedSession
from colab_cli.auth import PUBLIC_SCOPES
creds, _ = google.auth.default(scopes=PUBLIC_SCOPES)
r = AuthorizedSession(creds).get("https://openidconnect.googleapis.com/v1/userinfo", timeout=20)
r.raise_for_status()
print("account:", r.json().get("email", "<unknown>"))
PY
  run_colab usage
}

run_colab() {
  [[ -f "$cred" ]] || { echo "ERROR: '$profile' not signed in; run: $0 $profile login" >&2; exit 1; }
  GOOGLE_APPLICATION_CREDENTIALS="$cred" colab --auth adc --config "$sessions" "$@"
}

case "${1:-}" in
  login)  cmd_login ;;
  login-code) shift; cmd_login_code "$@" ;;
  whoami) cmd_whoami ;;
  "")     usage 1 ;;
  *)      run_colab "$@" ;;
esac
