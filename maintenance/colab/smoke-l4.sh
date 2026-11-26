#!/usr/bin/env bash
# Phase 0 smoke test: boot an L4 on a Colab profile, confirm it is Ada sm_89
# (same arch as the local RTX 4070), sample the real CU/hr rate, then stop it.
#   maintenance/colab/smoke-l4.sh [pro|ultra]   (default: pro)
set -euo pipefail
profile="${1:-pro}"
gpu="${COLAB_GPU:-L4}"
acct="$(cd -- "$(dirname -- "$0")" && pwd)/colab-acct.sh"
sess="l3ms-smoke-$$"

cleanup() { "$acct" "$profile" stop -s "$sess" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

t0=$(date +%s)
# L4 assignment often fails with "503 Service Unavailable" when the pool is dry
# (seen 2026-11-26 on pro while T4 assigned fine), so retry with backoff.
tries="${COLAB_TRIES:-6}"; wait_s="${COLAB_RETRY_WAIT:-90}"
for i in $(seq 1 "$tries"); do
  if "$acct" "$profile" new -s "$sess" --gpu "$gpu" 2>&1 | tail -2; then
    "$acct" "$profile" status -s "$sess" >/dev/null 2>&1 && break
  fi
  [[ $i -lt $tries ]] || { echo "ERROR: no $gpu capacity after $tries tries" >&2; exit 3; }
  echo "no $gpu capacity (try $i/$tries), retrying in ${wait_s}s"; sleep "$wait_s"
done
echo "boot: $(( $(date +%s) - t0 )) s"

# Hard timeout: on 2026-11-26 `colab exec` hit "Connection was lost" and then
# hung for ~1.5 h with the L4 still allocated (3.65 CU burned). Never block
# unbounded while a VM is billing; the EXIT trap releases it.
timeout --kill-after=15 "${COLAB_EXEC_TIMEOUT:-300}" "$acct" "$profile" exec -s "$sess" <<'PY' || echo "WARN: exec failed or timed out (rc=$?)"
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
print(subprocess.run(["bash", "-lc", "nvcc --version | tail -1; nproc; free -g | awk '/Mem/{print $2\" GiB RAM\"}'"],
                     capture_output=True, text=True).stdout.strip())
try:
    import torch
    cap = torch.cuda.get_device_capability(0)
    print("torch", torch.__version__, "cuda", torch.version.cuda, "capability", cap,
          "SM89_MATCH" if cap == (8, 9) else "SM_MISMATCH")
except Exception as e:
    print("torch check failed:", e)
PY

echo "--- usage while allocated ---"
"$acct" "$profile" usage
echo "wall: $(( $(date +%s) - t0 )) s"
