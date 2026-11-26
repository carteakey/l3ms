#!/usr/bin/env bash
# host-tuning-check.sh — non-destructive read-only host audit for L3MS on yeti-cachy
# (CachyOS / Alder Lake i5-12600K / 64 GB DDR5 / RTX 4070 12 GB).
#
# Never modifies sysctls, systemd services, or host files.
# Pass --print-remediation to print copy-pasteable commands for any warnings.

set -uo pipefail

PRINT_REMEDIATION=0

usage() {
  cat <<'EOF'
Usage: host-tuning-check.sh [--print-remediation] [-h|--help]

Non-destructive read-only audit of host inference hygiene:
  - DDR5 DIMM speed & memory / zram swap
  - CPU governor, EPP, turbo & Alder Lake P/E-core topology
  - vm.swappiness (warns if > 20)
  - Transparent Huge Pages (with Strata 2 MiB hugepage caveat)
  - ananicy-cpp & scx (sched_ext) status
  - NVIDIA driver & free VRAM
  - Local filesystem disk usage (df -hlT; skips remote/NFS mounts)

Options:
  --print-remediation   Print copy-pasteable sysctl/systemctl commands (no mutations)
  -h, --help            Show this help message
EOF
}

for arg in "$@"; do
  case "$arg" in
    --print-remediation) PRINT_REMEDIATION=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

WARNINGS=0
REMEDIATIONS=()

warn() {
  WARNINGS=$((WARNINGS + 1))
  printf '  [WARN] %s\n' "$1"
}

ok() {
  printf '  [OK]   %s\n' "$1"
}

info() {
  printf '  [INFO] %s\n' "$1"
}

add_remediation() {
  REMEDIATIONS+=("$1")
}

echo "=== L3MS Host Tuning Audit ($(hostname -s 2>/dev/null || echo host), $(date +%Y-%m-%d' '%H:%M:%S)) ==="

# ---------------------------------------------------------------------------
# 1. DDR5 DIMM Speed & Memory / zram
# ---------------------------------------------------------------------------
echo ""
echo "[1] Memory & DDR5 DIMM Speed"
if command -v inxi >/dev/null 2>&1; then
  mem_out="$(inxi -m --tty -c 0 2>/dev/null || inxi -m 2>/dev/null || true)"
  speeds="$(printf '%s\n' "$mem_out" | grep -oE 'speed: [0-9]+ MT/s' | sort -u | paste -sd ', ' -)"
  modules="$(printf '%s\n' "$mem_out" | grep -c 'Device-' || true)"
  ram_line="$(printf '%s\n' "$mem_out" | grep -m1 'System RAM:' | sed 's/^[[:space:]]*//')"
  [ -n "$ram_line" ] && info "$ram_line"
  if [ -n "$speeds" ]; then
    info "Detected DIMMs: ${modules:-?} module(s), ${speeds}"
    if printf '%s\n' "$speeds" | grep -qE '4800 MT/s'; then
      warn "DDR5 running at 4800 MT/s JEDEC floor (~60 GB/s CPU read ceiling; 4-DIMM 2DPC)."
      info "Note: BIOS Gear-2 tuning (5200-5400 MT/s) or 2x32 GB kit is a manual UEFI step."
    else
      ok "DDR5 running above 4800 MT/s JEDEC baseline ($speeds)."
    fi
  else
    info "Could not parse DIMM speed from inxi -m (run with root dmidecode if needed)."
  fi
elif [ "$(id -u)" -eq 0 ] && command -v dmidecode >/dev/null 2>&1; then
  dmi_speeds="$(dmidecode -t memory 2>/dev/null | grep -E 'Configured Memory Speed:' | grep -v Unknown | sort -u | tr '\n' '; ')"
  info "dmidecode memory speeds: ${dmi_speeds:-unavailable}"
else
  info "Neither inxi nor root dmidecode available; skipping DIMM speed check."
fi

if command -v swapon >/dev/null 2>&1; then
  zram_line="$(swapon --show --noheadings 2>/dev/null | head -n 1 || true)"
  if [ -n "$zram_line" ]; then
    info "Swap device: $zram_line"
  else
    info "No active swap devices."
  fi
fi

# ---------------------------------------------------------------------------
# 2. CPU Governor, EPP & Alder Lake P/E Topology
# ---------------------------------------------------------------------------
echo ""
echo "[2] CPU Governor & P/E-Core Topology"
govs=""
epps=""
for p in /sys/devices/system/cpu/cpufreq/policy*; do
  [ -d "$p" ] || continue
  govs="$govs $(cat "$p/scaling_governor" 2>/dev/null || true)"
  epps="$epps $(cat "$p/energy_performance_preference" 2>/dev/null || true)"
done
uniq_gov="$(printf '%s\n' $govs | sort -u | tr '\n' ' ' | sed 's/ $//')"
uniq_epp="$(printf '%s\n' $epps | sort -u | tr '\n' ' ' | sed 's/ $//')"
no_turbo="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo unknown)"
min_perf="$(cat /sys/devices/system/cpu/intel_pstate/min_perf_pct 2>/dev/null || echo unknown)"

if [ "$uniq_gov" = "performance" ]; then
  ok "CPU governor: performance (EPP: ${uniq_epp:-unknown}, no_turbo=$no_turbo, min_perf_pct=$min_perf)"
else
  warn "CPU governor is '${uniq_gov:-unknown}' (expected: performance; EPP: ${uniq_epp:-unknown})"
  add_remediation "# Pin CPU governor & EPP to performance (persists via cpupower.service):
sudo ./maintenance/cpu-power-state.sh apply-cpupower --yes"
fi

if [ "$no_turbo" != "0" ] && [ "$no_turbo" != "unknown" ]; then
  warn "intel_pstate no_turbo=$no_turbo (turbo boost is disabled!)"
  add_remediation "# Re-enable Intel turbo boost:
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo"
fi

if [ -r /sys/devices/cpu_core/cpus ] || [ -r /sys/devices/cpu_atom/cpus ]; then
  p_cpus="$(cat /sys/devices/cpu_core/cpus 2>/dev/null || echo n/a)"
  e_cpus="$(cat /sys/devices/cpu_atom/cpus 2>/dev/null || echo n/a)"
  info "Hybrid topology: P-cores=$p_cpus (cpu_range in llama-swap.yaml), E-cores=$e_cpus (ecores macro)"
else
  nproc_val="$(nproc 2>/dev/null || echo unknown)"
  info "CPU count: $nproc_val logical CPUs (hybrid sysfs nodes not present)"
fi

# ---------------------------------------------------------------------------
# 3. vm.swappiness
# ---------------------------------------------------------------------------
echo ""
echo "[3] vm.swappiness"
swappiness="$(cat /proc/sys/vm/swappiness 2>/dev/null || echo unknown)"
if [[ "$swappiness" =~ ^[0-9]+$ ]]; then
  if [ "$swappiness" -gt 20 ]; then
    warn "vm.swappiness=$swappiness (> 20; aggressive zram swapping can evict mmap'd model pages)"
    add_remediation "# Lower vm.swappiness to 10 (live + persistent across reboots):
sudo sysctl -w vm.swappiness=10
echo 'vm.swappiness = 10' | sudo tee /etc/sysctl.d/99-l3ms-inference.conf"
  else
    ok "vm.swappiness=$swappiness (<= 20)"
  fi
else
  info "vm.swappiness: $swappiness"
fi

# ---------------------------------------------------------------------------
# 4. Transparent Huge Pages (THP)
# ---------------------------------------------------------------------------
echo ""
echo "[4] Transparent Huge Pages (THP)"
thp_enabled="$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo unavailable)"
thp_defrag="$(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || echo unavailable)"
info "THP enabled: $thp_enabled"
info "THP defrag:  $thp_defrag"
if printf '%s\n' "$thp_enabled" | grep -q '\[always\]'; then
  info "Caveat: Strata uses 2 MiB hugepages (madvise/THP) for expert & KV tables. Before switching THP from [always] to [madvise] to cut khugepaged jitter, run a paired A/B benchmark on qwen38-flash-next-plat."
  add_remediation "# Optional (A/B test on Strata first due to 2 MiB hugepage usage):
echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled"
elif printf '%s\n' "$thp_enabled" | grep -q '\[madvise\]'; then
  ok "THP is [madvise] (explicit madvise 2 MiB hugepages active for Strata without global khugepaged stalls)."
else
  warn "THP is [never] — Strata 2 MiB hugepage allocations will fall back to 4 KiB pages (TLB miss penalty)."
  add_remediation "# Restore madvise THP so Strata 2 MiB hugepages work:
echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled"
fi

# ---------------------------------------------------------------------------
# 5. CachyOS Schedulers: ananicy-cpp & scx (sched_ext)
# ---------------------------------------------------------------------------
echo ""
echo "[5] Scheduler & Priority Daemons (ananicy-cpp / scx)"
ananicy_state="$(systemctl is-active ananicy-cpp.service 2>/dev/null || true)"
[ -z "$ananicy_state" ] && ananicy_state="not-found"
scx_svc_state="$(systemctl is-active scx.service 2>/dev/null || true)"
[ -z "$scx_svc_state" ] && scx_svc_state="not-found"
scx_kern_state="$(cat /sys/kernel/sched_ext/state 2>/dev/null || echo unavailable)"
scx_ops="$(cat /sys/kernel/sched_ext/root/ops 2>/dev/null || echo none)"

if [ "$ananicy_state" = "active" ]; then
  warn "ananicy-cpp.service is active (may override --prio 2 / nice levels on inference workers)."
  add_remediation "# Optional: pause ananicy-cpp during benchmarks or add a rule for llama-server/strata:
sudo systemctl stop ananicy-cpp.service
# (Re-enable with: sudo systemctl start ananicy-cpp.service)"
else
  ok "ananicy-cpp.service is $ananicy_state."
fi

info "scx.service: $scx_svc_state | kernel sched_ext: $scx_kern_state (ops: $scx_ops)"

# ---------------------------------------------------------------------------
# 6. NVIDIA Driver & Free VRAM
# ---------------------------------------------------------------------------
echo ""
echo "[6] NVIDIA GPU & VRAM"
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_csv="$(nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free,memory.total,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null || true)"
  if [ -n "$gpu_csv" ]; then
    while IFS= read -r line; do
      info "$line"
    done <<< "$gpu_csv"
  else
    warn "nvidia-smi returned no GPU rows."
  fi
else
  warn "nvidia-smi not found on PATH."
fi

# ---------------------------------------------------------------------------
# 7. Local Disk Usage (df -hlT: local filesystems only, never touches NFS)
# ---------------------------------------------------------------------------
echo ""
echo "[7] Local Disk Usage (df -hlT, key mounts)"
df_out="$(df -hlT 2>/dev/null || true)"
printf '%s\n' "$df_out" | awk 'NR==1 || ($7 == "/" || $7 == "/home" || $7 == "/mnt/dev" || $7 == "/mnt/media" || $7 == "/boot") { printf "  %s\n", $0 }'
while read -r fs fstype size used avail pct mnt; do
  pct_num="${pct%%%}"
  if [[ "$pct_num" =~ ^[0-9]+$ ]] && [ "$pct_num" -ge 90 ]; then
    warn "Mount $mnt ($fs) is at ${pct} used (${avail} free)."
  fi
done < <(printf '%s\n' "$df_out" | awk 'NR>1 && ($7 == "/" || $7 == "/home" || $7 == "/mnt/dev" || $7 == "/mnt/media" || $7 == "/boot") { print $1, $2, $3, $4, $5, $6, $7 }')

# ---------------------------------------------------------------------------
# Summary & Optional Remediation Output
# ---------------------------------------------------------------------------
echo ""
echo "=== Summary: $WARNINGS warning(s) (read-only audit; no host changes made) ==="

if [ "$PRINT_REMEDIATION" -eq 1 ]; then
  echo ""
  echo "=== Suggested Remediation Commands (copy-paste manually; not executed) ==="
  if [ "${#REMEDIATIONS[@]}" -eq 0 ]; then
    echo "  No remediation commands needed."
  else
    for cmd in "${REMEDIATIONS[@]}"; do
      printf '\n%s\n' "$cmd"
    done
  fi
elif [ "$WARNINGS" -gt 0 ]; then
  echo "Tip: re-run with --print-remediation for exact copy-pasteable sysctl/systemctl commands."
fi

exit 0
