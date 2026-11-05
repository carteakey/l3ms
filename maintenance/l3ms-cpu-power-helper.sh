#!/usr/bin/env bash
# l3ms-cpu-power - the root-owned half of maintenance/cpu-power-state.sh.
#
# Repo copy: maintenance/l3ms-cpu-power-helper.sh
# Installed copy: /usr/local/bin/l3ms-cpu-power, owned root:root, mode 0755.
#
# Why a copy at all: a sudoers grant that names a file inside the repo is a grant on root, because
# the agent can edit that file and then sudo-run it. Installing a root-owned copy, and granting
# exactly that path with a fixed verb list, is what makes NOPASSWD narrow instead of decorative.
# Every value this script writes is a literal in this file: no arguments are interpolated into a
# file, and the tuned profile is embedded here rather than read out of the repo, because a tuned
# profile can write [sysctl] and [bootloader] state and must not be agent-supplied.
#
# Install (one time, your password):
#   sudo install -m 0755 -o root -g root maintenance/l3ms-cpu-power-helper.sh /usr/local/bin/l3ms-cpu-power
#   sudo install -m 0440 -o root -g root maintenance/sudoers/l3ms-cpu-power /etc/sudoers.d/l3ms-cpu-power
#   sudo visudo -cf /etc/sudoers.d/l3ms-cpu-power
#
# Then an agent can do this without you:
#   sudo /usr/local/bin/l3ms-cpu-power apply
#   sudo /usr/local/bin/l3ms-cpu-power status
# and this only after it has been installed:
#   sudo /usr/local/bin/l3ms-cpu-power tuned
#   sudo /usr/local/bin/l3ms-cpu-power revert

set -uo pipefail

CONF="/etc/default/cpupower-service.conf"
STAMP="$(date +%Y%m%d-%H%M%S)"

gov_list() { for p in /sys/devices/system/cpu/cpufreq/policy*; do printf '%s ' "$(cat "$p/scaling_governor" 2>/dev/null)"; done; }
epp_list() { for p in /sys/devices/system/cpu/cpufreq/policy*; do printf '%s ' "$(cat "$p/energy_performance_preference" 2>/dev/null)"; done; }
uniq_words() { tr ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ' '; }

status() {
  echo "governor: $(gov_list | uniq_words)"
  echo "epp:      $(epp_list | uniq_words)"
  echo "intel_pstate: status=$(cat /sys/devices/system/cpu/intel_pstate/status 2>/dev/null) min_perf_pct=$(cat /sys/devices/system/cpu/intel_pstate/min_perf_pct 2>/dev/null) max_perf_pct=$(cat /sys/devices/system/cpu/intel_pstate/max_perf_pct 2>/dev/null) no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)"
  echo "cpupower.service: $(systemctl is-enabled cpupower.service 2>/dev/null) / $(systemctl is-active cpupower.service 2>/dev/null)"
  echo "conf active lines: $(grep -Ec '^[A-Z_]+=' "$CONF" 2>/dev/null || true)"
}

apply() {
  # Literal values only. The packaged helper (/usr/lib/systemd/scripts/cpupower) quotes the
  # variables it reads from this file, so a value here is an argument to cpupower, never a shell
  # word, and nothing a caller passes can reach the file.
  local tmp; tmp="$(mktemp)"
  if [ -f "$CONF" ]; then
    cp -a "$CONF" "$CONF.bak-$STAMP"
    echo "snapshot: $CONF -> $CONF.bak-$STAMP"
    grep -vE '^#?(GOVERNOR|MIN_FREQ|MAX_FREQ|FREQ|PERF_BIAS|EPP)=' "$CONF" > "$tmp" || true
  fi
  {
    printf '\n# Set by /usr/local/bin/l3ms-cpu-power %s\n' "$STAMP"
    printf "GOVERNOR='performance'\n"
    printf "EPP='performance'\n"
  } >> "$tmp"
  cat "$tmp" > "$CONF"
  rm -f "$tmp"
  chmod 0644 "$CONF"
  systemctl enable --now cpupower.service
  echo "applied:"
  status
}

# Only if tuned is ever installed. The profile text is embedded (see the reasoning above).
tuned() {
  if ! command -v tuned-adm >/dev/null; then
    echo "tuned is not installed. cachyos/tuned-cachy + cachyos/tuned-cachy-ppd, or extra/tuned + extra/tuned-ppd."
    return 1
  fi
  mkdir -p /etc/tuned/l3ms-bench
  if [ -f /etc/tuned/l3ms-bench/tuned.conf ]; then
    cp -a /etc/tuned/l3ms-bench/tuned.conf "/etc/tuned/l3ms-bench/tuned.conf.bak-$STAMP"
    echo "snapshot: /etc/tuned/l3ms-bench/tuned.conf -> tuned.conf.bak-$STAMP"
  fi
  cat > /etc/tuned/l3ms-bench/tuned.conf <<'CONF'
[main]
summary=L3MS bench state: governor performance, EPP performance, HWP floor pinned at 100%

[cpu]
governor=performance
energy_perf_bias=performance
min_perf_pct=100
energy_performance_preference=performance
boost=1

[vm]
transparent_hugepages=always
CONF
  chmod 0644 /etc/tuned/l3ms-bench/tuned.conf
  tuned-adm profile l3ms-bench
  systemctl mask power-profiles-daemon.service
  systemctl stop power-profiles-daemon.service 2>/dev/null || true
  echo "applied tuned profile l3ms-bench (embedded [cpu] block, no [acpi] platform_profile on this firmware):"
  status
}

revert() {
  if command -v tuned-adm >/dev/null; then
    tuned-adm off || true
    systemctl unmask power-profiles-daemon.service || true
  fi
  systemctl disable --now cpupower.service || true
  local latest
  latest="$(ls -1t "$CONF".bak-* 2>/dev/null | head -1)"
  if [ -n "$latest" ]; then
    cat "$latest" > "$CONF"
    echo "restored $latest -> $CONF"
  else
    echo "no $CONF.bak-* snapshot to restore"
  fi
  echo "reverted; the governor returns to intel_pstate's powersave default at the next write:"
  status
}

case "${1:-status}" in
  status) status ;;
  apply) apply ;;
  tuned) tuned ;;
  revert) revert ;;
  *) echo "usage: l3ms-cpu-power [status|apply|tuned|revert]"; exit 2 ;;
esac
