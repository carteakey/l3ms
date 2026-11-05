#!/usr/bin/env bash
# cpu-power-state.sh - put this box in the CPU power state a bench run needs, from a script.
#
# Ground truth on yeti-cachy (measured 2026-10-29):
#   - intel_pstate active, HWP enabled.  scaling_governor = powersave on all 16 logical CPUs.
#     "powersave" here is intel_cpufreq, the HWP governor: it is the kernel's default for
#     active mode, and nothing on this box writes anything else.
#   - energy_performance_preference = performance (so the EPP check looks clean while the
#     governor does not), intel_pstate min_perf_pct = 17, max_perf_pct = 100, no_turbo = 0,
#     hwp_dynamic_boost = 0.  The 17 is the point: the cores ramp to max instead of holding it.
#   - power-profiles-daemon 0.30 is installed and its ActiveProfile reads "performance",
#     so PPD is not what holds the governor.  intel-lpmd 0.1.1-2 is installed (ran at boot,
#     now inactive).  cpupower.service exists (cpupower 7.2.11-1) but is disabled and
#     /etc/default/cpupower-service.conf is untouched (all options commented).
#   - tuned is NOT installed: pacman -Q tuned -> not found, no /etc/tuned, no tuned-adm.
#     docs/bench-runbook.md's "sudo tuned-adm profile throughput-performance" line is the
#     CachyOS-recommended route, not a description of this box.
#
# Consequence for this repo's numbers: every Strata and llama.cpp t/s row recorded so far was
# measured with governor=powersave and min_perf_pct=17.  Compare runs against runs in the same
# state, and re-baseline the paired control after any change here.
#
# Modes (writes are dry-run unless --yes; every write snapshots its target first):
#   show                      print the whole power layering, services included
#   check                     exit 0 if a bench may quote t/s, 1 with the reasons (bench gate)
#   apply-cpupower --yes     no new packages: /etc/default/cpupower-service.conf + cpupower.service
#   install-tuned            print the pacman line (CachyOS ships tuned-cachy / tuned-cachy-ppd)
#   apply-tuned --yes [--ppd]  install maintenance/tuned/l3ms-bench as a tuned profile and select it
#   revert                    print the commands that undo either route
#
# Both apply routes need root.  Two ways to get it:
#   sudo ./maintenance/cpu-power-state.sh apply-cpupower --yes          # you type the password
#   sudo /usr/local/bin/l3ms-cpu-power apply                            # NOPASSWD, see maintenance/sudoers/
# The second one is a root-owned installed copy of maintenance/l3ms-cpu-power-helper.sh.  It exists
# because a sudoers grant on a file inside the repo is a grant on root (the agent can edit the file it
# is allowed to sudo-run), and because verbs like `systemctl *` are not restrictable: `systemctl edit`
# writes a drop-in that systemd then executes as root.  When the helper is installed this script prints
# that command instead of asking for a password, and exits 4.
#
# Modes (writes are dry-run unless --yes; every write snapshots its target first):
#   show                      print the whole power layering, services included
#   check                     exit 0 if a bench may quote t/s, 1 with the reasons (bench gate)
#   apply-cpupower --yes     no new packages: /etc/default/cpupower-service.conf + cpupower.service
#   install-tuned            print the pacman line (CachyOS ships tuned-cachy / tuned-cachy-ppd)
#   apply-tuned --yes [--ppd]  install maintenance/tuned/l3ms-bench as a tuned profile and select it
#   revert                    print the commands that undo either route
#
# Exit codes: 0 ok, 1 check failed or apply failed, 2 usage, 4 needs root (delegated command printed).

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_SRC="$REPO/maintenance/tuned/l3ms-bench/tuned.conf"
PROFILE_DIR="/etc/tuned/l3ms-bench"
CPUPOWER_CONF="/etc/default/cpupower-service.conf"
HELPER="/usr/local/bin/l3ms-cpu-power"
STAMP="$(date +%Y%m%d-%H%M%S)"

YES=0
WITH_PPD=0
MODE="${1:-show}"
for a in "$@"; do
  [ "$a" = "--yes" ] && YES=1
  [ "$a" = "--ppd" ] && WITH_PPD=1
done

snap() { # snapshot a file before overwriting it, if it exists and is not all-comments-default
  local f="$1"
  if [ -f "$f" ]; then
    cp -a "$f" "$f.bak-$STAMP"
    echo "   snapshot: $f -> $f.bak-$STAMP"
  fi
}

gov_list() {
  for p in /sys/devices/system/cpu/cpufreq/policy*; do
    printf '%s ' "$(cat "$p/scaling_governor" 2>/dev/null)"
  done
}

epp_list() {
  for p in /sys/devices/system/cpu/cpufreq/policy*; do
    printf '%s ' "$(cat "$p/energy_performance_preference" 2>/dev/null)"
  done
}

uniq_words() { tr ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ' '; }

pstate() { printf '%s=%s ' "$1" "$(cat "/sys/devices/system/cpu/intel_pstate/$1" 2>/dev/null || echo none)"; }

ppd_profile() {
  # busctl prints `s "performance"`; the value is the quoted token, not the first word.
  busctl --no-pager get-property net.hadess.PowerProfiles /net/hadess/PowerProfiles \
    net.hadess.PowerProfiles ActiveProfile 2>/dev/null | sed -n 's/.*"\([^"]*\)".*/\1/p' | head -1
}

count_active() { # grep -c prints 0 and exits 1 when nothing matched; keep one number
  local n
  n="$(grep -Ec '^[A-Z_]+=' "$1" 2>/dev/null)" || true
  printf '%s' "${n:-0}"
}

delegate() { # not root: hand over the NOPASSWD helper when installed, otherwise the sudo line
  local verb="$1" mode="$2"
  if [ -x "$HELPER" ]; then
    echo "not root. the installed grant runs this without a password:"
    echo "  sudo $HELPER $verb"
  else
    echo "not root and $HELPER is not installed, so this needs your password:"
    echo "  sudo $0 $mode --yes"
    echo "to stop being asked next time: see maintenance/sudoers/l3ms-cpu-power and maintenance/l3ms-cpu-power-helper.sh"
  fi
}

show() {
  echo "== cpufreq layer (per policy, 16 logical CPUs)"
  echo "   governor: $(gov_list | uniq_words)"
  echo "   epp:      $(epp_list | uniq_words)"
  echo "   available governors: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors 2>/dev/null)"
  echo "   available epp:      $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_available_preferences 2>/dev/null)"
  echo "   freq: min=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq 2>/dev/null) max=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null) cur0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null) avg0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_avg_freq 2>/dev/null)"
  echo "== intel_pstate (HWP)"
  echo "   $(pstate status; pstate min_perf_pct; pstate max_perf_pct; pstate no_turbo; pstate hwp_dynamic_boost)"
  echo "== profile managers"
  echo "   power-profiles-daemon: $(systemctl is-active power-profiles-daemon.service 2>/dev/null) profile=$(ppd_profile 2>/dev/null || echo none)"
  echo "   tuned:                 $(command -v tuned-adm >/dev/null && systemctl is-active tuned.service 2>/dev/null || echo 'not installed')"
  echo "   cpupower.service:      $(systemctl is-enabled cpupower.service 2>/dev/null) / $(systemctl is-active cpupower.service 2>/dev/null)"
  echo "   intel-lpmd:            $(systemctl is-active intel-lpmd.service 2>/dev/null)"
  echo "   cpupower conf:         $(count_active "$CPUPOWER_CONF") active setting lines ($( [ -f "$CPUPOWER_CONF" ] && echo packaged defaults, all commented || echo file absent ))"
  echo "== bench gate verdict"
  check
}

check() {
  local bad=0 g e minp perf_gov
  g="$(gov_list | uniq_words)"
  e="$(epp_list | uniq_words)"
  minp="$(cat /sys/devices/system/cpu/intel_pstate/min_perf_pct 2>/dev/null || echo none)"
  [ "$g" = "performance " ] && perf_gov=1 || perf_gov=0
  if [ "$perf_gov" -eq 0 ]; then
    echo "   governor: $g (expected: performance)  <- intel_pstate active-mode default, nothing writes it"
    bad=1
  fi
  if [ "$e" != "performance " ]; then
    echo "   epp: $e (expected: performance)"
    bad=1
  fi
  # min_perf_pct is the HWP floor.  With the powersave (HWP) governor a floor of 17 means the
  # cores ramp under a bench load.  With governor=performance the floor line is context, not a
  # verdict: tuned's profile writes min_perf_pct=100 explicitly, `cpupower frequency-set -g
  # performance` may leave the global line at its previous value, and which one actually pins
  # the cores has to be read off the numbers, not assumed.  It is reported either way.
  if [ "$perf_gov" -eq 1 ] && [ "$minp" != "100" ]; then
    echo "   note: governor=performance but intel_pstate min_perf_pct is $minp; check the row's tg probes for a ramp before trusting it."
  elif [ "$perf_gov" -eq 0 ] && [ "$minp" != "100" ]; then
    echo "   intel_pstate min_perf_pct: $minp (a bench state wants 100; 17 means the cores ramp instead of holding max)"
    bad=1
  fi
  if [ "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)" != "0" ]; then
    echo "   intel_pstate no_turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null) (expected: 0)"
    bad=1
  fi
  if [ "$bad" -eq 0 ]; then
    echo "   governor performance, EPP performance, no_turbo 0, min_perf_pct $minp: a t/s row taken now is comparable to other runs in this state."
    return 0
  fi
  echo "   fix it first: sudo $REPO/maintenance/cpu-power-state.sh apply-cpupower --yes  (no new packages)"
  [ -x "$HELPER" ] && echo "   or, with the installed grant: sudo $HELPER apply"
  echo "   or the tuned route: sudo $REPO/maintenance/cpu-power-state.sh install-tuned, then apply-tuned --yes"
  echo "   either way: the paired control has to be re-run in the new state before any older row is quoted."
  return 1
}

apply_cpupower() {
  # The distro unit already exists and reads this EnvironmentFile; the packaged helper
  # /usr/lib/systemd/scripts/cpupower turns GOVERNOR into `cpupower frequency-set -g` and
  # EPP into `cpupower set -e`, applied at boot (cpupower.service, WantedBy multi-user.target).
  local lines=("GOVERNOR='performance'" "EPP='performance'")
  echo "plan: write to $CPUPOWER_CONF"
  for l in "${lines[@]}"; do echo "   $l"; done
  echo "then: systemctl enable --now cpupower.service"
  if [ "$YES" -ne 1 ]; then echo "dry run (add --yes). nothing written."; return 0; fi
  snap "$CPUPOWER_CONF"
  local tmp; tmp="$(mktemp)"
  grep -E '^[A-Z_]+=' "$CPUPOWER_CONF" 2>/dev/null | grep -Ev '^(GOVERNOR|EPP)=' > "$tmp" || true
  # Keep the packaged header comments, drop commented option lines we are now setting.
  grep -vE '^#?(GOVERNOR|MIN_FREQ|MAX_FREQ|FREQ|PERF_BIAS|EPP)=' "$CPUPOWER_CONF" 2>/dev/null >> "$tmp" || true
  printf '\n# Set by l3ms maintenance/cpu-power-state.sh %s\n' "$STAMP" >> "$tmp"
  for l in "${lines[@]}"; do printf '%s\n' "$l" >> "$tmp"; done
  cat "$tmp" > "$CPUPOWER_CONF"
  rm -f "$tmp"
  systemctl enable --now cpupower.service && systemctl restart cpupower.service || return 1
  echo "== state after apply"
  echo "   governor: $(gov_list | uniq_words)"
  echo "   epp:      $(epp_list | uniq_words)"
  echo "   min_perf_pct: $(cat /sys/devices/system/cpu/intel_pstate/min_perf_pct 2>/dev/null)"
}

install_tuned() {
  cat <<'TXT'
CachyOS ships its own tuned in the cachyos repo (checked 2026-10-29 against the enabled repos):

  cachyos/tuned-cachy       2.27.0-3   tuned itself
  cachyos/tuned-cachy-ppd   2.27.0-3   PPD-compatible D-Bus front end, so KDE/GNOME's power
                                        slider keeps working and drives tuned instead

Upstream also has extra/tuned and extra/tuned-ppd at 2.28.0-1 if you would rather track
upstream. Pick one pair, not both:

  sudo pacman -S --noconfirm tuned-cachy tuned-cachy-ppd   # CachyOS build, keeps the slider
  sudo pacman -S --noconfirm tuned tuned-ppd               # upstream build

then:

  sudo ./maintenance/cpu-power-state.sh apply-tuned --yes

tuned-ppd replaces power-profiles-daemon's D-Bus names; if you install plain tuned without
-tuned-ppd, mask the daemon so it cannot write the profile back at the next session start:

  sudo systemctl mask power-profiles-daemon.service
TXT
}

apply_tuned() {
  if ! command -v tuned-adm >/dev/null; then
    echo "tuned is not installed. Run: sudo $REPO/maintenance/cpu-power-state.sh install-tuned"
    return 1
  fi
  if [ ! -f "$PROFILE_SRC" ]; then
    echo "missing $PROFILE_SRC"; return 1
  fi
  echo "plan: install $PROFILE_SRC as $PROFILE_DIR/tuned.conf, then tuned-adm profile l3ms-bench"
  if [ "$WITH_PPD" -eq 1 ]; then echo "      and mask power-profiles-daemon.service"; fi
  if [ "$YES" -ne 1 ]; then echo "dry run (add --yes). nothing written."; return 0; fi
  mkdir -p "$PROFILE_DIR"
  snap "$PROFILE_DIR/tuned.conf"
  install -m 0644 "$PROFILE_SRC" "$PROFILE_DIR/tuned.conf"
  tuned-adm profile l3ms-bench || return 1
  if [ "$WITH_PPD" -eq 1 ]; then systemctl mask power-profiles-daemon.service && systemctl stop power-profiles-daemon.service; fi
  echo "== state after apply"
  echo "   tuned active: $(tuned-adm active 2>&1 | head -1)"
  echo "   governor: $(gov_list | uniq_words)"
  echo "   epp:      $(epp_list | uniq_words)"
  echo "   min_perf_pct: $(cat /sys/devices/system/cpu/intel_pstate/min_perf_pct 2>/dev/null)"
}

revert() {
  cat <<'TXT'
tuned route:
  sudo tuned-adm off                       # tuned drops the profile (settings stay applied until the next writer)
  sudo systemctl disable --now tuned.service
  sudo rm -rf /etc/tuned/l3ms-bench
  sudo systemctl unmask power-profiles-daemon.service   # if you masked it
  sudo systemctl --user restart power-profiles-daemon.service 2>/dev/null || true

cpupower route:
  sudo systemctl disable --now cpupower.service
  restore the snapshot this script wrote: /etc/default/cpupower-service.conf.bak-<stamp>

Either way the governor returns to intel_pstate's powersave default, which is the state most
of this repo's recorded t/s rows were measured in.
TXT
}

case "$MODE" in
  show) show ;;
  check) check ;;
  apply-cpupower) [ "$(id -u)" -eq 0 ] || { delegate apply apply-cpupower; exit 4; }; apply_cpupower "${@:2}" ;;
  install-tuned) install_tuned ;;
  apply-tuned) [ "$(id -u)" -eq 0 ] || { delegate tuned apply-tuned; exit 4; }; apply_tuned "${@:2}" ;;
  revert) revert ;;
  *) echo "usage: $0 [show|check|apply-cpupower --yes|install-tuned|apply-tuned --yes [--ppd]|revert]"; exit 2 ;;
esac
