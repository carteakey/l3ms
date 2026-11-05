#!/usr/bin/env bash
# bench-env.sh — Standard system environment capture & preflight checks for benchmarks.
#
# Can be run directly or sourced by runners (run-llama-bench.sh, run-ik-llama-bench.sh,
# run-llama-fit-bench.sh, and standalone campaign scripts).
#
# Direct usage:
#   ./bench-models/bench-env.sh               # Run preflight warnings and print YAML frontmatter
#   ./bench-models/bench-env.sh --frontmatter # Print YAML frontmatter only
#   ./bench-models/bench-env.sh --json        # Print sysinfo JSON object only
#   ./bench-models/bench-env.sh --warn        # Print preflight warnings only
#
# Sourced usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/bench-env.sh"
#   bench_preflight_warn                      # Warns on non-optimal CPU/GPU/memory states
#   bench_emit_frontmatter                    # Emits YAML frontmatter (with embedded JSON)
#   bench_capture_sysinfo_json                # Emits JSON object of system state

bench_collect_python() {
  local action="${1:-all}"
  python3 - "$action" <<'PYEOF'
import os, sys, json, subprocess, re, glob

def collect_sysinfo():
    info = {}

    # Kernel & Host Session
    info["kernel"] = os.uname().release
    session = "unknown"
    try:
        if subprocess.run(["systemctl", "is-active", "--quiet", "graphical.target"]).returncode == 0:
            session = "graphical"
        elif subprocess.run(["systemctl", "is-active", "--quiet", "multi-user.target"]).returncode == 0:
            session = "multi-user (headless)"
    except Exception:
        pass
    info["session"] = session

    # Router state
    router_active = False
    try:
        router_active = (subprocess.run(["systemctl", "--user", "is-active", "--quiet", "llama-swap.service"]).returncode == 0)
    except Exception:
        pass
    info["llama_swap_active"] = router_active

    # CPU info
    cpu_model = "unknown"
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    info["cpu_model"] = cpu_model

    def read_sysfs(path):
        try:
            with open(path, "r") as f:
                return f.read().strip()
        except Exception:
            return None

    info["cpu_governor"] = read_sysfs("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") or "unknown"
    info["cpu_epp"] = read_sysfs("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference") or "unknown"
    # intel_pstate/HWP layer.  On Intel's active-mode driver the governor name is a label
    # (powersave == the HWP governor intel_cpufreq), so scaling_governor + EPP alone do not say
    # whether the cores hold max or ramp to it.  min_perf_pct is the floor HWP may not go under,
    # and 17 is the kernel default: a bench load ramps instead of starting at the ceiling.
    # Measured on yeti-cachy 2026-10-29: governor powersave, EPP performance, min_perf_pct 17.
    info["hwp_status"] = read_sysfs("/sys/devices/system/cpu/intel_pstate/status") or "unknown"
    info["hwp_min_perf_pct"] = read_sysfs("/sys/devices/system/cpu/intel_pstate/min_perf_pct")
    info["hwp_max_perf_pct"] = read_sysfs("/sys/devices/system/cpu/intel_pstate/max_perf_pct")
    info["hwp_no_turbo"] = read_sysfs("/sys/devices/system/cpu/intel_pstate/no_turbo")
    info["hwp_dynamic_boost"] = read_sysfs("/sys/devices/system/cpu/intel_pstate/hwp_dynamic_boost")
    gov_all = set()
    try:
        for p in glob.glob("/sys/devices/system/cpu/cpufreq/policy*"):
            g = read_sysfs(os.path.join(p, "scaling_governor"))
            if g:
                gov_all.add(g)
    except Exception:
        pass
    for key in ("hwp_min_perf_pct", "hwp_max_perf_pct", "hwp_no_turbo", "hwp_dynamic_boost"):
        try:
            info[key] = int(info[key])
        except (TypeError, ValueError):
            pass

    info["cpu_governors"] = ",".join(sorted(gov_all)) if gov_all else "unknown"

    # Power Profile (power-profiles-daemon via D-Bus / busctl)
    ppd = "unknown"
    try:
        out = subprocess.getoutput("busctl get-property net.hadess.PowerProfiles /net/hadess/PowerProfiles net.hadess.PowerProfiles ActiveProfile 2>/dev/null")
        m = re.search(r'"([^"]+)"', out)
        if m:
            ppd = m.group(1)
    except Exception:
        pass
    info["power_profile"] = ppd

    # CPU Temperature
    cpu_temp = None
    try:
        tz = read_sysfs("/sys/class/thermal/thermal_zone1/temp")
        if tz:
            cpu_temp = round(int(tz) / 1000.0, 1)
        else:
            sensors_out = subprocess.getoutput("sensors 2>/dev/null | awk '/Package id 0:/{print $4}'")
            if sensors_out:
                cpu_temp = float(sensors_out.replace("+", "").replace("°C", "").strip())
    except Exception:
        pass
    info["cpu_temp_c"] = cpu_temp

    # RAM speed and topology (via inxi -c 0 -m)
    ram_speed = None
    ram_topology = None
    try:
        inxi_out = subprocess.getoutput("inxi -c 0 -m 2>/dev/null")
        m_speed = re.search(r'actual:\s*(\d+)\s*MT/s', inxi_out)
        if m_speed:
            ram_speed = int(m_speed.group(1))

        m_mods = re.search(r'modules:\s*(\d+)', inxi_out)
        m_type = re.search(r'type:\s*(\S+)', inxi_out)
        m_sz = re.search(r'size:\s*(\d+\s*GiB)', inxi_out)
        if m_mods and m_type and m_sz:
            ram_topology = f"{m_mods.group(1)}x{m_sz.group(1)} {m_type.group(1)}"
    except Exception:
        pass
    info["ram_speed_mts"] = ram_speed
    info["ram_topology"] = ram_topology

    # Memory details from /proc/meminfo
    try:
        mem = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    k = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    if val.isdigit():
                        mem[k] = int(val)
        info["mem_total_gib"] = round(mem.get("MemTotal", 0) / 1048576, 2)
        info["mem_free_gib"] = round(mem.get("MemFree", 0) / 1048576, 2)
        info["mem_available_gib"] = round(mem.get("MemAvailable", 0) / 1048576, 2)
        info["mem_active_anon_gib"] = round(mem.get("Active(anon)", 0) / 1048576, 2)
        swap_used_kb = mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)
        info["zram_used_mib"] = round(swap_used_kb / 1024, 2)
    except Exception:
        pass

    info["thp"] = read_sysfs("/sys/kernel/mm/transparent_hugepage/enabled") or "unknown"

    # GPU details via nvidia-smi
    try:
        nv_out = subprocess.getoutput("nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,pcie.link.gen.current,pcie.link.gen.max --format=csv,noheader,nounits 2>/dev/null")
        if nv_out and "NVIDIA" in nv_out:
            fields = [x.strip() for x in nv_out.split(",")]
            if len(fields) >= 8:
                info["gpu_model"] = fields[0]
                info["gpu_driver"] = fields[1]
                info["vram_total_mib"] = int(fields[2])
                info["vram_free_baseline_mib"] = int(fields[3])
                info["vram_used_baseline_mib"] = int(fields[4])
                info["gpu_temp_c"] = float(fields[5])
                info["pcie_link"] = f"Gen{fields[6]} (max Gen{fields[7]})"
    except Exception:
        pass

    return info

def to_frontmatter(info):
    lines = [
        "---",
        "# System Environment Frontmatter (l3ms bench standard)",
        f"# bench_sys_json: {json.dumps(info, separators=(',', ':'))}",
        "sys_env:",
        f"  kernel: {json.dumps(info.get('kernel'))}",
        f"  session: {json.dumps(info.get('session'))}",
        f"  llama_swap_active: {json.dumps(info.get('llama_swap_active'))}",
        "cpu:",
        f"  model: {json.dumps(info.get('cpu_model'))}",
        f"  governor: {json.dumps(info.get('cpu_governor'))}",
        f"  governors_all: {json.dumps(info.get('cpu_governors'))}",
        f"  epp: {json.dumps(info.get('cpu_epp'))}",
        f"  hwp_status: {json.dumps(info.get('hwp_status'))}",
        f"  hwp_min_perf_pct: {json.dumps(info.get('hwp_min_perf_pct'))}",
        f"  hwp_max_perf_pct: {json.dumps(info.get('hwp_max_perf_pct'))}",
        f"  hwp_no_turbo: {json.dumps(info.get('hwp_no_turbo'))}",
        f"  hwp_dynamic_boost: {json.dumps(info.get('hwp_dynamic_boost'))}",
        f"  power_profile: {json.dumps(info.get('power_profile'))}",
        f"  temp_c: {info.get('cpu_temp_c')}",
        "memory:",
        f"  ram_speed_mts: {info.get('ram_speed_mts')}",
        f"  ram_topology: {json.dumps(info.get('ram_topology'))}",
        f"  mem_total_gib: {info.get('mem_total_gib')}",
        f"  mem_free_gib: {info.get('mem_free_gib')}",
        f"  mem_available_gib: {info.get('mem_available_gib')}",
        f"  mem_active_anon_gib: {info.get('mem_active_anon_gib')}",
        f"  zram_used_mib: {info.get('zram_used_mib')}",
        f"  thp: {json.dumps(info.get('thp'))}",
        "gpu:",
        f"  model: {json.dumps(info.get('gpu_model'))}",
        f"  driver: {json.dumps(info.get('gpu_driver'))}",
        f"  vram_total_mib: {info.get('vram_total_mib')}",
        f"  vram_free_baseline_mib: {info.get('vram_free_baseline_mib')}",
        f"  vram_used_baseline_mib: {info.get('vram_used_baseline_mib')}",
        f"  temp_c: {info.get('gpu_temp_c')}",
        f"  pcie_link: {json.dumps(info.get('pcie_link'))}",
        "  # pcie_link is sampled once, at harness start. An idle card reports Gen1 through ASPM",
        "  # downclocking (measured 2026-10-28: Gen1 sampled after the handoff, Gen4 16x with the model",
        "  # loaded), so re-check with `nvidia-smi -q | grep -A6 \"GPU Link Info\"` while a model is on the",
        "  # card before reading a low link gen as the reason an expert-streaming arm lost.",
        "---",
    ]
    return "\n".join(lines)

def run_preflight_checks(info):
    warnings = []
    if info.get("cpu_governor") != "performance":
        warnings.append(f"CPU governor is '{info.get('cpu_governor')}' (expected 'performance'). Decode t/s may be degraded by ~15-25%.")
        warnings.append("On intel_pstate active mode the governor is whatever the kernel left there; nothing on this box writes it. Set it with 'sudo ./maintenance/cpu-power-state.sh apply-cpupower --yes' (or the tuned route in docs/bench-runbook.md), then re-run the paired control before quoting any older row.")
    if info.get("cpu_epp") not in ("performance", "unknown"):
        warnings.append(f"CPU EPP is '{info.get('cpu_epp')}' (expected 'performance'). Speculative verification round latency may increase.")
    if info.get("hwp_status") == "active" and info.get("hwp_min_perf_pct") not in (None, 100) and info.get("cpu_governor") != "performance":
        warnings.append(f"intel_pstate min_perf_pct is {info.get('hwp_min_perf_pct')} while the governor is not performance: the cores ramp to max under load instead of holding it, so short probes read low.")
    if info.get("power_profile") not in ("performance", "unknown"):
        warnings.append(f"Power profile daemon profile is '{info.get('power_profile')}' (expected 'performance').")
    if info.get("llama_swap_active"):
        warnings.append("llama-swap.service is active. Stop with 'systemctl --user stop llama-swap.service' for full-VRAM isolation.")
    if (info.get("vram_used_baseline_mib") or 0) > 600:
        warnings.append(f"High baseline VRAM usage ({info.get('vram_used_baseline_mib')} MiB used). GPU memory may be constrained during bench.")
    if (info.get("mem_available_gib") or 0) < 30.0:
        warnings.append(f"Low available RAM ({info.get('mem_available_gib')} GiB). Large MoE models may experience SSD expert refaulting/spill.")
    if (info.get("cpu_temp_c") or 0) > 80.0:
        warnings.append(f"High CPU temperature ({info.get('cpu_temp_c')}°C). Potential thermal throttling.")
    if (info.get("gpu_temp_c") or 0) > 80.0:
        warnings.append(f"High GPU temperature ({info.get('gpu_temp_c')}°C). Potential thermal throttling.")
    return warnings

action = sys.argv[1] if len(sys.argv) > 1 else "all"
info = collect_sysinfo()

if action == "json":
    print(json.dumps(info))
elif action == "frontmatter":
    print(to_frontmatter(info))
elif action == "warn":
    for w in run_preflight_checks(info):
        print(f"[WARN] {w}", file=sys.stderr)
elif action == "all":
    for w in run_preflight_checks(info):
        print(f"[WARN] {w}", file=sys.stderr)
    print(to_frontmatter(info))
PYEOF
}

bench_preflight_warn() {
  bench_collect_python warn
}

bench_emit_frontmatter() {
  bench_collect_python frontmatter
}

bench_capture_sysinfo_json() {
  bench_collect_python json
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  case "${1:-}" in
    --json)
      bench_capture_sysinfo_json
      ;;
    --frontmatter)
      bench_emit_frontmatter
      ;;
    --warn)
      bench_preflight_warn
      ;;
    --help|-h)
      echo "Usage: $0 [--frontmatter | --json | --warn]"
      ;;
    *)
      bench_collect_python all
      ;;
  esac
fi
