"""Live hardware and host-tuning telemetry collector for the L3MS local dashboard.

Stdlib-only module. All disk checks strictly use `df -hlT` (local filesystems
only) so unreachable NFS mounts never block the server.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Callable, Iterable


DEFAULT_P_CORES = tuple(range(0, 12))
DEFAULT_E_CORES = tuple(range(12, 16))
DEFAULT_DF_TARGETS = ("/", "/home", "/mnt/dev")

# Inference process signatures to watch for NVMe spill via /proc/<pid>/io.
INFERENCE_CMD_PATTERNS = (
    ("polystrata", re.compile(r"(?:^|/)(?:strata-poly)(?:\s|$)|polystrata/terminal/serve/server\.py")),
    ("strata", re.compile(r"(?:^|/)(?:strata)(?:\s|$)|strata-lean/serve/server\.py")),
    ("llama-server", re.compile(r"(?:^|/)llama-server(?:\s|$)")),
    ("audiocpp", re.compile(r"(?:^|/)audiocpp_server(?:\s|$)")),
    ("speech.cpp", re.compile(r"(?:^|/)speech(?:\s+serve\b)")),
    ("comfyui", re.compile(r"ComfyUI/main\.py")),
)


def _to_float(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned or cleaned.upper() in {"N/A", "[N/A]", "[NOT SUPPORTED]", "NA"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_nvidia_smi_csv(text: str) -> dict[str, Any]:
    """Parse `nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits`."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return {"available": False, "error": "empty nvidia-smi output"}

    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) < 7:
        return {"available": False, "error": f"unexpected nvidia-smi columns ({len(parts)})"}

    name = parts[0]
    vram_used = _to_float(parts[1])
    vram_total = _to_float(parts[2])
    util_pct = _to_float(parts[3])
    temp_c = _to_float(parts[4])
    power_w = _to_float(parts[5])
    power_limit_w = _to_float(parts[6])

    vram_pct = None
    if vram_used is not None and vram_total and vram_total > 0:
        vram_pct = round((vram_used / vram_total) * 100.0, 1)

    return {
        "available": True,
        "name": name,
        "vram_used_mib": round(vram_used, 1) if vram_used is not None else None,
        "vram_total_mib": round(vram_total, 1) if vram_total is not None else None,
        "vram_pct": vram_pct,
        "util_pct": round(util_pct, 1) if util_pct is not None else None,
        "temp_c": round(temp_c, 1) if temp_c is not None else None,
        "power_w": round(power_w, 1) if power_w is not None else None,
        "power_limit_w": round(power_limit_w, 1) if power_limit_w is not None else None,
        "error": None,
    }


def parse_meminfo(text: str) -> dict[str, Any]:
    """Parse Linux `/proc/meminfo` into MiB metrics for RAM and swap/zram."""
    raw_kb: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r"^([^:]+):\s+(\d+)", line)
        if m:
            raw_kb[m.group(1)] = int(m.group(2))

    total_kb = raw_kb.get("MemTotal", 0)
    avail_kb = raw_kb.get("MemAvailable", raw_kb.get("MemFree", 0))
    used_kb = max(0, total_kb - avail_kb)
    cached_kb = raw_kb.get("Cached", 0) + raw_kb.get("Buffers", 0)
    swap_total_kb = raw_kb.get("SwapTotal", 0)
    swap_free_kb = raw_kb.get("SwapFree", 0)
    swap_used_kb = max(0, swap_total_kb - swap_free_kb)

    to_mib = lambda kb: round(kb / 1024.0, 1)
    used_pct = round((used_kb / total_kb) * 100.0, 1) if total_kb > 0 else 0.0
    swap_pct = round((swap_used_kb / swap_total_kb) * 100.0, 1) if swap_total_kb > 0 else 0.0

    return {
        "total_mib": to_mib(total_kb),
        "available_mib": to_mib(avail_kb),
        "used_mib": to_mib(used_kb),
        "used_pct": used_pct,
        "cached_mib": to_mib(cached_kb),
        "swap_total_mib": to_mib(swap_total_kb),
        "swap_free_mib": to_mib(swap_free_kb),
        "swap_used_mib": to_mib(swap_used_kb),
        "swap_pct": swap_pct,
    }


def parse_zram_mm_stat(text: str) -> dict[str, float] | None:
    """Parse `/sys/block/zram0/mm_stat` (orig_data_size compr_data_size mem_used_total ...)."""
    parts = text.strip().split()
    if len(parts) < 3:
        return None
    try:
        orig_bytes = int(parts[0])
        compr_bytes = int(parts[1])
        mem_used_bytes = int(parts[2])
    except ValueError:
        return None
    mib = 1024.0 * 1024.0
    return {
        "orig_mib": round(orig_bytes / mib, 1),
        "compr_mib": round(compr_bytes / mib, 1),
        "mem_used_mib": round(mem_used_bytes / mib, 1),
    }


def parse_proc_stat(text: str) -> dict[str, tuple[int, int]]:
    """Parse `/proc/stat` into `{cpu_label: (total_jiffies, idle_jiffies)}`."""
    result: dict[str, tuple[int, int]] = {}
    for line in text.splitlines():
        if not line.startswith("cpu"):
            continue
        parts = line.split()
        label = parts[0]
        if label != "cpu" and not label[3:].isdigit():
            continue
        try:
            nums = [int(x) for x in parts[1:]]
        except ValueError:
            continue
        if len(nums) < 4:
            continue
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        total = sum(nums)
        result[label] = (total, idle)
    return result


def compute_cpu_usage(
    prev: dict[str, tuple[int, int]] | None,
    curr: dict[str, tuple[int, int]],
    p_cores: Iterable[int] = DEFAULT_P_CORES,
    e_cores: Iterable[int] = DEFAULT_E_CORES,
) -> dict[str, Any]:
    """Compute aggregate, P-core (`0-11`), E-core (`12-15`), and per-core load %."""
    p_set = tuple(p_cores)
    e_set = tuple(e_cores)

    def _pct(label: str) -> float | None:
        if not prev or label not in prev or label not in curr:
            return None
        t0, i0 = prev[label]
        t1, i1 = curr[label]
        dt = t1 - t0
        di = i1 - i0
        if dt <= 0:
            return 0.0
        busy = max(0.0, min(1.0, (dt - di) / dt))
        return round(busy * 100.0, 1)

    def _group_pct(indices: tuple[int, ...]) -> tuple[float | None, int]:
        if not prev:
            present = sum(1 for idx in indices if f"cpu{idx}" in curr)
            return None, present
        dt_sum = 0
        di_sum = 0
        present = 0
        for idx in indices:
            key = f"cpu{idx}"
            if key in prev and key in curr:
                present += 1
                t0, i0 = prev[key]
                t1, i1 = curr[key]
                dt = t1 - t0
                di = i1 - i0
                if dt > 0:
                    dt_sum += dt
                    di_sum += max(0, di)
        if present == 0:
            return None, 0
        if dt_sum <= 0:
            return 0.0, present
        busy = max(0.0, min(1.0, (dt_sum - di_sum) / dt_sum))
        return round(busy * 100.0, 1), present

    per_core: dict[str, float] = {}
    for label in sorted(
        (k for k in curr if k != "cpu" and k[3:].isdigit()),
        key=lambda k: int(k[3:]),
    ):
        val = _pct(label)
        if val is not None:
            per_core[label] = val

    p_pct, p_count = _group_pct(p_set)
    e_pct, e_count = _group_pct(e_set)

    return {
        "total_pct": _pct("cpu"),
        "p_core_pct": p_pct,
        "e_core_pct": e_pct,
        "p_core_count": p_count,
        "e_core_count": e_count,
        "p_core_range": f"{p_set[0]}-{p_set[-1]}" if p_set else "",
        "e_core_range": f"{e_set[0]}-{e_set[-1]}" if e_set else "",
        "per_core": per_core,
    }


def parse_thp_enabled(text: str) -> str:
    """Parse `/sys/kernel/mm/transparent_hugepage/enabled` (e.g. `[always] madvise never`)."""
    m = re.search(r"\[([^\]]+)\]", text)
    if m:
        return m.group(1).strip()
    return text.strip() or "unknown"


def evaluate_governor(governors: list[str]) -> dict[str, Any]:
    """Evaluate CPU frequency scaling governors; healthy iff all cores use `performance`."""
    cleaned = [g.strip() for g in governors if g and g.strip()]
    if not cleaned:
        return {
            "governor": "unknown",
            "unique": [],
            "healthy": False,
            "alert": "CPU governor unavailable",
        }
    unique = sorted(set(cleaned))
    primary = unique[0] if len(unique) == 1 else "mixed:" + ",".join(unique)
    healthy = unique == ["performance"]
    alert = None if healthy else f"Governor is {primary} (expected performance)"
    return {
        "governor": primary,
        "unique": unique,
        "healthy": healthy,
        "alert": alert,
    }


def parse_proc_io(text: str) -> dict[str, int]:
    """Parse `/proc/<pid>/io` into byte counters (`read_bytes`, `write_bytes`, `rchar`, `wchar`)."""
    data: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r"^([a-z_]+):\s+(\d+)", line.strip())
        if m:
            data[m.group(1)] = int(m.group(2))
    return data


def parse_df_output(
    text: str,
    targets: Iterable[str] | None = DEFAULT_DF_TARGETS,
) -> list[dict[str, Any]]:
    """Parse `df -hlT` output into structured local filesystem rows."""
    target_list = list(targets) if targets is not None else None
    rows_by_mount: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []

    for idx, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if idx == 0 and stripped.startswith("Filesystem"):
            continue
        parts = stripped.split(None, 6)
        if len(parts) < 7:
            continue
        fs, fstype, size, used, avail, use_pct_raw, mount = parts
        try:
            use_pct = int(use_pct_raw.rstrip("%"))
        except ValueError:
            use_pct = 0
        row = {
            "filesystem": fs,
            "fstype": fstype,
            "size": size,
            "used": used,
            "avail": avail,
            "use_pct": use_pct,
            "mount": mount,
            "warning": use_pct >= 90,
            "critical": use_pct >= 95,
        }
        rows_by_mount[mount] = row
        all_rows.append(row)

    if target_list is None:
        return all_rows
    return [rows_by_mount[m] for m in target_list if m in rows_by_mount]


def classify_inference_cmdline(cmdline: str) -> str | None:
    """Return engine label if a `/proc/<pid>/cmdline` belongs to an L3MS inference process."""
    if not cmdline or "local-dashboard" in cmdline:
        return None
    for engine, pattern in INFERENCE_CMD_PATTERNS:
        if pattern.search(cmdline):
            return engine
    return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _run_cmd(cmd: list[str], timeout: float = 2.5) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, "", str(exc)


class TelemetryCollector:
    """Thread-safe hardware telemetry collector with delta tracking for CPU and NVMe I/O."""

    def __init__(
        self,
        proc_root: Path = Path("/proc"),
        sys_root: Path = Path("/sys"),
        df_targets: tuple[str, ...] = DEFAULT_DF_TARGETS,
        run_cmd: Callable[[list[str], float], tuple[int, str, str]] = _run_cmd,
        clock: Callable[[], float] = time.monotonic,
        df_cache_ttl_s: float = 15.0,
        service_cache_ttl_s: float = 30.0,
    ) -> None:
        self.proc_root = proc_root
        self.sys_root = sys_root
        self.df_targets = df_targets
        self._run_cmd = run_cmd
        self._clock = clock
        self._df_cache_ttl_s = df_cache_ttl_s
        self._service_cache_ttl_s = service_cache_ttl_s

        self._lock = threading.Lock()
        self._prev_stat: dict[str, tuple[int, int]] | None = None
        self._prev_io: dict[int, tuple[float, int]] = {}
        self._df_cache: tuple[float, list[dict[str, Any]]] | None = None
        self._svc_cache: tuple[float, dict[str, str]] | None = None

    def collect_gpu(self) -> dict[str, Any]:
        rc, out, err = self._run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            2.5,
        )
        if rc != 0:
            return {"available": False, "error": err.strip() or f"nvidia-smi exited {rc}"}
        return parse_nvidia_smi_csv(out)

    def collect_memory(self) -> dict[str, Any]:
        text = _read_text(self.proc_root / "meminfo")
        if text is None:
            return {"available": False, "error": "cannot read /proc/meminfo"}
        mem = parse_meminfo(text)
        mem["available"] = True
        zram_text = _read_text(self.sys_root / "block" / "zram0" / "mm_stat")
        zram = parse_zram_mm_stat(zram_text) if zram_text else None
        mem["zram"] = zram
        mem["zram_used_mib"] = zram["orig_mib"] if zram else mem["swap_used_mib"]
        return mem

    def collect_cpu(self) -> dict[str, Any]:
        stat_text = _read_text(self.proc_root / "stat")
        if stat_text is None:
            return {
                "total_pct": None,
                "p_core_pct": None,
                "e_core_pct": None,
                "p_core_count": 0,
                "e_core_count": 0,
                "per_core": {},
            }
        curr = parse_proc_stat(stat_text)
        with self._lock:
            prev = self._prev_stat
            self._prev_stat = curr
        if prev is None and self.proc_root == Path("/proc"):
            # Brief 60ms sample on first live call so the initial page load shows real utilization.
            time.sleep(0.06)
            second_text = _read_text(self.proc_root / "stat")
            if second_text is not None:
                second = parse_proc_stat(second_text)
                with self._lock:
                    self._prev_stat = second
                return compute_cpu_usage(curr, second)
        return compute_cpu_usage(prev, curr)

    def collect_governor_and_tuning(self) -> dict[str, Any]:
        cpu_dir = self.sys_root / "devices" / "system" / "cpu"
        govs: list[str] = []
        if cpu_dir.is_dir():
            for entry in sorted(cpu_dir.glob("cpu[0-9]*/cpufreq/scaling_governor")):
                val = _read_text(entry)
                if val:
                    govs.append(val.strip())
        gov_info = evaluate_governor(govs)

        swappiness_raw = _read_text(self.proc_root / "sys" / "vm" / "swappiness")
        try:
            swappiness = int(swappiness_raw.strip()) if swappiness_raw else None
        except ValueError:
            swappiness = None

        thp_raw = _read_text(self.sys_root / "kernel" / "mm" / "transparent_hugepage" / "enabled")
        thp = parse_thp_enabled(thp_raw) if thp_raw else "unknown"

        services = self._collect_tuning_services()

        return {
            "governor": gov_info["governor"],
            "governor_healthy": gov_info["healthy"],
            "governor_alert": gov_info["alert"],
            "vm_swappiness": swappiness,
            "swappiness_high": swappiness is not None and swappiness > 60,
            "thp": thp,
            "services": services,
        }

    def _collect_tuning_services(self) -> dict[str, str]:
        now = self._clock()
        with self._lock:
            if self._svc_cache and (now - self._svc_cache[0]) < self._service_cache_ttl_s:
                return dict(self._svc_cache[1])
        statuses: dict[str, str] = {}
        for svc in ("ananicy-cpp", "scx"):
            rc, out, _ = self._run_cmd(["systemctl", "is-active", svc], 1.5)
            state = out.strip()
            statuses[svc] = state if state else ("active" if rc == 0 else "unknown")
        with self._lock:
            self._svc_cache = (now, statuses)
        return dict(statuses)

    def collect_disks(self) -> list[dict[str, Any]]:
        now = self._clock()
        with self._lock:
            if self._df_cache and (now - self._df_cache[0]) < self._df_cache_ttl_s:
                return [dict(item) for item in self._df_cache[1]]
        # CRITICAL: Always use `df -hlT` (local filesystems only) to avoid hanging on dead NFS mounts.
        rc, out, _ = self._run_cmd(["df", "-hlT"], 3.0)
        rows = parse_df_output(out, targets=self.df_targets) if rc == 0 else []
        with self._lock:
            self._df_cache = (now, rows)
        return [dict(item) for item in rows]

    def collect_inference_io(self) -> dict[str, Any]:
        """Inspect `/proc/<pid>/io` read_bytes rate for active inference processes (NVMe spill detector)."""
        now = self._clock()
        processes: list[dict[str, Any]] = []
        seen_pids: set[int] = set()
        my_pid = os.getpid()

        try:
            entries = list(self.proc_root.iterdir())
        except OSError:
            entries = []

        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == my_pid:
                continue
            cmdline_raw = _read_text(entry / "cmdline")
            if not cmdline_raw:
                continue
            cmdline = cmdline_raw.replace("\x00", " ").strip()
            engine = classify_inference_cmdline(cmdline)
            if not engine:
                continue

            io_text = _read_text(entry / "io")
            if io_text is None:
                continue
            io_stats = parse_proc_io(io_text)
            read_bytes = io_stats.get("read_bytes", 0)
            seen_pids.add(pid)

            rss_mib = None
            status_text = _read_text(entry / "status")
            if status_text:
                m_rss = re.search(r"^VmRSS:\s+(\d+)\s+kB", status_text, re.MULTILINE)
                if m_rss:
                    rss_mib = round(int(m_rss.group(1)) / 1024.0, 1)

            with self._lock:
                prev = self._prev_io.get(pid)
                self._prev_io[pid] = (now, read_bytes)

            read_mb_s = 0.0
            if prev is not None:
                dt = now - prev[0]
                if dt > 0:
                    delta_bytes = max(0, read_bytes - prev[1])
                    read_mb_s = round((delta_bytes / (1024.0 * 1024.0)) / dt, 2)

            processes.append(
                {
                    "pid": pid,
                    "engine": engine,
                    "rss_mib": rss_mib,
                    "read_mb_s": read_mb_s,
                    "read_total_mib": round(read_bytes / (1024.0 * 1024.0), 1),
                }
            )

        with self._lock:
            stale = [pid for pid in self._prev_io if pid not in seen_pids]
            for pid in stale:
                del self._prev_io[pid]

        processes.sort(key=lambda p: (p["read_mb_s"], p["read_total_mib"]), reverse=True)
        total_rate = round(sum(p["read_mb_s"] for p in processes), 2)
        if not processes:
            spill_state = "idle"
        elif total_rate >= 100.0:
            spill_state = "spill"
        elif total_rate >= 25.0:
            spill_state = "elevated"
        else:
            spill_state = "ok"

        primary = processes[0] if processes else None
        return {
            "active_pid": primary["pid"] if primary else None,
            "active_engine": primary["engine"] if primary else None,
            "read_mb_s": total_rate,
            "spill_state": spill_state,
            "spill_alert": spill_state in {"elevated", "spill"},
            "processes": processes,
        }

    def snapshot(self) -> dict[str, Any]:
        """Collect a complete live telemetry snapshot."""
        return {
            "ts": time.time(),
            "gpu": self.collect_gpu(),
            "memory": self.collect_memory(),
            "cpu": self.collect_cpu(),
            "tuning": self.collect_governor_and_tuning(),
            "io": self.collect_inference_io(),
            "disks": self.collect_disks(),
        }
