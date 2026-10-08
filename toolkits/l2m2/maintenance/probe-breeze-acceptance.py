#!/usr/bin/env python3
"""Run a bounded native Breeze-TTS acceptance sequence.

This probes the native audio.cpp wrapper only. It is not evidence that the
L3MS coordinator can safely switch GPU owners. The default mode is a plan;
actual unit and inference actions require --run and an explicit, empty,
current-user-owned mode-0700 output directory.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path
from typing import Any, Callable, Iterable


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[3]
WRAPPER = REPO_ROOT / "toolkits/l2m2/media-models/generate-breeze-tts-2.sh"
LLAMA_CONFIG = Path("/home/kchauhan/repos/l3ms/llama-swap.yaml")
LLAMA_UNIT = "llama-swap.service"
COMFY_UNIT = "l2m2-comfyui.service"
UNIT_PROPERTIES = (
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "MainPID",
    "ControlGroup",
    "KillMode",
    "InvocationID",
)
INVOCATION_TIMEOUT_SECONDS = 120
PREFLIGHT_TIMEOUT_SECONDS = 30
COMMAND_TIMEOUT_SECONDS = 15
STOP_TIMEOUT_SECONDS = 30
MAX_WAVE_BYTES = 512 * 1024 * 1024
MAX_CHUNKS = 63
MAX_TOTAL_STREAM_BYTES = 512 * 1024 * 1024
MAX_CHILD_LOG_BYTES = 16 * 1024 * 1024
SEED = 42
MAX_TOKENS = 256
STREAM_FRAMES = 16
STREAM_LOOKAHEAD = 12
SYNTHETIC_REFERENCE_TEXT = (
    "At sunrise, a small red kite drifted above the quiet harbor."
)
SYNTHETIC_CLONE_TEXT = "Testing a new message from the harbor."
DESIGN_INSTRUCTION = "Speak warmly and clearly, with a light, playful tone."


class ProbeError(RuntimeError):
    """A fail-closed precondition or acceptance failure."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def parse_key_values(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def parse_csv_rows(output: str, expected_columns: int, label: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        values = [value.strip() for value in row]
        if not values or all(not value for value in values):
            continue
        if len(values) != expected_columns:
            raise ProbeError(f"could not parse {label} snapshot")
        rows.append(values)
    return rows


def parse_gpu_rows(output: str) -> list[dict[str, Any]]:
    raw_rows = parse_csv_rows(output, 5, "GPU")
    if not raw_rows:
        raise ProbeError("nvidia-smi returned no GPU rows")
    parsed: list[dict[str, Any]] = []
    for uuid, name, used, total, utilization in raw_rows:
        try:
            used_mib = int(used)
            total_mib = int(total)
            util_percent = int(utilization)
        except ValueError as error:
            raise ProbeError("nvidia-smi returned unknown GPU memory/utilization") from error
        if not uuid or used_mib < 0 or total_mib <= 0 or util_percent < 0:
            raise ProbeError("nvidia-smi returned invalid GPU metrics")
        parsed.append(
            {
                "uuid": uuid,
                "name": name,
                "memory_used_mib": used_mib,
                "memory_total_mib": total_mib,
                "utilization_percent": util_percent,
            }
        )
    return parsed


def parse_compute_rows(output: str) -> list[dict[str, Any]]:
    if any(line.strip().lower().startswith("no running processes") for line in output.splitlines()):
        return []
    raw_rows = parse_csv_rows(output, 3, "compute-process")
    parsed: list[dict[str, Any]] = []
    for pid_text, name, used_memory in raw_rows:
        try:
            pid = int(pid_text)
        except ValueError as error:
            raise ProbeError("nvidia-smi returned a compute claimant without a numeric PID") from error
        if pid <= 0 or used_memory.strip().upper() in {"N/A", "[N/A]", "UNKNOWN"}:
            raise ProbeError("nvidia-smi returned an unknown compute claimant")
        try:
            memory_mib = int(used_memory)
        except ValueError as error:
            raise ProbeError("nvidia-smi returned unknown per-process GPU memory") from error
        if memory_mib < 0:
            raise ProbeError("nvidia-smi returned invalid per-process GPU memory")
        parsed.append({"pid": pid, "process_name": name, "memory_used_mib": memory_mib})
    return parsed


def systemd_unit_snapshot(unit: str) -> dict[str, str]:
    command = [
        "systemctl",
        "--user",
        "show",
        "--no-pager",
        f"--property={','.join(UNIT_PROPERTIES)}",
        unit,
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise ProbeError(f"could not inspect exact user unit {unit}")
    values = parse_key_values(result.stdout)
    if any(key not in values for key in UNIT_PROPERTIES):
        raise ProbeError(f"incomplete systemd snapshot for {unit}")
    return values


def proc_observation(pid: int) -> dict[str, Any]:
    cgroup_path = Path(f"/proc/{pid}/cgroup")
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        cgroup_text = cgroup_path.read_text(encoding="utf-8")
        stat_text = stat_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ProbeError(f"could not verify GPU claimant PID {pid}") from error
    # The command name in /proc/PID/stat can contain spaces and parentheses.
    right_paren = stat_text.rfind(")")
    if right_paren < 0:
        raise ProbeError(f"could not parse process identity for GPU claimant PID {pid}")
    fields_after_name = stat_text[right_paren + 1 :].split()
    if len(fields_after_name) < 20:
        raise ProbeError(f"could not parse process start time for GPU claimant PID {pid}")
    try:
        process_group = int(fields_after_name[2])
        start_time = int(fields_after_name[19])
    except ValueError as error:
        raise ProbeError(f"could not parse process identity for GPU claimant PID {pid}") from error
    return {
        "pid": pid,
        "state": fields_after_name[0],
        "process_group": process_group,
        "start_time_ticks": start_time,
        "cgroup": [line.strip() for line in cgroup_text.splitlines() if line.strip()],
    }


def belongs_to_cgroup(observation: dict[str, Any], expected: str) -> bool:
    if not expected:
        return False
    normalized = expected.rstrip("/") or "/"
    if normalized == "/":
        return False
    for row in observation.get("cgroup", []):
        parts = row.split(":", 2)
        if len(parts) != 3:
            continue
        actual = parts[2].rstrip("/") or "/"
        if actual == normalized or actual.startswith(normalized.rstrip("/") + "/"):
            return True
    return False


def scan_cgroup_processes(expected: str) -> tuple[list[dict[str, Any]], bool]:
    if not expected:
        return [], True
    members: list[dict[str, Any]] = []
    complete = True
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return members, False
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            cgroup_text = (entry / "cgroup").read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except OSError:
            complete = False
            continue
        cgroup_lines = [line.strip() for line in cgroup_text.splitlines() if line.strip()]
        if not belongs_to_cgroup({"cgroup": cgroup_lines}, expected):
            continue
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except OSError:
            complete = False
            continue
        right_paren = stat_text.rfind(")")
        if right_paren < 0:
            complete = False
            continue
        fields = stat_text[right_paren + 1 :].split()
        if len(fields) < 20:
            complete = False
            continue
        try:
            process_group = int(fields[2])
            start_time = int(fields[19])
        except ValueError:
            complete = False
            continue
        members.append(
            {
                "pid": pid,
                "state": fields[0],
                "process_group": process_group,
                "start_time_ticks": start_time,
                "cgroup": cgroup_lines,
            }
        )
    members.sort(key=lambda member: member["pid"])
    return members, complete


def gpu_snapshot() -> dict[str, Any]:
    gpu_command = [
        "nvidia-smi",
        "--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    apps_command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu_result = subprocess.run(
            gpu_command,
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
        apps_result = subprocess.run(
            apps_command,
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ProbeError("could not read NVIDIA GPU state") from error
    if gpu_result.returncode != 0 or apps_result.returncode != 0:
        raise ProbeError("nvidia-smi failed; GPU ownership is unknown")
    gpus = parse_gpu_rows(gpu_result.stdout)
    apps = parse_compute_rows(apps_result.stdout)
    observations = []
    for app in apps:
        observations.append({**app, **proc_observation(app["pid"])})
    return {"gpus": gpus, "compute_processes": observations}


def assert_only_llama_swap_claimants(snapshot: dict[str, Any], cgroup: str) -> None:
    for process in snapshot["compute_processes"]:
        if not belongs_to_cgroup(process, cgroup):
            raise ProbeError(
                "pre-stop GPU claimant is not provably inside llama-swap.service cgroup"
            )


def assert_no_gpu_compute(snapshot: dict[str, Any]) -> None:
    if snapshot["compute_processes"]:
        raise ProbeError("GPU compute process remains active or has unknown ownership")


def require_reviewed_prestate(units: dict[str, dict[str, str]]) -> None:
    llama = units[LLAMA_UNIT]
    comfy = units[COMFY_UNIT]
    if llama["LoadState"] != "loaded" or llama["ActiveState"] != "active":
        raise ProbeError("llama-swap.service is not in the reviewed active prestate")
    if llama["SubState"] not in {"running", "listening"} or int(llama["MainPID"] or "0") <= 0:
        raise ProbeError("llama-swap.service has an uncertain active state")
    if not llama["ControlGroup"]:
        raise ProbeError("llama-swap.service has no observable control group")
    if llama["KillMode"] != "process":
        raise ProbeError("llama-swap.service KillMode changed from the reviewed process-only setting")
    if not llama["InvocationID"]:
        raise ProbeError("llama-swap.service has no observable invocation generation")
    if comfy["LoadState"] not in {"loaded", "not-found"} or comfy["ActiveState"] != "inactive":
        raise ProbeError("l2m2-comfyui.service is not in the reviewed inactive prestate")
    if comfy["SubState"] not in {"dead", "inactive"} or int(comfy["MainPID"] or "0") != 0:
        raise ProbeError("l2m2-comfyui.service has an uncertain inactive state")


def require_private_empty_output_dir(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise ProbeError("output directory must already exist") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ProbeError("output path must be a real directory, not a symlink")
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ProbeError("output directory must be owned by this user and have mode 0700")
    path = candidate.resolve(strict=True)
    if any(path.iterdir()):
        raise ProbeError("output directory must be empty; existing files are never overwritten")
    return path


def validate_wave_file(path: Path, root: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ProbeError(f"expected generated WAV is missing: {path.name}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProbeError(f"generated WAV is not a regular non-symlink file: {path.name}")
    if metadata.st_nlink != 1 or not resolved.is_relative_to(root):
        raise ProbeError(f"generated WAV escaped the private output directory: {path.name}")
    if metadata.st_size <= 44 or metadata.st_size > MAX_WAVE_BYTES:
        raise ProbeError(f"generated WAV has an invalid size: {path.name}")
    byte_order = "<"
    channels: int | None = None
    sample_width: int | None = None
    sample_rate: int | None = None
    block_align: int | None = None
    encoding: int | None = None
    bits: int | None = None
    data_size: int | None = None
    try:
        with path.open("rb") as audio_file:
            header = audio_file.read(12)
            if len(header) != 12 or header[8:12] != b"WAVE":
                raise ProbeError(f"generated output is not a RIFF/RIFX WAV: {path.name}")
            if header[:4] == b"RIFF":
                byte_order = "<"
            elif header[:4] == b"RIFX":
                byte_order = ">"
            else:
                raise ProbeError(f"generated output is not a RIFF/RIFX WAV: {path.name}")
            declared_size = struct.unpack(f"{byte_order}I", header[4:8])[0] + 8
            if declared_size != metadata.st_size:
                raise ProbeError(f"generated WAV has an invalid container length: {path.name}")
            offset = 12
            saw_format = False
            saw_data = False
            for _ in range(128):
                if offset == metadata.st_size:
                    break
                if metadata.st_size - offset < 8:
                    raise ProbeError(f"generated WAV has a truncated chunk header: {path.name}")
                audio_file.seek(offset)
                chunk_header = audio_file.read(8)
                if len(chunk_header) != 8:
                    raise ProbeError(f"generated WAV has a truncated chunk header: {path.name}")
                chunk_name = chunk_header[:4]
                chunk_size = struct.unpack(f"{byte_order}I", chunk_header[4:])[0]
                next_offset = offset + 8 + chunk_size + (chunk_size % 2)
                if next_offset > metadata.st_size:
                    raise ProbeError(f"generated WAV has a chunk beyond end of file: {path.name}")
                payload_offset = offset + 8
                if chunk_name == b"fmt ":
                    if saw_format or chunk_size < 16:
                        raise ProbeError(f"generated WAV has an invalid format chunk: {path.name}")
                    audio_file.seek(payload_offset)
                    format_chunk = audio_file.read(16)
                    if len(format_chunk) != 16:
                        raise ProbeError(f"generated WAV has a truncated format chunk: {path.name}")
                    (
                        encoding,
                        channels,
                        sample_rate,
                        byte_rate,
                        block_align,
                        bits,
                    ) = struct.unpack(f"{byte_order}HHIIHH", format_chunk)
                    sample_width = bits // 8
                    if (
                        encoding not in {1, 3}
                        or not 1 <= channels <= 8
                        or not 8000 <= sample_rate <= 384000
                        or bits not in {8, 16, 24, 32, 64}
                        or (encoding == 3 and bits not in {32, 64})
                        or (encoding == 1 and bits == 64)
                        or block_align != channels * sample_width
                        or byte_rate != sample_rate * block_align
                    ):
                        raise ProbeError(f"generated WAV has unsupported audio layout: {path.name}")
                    saw_format = True
                elif chunk_name == b"data":
                    if not saw_format or saw_data or chunk_size == 0 or chunk_size % (block_align or 1):
                        raise ProbeError(f"generated WAV has invalid or empty audio data: {path.name}")
                    saw_data = True
                    data_size = chunk_size
                    if encoding == 3:
                        audio_file.seek(payload_offset)
                        remaining = chunk_size
                        format_code = "f" if bits == 32 else "d"
                        scalar_bytes = bits // 8
                        unpacker = struct.Struct(f"{byte_order}{format_code}")
                        while remaining:
                            amount = min(remaining, 64 * 1024)
                            amount -= amount % scalar_bytes
                            raw = audio_file.read(amount)
                            if len(raw) != amount:
                                raise ProbeError(f"generated WAV float data is truncated: {path.name}")
                            if any(not math.isfinite(sample[0]) for sample in unpacker.iter_unpack(raw)):
                                raise ProbeError(f"generated WAV contains non-finite float samples: {path.name}")
                            remaining -= amount
                offset = next_offset
            if offset != metadata.st_size or not saw_format or not saw_data:
                raise ProbeError(f"generated WAV is empty or malformed: {path.name}")
    except OSError as error:
        raise ProbeError(f"generated output could not be read: {path.name}") from error
    assert channels is not None and sample_width is not None and sample_rate is not None
    assert data_size is not None
    frames = data_size // (block_align or 1)
    return {
        "path": str(path.relative_to(root)),
        "bytes": metadata.st_size,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate_hz": sample_rate,
        "frames": frames,
        "duration_seconds": round(frames / sample_rate, 6),
        "format": "IEEE-float" if encoding == 3 else "PCM",
    }


def validate_stream_chunks(final_wave: Path, root: Path) -> list[dict[str, Any]]:
    chunk_dir = Path(f"{final_wave}.chunks")
    try:
        metadata = chunk_dir.lstat()
        resolved = chunk_dir.resolve(strict=True)
    except OSError as error:
        raise ProbeError("streaming output has no chunks directory") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ProbeError("streaming chunks path is not a real directory")
    if not resolved.is_relative_to(root):
        raise ProbeError("streaming chunks directory escaped the private output directory")
    files = sorted(chunk_dir.iterdir(), key=lambda path: path.name)
    if not files or len(files) > MAX_CHUNKS:
        raise ProbeError("streaming output must contain between 1 and 63 chunks")
    if any(not re.fullmatch(r"chunk_[A-Za-z0-9_.-]+\.wav", path.name) for path in files):
        raise ProbeError("streaming output contains an unexpected chunk name")
    chunks = [validate_wave_file(path, resolved) for path in files]
    if sum(chunk["bytes"] for chunk in chunks) > MAX_TOTAL_STREAM_BYTES:
        raise ProbeError("streaming chunks exceed the 512 MiB total limit")
    return chunks


def leader_exited_unreaped(pid: int) -> bool:
    """Observe a child exit without releasing its PID/PGID for reuse."""
    try:
        result = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError as error:
        raise ProbeError("owned child was reaped outside the bounded process-group controller") from error
    return result is not None and result.si_pid == pid


def scan_process_group(pgid: int) -> tuple[list[dict[str, Any]], bool]:
    members: list[dict[str, Any]] = []
    complete = True
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return members, False
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except OSError:
            complete = False
            continue
        right_paren = stat_text.rfind(")")
        if right_paren < 0:
            complete = False
            continue
        fields = stat_text[right_paren + 1 :].split()
        if len(fields) < 3:
            complete = False
            continue
        try:
            process_group = int(fields[2])
        except ValueError:
            complete = False
            continue
        if process_group == pgid:
            members.append(
                {"pid": int(entry.name), "state": fields[0], "process_group": process_group}
            )
    return members, complete


def probe_reaped_group(pgid: int) -> dict[str, Any]:
    """After reaping, inspect only; signaling could hit a recycled PGID."""
    try:
        os.killpg(pgid, 0)
        group_exists = True
    except ProcessLookupError:
        group_exists = False
    except PermissionError:
        return {"clean": False, "group_exists": None, "members": [], "complete": False}
    members, complete = scan_process_group(pgid)
    live_members = [member for member in members if member["state"] not in {"Z", "X"}]
    return {
        "clean": complete and not live_members and (not group_exists or bool(members)),
        "group_exists": group_exists,
        "members": members,
        "complete": complete,
    }


def start_bounded_logged_child(
    command: list[str],
    log_path: Path,
    owned_groups: list[subprocess.Popen[bytes]],
) -> tuple[subprocess.Popen[bytes], threading.Thread, threading.Event, dict[str, Any]]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        bufsize=0,
    )
    owned_groups.append(process)
    output_event = threading.Event()
    log_state: dict[str, Any] = {"bytes_written": 0, "limit_exceeded": False, "error": None}

    def drain_output() -> None:
        if process.stdout is None:
            log_state["error"] = "missing-pipe"
            output_event.set()
            return
        try:
            with log_path.open("xb") as log_file:
                while True:
                    chunk = process.stdout.read(64 * 1024)
                    if not chunk:
                        break
                    remaining = MAX_CHILD_LOG_BYTES - log_state["bytes_written"]
                    keep = max(0, min(len(chunk), remaining))
                    if keep:
                        log_file.write(chunk[:keep])
                        log_state["bytes_written"] += keep
                    if keep < len(chunk):
                        log_state["limit_exceeded"] = True
                        output_event.set()
                log_file.flush()
                os.fsync(log_file.fileno())
        except OSError as error:
            log_state["error"] = type(error).__name__
            output_event.set()
        finally:
            process.stdout.close()

    output_thread = threading.Thread(
        target=drain_output,
        name=f"breeze-log-{process.pid}",
        daemon=True,
    )
    try:
        output_thread.start()
    except BaseException:
        # The group leader remains unreaped and tracked so the caller's finally
        # path can safely signal it before any PID/PGID reuse is possible.
        raise
    return process, output_thread, output_event, log_state


def clean_owned_process_group(
    process: subprocess.Popen[bytes],
    *,
    force: bool,
    watchdog: threading.Timer | None = None,
    output_thread: threading.Thread | None = None,
    grace_seconds: float = 3.0,
) -> dict[str, Any]:
    """Clean an owned group while its unreaped leader pins the PGID.

    Any timer is canceled and joined before final signals. Once ``process.wait``
    reaps the leader, this function performs no signal other than killpg(..., 0).
    """
    pgid = process.pid
    result: dict[str, Any] = {
        "clean": False,
        "leader_pid": process.pid,
        "process_group": pgid,
        "signals_sent": [],
        "reaped": process.returncode is not None,
    }
    if watchdog is not None:
        watchdog.cancel()
        watchdog.join()
    if process.returncode is not None:
        result.update(probe_reaped_group(pgid))
        result["reaped"] = True
        return result

    def send(signal_number: int) -> None:
        if process.returncode is not None:
            raise ProbeError("refusing to signal a process group after its leader was reaped")
        try:
            os.killpg(pgid, signal_number)
            result["signals_sent"].append(signal.Signals(signal_number).name)
        except ProcessLookupError:
            pass
        except OSError:
            result["signal_error"] = True

    def observations() -> tuple[bool, list[dict[str, Any]], bool]:
        leader_exited = leader_exited_unreaped(process.pid)
        members, complete = scan_process_group(pgid)
        descendants = [
            member
            for member in members
            if member["pid"] != process.pid and member["state"] not in {"Z", "X"}
        ]
        return leader_exited, descendants, complete

    leader_exited, descendants, complete = observations()
    if force or descendants or not complete:
        send(signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        leader_exited, descendants, complete = observations()
        if leader_exited and not descendants and complete:
            break
        time.sleep(0.05)

    leader_exited, descendants, complete = observations()
    if not leader_exited or descendants or not complete:
        send(signal.SIGKILL)
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            leader_exited, descendants, complete = observations()
            if leader_exited and not descendants and complete:
                break
            time.sleep(0.05)
    leader_exited, descendants, complete = observations()
    if output_thread is not None:
        output_thread.join(timeout=grace_seconds)
        if output_thread.is_alive():
            send(signal.SIGTERM)
            output_thread.join(timeout=grace_seconds)
        if output_thread.is_alive():
            send(signal.SIGKILL)
            output_thread.join(timeout=grace_seconds)
    if not leader_exited or descendants or not complete or (
        output_thread is not None and output_thread.is_alive()
    ):
        result.update(
            {
                "clean": False,
                "reaped": False,
                "leader_exited": leader_exited,
                "live_descendants": descendants,
                "proc_scan_complete": complete,
                "log_drained": output_thread is None or not output_thread.is_alive(),
            }
        )
        return result

    try:
        process.wait(timeout=0)
    except subprocess.TimeoutExpired:
        result["reaped"] = False
        return result
    result.update(probe_reaped_group(pgid))
    result["reaped"] = True
    result["log_drained"] = output_thread is None or not output_thread.is_alive()
    return result


def run_logged_command(
    command: list[str],
    log_path: Path,
    timeout_seconds: int,
    owned_groups: list[subprocess.Popen[bytes]],
    owned_log_threads: dict[int, threading.Thread],
) -> dict[str, Any]:
    started = time.monotonic()
    result_info: dict[str, Any] = {
        "command": command,
        "log": str(log_path),
        "started_utc": utc_now(),
        "timeout_seconds": timeout_seconds,
        "timed_out": False,
        "process_group_clean": True,
    }
    try:
        process, output_thread, output_event, log_state = start_bounded_logged_child(
            command, log_path, owned_groups
        )
        owned_log_threads[process.pid] = output_thread
    except OSError as error:
        result_info["duration_seconds"] = round(time.monotonic() - started, 3)
        result_info["spawn_error"] = type(error).__name__
        result_info["exit_code"] = None
        return result_info
    deadline = time.monotonic() + timeout_seconds
    try:
        while not leader_exited_unreaped(process.pid):
            if output_event.is_set() or time.monotonic() >= deadline:
                result_info["timed_out"] = time.monotonic() >= deadline
                break
            time.sleep(0.05)
    except BaseException:
        cleanup = clean_owned_process_group(
            process, force=True, output_thread=output_thread
        )
        result_info["cleanup"] = cleanup
        result_info["process_group_clean"] = cleanup["clean"]
        if cleanup["reaped"]:
            owned_groups.remove(process)
            owned_log_threads.pop(process.pid, None)
        raise
    cleanup = clean_owned_process_group(
        process,
        force=result_info["timed_out"] or output_event.is_set(),
        output_thread=output_thread,
    )
    result_info["cleanup"] = cleanup
    result_info["process_group_clean"] = cleanup["clean"]
    result_info["exit_code"] = process.returncode
    result_info["log_bytes"] = log_state["bytes_written"]
    result_info["log_limit_exceeded"] = log_state["limit_exceeded"]
    result_info["log_error"] = log_state["error"]
    if output_event.is_set() and log_state["error"]:
        result_info["log_error"] = log_state["error"]
    if cleanup["reaped"]:
        owned_groups.remove(process)
        owned_log_threads.pop(process.pid, None)
    result_info["duration_seconds"] = round(time.monotonic() - started, 3)
    return result_info


def run_inference(
    command: list[str],
    log_path: Path,
    manifest: dict[str, Any],
    invocation_name: str,
    before_poll: Callable[[int], bool],
    owned_groups: list[subprocess.Popen[bytes]],
    owned_log_threads: dict[int, threading.Thread],
) -> dict[str, Any]:
    """Run one model invocation, polling for competing GPU/unit owners."""
    started_mono = time.monotonic()
    item: dict[str, Any] = {
        "name": invocation_name,
        "command": command,
        "log": str(log_path),
        "started_utc": utc_now(),
        "timeout_seconds": INVOCATION_TIMEOUT_SECONDS,
        "timed_out": False,
        "process_group_clean": True,
        "retries": 0,
    }
    try:
        process, output_thread, output_event, log_state = start_bounded_logged_child(
            command, log_path, owned_groups
        )
        owned_log_threads[process.pid] = output_thread
    except OSError as error:
        item["spawn_error"] = type(error).__name__
        item["exit_code"] = None
        item["duration_seconds"] = round(time.monotonic() - started_mono, 3)
        manifest["invocations"].append(item)
        raise ProbeError(f"could not start {invocation_name}; no retry was attempted") from error

    failure: ProbeError | None = None
    item["gpu_compute_seen"] = False
    timeout_event = threading.Event()

    def timeout_group() -> None:
        timeout_event.set()
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    watchdog = threading.Timer(INVOCATION_TIMEOUT_SECONDS, timeout_group)
    watchdog.daemon = True
    watchdog.start()
    try:
        while not leader_exited_unreaped(process.pid):
            if output_event.is_set():
                reason = "log file exceeded its 16 MiB cap" if log_state["limit_exceeded"] else "log writer failed"
                failure = ProbeError(f"{invocation_name} {reason}; no retry was attempted")
                break
            try:
                if before_poll(process.pid):
                    item["gpu_compute_seen"] = True
            except ProbeError as error:
                failure = error
                break
            if timeout_event.is_set() or time.monotonic() - started_mono >= INVOCATION_TIMEOUT_SECONDS:
                item["timed_out"] = True
                failure = ProbeError(
                    f"{invocation_name} exceeded its 120-second limit; no retry was attempted"
                )
                break
            time.sleep(1.0)
        if output_event.is_set() and failure is None:
            reason = "log file exceeded its 16 MiB cap" if log_state["limit_exceeded"] else "log writer failed"
            failure = ProbeError(f"{invocation_name} {reason}; no retry was attempted")
        if timeout_event.is_set() and failure is None:
            item["timed_out"] = True
            failure = ProbeError(
                f"{invocation_name} exceeded its 120-second limit; no retry was attempted"
            )
    except BaseException:
        cleanup = clean_owned_process_group(
            process, force=True, watchdog=watchdog, output_thread=output_thread
        )
        item["cleanup"] = cleanup
        item["process_group_clean"] = cleanup["clean"]
        item["exit_code"] = process.returncode
        item["log_bytes"] = log_state["bytes_written"]
        item["log_limit_exceeded"] = log_state["limit_exceeded"]
        item["log_error"] = log_state["error"]
        item["duration_seconds"] = round(time.monotonic() - started_mono, 3)
        manifest["invocations"].append(item)
        if cleanup["reaped"]:
            owned_groups.remove(process)
            owned_log_threads.pop(process.pid, None)
        if not cleanup["clean"]:
            manifest["child_cleanup_proven"] = False
        raise
    cleanup = clean_owned_process_group(
        process,
        force=failure is not None,
        watchdog=watchdog,
        output_thread=output_thread,
    )
    item["cleanup"] = cleanup
    item["process_group_clean"] = cleanup["clean"]
    item["exit_code"] = process.returncode
    item["log_bytes"] = log_state["bytes_written"]
    item["log_limit_exceeded"] = log_state["limit_exceeded"]
    item["log_error"] = log_state["error"]
    item["duration_seconds"] = round(time.monotonic() - started_mono, 3)
    manifest["invocations"].append(item)
    if cleanup["reaped"]:
        owned_groups.remove(process)
        owned_log_threads.pop(process.pid, None)
    if not item["process_group_clean"]:
        manifest["child_cleanup_proven"] = False
        raise ProbeError(f"{invocation_name} left an unconfirmed process group") from failure
    if failure is not None:
        raise failure
    if item["exit_code"] != 0:
        raise ProbeError(
            f"{invocation_name} exited unsuccessfully with status {item['exit_code']}; no retry was attempted"
        )
    if not item["gpu_compute_seen"]:
        raise ProbeError(f"{invocation_name} showed no owned GPU compute activity")
    return item


def current_state_snapshot(
    label: str, expected_cgroups: dict[str, str] | None = None
) -> dict[str, Any]:
    units = {unit: systemd_unit_snapshot(unit) for unit in (LLAMA_UNIT, COMFY_UNIT)}
    gpu = gpu_snapshot()
    expected_cgroups = expected_cgroups or {}
    unit_cgroup_processes: dict[str, list[dict[str, Any]]] = {}
    main_processes: dict[str, dict[str, Any] | None] = {}
    for unit, state in units.items():
        path = expected_cgroups.get(unit, state["ControlGroup"])
        members, complete = scan_cgroup_processes(path)
        if not complete:
            raise ProbeError(f"could not prove process membership for {unit} cgroup")
        unit_cgroup_processes[unit] = members
        main_pid = int(state["MainPID"] or "0")
        main_processes[unit] = proc_observation(main_pid) if main_pid > 0 else None
    return {
        "label": label,
        "at_utc": utc_now(),
        "units": units,
        "unit_cgroup_processes": unit_cgroup_processes,
        "main_processes": main_processes,
        "gpu": gpu,
    }


def hash_llama_config() -> dict[str, Any]:
    try:
        metadata = LLAMA_CONFIG.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ProbeError("llama-swap YAML path is not a regular non-symlink file")
        digest = hashlib.sha256(LLAMA_CONFIG.read_bytes()).hexdigest()
    except OSError as error:
        raise ProbeError("could not capture the original llama-swap YAML hash") from error
    return {
        "path": str(LLAMA_CONFIG),
        "sha256": digest,
        "size_bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
    }


def require_comfy_unchanged(snapshot: dict[str, Any], prestate: dict[str, str]) -> None:
    current = snapshot["units"][COMFY_UNIT]
    for key in (
        "LoadState",
        "ActiveState",
        "SubState",
        "UnitFileState",
        "MainPID",
        "ControlGroup",
        "KillMode",
    ):
        if current[key] != prestate[key]:
            raise ProbeError(f"{COMFY_UNIT} changed during the acceptance run")


def require_same_llama_generation(
    initial: dict[str, Any], fresh: dict[str, Any]
) -> None:
    before_unit = initial["units"][LLAMA_UNIT]
    fresh_unit = fresh["units"][LLAMA_UNIT]
    generation_fields = (
        "LoadState",
        "ActiveState",
        "SubState",
        "UnitFileState",
        "MainPID",
        "ControlGroup",
        "KillMode",
        "InvocationID",
    )
    if any(before_unit[key] != fresh_unit[key] for key in generation_fields):
        raise ProbeError("llama-swap.service generation changed before stop")
    if not before_unit["InvocationID"]:
        raise ProbeError("llama-swap.service generation is unknown")
    before_main = initial["main_processes"].get(LLAMA_UNIT)
    fresh_main = fresh["main_processes"].get(LLAMA_UNIT)
    if (
        before_main is None
        or fresh_main is None
        or before_main["pid"] != fresh_main["pid"]
        or before_main["start_time_ticks"] != fresh_main["start_time_ticks"]
    ):
        raise ProbeError("llama-swap.service MainPID birth generation changed before stop")

    def cgroup_identity(snapshot: dict[str, Any]) -> list[tuple[int, int, int]]:
        return sorted(
            (
                member["pid"],
                member["start_time_ticks"],
                member["process_group"],
            )
            for member in snapshot["unit_cgroup_processes"][LLAMA_UNIT]
        )

    if cgroup_identity(initial) != cgroup_identity(fresh):
        raise ProbeError("llama-swap.service cgroup membership changed before stop")

    def gpu_identity(snapshot: dict[str, Any]) -> tuple[list[tuple[str, int]], list[tuple[int, str, int, int]]]:
        gpus = sorted(
            (gpu["uuid"], gpu["memory_used_mib"]) for gpu in snapshot["gpu"]["gpus"]
        )
        processes = sorted(
            (
                process["pid"],
                process["process_name"],
                process["process_group"],
                process["start_time_ticks"],
            )
            for process in snapshot["gpu"]["compute_processes"]
        )
        return gpus, processes

    if gpu_identity(initial) != gpu_identity(fresh):
        raise ProbeError("GPU memory or compute claimants changed before stop")


def require_empty_original_cgroup(snapshot: dict[str, Any]) -> None:
    members = snapshot.get("unit_cgroup_processes", {}).get(LLAMA_UNIT)
    if members is None:
        raise ProbeError("original llama-swap cgroup membership is unavailable")
    if members:
        raise ProbeError("original llama-swap cgroup still contains processes")


def require_llama_stopped(snapshot: dict[str, Any]) -> None:
    unit = snapshot["units"][LLAMA_UNIT]
    if unit["ActiveState"] != "inactive" or unit["SubState"] not in {"dead", "inactive"}:
        raise ProbeError("llama-swap.service is no longer inactive; another owner may have changed it")
    if int(unit["MainPID"] or "0") != 0:
        raise ProbeError("llama-swap.service still has a main process after stop")


def wait_for_llama_stop(manifest: dict[str, Any], timeout_seconds: int = STOP_TIMEOUT_SECONDS) -> None:
    deadline = time.monotonic() + timeout_seconds
    original_cgroup = manifest["prestate"]["units"][LLAMA_UNIT]["ControlGroup"]
    cgroups = {LLAMA_UNIT: original_cgroup}
    while True:
        snapshot = current_state_snapshot("waiting-for-llama-swap-stop", cgroups)
        manifest["snapshots"].append(snapshot)
        require_comfy_unchanged(snapshot, manifest["prestate"]["units"][COMFY_UNIT])
        if snapshot["units"][LLAMA_UNIT]["ActiveState"] == "inactive":
            require_llama_stopped(snapshot)
            if not snapshot["gpu"]["compute_processes"] and not snapshot["unit_cgroup_processes"][LLAMA_UNIT]:
                return
        if time.monotonic() >= deadline:
            if snapshot["units"][LLAMA_UNIT]["ActiveState"] != "inactive":
                raise ProbeError("llama-swap.service did not become inactive within 30 seconds")
            if snapshot["gpu"]["compute_processes"]:
                raise ProbeError("GPU compute claimants remain after stopping llama-swap.service")
            raise ProbeError("original llama-swap.service cgroup still contains processes after stop")
        time.sleep(1.0)


def make_command(
    task: str,
    mode: str,
    output: Path | None = None,
    reference: Path | None = None,
    *,
    check: bool = False,
) -> list[str]:
    command = [str(WRAPPER), "--task", task]
    if check:
        command.append("--check")
        if reference is not None:
            command.extend(["--voice-ref", str(reference), "--reference-text", SYNTHETIC_REFERENCE_TEXT])
        return command
    if output is None:
        raise ValueError("output path is required for synthesis")
    text = SYNTHETIC_CLONE_TEXT if task == "clon" else SYNTHETIC_REFERENCE_TEXT
    command.extend(
        [
            "--text",
            text,
            "--mode",
            mode,
            "--max-tokens",
            str(MAX_TOKENS),
            "--seed",
            str(SEED),
            "--output",
            str(output),
            "--out-dir",
            str(output.parent),
        ]
    )
    if task == "design":
        command.extend(["--instruction", DESIGN_INSTRUCTION])
    if mode == "streaming":
        command.extend(
            [
                "--stream-frames",
                str(STREAM_FRAMES),
                "--stream-lookahead",
                str(STREAM_LOOKAHEAD),
            ]
        )
    if task == "clon":
        if reference is None:
            raise ValueError("clone reference is required")
        command.extend(
            ["--voice-ref", str(reference), "--reference-text", SYNTHETIC_REFERENCE_TEXT]
        )
    return command


def require_target_absent(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise ProbeError(f"refusing to overwrite existing output path: {path.name}")


def stop_llama_swap(manifest: dict[str, Any]) -> None:
    command = ["systemctl", "--user", "stop", LLAMA_UNIT]
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=STOP_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        manifest["service_actions"].append(
            {
                "action": "stop",
                "command": command,
                "duration_seconds": round(time.monotonic() - started, 3),
                "timed_out": True,
            }
        )
        raise ProbeError("systemd stop of llama-swap.service timed out")
    manifest["service_actions"].append(
        {
            "action": "stop",
            "command": command,
            "duration_seconds": round(time.monotonic() - started, 3),
            "exit_code": result.returncode,
            "stderr": result.stderr.strip(),
        }
    )
    if result.returncode != 0:
        raise ProbeError("systemd could not stop llama-swap.service")


def restore_llama_swap_if_needed(
    was_active: bool, manifest: dict[str, Any], cleanup_proven: bool
) -> None:
    if not was_active:
        manifest["restoration"] = {"attempted": False, "reason": "unit was not active at prestate"}
        return
    try:
        current = systemd_unit_snapshot(LLAMA_UNIT)
        if current["ActiveState"] != "active":
            if not cleanup_proven:
                manifest["restoration"] = {
                    "attempted": False,
                    "blocked": True,
                    "reason": "owned child process-group cleanup is unconfirmed",
                }
                raise ProbeError("service start blocked because owned process cleanup is uncertain")
            original_cgroup = manifest["prestate"]["units"][LLAMA_UNIT]["ControlGroup"]
            cgroups = {LLAMA_UNIT: original_cgroup}
            pre_restore = current_state_snapshot("before-service-restore", cgroups)
            manifest["snapshots"].append(pre_restore)
            require_comfy_unchanged(
                pre_restore, manifest["prestate"]["units"][COMFY_UNIT]
            )
            require_llama_stopped(pre_restore)
            assert_no_gpu_compute(pre_restore["gpu"])
            require_empty_original_cgroup(pre_restore)
            expected_hash = manifest["llama_swap_config"]["before"]["sha256"]
            current_hash = hash_llama_config()
            manifest["llama_swap_config"]["before_restore"] = current_hash
            if current_hash["sha256"] != expected_hash:
                manifest["restoration"] = {
                    "attempted": False,
                    "blocked": True,
                    "reason": "llama-swap YAML changed; refusing to start an unreviewed config",
                }
                raise ProbeError("service start blocked because llama-swap YAML changed")
            command = ["systemctl", "--user", "start", LLAMA_UNIT]
            started = time.monotonic()
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=STOP_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                manifest["service_actions"].append(
                    {
                        "action": "start",
                        "command": command,
                        "duration_seconds": round(time.monotonic() - started, 3),
                        "timed_out": True,
                    }
                )
                raise ProbeError("restoration start of llama-swap.service timed out")
            manifest["service_actions"].append(
                {
                    "action": "start",
                    "command": command,
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "exit_code": result.returncode,
                    "stderr": result.stderr.strip(),
                }
            )
            manifest["restoration"] = {
                "attempted": True,
                "command": command,
                "exit_code": result.returncode,
            }
            if result.returncode != 0:
                raise ProbeError("failed to restore the previously active llama-swap.service")
        else:
            manifest["restoration"] = {
                "attempted": False,
                "reason": "already active; no additional claimant was started",
            }
        restored = systemd_unit_snapshot(LLAMA_UNIT)
        manifest["restoration"]["post_state"] = restored
        before = manifest["prestate"]["units"][LLAMA_UNIT]
        stable_fields = (
            "LoadState",
            "ActiveState",
            "SubState",
            "UnitFileState",
            "ControlGroup",
            "KillMode",
        )
        if any(restored[key] != before[key] for key in stable_fields) or int(restored["MainPID"] or "0") <= 0:
            raise ProbeError("llama-swap.service did not return to its active prestate")
    except ProbeError as error:
        if not manifest.get("restoration", {}).get("attempted"):
            manifest["restoration"] = {
                "attempted": False,
                "blocked": True,
                "reason": str(error),
            }
        raise
    except (OSError, subprocess.TimeoutExpired) as error:
        manifest["restoration"] = {
            "attempted": True,
            "error": type(error).__name__,
        }
        raise ProbeError("could not verify restoration of llama-swap.service") from error


def copy_audit_script(output_dir: Path, source: Path, destination_name: str) -> dict[str, Any]:
    destination = output_dir / destination_name
    require_target_absent(destination)
    if source.is_symlink() or not source.is_file():
        raise ProbeError(f"audit source is not a regular file: {source}")
    shutil.copy2(source, destination, follow_symlinks=False)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {"path": destination.name, "sha256": digest, "source": str(source)}


def save_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    validate_manifest_schema(manifest)
    path = output_dir / "manifest.json"
    with path.open("x", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
        manifest_file.flush()
        os.fsync(manifest_file.fileno())


def validate_manifest_schema(manifest: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "purpose",
        "started_utc",
        "finished_utc",
        "policy",
        "prestate",
        "llama_swap_config",
        "snapshots",
        "service_actions",
        "invocations",
        "artifacts",
        "errors",
        "result",
        "restoration",
    }
    if manifest.get("schema_version") != 1 or not required.issubset(manifest):
        raise ProbeError("acceptance manifest does not match schema version 1")
    if manifest["policy"].get("automatic_retry") is not False:
        raise ProbeError("acceptance manifest must state that automatic retries are disabled")
    if not all(isinstance(row, dict) for row in manifest["snapshots"]):
        raise ProbeError("acceptance manifest contains a malformed state snapshot")
    for invocation in manifest["invocations"]:
        if not isinstance(invocation.get("command"), list) or not invocation.get("log"):
            raise ProbeError("acceptance manifest contains an invocation without command/log metadata")
        if invocation.get("retries", 0) != 0:
            raise ProbeError("acceptance manifest records a forbidden inference retry")


def execute_probe(output_dir: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "bounded native Breeze-TTS acceptance; not coordinator GPU-switch proof",
        "started_utc": utc_now(),
        "host": {"uid": os.geteuid(), "pid": os.getpid()},
        "policy": {
            "service_scope": "user manager only",
            "observed_units": [LLAMA_UNIT, COMFY_UNIT],
            "unit_changed": LLAMA_UNIT,
            "other_units_changed": [],
            "model_invocation_timeout_seconds": INVOCATION_TIMEOUT_SECONDS,
            "seed": SEED,
            "max_tokens": MAX_TOKENS,
            "stream_frames": STREAM_FRAMES,
            "stream_lookahead": STREAM_LOOKAHEAD,
            "automatic_retry": False,
            "output_overwrite": False,
            "reference": "first generated standard WAV only",
            "transcript": "public synthetic text embedded in this script",
        },
        "script_copies": [],
        "prestate": {},
        "llama_swap_config": {},
        "snapshots": [],
        "service_actions": [],
        "invocations": [],
        "artifacts": [],
        "errors": [],
        "child_cleanup_proven": True,
        "stop_attempted": False,
        "restoration": {"attempted": False},
    }
    was_active = False
    output_dir_ready = False
    owned_groups: list[subprocess.Popen[bytes]] = []
    owned_log_threads: dict[int, threading.Thread] = {}
    try:
        output_dir_ready = True
        manifest["script_copies"].append(
            copy_audit_script(output_dir, SCRIPT, "probe-breeze-acceptance.py.snapshot")
        )
        manifest["script_copies"].append(
            copy_audit_script(output_dir, WRAPPER, "generate-breeze-tts-2.sh.snapshot")
        )
        pre = current_state_snapshot("before-stop")
        manifest["prestate"] = pre
        manifest["snapshots"].append(pre)
        config_before = hash_llama_config()
        manifest["llama_swap_config"]["before"] = config_before
        require_reviewed_prestate(pre["units"])
        assert_only_llama_swap_claimants(
            pre["gpu"], pre["units"][LLAMA_UNIT]["ControlGroup"]
        )
        was_active = True

        check_log = output_dir / "preflight-tts.log"
        preflight = run_logged_command(
            make_command("tts", "offline", check=True),
            check_log,
            PREFLIGHT_TIMEOUT_SECONDS,
            owned_groups,
            owned_log_threads,
        )
        manifest["invocations"].append({"name": "preflight-tts", **preflight})
        if not preflight.get("process_group_clean", False):
            manifest["child_cleanup_proven"] = False
            raise ProbeError("Breeze preflight left a child process group; no unit was stopped")
        if (
            preflight.get("exit_code") != 0
            or preflight.get("timed_out")
            or preflight.get("log_limit_exceeded")
            or preflight.get("log_error")
        ):
            raise ProbeError("read-only Breeze preflight failed; no unit was stopped")
        config_after_preflight = hash_llama_config()
        manifest["llama_swap_config"]["after_preflight"] = config_after_preflight
        if config_after_preflight["sha256"] != config_before["sha256"]:
            raise ProbeError("llama-swap YAML changed during preflight; no unit was stopped")

        original_cgroup = pre["units"][LLAMA_UNIT]["ControlGroup"]
        cgroups = {LLAMA_UNIT: original_cgroup}
        fresh_pre_stop = current_state_snapshot("immediately-before-stop", cgroups)
        manifest["snapshots"].append(fresh_pre_stop)
        require_reviewed_prestate(fresh_pre_stop["units"])
        require_comfy_unchanged(fresh_pre_stop, pre["units"][COMFY_UNIT])
        require_same_llama_generation(pre, fresh_pre_stop)
        assert_only_llama_swap_claimants(
            fresh_pre_stop["gpu"], original_cgroup
        )
        config_pre_stop = hash_llama_config()
        manifest["llama_swap_config"]["immediately_before_stop"] = config_pre_stop
        if config_pre_stop["sha256"] != config_before["sha256"]:
            raise ProbeError("llama-swap YAML changed after preflight; no unit was stopped")

        manifest["stop_attempted"] = True
        stop_llama_swap(manifest)
        manifest["snapshots"].append({"label": "stop-requested", "at_utc": utc_now()})
        wait_for_llama_stop(manifest)
        idle = current_state_snapshot("gpu-idle-before-inference", cgroups)
        manifest["snapshots"].append(idle)
        require_llama_stopped(idle)
        require_comfy_unchanged(idle, pre["units"][COMFY_UNIT])
        assert_no_gpu_compute(idle["gpu"])
        require_empty_original_cgroup(idle)

        def ensure_exclusive(process_group: int) -> bool:
            snapshot = current_state_snapshot("during-inference", cgroups)
            manifest["snapshots"].append(snapshot)
            require_llama_stopped(snapshot)
            require_comfy_unchanged(snapshot, pre["units"][COMFY_UNIT])
            require_empty_original_cgroup(snapshot)
            for process in snapshot["gpu"]["compute_processes"]:
                if process["process_group"] != process_group:
                    raise ProbeError("an unknown GPU compute claimant appeared during inference")
            return bool(snapshot["gpu"]["compute_processes"])

        generated: dict[str, Path] = {}
        plans = [
            ("standard", "tts", "offline"),
            ("design", "design", "offline"),
            ("streaming", "tts", "streaming"),
        ]
        for name, task, mode in plans:
            target = output_dir / f"{name}.wav"
            require_target_absent(target)
            invocation = run_inference(
                make_command(task, mode, target),
                output_dir / f"{name}.log",
                manifest,
                name,
                ensure_exclusive,
                owned_groups,
                owned_log_threads,
            )
            artifact = validate_wave_file(target, output_dir)
            if mode == "streaming":
                chunks = validate_stream_chunks(target, output_dir)
                artifact["chunks"] = chunks
            invocation["validated_artifact"] = artifact
            manifest["artifacts"].append(artifact)
            generated[name] = target
            after = current_state_snapshot(f"after-{name}", cgroups)
            manifest["snapshots"].append(after)
            require_llama_stopped(after)
            require_comfy_unchanged(after, pre["units"][COMFY_UNIT])
            assert_no_gpu_compute(after["gpu"])
            require_empty_original_cgroup(after)

        clone_reference = generated["standard"]
        validate_wave_file(clone_reference, output_dir)
        clone_check = run_logged_command(
            make_command("clon", "offline", reference=clone_reference, check=True),
            output_dir / "preflight-clone.log",
            PREFLIGHT_TIMEOUT_SECONDS,
            owned_groups,
            owned_log_threads,
        )
        manifest["invocations"].append({"name": "preflight-clone", **clone_check})
        if not clone_check.get("process_group_clean", False):
            manifest["child_cleanup_proven"] = False
            raise ProbeError("clone preflight left a child process group; clone was not started")
        if (
            clone_check.get("exit_code") != 0
            or clone_check.get("timed_out")
            or clone_check.get("log_limit_exceeded")
            or clone_check.get("log_error")
        ):
            raise ProbeError("read-only clone preflight failed; no clone inference was retried")

        target = output_dir / "clone.wav"
        require_target_absent(target)
        invocation = run_inference(
            make_command("clon", "offline", target, clone_reference),
            output_dir / "clone.log",
            manifest,
            "clone",
            ensure_exclusive,
            owned_groups,
            owned_log_threads,
        )
        artifact = validate_wave_file(target, output_dir)
        invocation["validated_artifact"] = artifact
        manifest["artifacts"].append(artifact)
        after = current_state_snapshot("after-clone", cgroups)
        manifest["snapshots"].append(after)
        require_llama_stopped(after)
        require_comfy_unchanged(after, pre["units"][COMFY_UNIT])
        assert_no_gpu_compute(after["gpu"])
        require_empty_original_cgroup(after)
        manifest["result"] = "passed"
    except BaseException as error:
        manifest["result"] = "failed"
        manifest["errors"].append(
            {"type": type(error).__name__, "message": str(error), "at_utc": utc_now()}
        )
        raise
    finally:
        for caught_signal in (signal.SIGINT, signal.SIGTERM):
            signal.signal(caught_signal, signal.SIG_IGN)
        cleanup_records = []
        for process in list(owned_groups):
            try:
                cleanup = clean_owned_process_group(
                    process,
                    force=True,
                    output_thread=owned_log_threads.get(process.pid),
                )
            except BaseException as error:
                cleanup = {
                    "clean": False,
                    "reaped": process.returncode is not None,
                    "error": type(error).__name__,
                }
            cleanup_records.append(cleanup)
            if cleanup.get("reaped"):
                owned_groups.remove(process)
                owned_log_threads.pop(process.pid, None)
            if not cleanup.get("clean"):
                manifest["child_cleanup_proven"] = False
        if cleanup_records:
            manifest["final_owned_group_cleanup"] = cleanup_records
            cleanup_by_pid = {
                record["leader_pid"]: record
                for record in cleanup_records
                if record.get("leader_pid") is not None
            }
            for invocation in manifest["invocations"]:
                previous = invocation.get("cleanup", {})
                leader_pid = previous.get("leader_pid")
                if leader_pid in cleanup_by_pid:
                    invocation["cleanup"] = cleanup_by_pid[leader_pid]
                    invocation["process_group_clean"] = cleanup_by_pid[leader_pid]["clean"]
        manifest["child_cleanup_proven"] = (
            all(
                invocation.get("process_group_clean", True)
                for invocation in manifest["invocations"]
            )
            and all(record.get("clean", False) for record in cleanup_records)
            and not owned_groups
            and not owned_log_threads
        )
        if manifest["stop_attempted"]:
            try:
                after_hash = hash_llama_config()
                manifest["llama_swap_config"]["after_run"] = after_hash
                if after_hash["sha256"] != manifest["llama_swap_config"].get("before", {}).get("sha256"):
                    manifest["errors"].append(
                        {
                            "type": "ConfigChanged",
                            "message": "llama-swap YAML changed during the acceptance run",
                            "at_utc": utc_now(),
                        }
                    )
                    manifest["result"] = "failed-external-config-change"
            except ProbeError as error:
                manifest["llama_swap_config"]["after_run_error"] = str(error)
                manifest["result"] = "failed-config-snapshot"
            try:
                restore_llama_swap_if_needed(
                    was_active, manifest, manifest["child_cleanup_proven"]
                )
            except ProbeError as error:
                manifest["errors"].append(
                    {"type": type(error).__name__, "message": str(error), "at_utc": utc_now()}
                )
                manifest["result"] = "failed-restoration"
            try:
                original_cgroup = manifest["prestate"]["units"][LLAMA_UNIT]["ControlGroup"]
                post = current_state_snapshot(
                    "after-restoration", {LLAMA_UNIT: original_cgroup}
                )
                manifest["poststate"] = post
                manifest["snapshots"].append(post)
                if manifest.get("prestate"):
                    require_comfy_unchanged(
                        post, manifest["prestate"]["units"][COMFY_UNIT]
                    )
                final_hash = hash_llama_config()
                manifest["llama_swap_config"]["after_restore"] = final_hash
                if final_hash["sha256"] != manifest["llama_swap_config"].get("before", {}).get("sha256"):
                    manifest["result"] = "failed-external-config-change"
            except ProbeError as error:
                manifest["poststate_error"] = str(error)
                manifest["result"] = "failed-restoration"
        else:
            manifest["restoration"] = {
                "attempted": False,
                "reason": "llama-swap stop was not attempted",
            }
        manifest["finished_utc"] = utc_now()
        if output_dir_ready:
            try:
                save_manifest(output_dir, manifest)
            except OSError as error:
                print(
                    f"ERROR: could not preserve acceptance manifest ({type(error).__name__})",
                    file=sys.stderr,
                )
    return manifest


def write_pcm_wave(path: Path, frames: int = 16) -> None:
    format_chunk = struct.pack("<HHIIHH", 1, 1, 24000, 48000, 2, 16)
    data = b"\x00\x00" * frames
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(format_chunk))
        + format_chunk
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def write_float_wave(path: Path, samples: tuple[float, ...]) -> None:
    format_chunk = struct.pack("<HHIIHH", 3, 1, 24000, 96000, 4, 32)
    data = b"".join(struct.pack("<f", sample) for sample in samples)
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(format_chunk))
        + format_chunk
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


class OfflineFixtureTests(unittest.TestCase):
    def test_absent_comfy_unit_is_inactive_but_unknown_load_state_is_not(self) -> None:
        units = self.restore_manifest_fixture()["prestate"]["units"]
        units[LLAMA_UNIT]["InvocationID"] = "fixture-generation"
        units[COMFY_UNIT]["LoadState"] = "not-found"
        units[COMFY_UNIT]["ControlGroup"] = ""
        require_reviewed_prestate(units)
        for state in ("error", "bad-setting", "masked"):
            units[COMFY_UNIT]["LoadState"] = state
            with self.assertRaises(ProbeError):
                require_reviewed_prestate(units)

    def test_systemd_and_gpu_csv_parsers(self) -> None:
        props = parse_key_values("LoadState=loaded\nActiveState=active\n")
        self.assertEqual(props["ActiveState"], "active")
        gpus = parse_gpu_rows("GPU-a, Fixture GPU, 128, 24000, 1\n")
        self.assertEqual(gpus[0]["memory_used_mib"], 128)
        processes = parse_compute_rows("314, audio.cpp, 1024\n")
        self.assertEqual(processes[0]["pid"], 314)
        self.assertEqual(parse_compute_rows(""), [])
        with self.assertRaises(ProbeError):
            parse_compute_rows("unknown, process, N/A\n")

    def test_cgroup_match_is_component_bounded(self) -> None:
        process = {"cgroup": ["0::/user.slice/app.slice/llama-swap.service/worker"]}
        self.assertTrue(belongs_to_cgroup(process, "/user.slice/app.slice/llama-swap.service"))
        self.assertFalse(belongs_to_cgroup(process, "/user.slice/app.slice/llama-swap"))
        self.assertFalse(belongs_to_cgroup(process, "/user.slice/app.slice/other.service"))

    def test_pre_stop_revalidation_rejects_new_unit_generation_or_gpu_memory(self) -> None:
        initial = self.generation_snapshot_fixture()
        require_same_llama_generation(initial, self.generation_snapshot_fixture())
        changed_invocation = self.generation_snapshot_fixture()
        changed_invocation["units"][LLAMA_UNIT]["InvocationID"] = "new-generation"
        with self.assertRaises(ProbeError):
            require_same_llama_generation(initial, changed_invocation)
        changed_birth = self.generation_snapshot_fixture()
        changed_birth["main_processes"][LLAMA_UNIT]["start_time_ticks"] += 1
        with self.assertRaises(ProbeError):
            require_same_llama_generation(initial, changed_birth)
        changed_memory = self.generation_snapshot_fixture()
        changed_memory["gpu"]["gpus"][0]["memory_used_mib"] += 1
        with self.assertRaises(ProbeError):
            require_same_llama_generation(initial, changed_memory)
        changed_cgroup = self.generation_snapshot_fixture()
        changed_cgroup["unit_cgroup_processes"][LLAMA_UNIT].append(
            {
                "pid": 43,
                "state": "S",
                "process_group": 42,
                "start_time_ticks": 101,
            }
        )
        with self.assertRaises(ProbeError):
            require_same_llama_generation(initial, changed_cgroup)

    def test_original_cgroup_must_be_empty_before_model_or_restore(self) -> None:
        snapshot = {"unit_cgroup_processes": {LLAMA_UNIT: []}}
        require_empty_original_cgroup(snapshot)
        snapshot["unit_cgroup_processes"][LLAMA_UNIT] = [
            {"pid": 99, "state": "S", "start_time_ticks": 7}
        ]
        with self.assertRaises(ProbeError):
            require_empty_original_cgroup(snapshot)

    def test_nonempty_valid_wave_and_stream_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final_wave = root / "stream.wav"
            write_pcm_wave(final_wave)
            info = validate_wave_file(final_wave, root)
            self.assertGreater(info["frames"], 0)
            chunks_dir = Path(f"{final_wave}.chunks")
            chunks_dir.mkdir()
            write_pcm_wave(chunks_dir / "chunk_0001.wav")
            self.assertEqual(len(validate_stream_chunks(final_wave, root)), 1)
            (chunks_dir / "chunk_0002.wav").write_bytes(b"bad")
            with self.assertRaises(ProbeError):
                validate_stream_chunks(final_wave, root)

    def test_accepts_ieee_float_and_rejects_non_finite_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "float.wav"
            write_float_wave(valid, (0.0, 0.25, -0.5))
            self.assertEqual(validate_wave_file(valid, root)["format"], "IEEE-float")
            invalid = root / "non-finite.wav"
            write_float_wave(invalid, (0.0, float("nan")))
            with self.assertRaises(ProbeError):
                validate_wave_file(invalid, root)

    def test_reaped_process_group_cleanup_never_sends_a_signal(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        process.wait()
        calls: list[tuple[int, int]] = []

        def observe_only(pgid: int, signal_number: int) -> None:
            calls.append((pgid, signal_number))
            if signal_number != 0:
                raise AssertionError("cleanup signaled after reaping the leader")
            raise ProcessLookupError

        with mock.patch("os.killpg", side_effect=observe_only):
            result = clean_owned_process_group(process, force=True)
        self.assertTrue(result["clean"])
        self.assertEqual(calls, [(process.pid, 0)])

    def test_cleanup_cancels_and_joins_watchdog_before_post_reap_probe(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        process.wait()
        watchdog = mock.Mock()
        with mock.patch("os.killpg", side_effect=ProcessLookupError) as killpg:
            result = clean_owned_process_group(
                process, force=True, watchdog=watchdog
            )
        self.assertTrue(result["clean"])
        watchdog.cancel.assert_called_once_with()
        watchdog.join.assert_called_once_with()
        killpg.assert_called_once_with(process.pid, 0)

    def test_child_log_cap_is_enforced_without_unbounded_file_growth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_path = root / "bounded.log"
            owned_groups: list[subprocess.Popen[bytes]] = []
            owned_log_threads: dict[int, threading.Thread] = {}
            with mock.patch(__name__ + ".MAX_CHILD_LOG_BYTES", 8):
                result = run_logged_command(
                    [sys.executable, "-c", "import sys; sys.stdout.write('x' * 1000000)"],
                    log_path,
                    5,
                    owned_groups,
                    owned_log_threads,
                )
            self.assertTrue(result["log_limit_exceeded"])
            self.assertLessEqual(log_path.stat().st_size, 8)
            self.assertTrue(result["process_group_clean"])
            self.assertEqual(owned_groups, [])
            self.assertEqual(owned_log_threads, {})

    def test_thread_start_failure_keeps_unrecorded_child_cleanup_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            output_dir.chmod(0o700)
            llama = self.generation_snapshot_fixture()
            units = llama["units"]
            units[COMFY_UNIT] = {
                "LoadState": "not-found",
                "ActiveState": "inactive",
                "SubState": "dead",
                "UnitFileState": "not-found",
                "MainPID": "0",
                "ControlGroup": "",
                "KillMode": "control-group",
                "InvocationID": "",
            }
            snapshot = {
                "label": "before-stop-fixture",
                "at_utc": "fixture",
                "units": units,
                "unit_cgroup_processes": llama["unit_cgroup_processes"],
                "main_processes": llama["main_processes"],
                "gpu": {"gpus": [], "compute_processes": []},
            }
            spawned_child = mock.Mock(pid=4242, returncode=None, stdout=mock.Mock())
            saved: dict[str, Any] = {}
            uncertain_cleanup = {
                "clean": False,
                "reaped": False,
                "leader_pid": spawned_child.pid,
            }

            def capture_manifest(_output_dir: Path, manifest: dict[str, Any]) -> None:
                saved.update(manifest)

            with (
                mock.patch(__name__ + ".current_state_snapshot", return_value=snapshot),
                mock.patch(__name__ + ".hash_llama_config", return_value={"sha256": "fixture"}),
                mock.patch("subprocess.Popen", return_value=spawned_child),
                mock.patch.object(threading.Thread, "start", side_effect=RuntimeError("thread start failed")),
                mock.patch(
                    __name__ + ".clean_owned_process_group",
                    return_value=uncertain_cleanup,
                ) as cleanup,
                mock.patch(__name__ + ".save_manifest", side_effect=capture_manifest),
            ):
                with self.assertRaisesRegex(RuntimeError, "thread start failed"):
                    execute_probe(output_dir)

            cleanup.assert_called_once()
            self.assertEqual(saved["invocations"], [])
            self.assertEqual(saved["final_owned_group_cleanup"], [uncertain_cleanup])
            self.assertFalse(saved["child_cleanup_proven"])
            self.assertFalse(saved["stop_attempted"])
            self.assertEqual(saved["service_actions"], [])

    def test_restore_is_blocked_when_owned_process_cleanup_is_uncertain(self) -> None:
        manifest = self.restore_manifest_fixture()
        with mock.patch(__name__ + ".systemd_unit_snapshot", return_value={"ActiveState": "inactive"}), mock.patch(
            "subprocess.run"
        ) as run:
            with self.assertRaises(ProbeError):
                restore_llama_swap_if_needed(True, manifest, cleanup_proven=False)
        run.assert_not_called()
        self.assertTrue(manifest["restoration"]["blocked"])

    def test_restore_is_blocked_when_gpu_claimant_remains(self) -> None:
        manifest = self.restore_manifest_fixture()
        inactive = {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "enabled",
            "MainPID": "0",
            "ControlGroup": "/user.slice/app.slice/llama-swap.service",
            "KillMode": "process",
        }
        comfy = self.restore_manifest_fixture()["prestate"]["units"][COMFY_UNIT]
        snapshot = {
            "label": "before-service-restore",
            "units": {LLAMA_UNIT: inactive, COMFY_UNIT: comfy},
            "gpu": {"compute_processes": [{"pid": 123, "process_group": 123}]},
        }
        with mock.patch(
            __name__ + ".systemd_unit_snapshot", return_value=inactive
        ), mock.patch(__name__ + ".current_state_snapshot", return_value=snapshot), mock.patch(
            __name__ + ".hash_llama_config", return_value={"sha256": "before"}
        ), mock.patch("subprocess.run") as run:
            with self.assertRaises(ProbeError):
                restore_llama_swap_if_needed(True, manifest, cleanup_proven=True)
        run.assert_not_called()
        self.assertTrue(manifest["restoration"]["blocked"])

    def test_restore_is_blocked_when_original_service_cgroup_is_not_empty(self) -> None:
        manifest = self.restore_manifest_fixture()
        inactive = {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "enabled",
            "MainPID": "0",
            "ControlGroup": "/user.slice/app.slice/llama-swap.service",
            "KillMode": "process",
            "InvocationID": "old-generation",
        }
        comfy = self.restore_manifest_fixture()["prestate"]["units"][COMFY_UNIT]
        snapshot = {
            "units": {LLAMA_UNIT: inactive, COMFY_UNIT: comfy},
            "unit_cgroup_processes": {
                LLAMA_UNIT: [{"pid": 456, "state": "S"}],
                COMFY_UNIT: [],
            },
            "gpu": {"compute_processes": []},
        }
        with mock.patch(
            __name__ + ".systemd_unit_snapshot", return_value=inactive
        ), mock.patch(__name__ + ".current_state_snapshot", return_value=snapshot), mock.patch(
            __name__ + ".hash_llama_config", return_value={"sha256": "before"}
        ), mock.patch("subprocess.run") as run:
            with self.assertRaises(ProbeError):
                restore_llama_swap_if_needed(True, manifest, cleanup_proven=True)
        run.assert_not_called()
        self.assertTrue(manifest["restoration"]["blocked"])

    def test_restore_does_not_start_a_unit_that_was_not_active(self) -> None:
        manifest = self.restore_manifest_fixture()
        with mock.patch("subprocess.run") as run:
            restore_llama_swap_if_needed(False, manifest, cleanup_proven=True)
        run.assert_not_called()
        self.assertFalse(manifest["restoration"]["attempted"])

    def test_manifest_schema_rejects_retry_and_missing_audit_fields(self) -> None:
        manifest = {
            "schema_version": 1,
            "purpose": "fixture",
            "started_utc": "now",
            "finished_utc": "later",
            "policy": {"automatic_retry": False},
            "prestate": {},
            "llama_swap_config": {},
            "snapshots": [],
            "service_actions": [],
            "invocations": [{"command": ["wrapper"], "log": "x.log", "retries": 1}],
            "artifacts": [],
            "errors": [],
            "result": "failed",
            "restoration": {},
        }
        with self.assertRaises(ProbeError):
            validate_manifest_schema(manifest)
        manifest["invocations"][0]["retries"] = 0
        validate_manifest_schema(manifest)

    @staticmethod
    def restore_manifest_fixture() -> dict[str, Any]:
        return {
            "prestate": {
                "units": {
                    LLAMA_UNIT: {
                        "LoadState": "loaded",
                        "ActiveState": "active",
                        "SubState": "running",
                        "UnitFileState": "enabled",
                        "MainPID": "42",
                        "ControlGroup": "/user.slice/app.slice/llama-swap.service",
                        "KillMode": "process",
                    },
                    COMFY_UNIT: {
                        "LoadState": "loaded",
                        "ActiveState": "inactive",
                        "SubState": "dead",
                        "UnitFileState": "disabled",
                        "MainPID": "0",
                        "ControlGroup": "/user.slice/app.slice/l2m2-comfyui.service",
                        "KillMode": "control-group",
                    },
                }
            },
            "llama_swap_config": {"before": {"sha256": "before"}},
            "snapshots": [],
            "service_actions": [],
            "restoration": {"attempted": False},
        }

    @staticmethod
    def generation_snapshot_fixture() -> dict[str, Any]:
        unit = {
            "LoadState": "loaded",
            "ActiveState": "active",
            "SubState": "running",
            "UnitFileState": "enabled",
            "MainPID": "42",
            "ControlGroup": "/user.slice/app.slice/llama-swap.service",
            "KillMode": "process",
            "InvocationID": "generation-one",
        }
        process = {
            "pid": 42,
            "state": "S",
            "process_group": 42,
            "start_time_ticks": 100,
            "cgroup": ["0::/user.slice/app.slice/llama-swap.service"],
        }
        return {
            "units": {LLAMA_UNIT: unit},
            "main_processes": {LLAMA_UNIT: process.copy()},
            "unit_cgroup_processes": {LLAMA_UNIT: [process.copy()]},
            "gpu": {
                "gpus": [{"uuid": "GPU-fixture", "memory_used_mib": 44}],
                "compute_processes": [],
            },
        }

    def test_private_output_must_be_empty_and_owned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "private"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            self.assertEqual(require_private_empty_output_dir(str(root)), root.resolve())
            (root / "existing.wav").write_bytes(b"keep")
            with self.assertRaises(ProbeError):
                require_private_empty_output_dir(str(root))


def print_plan() -> None:
    print("Native Breeze-TTS acceptance plan (no service or GPU changes):")
    print(f"  Observe only: {LLAMA_UNIT}, {COMFY_UNIT}, nvidia-smi GPU/process snapshots")
    print("  Require reviewed prestate: llama-swap active; ComfyUI inactive")
    print("  Pre-stop: reject any compute PID outside llama-swap.service cgroup")
    print("  Stop only llama-swap; wait for inactive unit and no compute PIDs")
    print("  Run standard, voice-design, streaming, then clone (standard WAV reference)")
    print("  Each inference: deterministic seed 42, max_tokens 256, 120-second timeout, no retry")
    print("  Validate non-empty WAVs and bounded streaming chunks; retain private audit artifacts")
    print("  Finally clean owned process group, then restore llama-swap only if initially active")
    print("  This is not coordinator GPU-switch evidence")


def require_safe_process_api() -> None:
    required = ("P_PID", "WEXITED", "WNOHANG", "WNOWAIT")
    if sys.platform != "linux" or not all(hasattr(os, name) for name in required):
        raise ProbeError("bounded process cleanup requires Linux waitid(WNOWAIT) support")


def handle_interrupt(signum: int, _frame: Any) -> None:
    raise ProbeError(f"received {signal.Signals(signum).name}; cleaning owned work before restore")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="perform the gated acceptance run")
    mode.add_argument("--self-test", action="store_true", help="run offline parser/artifact fixtures")
    parser.add_argument(
        "--output-dir",
        help="existing empty current-user mode-0700 directory for logs, metadata, and generated audio",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(OfflineFixtureTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    if not args.run:
        print_plan()
        print("To enable execution after review: pass --run --output-dir /path/to/private-empty-dir")
        return 0
    if not args.output_dir:
        parser.error("--run requires --output-dir")
    try:
        require_safe_process_api()
    except ProbeError as error:
        parser.error(str(error))
    try:
        output_dir = require_private_empty_output_dir(args.output_dir)
    except ProbeError as error:
        parser.error(str(error))
    if not WRAPPER.is_file() or not os.access(WRAPPER, os.X_OK):
        parser.error(f"Breeze wrapper is unavailable or not executable: {WRAPPER}")
    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)
    try:
        manifest = execute_probe(output_dir)
    except BaseException as error:
        print(f"Acceptance run failed safely: {error}", file=sys.stderr)
        print(f"Audit output: {output_dir}", file=sys.stderr)
        return 1
    print(f"Acceptance result: {manifest['result']}")
    print(f"Audit output: {output_dir}")
    return 0 if manifest.get("result") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
