#!/usr/bin/env python3
"""Opt-in, bounded CPU-only Nagi-Smol acceptance through the foreground coordinator.

Default mode prints the historical plan and performs no model work. ``--run``
is retired with the generic coordinator file-submit entry point; use direct
CPU decision acceptance instead. Helpers remain for bounded direct-run probes.
All model/cache access is forced offline; no service manager or GPU utility is
used. Uncertain cleanup leaves the private fixture in place.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
TEMP_PARENT = Path("/tmp")
TEMP_PREFIX = "l3ms-nagi-accept-"
VENV_RELATIVE = Path("toolkits/l3ds/.venv")
MAX_TIMEOUT_SECONDS = 600
DEFAULT_TIMEOUT_SECONDS = 240
STARTUP_TIMEOUT_SECONDS = 15
CLEANUP_TIMEOUT_SECONDS = 35
CLIENT_TIMEOUT_SECONDS = 15
STATUS_POLL_SECONDS = 0.5
MAX_CLI_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_HISTORY_FILE_BYTES = 16 * 1024 * 1024
MAX_RESULT_PAGES = 256
RESULT_PAGE_BYTES = 128 * 1024
MAX_TRACKED_PROCESSES = 256
CLI_CLEANUP_TIMEOUT_SECONDS = 2

# If a short-lived CLI client cannot be proven stopped, keep the fixture just
# as we do for the foreground coordinator. This is a single-run script.
UNVERIFIED_CLIENT_CLEANUP = False


class ProbeFailure(Exception):
    """A fixed-category harness failure; never holds subprocess output."""

    def __init__(self, category: str):
        self.category = category
        super().__init__(category)


@dataclass(frozen=True)
class ProcRecord:
    pid: int
    ppid: int
    pgrp: int
    start_ticks: int


def parse_proc_stat(pid: int, text: str) -> ProcRecord | None:
    closing = text.rfind(")")
    if closing < 0:
        return None
    fields = text[closing + 1 :].strip().split()
    if len(fields) <= 19:
        return None
    try:
        return ProcRecord(
            pid=pid,
            ppid=int(fields[1]),
            pgrp=int(fields[2]),
            start_ticks=int(fields[19]),
        )
    except ValueError:
        return None


def current_record(pid: int) -> ProcRecord | None:
    path = Path("/proc") / str(pid)
    try:
        metadata = path.stat()
    except FileNotFoundError:
        return None
    except OSError:
        raise ProbeFailure("process identity could not be verified") from None
    if metadata.st_uid != os.getuid():
        # The tracked PID was reused by a different UID, so the original
        # same-UID generation is gone.
        return None
    try:
        text = (path / "stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError):
        raise ProbeFailure("process identity could not be verified") from None
    record = parse_proc_stat(pid, text)
    if record is None:
        raise ProbeFailure("process identity could not be verified")
    return record


def process_table() -> tuple[dict[int, ProcRecord], bool]:
    records: dict[int, ProcRecord] = {}
    certain = True
    try:
        names = os.listdir("/proc")
    except OSError:
        return records, False
    for name in names:
        if not name.isdecimal():
            continue
        pid = int(name)
        path = Path("/proc") / name
        try:
            metadata = path.stat()
            if metadata.st_uid != os.getuid():
                continue
            text = (path / "stat").read_text(encoding="ascii")
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError):
            certain = False
            continue
        record = parse_proc_stat(pid, text)
        if record is None:
            certain = False
        else:
            records[pid] = record
    return records, certain


def capture_owned_tree(
    daemon_pid: int, observed: set[tuple[int, int, int]]
) -> bool:
    records, certain = process_table()
    if not certain:
        return False
    root = records.get(daemon_pid)
    if root is not None:
        observed.add((root.pid, root.start_ticks, root.pgrp))
    children: dict[int, list[ProcRecord]] = {}
    for record in records.values():
        children.setdefault(record.ppid, []).append(record)
    pending = [daemon_pid]
    visited = {daemon_pid}
    while pending:
        parent = pending.pop()
        for record in children.get(parent, []):
            if record.pid in visited:
                continue
            visited.add(record.pid)
            pending.append(record.pid)
            observed.add((record.pid, record.start_ticks, record.pgrp))
            if len(observed) > MAX_TRACKED_PROCESSES:
                return False
    # New-session subprocesses inherit a unique process-group ID. Capture all
    # members even if they are briefly reparented before the next tree walk.
    for record in records.values():
        if record.pgrp == daemon_pid:
            observed.add((record.pid, record.start_ticks, record.pgrp))
            if len(observed) > MAX_TRACKED_PROCESSES:
                return False
    return True


def owned_identities_gone(
    daemon_pid: int,
    daemon_start_ticks: int,
    observed: set[tuple[int, int, int]],
) -> bool:
    try:
        for pid, start_ticks, _ in observed:
            if pid <= 0 or start_ticks <= 0:
                return False
            record = current_record(pid)
            if record is not None and record.start_ticks == start_ticks:
                return False
        record = current_record(daemon_pid)
        if record is not None and record.start_ticks == daemon_start_ticks:
            return False
    except ProbeFailure:
        return False
    groups = {pgrp for _, _, pgrp in observed}
    groups.add(daemon_pid)
    for pgrp in groups:
        if pgrp <= 0:
            return False
        try:
            os.killpg(pgrp, 0)
        except ProcessLookupError:
            continue
        except OSError:
            return False
        return False
    return True


def verify_private_directory(path: Path) -> None:
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ProbeFailure("private directory could not be verified")


def make_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700)
    os.chmod(path, 0o700)
    verify_private_directory(path)


def write_private_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if len(payload) > 16 * 1024:
        raise ProbeFailure("private configuration exceeded its bound")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise ProbeFailure("private fixture file could not be created") from error
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise ProbeFailure("private fixture file could not be verified")


def copy_regular_file(source: Path, destination: Path) -> None:
    try:
        metadata = source.lstat()
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise ProbeFailure("native fixture source is unavailable")
        shutil.copyfile(source, destination)
    except OSError as error:
        raise ProbeFailure("native fixture source is unavailable") from error


def prepare_cpu_repository(fixture_root: Path) -> Path:
    source_l3ds = REPO_ROOT / "toolkits/l3ds"
    source_venv = (REPO_ROOT / VENV_RELATIVE).resolve(strict=True)
    interpreter = source_venv / "bin/python"
    if not source_venv.is_dir() or not (source_venv / "pyvenv.cfg").is_file():
        raise ProbeFailure("pinned Nagi environment is unavailable")
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ProbeFailure("pinned Nagi interpreter is unavailable")

    copy_regular_file(REPO_ROOT / "Cargo.toml", fixture_root / "Cargo.toml")
    copy_regular_file(REPO_ROOT / "llama-swap.yaml", fixture_root / "llama-swap.yaml")
    (fixture_root / "bench-models").mkdir(mode=0o700)

    media_root = fixture_root / "toolkits/l2m2"
    decisions_root = fixture_root / "toolkits/l3ds"
    (fixture_root / "toolkits").mkdir(mode=0o700)
    media_root.mkdir(mode=0o700)
    decisions_root.mkdir(mode=0o700)
    (decisions_root / "decision-models").mkdir(mode=0o700)
    copy_regular_file(
        REPO_ROOT / "toolkits/l2m2/media-runtimes.json",
        media_root / "media-runtimes.json",
    )
    copy_regular_file(
        source_l3ds / "decision-models/runner.py",
        decisions_root / "decision-models" / "runner.py",
    )

    try:
        manifest = json.loads(
            (source_l3ds / "decision-runtimes.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProbeFailure("Nagi manifest is unavailable") from error
    profiles = [profile for profile in manifest.get("profiles", []) if profile.get("id") == "nagi-smol"]
    if len(profiles) != 1:
        raise ProbeFailure("Nagi-Smol profile is unavailable")
    profile = dict(profiles[0])
    profile["device"] = "cpu"
    profile.pop("fallback_device", None)
    profile["entrypoint"] = "decision-models/runner.py"
    write_private_json(
        decisions_root / "decision-runtimes.json",
        {"version": manifest.get("version", "1.0"), "profiles": [profile]},
    )
    try:
        os.symlink(source_venv, decisions_root / ".venv", target_is_directory=True)
    except OSError as error:
        raise ProbeFailure("pinned Nagi environment could not be linked") from error
    return decisions_root


def private_child_environment(fixture_root: Path, data_root: Path) -> dict[str, str]:
    inherited = {
        key: os.environ[key]
        for key in (
            "HOME",
            "XDG_CACHE_HOME",
            "HF_HOME",
            "HF_HUB_CACHE",
            "HF_XET_CACHE",
        )
        if key in os.environ
    }
    if not inherited.get("HOME"):
        raise ProbeFailure("home directory is unavailable")
    inherited.update(
        {
            "PATH": "/usr/bin:/bin",
            "L3MS_ROOT": str(fixture_root),
            "L3MS_DATA_DIR": str(data_root),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "none",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
        }
    )
    return inherited


def daemon_config(
    fixture_root: Path,
    socket_dir: Path,
    lock_dir: Path,
    history_dir: Path,
    standalone_dir: Path,
    session_seconds: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "repository_root": str(fixture_root),
        "socket_directory": str(socket_dir),
        "lock_directory": str(lock_dir),
        "data_directory": str(history_dir),
        "standalone_data_directory": str(standalone_dir),
        "runtimes": ["nagi"],
        "queue_capacity": 1,
        "peer_timeout_ms": 15_000,
        "max_connections": 1024,
        "session_timeout_seconds": session_seconds,
        "recovery_timeout_seconds": 30,
    }


def proc_start_record(
    process: subprocess.Popen[bytes], deadline: float, category: str
) -> ProcRecord:
    while time.monotonic() < deadline:
        try:
            record = current_record(process.pid)
        except ProbeFailure:
            if process.poll() is not None:
                break
            raise
        if record is not None:
            if record.pgrp != process.pid or record.ppid != os.getpid():
                raise ProbeFailure(category)
            return record
        if process.poll() is not None:
            break
        time.sleep(0.01)
    raise ProbeFailure(category)


def wait_for_socket(
    process: subprocess.Popen[bytes], socket_path: Path, deadline: float
) -> None:
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ProbeFailure("foreground coordinator exited before readiness")
        try:
            metadata = socket_path.lstat()
        except FileNotFoundError:
            time.sleep(0.05)
            continue
        if (
            not stat.S_ISSOCK(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise ProbeFailure("coordinator socket identity or permissions are invalid")
        return
    raise ProbeFailure("foreground coordinator readiness deadline expired")


def run_cli(
    binary: Path,
    args: list[str],
    cwd: Path,
    environment: dict[str, str],
    timeout: float,
) -> dict[str, Any] | None:
    global UNVERIFIED_CLIENT_CLEANUP
    process: subprocess.Popen[bytes] | None = None
    identity: ProcRecord | None = None
    observed: set[tuple[int, int, int]] = set()
    capture_certain = True
    try:
        process = subprocess.Popen(
            [str(binary), *args],
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
            bufsize=0,
        )
    except OSError:
        return None
    deadline = time.monotonic() + max(0.1, timeout)
    try:
        identity = proc_start_record(
            process,
            min(deadline, time.monotonic() + 1.0),
            "coordinator client identity could not be verified",
        )
        observed.add((identity.pid, identity.start_ticks, identity.pgrp))
    except ProbeFailure:
        UNVERIFIED_CLIENT_CLEANUP = True
        return None

    assert process.stdout is not None
    try:
        os.set_blocking(process.stdout.fileno(), False)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            output = bytearray()
            oversized = False
            eof = False
            next_process_capture = time.monotonic() + 0.5
            while not eof or process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if not stop_owned_daemon(
                        process,
                        identity,
                        observed,
                        wait_seconds=CLI_CLEANUP_TIMEOUT_SECONDS,
                    ):
                        UNVERIFIED_CLIENT_CLEANUP = True
                    return None
                try:
                    if eof:
                        time.sleep(min(0.05, remaining))
                        events = []
                    else:
                        events = selector.select(min(0.1, remaining))
                    for key, _ in events:
                        chunk = os.read(key.fileobj.fileno(), 8192)
                        if not chunk:
                            eof = True
                            selector.unregister(key.fileobj)
                            continue
                        if not oversized:
                            if len(output) + len(chunk) > MAX_CLI_OUTPUT_BYTES:
                                output.clear()
                                oversized = True
                            else:
                                output.extend(chunk)
                except (OSError, ValueError):
                    if process.poll() is None:
                        if not stop_owned_daemon(
                            process,
                            identity,
                            observed,
                            wait_seconds=CLI_CLEANUP_TIMEOUT_SECONDS,
                        ):
                            UNVERIFIED_CLIENT_CLEANUP = True
                    else:
                        try:
                            process.wait(timeout=0)
                        except subprocess.TimeoutExpired:
                            UNVERIFIED_CLIENT_CLEANUP = True
                        if not owned_identities_gone(
                            process.pid, identity.start_ticks, observed
                        ):
                            UNVERIFIED_CLIENT_CLEANUP = True
                    return None
                if process.poll() is None and time.monotonic() >= next_process_capture:
                    capture_certain = capture_owned_tree(process.pid, observed) and capture_certain
                    next_process_capture = time.monotonic() + 0.5
            try:
                returncode = process.wait(timeout=0)
            except subprocess.TimeoutExpired:
                return None
            capture_certain = capture_owned_tree(process.pid, observed) and capture_certain
            if not capture_certain or not owned_identities_gone(
                process.pid, identity.start_ticks, observed
            ):
                UNVERIFIED_CLIENT_CLEANUP = True
                return None
    except BaseException:
        if process.poll() is None:
            if identity is None or not stop_owned_daemon(
                process,
                identity,
                observed,
                wait_seconds=CLI_CLEANUP_TIMEOUT_SECONDS,
            ):
                UNVERIFIED_CLIENT_CLEANUP = True
        else:
            try:
                process.wait(timeout=0)
            except subprocess.TimeoutExpired:
                UNVERIFIED_CLIENT_CLEANUP = True
            if identity is None or not owned_identities_gone(
                process.pid, identity.start_ticks, observed
            ):
                UNVERIFIED_CLIENT_CLEANUP = True
        raise
    finally:
        process.stdout.close()
    if returncode != 0 or oversized:
        return None
    try:
        value = json.loads(output)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def query_status(
    binary: Path,
    socket_dir: Path,
    fixture_root: Path,
    environment: dict[str, str],
    request_id: str,
    deadline: float,
) -> dict[str, Any] | None:
    return run_cli(
        binary,
        [
            "--coordinator-status",
            request_id,
            "--coordinator-socket-dir",
            str(socket_dir),
        ],
        fixture_root,
        environment,
        min(CLIENT_TIMEOUT_SECONDS, max(0.1, deadline - time.monotonic())),
    )


def await_success(
    binary: Path,
    daemon: subprocess.Popen[bytes],
    socket_dir: Path,
    fixture_root: Path,
    environment: dict[str, str],
    request_id: str,
    deadline: float,
    observed: set[tuple[int, int, int]],
) -> tuple[int, str, bool]:
    status_queries = 0
    not_found = 0
    process_evidence_certain = True
    while time.monotonic() < deadline:
        if daemon.poll() is not None:
            raise ProbeFailure("foreground coordinator exited during request")
        if not capture_owned_tree(daemon.pid, observed):
            process_evidence_certain = False
        response = query_status(
            binary,
            socket_dir,
            fixture_root,
            environment,
            request_id,
            deadline,
        )
        status_queries += 1
        if response and response.get("type") == "status" and response.get("request_id") == request_id:
            state = response.get("state")
            if state == "succeeded":
                return status_queries, state, process_evidence_certain
            if state in {"failed", "cancelled", "interrupted"}:
                raise ProbeFailure("Nagi coordinator request did not succeed")
            if state == "not_found":
                not_found += 1
                if not_found >= 4:
                    raise ProbeFailure("submitted request was not found; no resubmission was attempted")
            elif state not in {"queued", "running"}:
                raise ProbeFailure("coordinator status response was invalid")
        time.sleep(STATUS_POLL_SECONDS)
    raise ProbeFailure("Nagi coordinator request deadline expired; no resubmission was attempted")


def fetch_result_pages(
    binary: Path,
    socket_dir: Path,
    fixture_root: Path,
    environment: dict[str, str],
    request_id: str,
    deadline: float,
) -> tuple[int, int]:
    cursor = 0
    page_count = 0
    event_count = 0
    while page_count < MAX_RESULT_PAGES and time.monotonic() < deadline:
        response = run_cli(
            binary,
            [
                "--request-results",
                request_id,
                "--coordinator-socket-dir",
                str(socket_dir),
                "--after-cursor",
                str(cursor),
                "--result-page-bytes",
                str(RESULT_PAGE_BYTES),
            ],
            fixture_root,
            environment,
            min(CLIENT_TIMEOUT_SECONDS, max(0.1, deadline - time.monotonic())),
        )
        if (
            response is None
            or response.get("type") != "results"
            or response.get("request_id") != request_id
            or response.get("state") != "succeeded"
        ):
            raise ProbeFailure("coordinator result page could not be validated")
        page = response.get("page")
        if not isinstance(page, dict) or page.get("request_id") != request_id:
            raise ProbeFailure("coordinator result page could not be validated")
        if page.get("availability") != "available" or page.get("terminal") != {"status": "succeeded"}:
            raise ProbeFailure("coordinator result page is incomplete")
        events = page.get("events")
        next_cursor = page.get("next_cursor")
        latest_cursor = page.get("latest_cursor")
        if (
            not isinstance(events, list)
            or not isinstance(next_cursor, int)
            or not isinstance(latest_cursor, int)
            or next_cursor < cursor
            or next_cursor > latest_cursor
        ):
            raise ProbeFailure("coordinator result cursor is invalid")
        event_count += len(events)
        page_count += 1
        if next_cursor == latest_cursor:
            if event_count == 0:
                raise ProbeFailure("coordinator returned no native decision result")
            return page_count, event_count
        if next_cursor == cursor or not events:
            raise ProbeFailure("coordinator result cursor did not advance")
        cursor = next_cursor
    raise ProbeFailure("coordinator result pagination exceeded its bound")


def fetch_and_validate_output(
    binary: Path,
    socket_dir: Path,
    fixture_root: Path,
    environment: dict[str, str],
    request_id: str,
    deadline: float,
) -> tuple[int, int]:
    response = run_cli(
        binary,
        [
            "--request-output",
            request_id,
            "--request-lane",
            "decisions",
            "--coordinator-socket-dir",
            str(socket_dir),
        ],
        fixture_root,
        environment,
        min(CLIENT_TIMEOUT_SECONDS, max(0.1, deadline - time.monotonic())),
    )
    if (
        response is None
        or response.get("request_id") != request_id
        or response.get("complete") is not True
        or response.get("terminal") != {"status": "succeeded"}
    ):
        raise ProbeFailure("coordinator native output is incomplete")
    output = response.get("output")
    if not isinstance(output, dict) or output.get("lane") != "decisions":
        raise ProbeFailure("coordinator native output has the wrong lane")
    answers = output.get("answers")
    if not isinstance(answers, list) or not answers:
        raise ProbeFailure("coordinator native decision is missing")
    criteria = {"accept", "reject"}
    valid = any(
        isinstance(answer, dict)
        and isinstance(answer.get("route"), dict)
        and answer["route"].get("choice") in criteria
        for answer in answers
    )
    if not valid:
        raise ProbeFailure("coordinator native decision is invalid")
    metadata = response.get("native_decision_metadata")
    if not isinstance(metadata, dict) or set(metadata) - {"native_latency_ns", "native_usage"}:
        raise ProbeFailure("native decision metadata is missing or invalid")
    latency = metadata.get("native_latency_ns")
    if type(latency) is not int or not 0 < latency <= (2**64 - 1):
        raise ProbeFailure("native decision latency is missing or invalid")
    if "native_usage" in metadata and not isinstance(metadata["native_usage"], dict):
        raise ProbeFailure("native decision usage is invalid")
    return len(answers), latency


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def validate_durable_history(history_dir: Path, request_id: str) -> tuple[int, int]:
    path = history_dir / "jobs.json"
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or before.st_size > MAX_HISTORY_FILE_BYTES
        ):
            raise ProbeFailure("durable request history could not be verified")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags)
    except ProbeFailure:
        raise
    except OSError:
        raise ProbeFailure("durable request history could not be verified") from None

    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or opened.st_size > MAX_HISTORY_FILE_BYTES
        ):
            raise ProbeFailure("durable request history could not be verified")
        chunks = bytearray()
        while len(chunks) <= MAX_HISTORY_FILE_BYTES:
            chunk = os.read(descriptor, min(64 * 1024, MAX_HISTORY_FILE_BYTES + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
        if len(chunks) > MAX_HISTORY_FILE_BYTES:
            raise ProbeFailure("durable request history exceeded its bound")
    except OSError:
        raise ProbeFailure("durable request history could not be read") from None
    finally:
        os.close(descriptor)

    try:
        rows = json.loads(chunks.decode("utf-8"), object_pairs_hook=reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ProbeFailure("durable request history was malformed") from None
    if not isinstance(rows, list) or len(rows) > 200 or not all(isinstance(row, dict) for row in rows):
        raise ProbeFailure("durable request history was malformed")
    matches = [row for row in rows if row.get("request_id") == request_id]
    if len(matches) != 1:
        raise ProbeFailure("caller ID was not uniquely retained in durable history")
    row = matches[0]
    if not (
        row.get("toolkit") == "l3ds"
        and row.get("profile") == "nagi-smol"
        and row.get("task") == "decision"
        and row.get("result_state") == "succeeded"
        and row.get("exit") == "0"
        and row.get("mode") == "cross-toolkit"
        and row.get("script_path") == ""
        and isinstance(row.get("finished_at"), str)
    ):
        raise ProbeFailure("durable caller history was not terminal and successful")
    return len(rows), len(matches)


def stop_owned_daemon(
    process: subprocess.Popen[bytes] | None,
    identity: ProcRecord | None,
    observed: set[tuple[int, int, int]],
    wait_seconds: float = CLEANUP_TIMEOUT_SECONDS,
) -> bool:
    if process is None:
        return True
    if identity is None:
        return False
    returncode = process.poll()
    if returncode is None:
        try:
            current = current_record(process.pid)
        except ProbeFailure:
            return False
        if current != identity:
            return False
        try:
            process.send_signal(signal.SIGTERM)
        except OSError:
            return False
        try:
            returncode = process.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            return False
    else:
        try:
            returncode = process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            return False
    return owned_identities_gone(process.pid, identity.start_ticks, observed)


def safe_remove_fixture(path: Path) -> bool:
    try:
        metadata = path.lstat()
        if (
            path.parent != TEMP_PARENT
            or not path.name.startswith(TEMP_PREFIX)
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            return False
        shutil.rmtree(path)
    except OSError:
        return False
    return not path.exists()


def validate_binary(path: Path) -> Path:
    expected_parent = (REPO_ROOT / "target/debug").resolve()
    if not path.is_absolute():
        raise ProbeFailure("built L3MS CLI path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ProbeFailure("built L3MS CLI is unavailable") from error
    if (
        resolved.parent != expected_parent
        or not resolved.is_file()
        or not os.access(resolved, os.X_OK)
        or path.is_symlink()
    ):
        raise ProbeFailure("built L3MS CLI could not be verified")
    return resolved


def run_acceptance(binary: Path, timeout_seconds: int) -> dict[str, Any]:
    global UNVERIFIED_CLIENT_CLEANUP
    UNVERIFIED_CLIENT_CLEANUP = False
    if not sys.platform.startswith("linux"):
        return {"ok": False, "error": "foreground CPU coordinator requires Linux"}
    fixture_root: Path | None = None
    try:
        fixture_root = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX, dir=TEMP_PARENT))
        os.chmod(fixture_root, 0o700)
        verify_private_directory(fixture_root)
    except (OSError, ProbeFailure):
        removed = fixture_root is not None and safe_remove_fixture(fixture_root)
        return {
            "ok": False,
            "error": "private temporary directory is unavailable",
            "temporary_fixture_retained": fixture_root is not None and not removed,
            "temporary_fixture_removed": removed,
        }

    daemon: subprocess.Popen[bytes] | None = None
    daemon_identity: ProcRecord | None = None
    observed: set[tuple[int, int, int]] = set()
    request_verified = False
    process_evidence_certain = True
    failure_category: str | None = None
    evidence: dict[str, Any] = {}
    try:
        prepare_cpu_repository(fixture_root)
        socket_dir = fixture_root / "socket"
        lock_dir = fixture_root / "locks"
        history_dir = fixture_root / "history"
        standalone_dir = fixture_root / "standalone-history"
        settings_dir = fixture_root / "settings"
        for directory in (socket_dir, lock_dir, history_dir, standalone_dir, settings_dir):
            make_private_directory(directory)

        environment = private_child_environment(fixture_root, settings_dir)
        session_seconds = min(timeout_seconds + 60, 660)
        config_path = fixture_root / "coordinator.json"
        write_private_json(
            config_path,
            daemon_config(
                fixture_root,
                socket_dir,
                lock_dir,
                history_dir,
                standalone_dir,
                session_seconds,
            ),
        )
        request_id = "nagi-smol-accept-" + uuid.uuid4().hex
        request_path = fixture_root / "submit.json"
        write_private_json(
            request_path,
            {
                "version": 1,
                "message": {
                    "type": "submit",
                    "request_id": request_id,
                    "timeout_ms": timeout_seconds * 1000,
                    "request": {
                        "lane": "decisions",
                        "profile_id": "nagi-smol",
                        "state": {"acceptance_probe": "synthetic-local-cpu"},
                        "questions": {
                            "route": {
                                "type": "choice",
                                "criteria": {
                                    "accept": "The local CPU decision path is available.",
                                    "reject": "The local CPU decision path is unavailable.",
                                },
                            }
                        },
                    },
                },
            },
        )
        daemon = subprocess.Popen(
            [str(binary), "--coordinator-cpu-daemon", str(config_path)],
            cwd=fixture_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
        daemon_identity = proc_start_record(
            daemon,
            time.monotonic() + 1.0,
            "foreground coordinator identity could not be verified",
        )
        observed.add((daemon_identity.pid, daemon_identity.start_ticks, daemon_identity.pgrp))
        wait_for_socket(
            daemon,
            socket_dir / "coordinator.sock",
            time.monotonic() + STARTUP_TIMEOUT_SECONDS,
        )

        submit_reply = run_cli(
            binary,
            [
                "--request-submit",
                str(request_path),
                "--coordinator-socket-dir",
                str(socket_dir),
            ],
            fixture_root,
            environment,
            CLIENT_TIMEOUT_SECONDS,
        )
        submit_acknowledged = bool(
            submit_reply
            and submit_reply.get("type") == "accepted"
            and submit_reply.get("operation") == "submit"
            and submit_reply.get("request_id") == request_id
        )
        # A lost or malformed submit reply is reconciled only by the same-ID
        # read-only status query below. The submit operation is never repeated.
        deadline = time.monotonic() + timeout_seconds
        status_queries, terminal_state, status_process_evidence = await_success(
            binary,
            daemon,
            socket_dir,
            fixture_root,
            environment,
            request_id,
            deadline,
            observed,
        )
        process_evidence_certain = process_evidence_certain and status_process_evidence
        page_count, event_count = fetch_result_pages(
            binary,
            socket_dir,
            fixture_root,
            environment,
            request_id,
            deadline,
        )
        history_rows, history_matches = validate_durable_history(history_dir, request_id)
        answer_count, native_latency_ns = fetch_and_validate_output(
            binary,
            socket_dir,
            fixture_root,
            environment,
            request_id,
            deadline,
        )
        request_verified = True
        evidence = {
            "inference_submissions": 1,
            "submit_reply_confirmed": submit_acknowledged,
            "caller_id_matches": True,
            "terminal_state": terminal_state,
            "status_queries": status_queries,
            "result_pages": page_count,
            "result_events": event_count,
            "decision_answers": answer_count,
            "native_decision_validated": True,
            "native_latency_ns": native_latency_ns,
            "native_metadata_validated": True,
            "durable_history_rows": history_rows,
            "durable_terminal_matches": history_matches,
            "offline": True,
            "cuda_visible_devices_empty": True,
            "threads_per_library": 2,
        }
    except ProbeFailure as error:
        failure_category = error.category
    except Exception:
        # Do not expose exception text: it can contain request data, native
        # paths, environment values, or model/runtime details.
        failure_category = "acceptance harness failed"
    finally:
        cleanup_proven = False
        if daemon is not None and daemon_identity is not None:
            capture_certain = capture_owned_tree(daemon.pid, observed)
            stopped = stop_owned_daemon(daemon, daemon_identity, observed)
            cleanup_proven = (
                capture_certain
                and process_evidence_certain
                and stopped
                and owned_identities_gone(daemon.pid, daemon_identity.start_ticks, observed)
                and not UNVERIFIED_CLIENT_CLEANUP
            )
        elif daemon is None:
            cleanup_proven = not UNVERIFIED_CLIENT_CLEANUP

        fixture_removed = False
        if cleanup_proven and fixture_root is not None:
            fixture_removed = safe_remove_fixture(fixture_root)
            cleanup_proven = fixture_removed

    result = {"ok": request_verified and cleanup_proven, **evidence}
    if failure_category is not None:
        result["error"] = failure_category
    if cleanup_proven:
        if daemon is not None:
            result["daemon_stopped"] = daemon.poll() is not None
            result["owned_process_groups_empty"] = True
        result["temporary_fixture_removed"] = fixture_removed
        result["temporary_fixture_retained"] = False
    else:
        result["error"] = result.get("error", "acceptance or cleanup proof incomplete")
        result["cleanup_status"] = "unverified"
        result["temporary_fixture_retained"] = fixture_root is not None
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="retired coordinator path: refuses before any model or service work",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"request deadline, 30..{MAX_TIMEOUT_SECONDS} seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=REPO_ROOT / "target/debug/l3ms",
        help="built repository target/debug/l3ms executable",
    )
    args = parser.parse_args()
    if not 30 <= args.timeout_seconds <= MAX_TIMEOUT_SECONDS:
        parser.error(f"--timeout-seconds must be between 30 and {MAX_TIMEOUT_SECONDS}")
    return args


def main() -> int:
    args = parse_args()
    if not args.run:
        print(
            json.dumps(
                {
                    "mode": "plan_only",
                    "run_required": True,
                    "model_loads": 0,
                    "network_access": "offline",
                    "cuda_visible_devices": "empty",
                    "coordinator": "foreground_cpu_nagi_only",
                    "submissions_per_run": 1,
                },
                separators=(",", ":"),
            )
        )
        return 0
    result = {
        "ok": False,
        "error": "historical coordinator file-submit probe retired; use direct CPU decisions",
        "temporary_fixture_retained": False,
        "model_loads": 0,
    }
    print(json.dumps(result, separators=(",", ":")))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
