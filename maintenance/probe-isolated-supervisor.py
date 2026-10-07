#!/usr/bin/env python3
"""Opt-in, model-free probe for the hidden isolated-engine supervisor.

The default invocation is help-only. With --run, this starts one uniquely
named transient user unit containing only target/debug/l3ms and a Python
stdlib loopback HTTP fixture. The unit uses PrivateNetwork=yes and one manager-
opened coordinator namespace descriptor. No installed unit, real engine,
model, GPU, network download, or host TCP listener is touched. Not run by CI.
Manager output requests selected properties and is size-checked after capture;
that check rejects anomalous output but is not a pre-allocation memory bound.
"""

import argparse
import json
import os
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


HELPER_MARKER = "--l3ms-isolated-engine-supervisor-v1"
ENGINE_SOCKET_NAME = "engine.sock"
MANAGER_TIMEOUT_SECONDS = 5.0
UNIT_RUNTIME_SECONDS = 20
UNIT_STOP_SECONDS = 2
UNIT_START_SECONDS = 5
READY_WAIT_SECONDS = 8.0
READY_POLL_SECONDS = 0.2
MAX_READY_POLLS = 40
MAX_MANAGER_OUTPUT_BYTES = 64 * 1024
MAX_HTTP_RESPONSE_BYTES = 4096
NATIVE_EXCHANGE_SECONDS = 5.0
MAX_HEALTH_ATTEMPTS = 40
HEALTH_RETRY_SECONDS = 0.1

BACKEND_CODE = r"""
import http.server
import fcntl
import json
import os
import socket
import struct
import sys

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/probe":
            self.send_error(404)
            return
        namespace = os.stat("/proc/self/ns/net")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control:
            request = struct.pack("256s", b"lo")
            flags = struct.unpack("H", fcntl.ioctl(control.fileno(), 0x8913, request)[16:18])[0]
        body = json.dumps({
            "pid": os.getpid(),
            "namespace_device": namespace.st_dev,
            "namespace_inode": namespace.st_ino,
            "loopback_up": bool(flags & 1),
        }, separators=(",", ":")).encode("ascii")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_arguments):
        pass

server = http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler)
server.serve_forever()
"""


class ProbeFailure(Exception):
    """A fixed diagnostic with no captured argv, manager output, or payload."""


class BackendNotReady(Exception):
    """An idempotent proof GET may be repeated within its shared deadline."""


def manager_command(*arguments, timeout=MANAGER_TIMEOUT_SECONDS):
    # systemctl calls below request only selected properties. subprocess.run
    # captures output before these size checks, so the limit rejects anomalous
    # manager responses but is not a pre-allocation memory bound.
    bounded_timeout = min(float(timeout), MANAGER_TIMEOUT_SECONDS)
    try:
        result = subprocess.run(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=bounded_timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ProbeFailure("manager command failed or timed out") from None
    if (
        result.returncode != 0
        or len(result.stdout) > MAX_MANAGER_OUTPUT_BYTES
        or len(result.stderr) > MAX_MANAGER_OUTPUT_BYTES
    ):
        raise ProbeFailure("manager command failed or exceeded its output limit")
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        raise ProbeFailure("manager response was invalid") from None


def unit_snapshot(unit, timeout=MANAGER_TIMEOUT_SECONDS):
    output = manager_command(
        "systemctl",
        "--user",
        "show",
        unit,
        "--property=LoadState",
        "--property=ActiveState",
        "--property=SubState",
        "--property=MainPID",
        "--property=InvocationID",
        "--property=Type",
        "--property=PrivateNetwork",
        "--property=JoinsNamespaceOf",
        "--property=ControlGroup",
        "--property=KillMode",
        "--property=Restart",
        "--property=Transient",
        timeout=timeout,
    )
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


def valid_invocation_id(value):
    return (
        len(value) == 32
        and all(character in "0123456789abcdefABCDEF" for character in value)
        and any(character != "0" for character in value)
    )


def namespace_identity(pid):
    metadata = os.stat(f"/proc/{pid}/ns/net")
    return (metadata.st_dev, metadata.st_ino)


def process_start_time(pid):
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").rsplit(")", 1)
    if len(fields) != 2:
        raise ProbeFailure("harness process identity unavailable")
    remaining = fields[1].split()
    if len(remaining) <= 19:
        raise ProbeFailure("harness process identity unavailable")
    value = int(remaining[19])
    if value <= 0:
        raise ProbeFailure("harness process identity unavailable")
    return value


def process_parent(pid):
    for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    raise ProbeFailure("backend process identity unavailable")


def process_cgroup(pid):
    value = Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii")
    if not value or len(value) > MAX_MANAGER_OUTPUT_BYTES:
        raise ProbeFailure("backend process identity unavailable")
    return value


def proc_cgroup_path(value):
    entries = [line for line in value.splitlines() if line.startswith("0::")]
    if len(entries) != 1:
        raise ProbeFailure("backend cgroup identity unavailable")
    path = entries[0][3:]
    if not path.startswith("/") or any(part in ("", ".", "..") for part in path.split("/")[1:]):
        raise ProbeFailure("backend cgroup identity unavailable")
    return path


def process_identity_presence(pid, expected_start_time):
    """Return True for the exact process, False if gone/reused, None if unknown."""
    try:
        actual_start_time = process_start_time(pid)
    except FileNotFoundError:
        return False
    except (OSError, ProbeFailure, ValueError, UnicodeDecodeError):
        return None
    return actual_start_time == expected_start_time


def cgroup_is_empty(cgroup):
    if not cgroup or not cgroup.startswith("/") or "\x00" in cgroup:
        return False
    components = cgroup.split("/")[1:]
    if not components or any(component in ("", ".", "..") for component in components):
        return False
    root = Path("/sys/fs/cgroup")
    try:
        root_metadata = root.lstat()
    except OSError:
        return False
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_ISLNK(root_metadata.st_mode)
        or not (root / "cgroup.controllers").is_file()
    ):
        return False
    target = root
    for component in components:
        target = target / component
        try:
            metadata = target.lstat()
        except FileNotFoundError:
            return True
        except OSError:
            return False
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            return False
    seen = 0

    def walk_error(error):
        raise error

    try:
        for directory, subdirectories, files in os.walk(
            target,
            followlinks=False,
            onerror=walk_error,
        ):
            seen += 1
            if seen > 256 or len(subdirectories) > 256:
                return False
            if "cgroup.procs" not in files:
                return False
            contents = Path(directory, "cgroup.procs").read_text(encoding="ascii")
            if len(contents) > MAX_MANAGER_OUTPUT_BYTES or contents.strip():
                return False
            if any((Path(directory) / child).is_symlink() for child in subdirectories):
                return False
    except OSError:
        return False
    return True


def make_private_directories():
    root = None
    root_identity = None
    try:
        root = Path(tempfile.mkdtemp(prefix="l3ms-supervisor-probe-", dir="/tmp"))
        os.chmod(root, 0o700)
        root_metadata = root.lstat()
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or root_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(root_metadata.st_mode) != 0o700
        ):
            raise ProbeFailure("private probe directory unavailable")
        root_identity = (root_metadata.st_dev, root_metadata.st_ino)
        transport = root / "transport"
        transport.mkdir(mode=0o700)
        os.chmod(transport, 0o700)
        transport_metadata = transport.lstat()
        if (
            not stat.S_ISDIR(transport_metadata.st_mode)
            or transport_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(transport_metadata.st_mode) != 0o700
        ):
            raise ProbeFailure("private transport directory unavailable")
        return root, root_identity, transport
    except ProbeFailure:
        if root is not None and root_identity is not None:
            if not remove_private_directory(root, root_identity):
                raise ProbeFailure("private probe directory cleanup uncertain") from None
        raise
    except OSError:
        if root is not None and root_identity is not None:
            if not remove_private_directory(root, root_identity):
                raise ProbeFailure("private probe directory cleanup uncertain") from None
        raise ProbeFailure("private probe directory unavailable") from None


def remove_private_directory(root, identity):
    try:
        metadata = root.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or (metadata.st_dev, metadata.st_ino) != identity
        ):
            return False
        shutil.rmtree(root)
        return not root.exists()
    except OSError:
        return False


def wait_for_ready(unit, captured_invocation, captured_cgroup):
    deadline = time.monotonic() + READY_WAIT_SECONDS
    last = {}
    for _attempt in range(MAX_READY_POLLS):
        if time.monotonic() >= deadline:
            break
        remaining = deadline - time.monotonic()
        last = unit_snapshot(unit, timeout=min(MANAGER_TIMEOUT_SECONDS, remaining))
        if last.get("LoadState") == "not-found":
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            continue

        invocation = last.get("InvocationID", "")
        if valid_invocation_id(invocation):
            if captured_invocation[0] is not None and captured_invocation[0] != invocation:
                raise ProbeFailure("transient unit generation changed")
            captured_invocation[0] = invocation
            cgroup = last.get("ControlGroup", "")
            if cgroup:
                if captured_cgroup[0] is not None and captured_cgroup[0] != cgroup:
                    raise ProbeFailure("transient unit control group changed")
                captured_cgroup[0] = cgroup

        state = last.get("ActiveState", "")
        if state == "active":
            if not valid_invocation_id(captured_invocation[0] or ""):
                raise ProbeFailure("transient unit identity unavailable")
            if last.get("SubState") != "running":
                raise ProbeFailure("supervisor readiness was not confirmed")
            if last.get("Type") != "notify":
                raise ProbeFailure("transient unit type was not notify")
            if last.get("PrivateNetwork") != "yes" or last.get("JoinsNamespaceOf", ""):
                raise ProbeFailure("transient unit network isolation was not confirmed")
            if last.get("KillMode") != "control-group" or last.get("Restart") != "no":
                raise ProbeFailure("transient unit cleanup policy was not confirmed")
            if last.get("Transient", "") not in ("", "yes"):
                raise ProbeFailure("transient unit ownership was not confirmed")
            try:
                helper_pid = int(last.get("MainPID", "0"))
            except ValueError:
                helper_pid = 0
            if helper_pid <= 1:
                raise ProbeFailure("supervisor process identity unavailable")
            if captured_cgroup[0] is None:
                raise ProbeFailure("transient unit control group was unavailable")
            return helper_pid, captured_cgroup[0]

        if state in ("failed", "inactive") and captured_invocation[0] is not None:
            raise ProbeFailure("supervisor did not report readiness")
        time.sleep(min(READY_POLL_SECONDS, max(0.0, deadline - time.monotonic())))
    raise ProbeFailure("supervisor readiness timed out")


def parse_http_response(response):
    if not response or len(response) > MAX_HTTP_RESPONSE_BYTES:
        raise ProbeFailure("backend HTTP response was invalid")
    marker = response.find(b"\r\n\r\n")
    if marker < 0:
        raise ProbeFailure("backend HTTP response was invalid")
    header_lines = response[:marker].split(b"\r\n")
    if any(
        not line or any(byte < 0x20 or byte > 0x7E for byte in line)
        for line in header_lines
    ):
        raise ProbeFailure("backend HTTP response was invalid")
    lines = [line.decode("ascii") for line in header_lines]
    if not lines or lines[0] != "HTTP/1.0 200 OK":
        raise ProbeFailure("backend HTTP response was invalid")
    headers = {}
    for line in lines[1:]:
        if not line or ":" not in line or line[0] in " \t":
            raise ProbeFailure("backend HTTP response was invalid")
        name, value = line.split(":", 1)
        normalized = name.lower()
        token_punctuation = "!#$%&'*+-.^_`|~"
        if (
            not normalized
            or any(
                not character.isascii()
                or not (character.isalnum() or character in token_punctuation)
                for character in normalized
            )
        ):
            raise ProbeFailure("backend HTTP response was invalid")
        if normalized in headers:
            raise ProbeFailure("backend HTTP response was invalid")
        headers[normalized] = value.strip(" ")
    if "transfer-encoding" in headers or headers.get("content-type") != "application/json":
        raise ProbeFailure("backend HTTP response was invalid")
    if headers.get("connection", "").lower() != "close":
        raise ProbeFailure("backend HTTP response was invalid")
    content_length = headers.get("content-length", "")
    if not content_length.isdigit():
        raise ProbeFailure("backend HTTP response was invalid")
    body = response[marker + 4 :]
    if int(content_length) != len(body) or len(body) > 2048:
        raise ProbeFailure("backend HTTP response was invalid")
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        proof = json.loads(body.decode("ascii"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ProbeFailure("backend HTTP response was invalid") from None
    expected = {"pid", "namespace_device", "namespace_inode"}
    if (
        not isinstance(proof, dict)
        or set(proof) != expected | {"loopback_up"}
        or any(type(proof[name]) is not int for name in expected)
        or proof["pid"] <= 1
        or proof["namespace_device"] < 0
        or proof["namespace_inode"] <= 0
        or proof["loopback_up"] is not True
    ):
        raise ProbeFailure("backend namespace or loopback proof was invalid")
    return proof


def parser_self_test():
    body = b'{"pid":2,"namespace_device":1,"namespace_inode":2,"loopback_up":true}'
    head = (
        b"HTTP/1.0 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n".encode("ascii")
        + b"Connection: close\r\n\r\n"
    )
    valid = head + body
    if parse_http_response(valid) != {
        "pid": 2,
        "namespace_device": 1,
        "namespace_inode": 2,
        "loopback_up": True,
    }:
        raise ProbeFailure("HTTP parser self-test failed")

    duplicate_length = head.replace(
        b"Content-Length:", b"Content-Length: 2\r\nContent-Length:", 1
    ) + body
    transfer_encoding = head.replace(
        b"Content-Type:", b"Transfer-Encoding: chunked\r\nContent-Type:", 1
    ) + body
    trailing_bytes = valid + b"x"
    truncated_body = valid[:-1]
    bare_lf = valid.replace(b"\r\n", b"\n")
    control_character = head.replace(
        b"Connection:", b"X-Test: bad\x01\r\nConnection:", 1
    ) + body
    obs_fold = head.replace(
        b"Connection:", b" folded\r\nConnection:", 1
    ) + body
    duplicate_json_body = (
        b'{"pid":2,"pid":2,"namespace_device":1,"namespace_inode":2,'
        b'"loopback_up":true}'
    )
    duplicate_json = (
        b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(duplicate_json_body)}\r\n".encode("ascii")
        + b"Connection: close\r\n\r\n"
        + duplicate_json_body
    )
    malformed = (
        duplicate_length,
        transfer_encoding,
        trailing_bytes,
        truncated_body,
        bare_lf,
        control_character,
        obs_fold,
        duplicate_json,
    )
    for response in malformed:
        try:
            parse_http_response(response)
        except ProbeFailure:
            continue
        raise ProbeFailure("HTTP parser accepted malformed framing")
    print(json.dumps({"status": "pass", "parser_cases": 9}, separators=(",", ":")))
    return 0


def send_native_get(engine_socket, helper_pid, deadline):
    request = (
        b"GET /probe HTTP/1.0\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Content-Length: 0\r\n"
        b"Connection: close\r\n\r\n"
    )
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BackendNotReady()
            connection.settimeout(remaining)
            connection.connect(os.fspath(engine_socket))
            if not hasattr(socket, "SO_PEERCRED"):
                raise ProbeFailure("engine transport credentials were unavailable")
            credentials = connection.getsockopt(
                socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
            )
            peer_pid, peer_uid, _peer_gid = struct.unpack("3i", credentials)
            if peer_pid != helper_pid or peer_uid != os.geteuid():
                raise ProbeFailure("engine transport peer identity was invalid")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BackendNotReady()
            connection.settimeout(remaining)
            connection.sendall(request)
            connection.shutdown(socket.SHUT_WR)
            response = bytearray()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BackendNotReady()
                connection.settimeout(remaining)
                chunk = connection.recv(1024)
                if not chunk:
                    break
                response.extend(chunk)
                if len(response) > MAX_HTTP_RESPONSE_BYTES:
                    raise ProbeFailure("backend HTTP response exceeded its limit")
    except (ProbeFailure, BackendNotReady):
        raise
    except (OSError, TimeoutError):
        raise BackendNotReady() from None
    if not response:
        raise BackendNotReady()
    return request, bytes(response)


def probe_backend_until_ready(engine_socket, helper_pid):
    # Only this side-effect-free GET may repeat while the loopback fixture
    # starts. Never turn this into an inference or mutating request.
    deadline = time.monotonic() + NATIVE_EXCHANGE_SECONDS
    for attempt in range(1, MAX_HEALTH_ATTEMPTS + 1):
        try:
            request, response = send_native_get(engine_socket, helper_pid, deadline)
        except BackendNotReady:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or attempt == MAX_HEALTH_ATTEMPTS:
                raise ProbeFailure("backend health probe timed out") from None
            time.sleep(min(HEALTH_RETRY_SECONDS, remaining))
            continue
        return request, response, parse_http_response(response), attempt
    raise ProbeFailure("backend health probe timed out")


def verify_owned_generation(unit, generation, helper_pid, cgroup):
    current = unit_snapshot(unit)
    try:
        current_pid = int(current.get("MainPID", "0"))
    except ValueError:
        current_pid = 0
    if (
        current.get("LoadState") == "not-found"
        or current.get("InvocationID") != generation
        or current.get("ActiveState") != "active"
        or current.get("SubState") != "running"
        or current.get("Type") != "notify"
        or current.get("PrivateNetwork") != "yes"
        or bool(current.get("JoinsNamespaceOf", ""))
        or current.get("ControlGroup") != cgroup
        or current.get("KillMode") != "control-group"
        or current.get("Restart") != "no"
        or current.get("Transient", "") not in ("", "yes")
        or current_pid != helper_pid
    ):
        raise ProbeFailure("transient unit generation changed during proof")


def owned_processes_gone(owned_processes):
    roles = {role for role, _pid, _start_time in owned_processes}
    if not {"helper", "backend"}.issubset(roles):
        return False
    return all(
        process_identity_presence(pid, start_time) is False
        for _role, pid, start_time in owned_processes
    )


def confirm_fixture_stopped(cgroup, owned_processes):
    deadline = time.monotonic() + 1.0
    while True:
        if (
            cgroup is not None
            and cgroup_is_empty(cgroup)
            and owned_processes_gone(owned_processes)
        ):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.1, remaining))


def cleanup_unit(
    unit,
    start_attempted,
    start_acknowledged,
    captured_invocation,
    captured_cgroup,
    owned_processes,
):
    if not start_attempted:
        return True
    # An unknown start acknowledgment or unknown generation is never license
    # to stop a unit by its name. RuntimeMaxSec is the recovery bound.
    if not start_acknowledged:
        return False
    generation = captured_invocation[0]
    if not valid_invocation_id(generation or ""):
        return False
    try:
        current = unit_snapshot(unit)
    except ProbeFailure:
        return False
    if current.get("LoadState") != "not-found":
        if current.get("InvocationID") != generation:
            return False
        state = current.get("ActiveState", "")
        if state not in ("inactive", "failed"):
            if state != "active":
                return False
            helper_identities = [
                (pid, start_time)
                for role, pid, start_time in owned_processes
                if role == "helper"
            ]
            if len(helper_identities) != 1:
                return False
            helper_pid = helper_identities[0][0]
            if (
                current.get("ActiveState") != "active"
                or current.get("SubState") != "running"
                or current.get("Type") != "notify"
                or current.get("PrivateNetwork") != "yes"
                or bool(current.get("JoinsNamespaceOf", ""))
                or current.get("ControlGroup") != captured_cgroup[0]
                or current.get("KillMode") != "control-group"
                or current.get("Restart") != "no"
                or current.get("Transient", "") not in ("", "yes")
                or current.get("MainPID") != str(helper_pid)
            ):
                return False
            try:
                immediate = unit_snapshot(unit)
                if (
                    immediate.get("InvocationID") != generation
                    or immediate.get("ActiveState") != "active"
                    or immediate.get("SubState") != "running"
                    or immediate.get("Type") != "notify"
                    or immediate.get("PrivateNetwork") != "yes"
                    or bool(immediate.get("JoinsNamespaceOf", ""))
                    or immediate.get("ControlGroup") != captured_cgroup[0]
                    or immediate.get("MainPID") != str(helper_pid)
                    or immediate.get("KillMode") != "control-group"
                    or immediate.get("Restart") != "no"
                    or immediate.get("Transient", "") not in ("", "yes")
                ):
                    return False
                manager_command("systemctl", "--user", "stop", unit)
                after = unit_snapshot(unit)
            except ProbeFailure:
                return False
            if after.get("LoadState") != "not-found" and (
                after.get("InvocationID") != generation
                or after.get("ActiveState") not in ("inactive", "failed")
            ):
                return False
    return confirm_fixture_stopped(captured_cgroup[0], owned_processes)


def run_probe():
    root = None
    root_identity = None
    transport = None
    unit = None
    start_attempted = False
    start_acknowledged = False
    captured_invocation = [None]
    captured_cgroup = [None]
    owned_processes = []
    primary_failure = None
    cleanup_ok = True
    proof = None

    try:
        if not sys.platform.startswith("linux"):
            raise ProbeFailure("probe requires Linux")
        if shutil.which("systemctl") is None or shutil.which("systemd-run") is None:
            raise ProbeFailure("user manager commands are unavailable")
        repository = Path(__file__).resolve().parents[1]
        binary = repository / "target" / "debug" / "l3ms"
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ProbeFailure("target/debug/l3ms is unavailable; build it explicitly first")

        harness_pid = os.getpid()
        harness_uid = os.geteuid()
        harness_start = process_start_time(harness_pid)
        host_namespace = namespace_identity(harness_pid)
        suffix = uuid.uuid4().hex
        unit = f"l3ms-supervisor-probe-{suffix}.service"
        backend_port = 40000 + (int(suffix[:8], 16) % 20000)
        root, root_identity, transport = make_private_directories()
        if len(os.fsencode(transport / ENGINE_SOCKET_NAME)) > 100:
            raise ProbeFailure("private transport path exceeded its limit")

        before = unit_snapshot(unit)
        if before.get("LoadState") != "not-found":
            raise ProbeFailure("unique probe unit name was not absent")

        backend_arguments = ["-c", BACKEND_CODE, str(backend_port)]
        helper_arguments = [
            HELPER_MARKER,
            os.fspath(transport),
            str(backend_port),
            str(harness_pid),
            str(harness_uid),
            str(harness_start),
            str(host_namespace[0]),
            str(host_namespace[1]),
            os.fspath(Path(sys.executable).resolve()),
            os.fspath(root),
            str(len(backend_arguments)),
            *backend_arguments,
        ]
        open_file = (
            f"--property=OpenFile=/proc/{harness_pid}/ns/net:"
            "l3ms-coordinator-netns:read-only"
        )
        start_command = [
            "systemd-run",
            "--user",
            "--no-ask-password",
            "--quiet",
            "--no-block",
            "--collect",
            "--expand-environment=no",
            "--service-type=notify",
            f"--unit={unit}",
            f"--property=WorkingDirectory={root}",
            "--property=PrivateNetwork=yes",
            "--property=NotifyAccess=main",
            "--property=KillMode=control-group",
            "--property=Restart=no",
            f"--property=RuntimeMaxSec={UNIT_RUNTIME_SECONDS}s",
            f"--property=TimeoutStartSec={UNIT_START_SECONDS}s",
            f"--property=TimeoutStopSec={UNIT_STOP_SECONDS}s",
            open_file,
            "--",
            os.fspath(binary),
            *helper_arguments,
        ]
        start_attempted = True
        try:
            manager_command(*start_command)
        except ProbeFailure:
            raise ProbeFailure("unit start acknowledgment is unknown") from None
        start_acknowledged = True

        helper_pid, unit_cgroup = wait_for_ready(
            unit, captured_invocation, captured_cgroup
        )
        helper_start = process_start_time(helper_pid)
        owned_processes.append(("helper", helper_pid, helper_start))
        helper_namespace = namespace_identity(helper_pid)
        if helper_namespace == host_namespace:
            raise ProbeFailure("supervisor network namespace was not isolated")
        if namespace_identity(harness_pid) != host_namespace:
            raise ProbeFailure("harness network namespace changed")
        if process_start_time(harness_pid) != harness_start:
            raise ProbeFailure("harness process identity changed")

        socket_path = transport / ENGINE_SOCKET_NAME
        socket_metadata = socket_path.lstat()
        if (
            not stat.S_ISSOCK(socket_metadata.st_mode)
            or socket_metadata.st_uid != harness_uid
            or stat.S_IMODE(socket_metadata.st_mode) != 0o600
        ):
            raise ProbeFailure("engine transport socket was not ready")

        request, response, backend_proof, health_attempts = probe_backend_until_ready(
            socket_path, helper_pid
        )
        backend_pid = backend_proof["pid"]
        backend_start = process_start_time(backend_pid)
        owned_processes.append(("backend", backend_pid, backend_start))
        backend_namespace = namespace_identity(backend_pid)
        if backend_namespace != (
            backend_proof["namespace_device"],
            backend_proof["namespace_inode"],
        ):
            raise ProbeFailure("backend namespace response did not match proc")
        if backend_namespace != helper_namespace or backend_namespace == host_namespace:
            raise ProbeFailure("backend and supervisor namespaces did not match")
        if process_parent(backend_pid) != helper_pid:
            raise ProbeFailure("backend was not a supervisor child")
        helper_cgroup = process_cgroup(helper_pid)
        backend_cgroup = process_cgroup(backend_pid)
        if (
            proc_cgroup_path(helper_cgroup) != unit_cgroup
            or proc_cgroup_path(backend_cgroup) != unit_cgroup
        ):
            raise ProbeFailure("backend was not in the supervisor unit")
        if process_start_time(helper_pid) != helper_start:
            raise ProbeFailure("supervisor process identity changed")
        if process_start_time(backend_pid) != backend_start:
            raise ProbeFailure("backend process identity changed")
        verify_owned_generation(
            unit,
            captured_invocation[0],
            helper_pid,
            unit_cgroup,
        )

        proof = {
            "status": "pass",
            "manager_ready": True,
            "named_namespace_fd_handoff": True,
            "harness_pid": harness_pid,
            "helper_pid": helper_pid,
            "helper_start_time_ticks": helper_start,
            "backend_pid": backend_pid,
            "backend_start_time_ticks": backend_start,
            "host_netns_dev_ino": list(host_namespace),
            "helper_netns_dev_ino": list(helper_namespace),
            "backend_netns_dev_ino": list(backend_namespace),
            "engine_socket_peer_pid": helper_pid,
            "coordinator_client_pid": harness_pid,
            "backend_parent_pid": helper_pid,
            "request_bytes": len(request),
            "response_bytes": len(response),
            "health_get_attempts": health_attempts,
            "invocation_id_captured": valid_invocation_id(captured_invocation[0] or ""),
        }
    except ProbeFailure as error:
        primary_failure = error
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
        primary_failure = ProbeFailure("probe observation unavailable")
    except Exception:
        primary_failure = ProbeFailure("probe observation unavailable")
    finally:
        try:
            cleanup_ok = cleanup_unit(
                unit,
                start_attempted,
                start_acknowledged,
                captured_invocation,
                captured_cgroup,
                owned_processes,
            )
        except Exception:
            cleanup_ok = False
        if cleanup_ok and proof is not None:
            helper_backend_gone = owned_processes_gone(owned_processes)
            unit_cgroup_empty = (
                captured_cgroup[0] is not None
                and cgroup_is_empty(captured_cgroup[0])
            )
            cleanup_ok = helper_backend_gone and unit_cgroup_empty
            if cleanup_ok:
                proof["helper_backend_identities_gone"] = True
                proof["owned_cgroup_empty_or_removed"] = True
        if root is not None and cleanup_ok:
            cleanup_ok = remove_private_directory(root, root_identity)

    if not cleanup_ok:
        raise ProbeFailure(
            "probe cleanup uncertain; wait for the bounded unit lifetime to expire before retrying"
        )
    if primary_failure is not None:
        raise primary_failure
    if proof is None:
        raise ProbeFailure("probe observation unavailable")
    print(json.dumps(proof, sort_keys=True, separators=(",", ":")))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--run",
        action="store_true",
        help="explicitly start one bounded model-free supervisor fixture",
    )
    modes.add_argument(
        "--self-test",
        action="store_true",
        help="test strict HTTP response parsing without starting a unit",
    )
    args = parser.parse_args()
    if args.self_test:
        try:
            return parser_self_test()
        except ProbeFailure as error:
            print(str(error), file=sys.stderr)
            return 1
    if not args.run:
        parser.print_help()
        return 2
    try:
        return run_probe()
    except ProbeFailure as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception:
        print("probe observation unavailable", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
