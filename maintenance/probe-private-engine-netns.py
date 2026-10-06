#!/usr/bin/env python3
"""Opt-in, model-free systemd-user namespace topology acceptance probe.

Starts only two uniquely named /usr/bin/sleep fixtures, each with a ten-second
manager lifetime limit. No GPU, real engines, downloads or existing services.
Stops only fixtures whose captured InvocationID still matches. Not run by CI.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid


def command(*arguments):
    result = subprocess.run(
        arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=5, check=False,
    )
    if result.returncode:
        raise RuntimeError("probe command failed")
    return result.stdout


def snapshot(unit):
    output = command(
        "systemctl", "--user", "show", unit,
        "--property=LoadState", "--property=ActiveState",
        "--property=MainPID", "--property=InvocationID",
        "--property=PrivateNetwork", "--property=JoinsNamespaceOf",
    )
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


def namespace(pid):
    metadata = os.stat(f"/proc/{pid}/ns/net")
    return [metadata.st_dev, metadata.st_ino]


def run(with_open_file, delay_seconds, join):
    suffix = uuid.uuid4().hex
    engine = f"l3ms-netns-probe-{suffix}-engine.service"
    bridge = f"l3ms-netns-probe-{suffix}-bridge.service"
    owned = []
    cleanup_ok = True
    try:
        for unit in (engine, bridge):
            if snapshot(unit).get("LoadState") != "not-found":
                raise RuntimeError("probe name collision")
        common = (
            "--property=PrivateNetwork=yes", "--property=RuntimeMaxSec=10s",
            "--property=TimeoutStopSec=1s", "--property=KillMode=control-group",
        )
        owned.append((engine, None))
        command("systemd-run", "--user", "--collect", "--service-type=exec",
                f"--unit={engine}", *common, "/usr/bin/sleep", "8")
        first = snapshot(engine)
        generation = first.get("InvocationID", "")
        if len(generation) != 32 or int(first.get("MainPID", "0")) <= 1:
            raise RuntimeError("engine fixture identity unavailable")
        owned[-1] = (engine, generation)
        engine_pid = int(first["MainPID"])
        time.sleep(delay_seconds)
        properties = [f"--property=JoinsNamespaceOf={engine}"] if join else []
        if with_open_file:
            properties.append(f"--property=OpenFile=/proc/{engine_pid}/ns/net:l3ms-engine-netns:read-only")
        owned.append((bridge, None))
        command("systemd-run", "--user", "--collect", "--service-type=exec",
                f"--unit={bridge}", *common, *properties, "/usr/bin/sleep", "8")
        second = snapshot(bridge)
        generation = second.get("InvocationID", "")
        if len(generation) != 32 or int(second.get("MainPID", "0")) <= 1:
            raise RuntimeError("bridge fixture identity unavailable")
        owned[-1] = (bridge, generation)
        host_ns = namespace(os.getpid())
        engine_ns = namespace(engine_pid)
        bridge_ns = namespace(int(second["MainPID"]))
        print(json.dumps({
            "with_open_file": with_open_file,
            "delay_seconds": delay_seconds, "join_requested": join,
            "host_namespace": host_ns, "engine_namespace": engine_ns,
            "bridge_namespace": bridge_ns,
            "engine_private_network": first.get("PrivateNetwork"),
            "bridge_private_network": second.get("PrivateNetwork"),
            "joins_expected_engine": second.get("JoinsNamespaceOf") == engine,
        }, sort_keys=True))
        if engine_ns == host_ns:
            raise RuntimeError("engine fixture is not isolated")
        if join and bridge_ns != engine_ns:
            raise RuntimeError("bridge did not join the exact engine namespace")
        if not join and (bridge_ns == host_ns or bridge_ns == engine_ns):
            raise RuntimeError("unjoined bridge did not get its own private namespace")
    finally:
        # Stop bridge before engine, only while each exact generation is ours.
        # If identity is unavailable/replaced, rely on RuntimeMaxSec and report
        # uncertainty instead of stopping an arbitrary unit by name.
        for unit, generation in reversed(owned):
            try:
                current = snapshot(unit)
                if current.get("LoadState") == "not-found":
                    continue
                if generation is None or current.get("InvocationID") != generation:
                    cleanup_ok = False
                    continue
                command("systemctl", "--user", "stop", unit)
                after = snapshot(unit)
                if after.get("ActiveState") not in ("inactive", "failed"):
                    cleanup_ok = False
            except (OSError, ValueError, subprocess.TimeoutExpired, RuntimeError):
                cleanup_ok = False
        if not cleanup_ok:
            raise RuntimeError("probe cleanup uncertain; bounded fixture lifetime remains")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="explicitly start bounded model-free fixtures")
    parser.add_argument("--with-open-file", action="store_true", help="also test manager-opened engine namespace FD")
    parser.add_argument("--delay-seconds", type=int, choices=(0, 1, 2), default=0,
                        help="test startup-order timing independently")
    parser.add_argument("--no-join", action="store_true", help="control: request independent private namespaces")
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 2
    try:
        run(args.with_open_file, args.delay_seconds, not args.no_join)
    except (OSError, ValueError, subprocess.TimeoutExpired, RuntimeError) as error:
        # Only fixed diagnostic reasons, never captured manager output.
        reason = str(error) if isinstance(error, RuntimeError) else "probe observation unavailable"
        print(reason, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
