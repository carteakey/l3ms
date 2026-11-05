#!/usr/bin/env python3
"""
Token counter and audit tool for Strata under L3MS.
Inspects live session metrics, current active deployment logs, and historical host logs.
"""

import argparse
import glob
import json
import os
import re
import sys
import urllib.error
import urllib.request

LOG_PATTERN = re.compile(
    r"strata serve: prompt (?P<prompt>\d+) tokens = (?P<reused>\d+) reused \+ (?P<read>\d+) read.*?, (?P<gen>\d+) generated"
)

def get_live_metrics(port=10022):
    """Query live server metrics from Strata's REST API endpoint."""
    url = f"http://127.0.0.1:{port}/metrics"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "strata-token-counter/1.0"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        pass
    return None

def find_strata_port():
    """Detect running strata server port from process list if not default."""
    try:
        import subprocess
        out = subprocess.check_output(["pgrep", "-fa", "server.py.*--port"], text=True)
        m = re.search(r"--port\s+(\d+)", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 10022

def parse_log_file(filepath):
    """Parse a single strata log file and aggregate token statistics."""
    stats = {
        "requests": 0,
        "prompt_tokens": 0,
        "reused_tokens": 0,
        "read_tokens": 0,
        "generated_tokens": 0,
        "total_tokens": 0,
    }
    if not os.path.exists(filepath):
        return stats

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if m:
                p = int(m.group("prompt"))
                r = int(m.group("reused"))
                rd = int(m.group("read"))
                g = int(m.group("gen"))
                stats["requests"] += 1
                stats["prompt_tokens"] += p
                stats["reused_tokens"] += r
                stats["read_tokens"] += rd
                stats["generated_tokens"] += g

    stats["total_tokens"] = stats["prompt_tokens"] + stats["generated_tokens"]
    return stats

def collect_stats(base_dir):
    """Collect statistics across live, current deployment, and historical logs."""
    port = find_strata_port()
    live = get_live_metrics(port)

    # Active deployment (strata-lean)
    lean_dir = os.path.join(base_dir, "vendor", "strata-lean")
    lean_files = sorted(glob.glob(os.path.join(lean_dir, "*.log")))
    # Filter out build logs or non-strata logs
    lean_files = [f for f in lean_files if "build" not in f and "bench" not in f]

    lean_breakdown = {}
    lean_totals = {
        "requests": 0,
        "prompt_tokens": 0,
        "reused_tokens": 0,
        "read_tokens": 0,
        "generated_tokens": 0,
        "total_tokens": 0,
    }

    for f in lean_files:
        rel = os.path.relpath(f, base_dir)
        st = parse_log_file(f)
        if st["requests"] > 0:
            lean_breakdown[rel] = st
            for k in lean_totals:
                lean_totals[k] += st[k]

    # Historical host logs (all strata versions in vendor, deduplicated by inode)
    all_vendor_logs = sorted(glob.glob(os.path.join(base_dir, "vendor", "**", "*.log"), recursive=True))
    host_logs = []
    seen_inodes = set()

    for f in all_vendor_logs:
        # Exclude external community logs, build-time outputs, benchmark scripts
        if "community" in f or "build" in f or "benches" in f or "bench-results" in f or "llama.cpp" in f:
            continue
        try:
            ino = os.stat(f).st_ino
            if ino in seen_inodes:
                continue
            seen_inodes.add(ino)
        except OSError:
            continue
        host_logs.append(f)

    host_breakdown = {}
    host_totals = {
        "requests": 0,
        "prompt_tokens": 0,
        "reused_tokens": 0,
        "read_tokens": 0,
        "generated_tokens": 0,
        "total_tokens": 0,
    }

    for f in host_logs:
        rel = os.path.relpath(f, base_dir)
        st = parse_log_file(f)
        if st["requests"] > 0:
            host_breakdown[rel] = st
            for k in host_totals:
                host_totals[k] += st[k]

    return {
        "live": live,
        "live_port": port,
        "lean": {
            "breakdown": lean_breakdown,
            "totals": lean_totals,
        },
        "host_history": {
            "breakdown": host_breakdown,
            "totals": host_totals,
        },
    }

def print_markdown_report(data):
    """Format and print results as clean markdown."""
    print("# Strata Token Accounting Report\n")

    # 1. Live Session
    live = data.get("live")
    print("## 1. Active Running Session")
    if live and "totals" in live:
        tot = live["totals"]
        lv = live.get("live", {})
        state = lv.get("state", "idle")
        in_flight_p = lv.get("prompt_tokens") or 0
        in_flight_g = lv.get("generated") or 0

        completed_p = tot.get("prompt_tokens", 0)
        completed_r = tot.get("reused", 0)
        completed_g = tot.get("output_tokens", 0)
        session_total = completed_p + completed_g + in_flight_p + in_flight_g

        print(f"- **Engine Status**: `{state}` (Port: `{data.get('live_port')}`)")
        print(f"- **Completed Requests**: {tot.get('requests', 0):,}")
        print(f"- **Completed Prompt Tokens**: {completed_p:,} (Reused: {completed_r:,}, Read: {completed_p - completed_r:,})")
        print(f"- **Completed Output Tokens**: {completed_g:,}")
        if state == "generating" or in_flight_p > 0 or in_flight_g > 0:
            phase = lv.get("phase", "processing")
            print(f"- **In-Flight Request**: Phase: `{phase}`, Prompt: {in_flight_p:,}, Generated: {in_flight_g:,}")
        print(f"- **Total Session Tokens**: **{session_total:,}**\n")
    else:
        print("*No active Strata server instance currently responding on local ports.*\n")

    # 2. Current Active Deployment
    lean = data["lean"]
    print("## 2. Active Deployment (`vendor/strata-lean`)")
    lt = lean["totals"]
    print(f"- **Total Requests**: {lt['requests']:,}")
    print(f"- **Prompt Tokens**: {lt['prompt_tokens']:,} (Reused: {lt['reused_tokens']:,}, Read: {lt['read_tokens']:,})")
    print(f"- **Generated Output Tokens**: {lt['generated_tokens']:,}")
    print(f"- **Total Deployment Tokens**: **{lt['total_tokens']:,}**\n")

    print("| Log / Model Configuration | Reqs | Prompt Tokens | Read Tokens | Generated | Total Tokens |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for path, st in lean["breakdown"].items():
        base = os.path.basename(path)
        print(f"| `{base}` | {st['requests']:,} | {st['prompt_tokens']:,} | {st['read_tokens']:,} | {st['generated_tokens']:,} | **{st['total_tokens']:,}** |")
    print()

    # 3. All-Time Host History
    hist = data["host_history"]
    ht = hist["totals"]
    print("## 3. All-Time Strata Host History (All Logged Versions)")
    print(f"- **Total Historical Requests**: {ht['requests']:,}")
    print(f"- **Total Prompt Tokens**: {ht['prompt_tokens']:,} (Reused: {ht['reused_tokens']:,}, Read: {ht['read_tokens']:,})")
    print(f"- **Total Output Generated**: {ht['generated_tokens']:,}")
    print(f"- **Grand Total Processed**: **{ht['total_tokens']:,} (~{ht['total_tokens']/1e6:.1f} M)**\n")

    print("| Deployment Worktree | File | Reqs | Prompt Tokens | Read Tokens | Generated | Total Tokens |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    for path, st in hist["breakdown"].items():
        parts = path.split(os.sep)
        vendor_ver = parts[1] if len(parts) > 1 else "vendor"
        base = parts[-1]
        print(f"| `{vendor_ver}` | `{base}` | {st['requests']:,} | {st['prompt_tokens']:,} | {st['read_tokens']:,} | {st['generated_tokens']:,} | **{st['total_tokens']:,}** |")
    print(f"| **Grand Total** | | **{ht['requests']:,}** | **{ht['prompt_tokens']:,}** | **{ht['read_tokens']:,}** | **{ht['generated_tokens']:,}** | **{ht['total_tokens']:,}** |")

def main():
    parser = argparse.ArgumentParser(description="Audit token throughput for Strata under L3MS")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")
    parser.add_argument("--repo-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")),
                        help="Root directory of l3ms repository")
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    stats = collect_stats(repo_root)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print_markdown_report(stats)

if __name__ == "__main__":
    main()
