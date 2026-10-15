#!/usr/bin/env python3
"""Benchmark one local decision profile with a suite from `suites/`.

Replaces the retired `l3ds --bench`: look the profile up in
`decision-runtimes.json`, then run `decision-models/runner.py` with its engine,
model and device. Router-hosted profiles (`llamacpp`) are benchmarked through
llama-swap, not here.

  python3 decision-models/bench_profile.py --profile nagi-smol-cpu \
      --suite routing-gate --iterations 10 [--dry-run]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def interpreter() -> str:
    venv = ROOT / ".venv-decisions" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def build_command(profile_id: str, suite: str, iterations: int) -> list[str]:
    manifest = json.loads((ROOT / "decision-runtimes.json").read_text())
    profile = next((p for p in manifest["profiles"] if p["id"] == profile_id), None)
    if profile is None:
        raise SystemExit(f"unknown decision profile: {profile_id}")
    if profile["engine"] == "llamacpp":
        raise SystemExit(f"{profile_id} is served by llama-swap; benchmark it through the router")
    suite_path = Path(suite)
    if not suite_path.is_file():
        suite_path = ROOT / "suites" / (suite if suite.endswith(".json") else f"{suite}.json")
    if not suite_path.is_file():
        raise SystemExit(f"suite not found: {suite}")
    return [
        interpreter(), str(ROOT / profile.get("entrypoint", "decision-models/runner.py")),
        "--engine", profile["engine"], "--model", profile["model"],
        "--device", profile["device"],
        "--bench-suite", str(suite_path), "--iterations", str(iterations),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--suite", default="routing-gate")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="print the command only")
    args = parser.parse_args()
    command = build_command(args.profile, args.suite, args.iterations)
    if args.dry_run:
        print(" ".join(command))
        return 0
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
