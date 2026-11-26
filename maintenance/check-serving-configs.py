#!/usr/bin/env python3
"""Validate llama-swap.yaml and Strata/PolyStrata serving configs.

Stdlib-only guardrail (no PyYAML required). Runs cleanly both in CI (where
gitignored vendor/ trees are absent) and locally on the serving host.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_TAILSCALE_HOST = "yeti-cachy.llama-alnair.ts.net"


def parse_llama_swap(swap_path: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    macros: dict[str, str] = {}
    models: dict[str, dict[str, str]] = {}
    section: str | None = None
    current: str | None = None
    in_cmd = False

    for line in swap_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" "):
            section = stripped.rstrip(":")
            current = None
            in_cmd = False
            continue
        if section == "macros":
            m = re.match(r'^\s{2}"([^"]+)":\s*"([^"]+)"', line)
            if m:
                macros[m.group(1)] = m.group(2)
            continue
        if section == "models":
            m = re.match(r'^\s{2}"([^"]+)":\s*$', line)
            if m:
                current = m.group(1)
                models[current] = {"proxy": "", "cmd": ""}
                in_cmd = False
                continue
            if current is None:
                continue
            if re.match(r"^\s{4}proxy:\s*(.+)$", line):
                models[current]["proxy"] = line.split("proxy:", 1)[1].strip()
                in_cmd = False
            elif re.match(r"^\s{4}cmd:\s*", line):
                in_cmd = True
                rest = line.split("cmd:", 1)[1].strip()
                if rest and rest != "|":
                    models[current]["cmd"] += " " + rest
            elif in_cmd and line.startswith("      "):
                models[current]["cmd"] += " " + stripped
            elif re.match(r"^\s{4}\S", line):
                in_cmd = False

    return macros, models


def expand_macros(text: str, macros: dict[str, str], root: Path) -> str:
    out = text
    for k, v in macros.items():
        out = out.replace(f"${{{k}}}", v)
    out = out.replace("${env.L3MS_ROOT}", str(root))
    out = out.replace("${env.HOME}", str(Path.home()))
    return out


def validate_configs(
    root: Path = ROOT,
    *,
    local_paths: bool = False,
    expected_host: str = EXPECTED_TAILSCALE_HOST,
) -> list[str]:
    errors: list[str] = []
    swap_path = root / "llama-swap.yaml"
    if not swap_path.exists():
        return [f"Missing {swap_path}"]

    macros, models = parse_llama_swap(swap_path)
    if "inference_port" not in macros:
        errors.append("llama-swap.yaml: missing macros.inference_port")

    referenced_strata_configs: set[Path] = set()
    referenced_other_configs: set[Path] = set()
    for name, cfg in models.items():
        raw_cmd = cfg["cmd"]
        raw_proxy = cfg["proxy"]
        if "${inference_port}" in raw_cmd or "${inference_port}" in raw_proxy:
            if raw_proxy != "http://127.0.0.1:${inference_port}":
                errors.append(
                    f"llama-swap.yaml [{name}]: proxy must be http://127.0.0.1:${{inference_port}} (got {raw_proxy!r})"
                )
            if "--port ${inference_port}" not in raw_cmd:
                errors.append(f"llama-swap.yaml [{name}]: missing '--port ${{inference_port}}'")
            if "--host 0.0.0.0" not in raw_cmd:
                errors.append(f"llama-swap.yaml [{name}]: missing '--host 0.0.0.0' (breaks Tailscale Web UI)")

        expanded_cmd = expand_macros(raw_cmd, macros, root)
        is_strata = "--engine strata" in raw_cmd or "strata" in raw_cmd
        for cfg_str in re.findall(r"--config\s+([^\s]+)", expanded_cmd):
            if is_strata:
                referenced_strata_configs.add(Path(cfg_str))
            else:
                referenced_other_configs.add(Path(cfg_str))

        if local_paths:
            for flag in ("-m", "--model", "--mmproj", "--spec-draft-model"):
                for match in re.findall(rf"{re.escape(flag)}\s+([^\s]+)", expanded_cmd):
                    p = Path(match)
                    if not p.exists():
                        errors.append(f"llama-swap.yaml [{name}]: missing {flag} path {p}")

    # Validate tracked Strata configs + any present referenced JSON configs
    json_targets: set[Path] = set()
    tracked_strata_dir = root / "maintenance" / "configs" / "strata"
    if tracked_strata_dir.exists():
        json_targets.update(tracked_strata_dir.glob("*.json"))
    for cfg_path in referenced_strata_configs | referenced_other_configs:
        if cfg_path.exists():
            json_targets.add(cfg_path)
        elif local_paths:
            errors.append(f"Missing referenced --config file: {cfg_path}")

    for cfg_path in sorted(json_targets):
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{cfg_path}: invalid JSON ({exc})")
            continue

        # Enforce allowed_hosts only on active Strata / PolyStrata configs
        if cfg_path in referenced_strata_configs:
            allowed = data.get("allowed_hosts")
            if not isinstance(allowed, list) or expected_host not in allowed:
                errors.append(
                    f"{cfg_path}: missing {expected_host!r} in allowed_hosts (breaks Tailscale DNS rebinding check)"
                )
            if local_paths:
                for key in ("tokenizer",):
                    val = data.get(key)
                    if isinstance(val, str) and not Path(val).exists():
                        errors.append(f"{cfg_path}: missing {key} path {val}")
                vision = data.get("vision")
                if isinstance(vision, dict):
                    for vk in ("mmproj", "model"):
                        vv = vision.get(vk)
                        if isinstance(vv, str) and not Path(vv).exists():
                            errors.append(f"{cfg_path}: missing vision.{vk} path {vv}")
                args = data.get("args")
                if isinstance(args, list):
                    for i, token in enumerate(args):
                        if token in ("--model", "--ple-model") and i + 1 < len(args):
                            mp = Path(args[i + 1])
                            if not mp.exists():
                                errors.append(f"{cfg_path}: missing {token} path {mp}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate L3MS llama-swap and Strata/PolyStrata configs")
    parser.add_argument(
        "--local-paths",
        action="store_true",
        help="Also verify referenced model files and vendor configs exist on this host",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root")
    args = parser.parse_args()

    errors = validate_configs(root=args.root, local_paths=args.local_paths)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        print(f"Serving config check failed ({len(errors)} issue(s)).", file=sys.stderr)
        return 1

    mode = "with local path verification" if args.local_paths else "syntax & invariants"
    print(f"Serving config check passed ({mode}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
