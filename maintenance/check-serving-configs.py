#!/usr/bin/env python3
"""Validate llama-swap.yaml and Strata/PolyStrata serving configs.

Stdlib-only guardrail (no PyYAML required). Runs cleanly both in CI (where
gitignored vendor/ trees are absent) and locally on the serving host.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_TAILSCALE_HOST = "yeti-cachy.llama-alnair.ts.net"


def _unquote(raw: str) -> str:
    s = raw.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s


def parse_llama_swap_full(
    swap_path: Path,
) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    macros: dict[str, str] = {}
    models: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {}
    section: str | None = None
    current_model: str | None = None
    current_group: str | None = None
    model_sub: str | None = None  # "cmd" | "aliases" | None
    group_sub: str | None = None  # "members" | None

    for line in swap_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" "):
            section = stripped.rstrip(":")
            current_model = None
            current_group = None
            model_sub = None
            group_sub = None
            continue

        if section == "macros":
            m = re.match(r'^\s{2}"([^"]+)":\s*"([^"]+)"', line)
            if m:
                macros[m.group(1)] = m.group(2)
            continue

        if section == "models":
            m = re.match(r'^\s{2}"([^"]+)":\s*$', line)
            if m:
                current_model = m.group(1)
                models[current_model] = {"proxy": "", "cmd": "", "aliases": []}
                model_sub = None
                continue
            if current_model is None:
                continue
            if re.match(r"^\s{4}proxy:\s*(.+)$", line):
                models[current_model]["proxy"] = line.split("proxy:", 1)[1].strip()
                model_sub = None
            elif re.match(r"^\s{4}cmd:\s*", line):
                model_sub = "cmd"
                rest = line.split("cmd:", 1)[1].strip()
                if rest and rest not in ("|", ">", ">-", "|-"):
                    models[current_model]["cmd"] += " " + rest
            elif re.match(r"^\s{4}aliases:\s*(.*)$", line):
                rest = line.split("aliases:", 1)[1].strip()
                if rest.startswith("[") and rest.endswith("]"):
                    inner = rest[1:-1].strip()
                    if inner:
                        for item in inner.split(","):
                            val = _unquote(item)
                            if val:
                                models[current_model]["aliases"].append(val)
                    model_sub = None
                else:
                    model_sub = "aliases"
            elif model_sub == "cmd" and line.startswith("      "):
                models[current_model]["cmd"] += " " + stripped
            elif model_sub == "aliases" and re.match(r"^\s{6}-\s+(.+)$", line):
                val = _unquote(re.match(r"^\s{6}-\s+(.+)$", line).group(1))
                if val:
                    models[current_model]["aliases"].append(val)
            elif re.match(r"^\s{4}\S", line):
                model_sub = None
            continue

        if section == "groups":
            gm = re.match(r'^\s{2}("[^"]+"|[A-Za-z0-9_.-]+):\s*$', line)
            if gm:
                current_group = _unquote(gm.group(1))
                groups[current_group] = {"swap": True, "exclusive": True, "members": []}
                group_sub = None
                continue
            if current_group is None:
                continue
            if re.match(r"^\s{4}(swap|exclusive):\s*(true|false)\s*$", line):
                km = re.match(r"^\s{4}(swap|exclusive):\s*(true|false)\s*$", line)
                assert km is not None
                groups[current_group][km.group(1)] = km.group(2) == "true"
                group_sub = None
            elif re.match(r"^\s{4}members:\s*$", line):
                group_sub = "members"
            elif group_sub == "members" and re.match(r"^\s{6}-\s+(.+)$", line):
                val = _unquote(re.match(r"^\s{6}-\s+(.+)$", line).group(1))
                if val:
                    groups[current_group]["members"].append(val)
            elif re.match(r"^\s{4}\S", line):
                group_sub = None

    return macros, models, groups


def parse_llama_swap(swap_path: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    macros, models, _groups = parse_llama_swap_full(swap_path)
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

    macros, models, groups = parse_llama_swap_full(swap_path)
    if "inference_port" not in macros:
        errors.append("llama-swap.yaml: missing macros.inference_port")

    alias_owners: dict[str, str] = {}
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

        for alias in cfg.get("aliases", []):
            if alias in models:
                errors.append(
                    f"llama-swap.yaml [{name}]: alias {alias!r} collides with existing model ID"
                )
            elif alias in alias_owners:
                errors.append(
                    f"llama-swap.yaml [{name}]: duplicate alias {alias!r} (already defined on {alias_owners[alias]!r})"
                )
            else:
                alias_owners[alias] = name

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

    for group_name, gcfg in groups.items():
        for member in gcfg.get("members", []):
            if member not in models:
                if member in alias_owners:
                    errors.append(
                        f"llama-swap.yaml [groups.{group_name}]: member {member!r} is an alias of {alias_owners[member]!r} (use canonical model ID)"
                    )
                else:
                    errors.append(
                        f"llama-swap.yaml [groups.{group_name}]: unknown member {member!r}"
                    )
                continue
            if not gcfg.get("exclusive", True):
                mcfg = models[member]
                if "${inference_port}" in mcfg["cmd"] or "${inference_port}" in mcfg["proxy"]:
                    errors.append(
                        f"llama-swap.yaml [groups.{group_name}]: non-exclusive member {member!r} must not bind shared ${{inference_port}} (use ${{PORT}})"
                    )

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
