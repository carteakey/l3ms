#!/usr/bin/env python3
"""L3MS Model Inventory & Cleanup Auditor.

Scans homelab model storage mounts (/home/kchauhan/models, /mnt/dev/models,
/mnt/media/models), cross-references llama-swap.yaml, Strata/PolyStrata JSON
configs, and ComfyUI model directories, and reports active vs unreferenced
files. Multi-shard GGUFs (e.g. -00001-of-00003.gguf .. -00003-of-00003.gguf)
are protected as a group whenever any shard is actively served.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]

SCAN_ROOTS = [
    Path("/home/kchauhan/models"),
    Path("/mnt/dev/models"),
    Path("/mnt/media/models"),
]

DISK_MOUNTS = ["/home", "/mnt/dev", "/mnt/media"]
MODEL_EXTENSIONS = {".gguf", ".safetensors", ".bin", ".pt", ".pth"}
SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")


def _iter_strings(obj: object) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_iter_strings(item))
    elif isinstance(obj, dict):
        for val in obj.values():
            out.extend(_iter_strings(val))
    return out


def shard_group_key(path_str: str) -> tuple[str, str] | None:
    """Return (prefix_without_shard_idx, total_shards) for multi-shard GGUFs."""
    m = SHARD_RE.match(path_str)
    if not m:
        return None
    return (m.group(1), m.group(3))


def is_path_served(
    raw_path: str,
    real_path: str,
    served_paths: set[str],
    served_shard_keys: set[tuple[str, str]],
) -> bool:
    """Return True if a file or any sibling shard of its multi-shard GGUF is served."""
    if real_path in served_paths or raw_path in served_paths:
        return True
    for candidate in (raw_path, real_path):
        key = shard_group_key(candidate)
        if key is not None and key in served_shard_keys:
            return True
    return False


def load_llama_swap_models(root: Path = ROOT) -> dict[str, dict[str, object]]:
    swap_file = root / "llama-swap.yaml"
    if not swap_file.exists():
        return {}
    models: dict[str, dict[str, object]] = {}
    macros: dict[str, str] = {}
    current_model: str | None = None
    section: str | None = None
    try:
        with open(swap_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if not line.startswith(" "):
                    section = stripped.rstrip(":")
                    current_model = None
                    continue
                if section == "macros":
                    macro_match = re.match(r'^\s{2}"([^"]+)":\s*"([^"]+)"', line)
                    if macro_match:
                        macros[macro_match.group(1)] = macro_match.group(2)
                    continue
                if section == "models":
                    m_match = re.match(r'^\s{2}"([^"]+)":\s*$', line)
                    if m_match:
                        current_model = m_match.group(1)
                        models[current_model] = {"cmd": ""}
                    elif current_model and stripped.startswith("cmd:"):
                        continue
                    elif current_model and line.startswith("      "):
                        models[current_model]["cmd"] = str(models[current_model]["cmd"]) + " " + stripped
    except Exception as e:
        print(f"Warning: Failed to parse llama-swap.yaml: {e}", file=sys.stderr)

    # Resolve macros in commands and extract referenced config files
    for m_cfg in models.values():
        cmd = str(m_cfg.get("cmd", ""))
        for k, v in macros.items():
            cmd = cmd.replace(f"${{{k}}}", v)
            cmd = cmd.replace(f"${k}", v)
        cmd = cmd.replace("${env.L3MS_ROOT}", str(root))
        cmd = cmd.replace("${env.HOME}", str(Path.home()))
        m_cfg["cmd"] = cmd

        for cfg_path_str in re.findall(r"--config\s+([^\s]+)", cmd):
            cfg_path = Path(cfg_path_str)
            if cfg_path.exists():
                try:
                    with open(cfg_path, "r", encoding="utf-8") as jf:
                        cfg_json = json.load(jf)
                    m_cfg["cmd"] = str(m_cfg["cmd"]) + " " + " ".join(_iter_strings(cfg_json))
                except Exception:
                    pass

    return models


def load_media_runtimes(root: Path = ROOT) -> list[dict[str, object]]:
    for rel in (Path("media-runtimes.json"), Path("toolkits/l2m2/media-runtimes.json")):
        media_file = root / rel
        if media_file.exists():
            try:
                with open(media_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("profiles", [])
            except Exception as e:
                print(f"Warning: Failed to parse {media_file}: {e}", file=sys.stderr)
                return []
    return []


def get_disk_usage(mounts: list[str] = DISK_MOUNTS) -> dict[str, dict[str, str]]:
    disks: dict[str, dict[str, str]] = {}
    for mount in mounts:
        if os.path.exists(mount) and (mount == "/home" or os.path.ismount(mount)):
            stat = os.statvfs(mount)
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bavail * stat.f_frsize
            used = total - free
            disks[mount] = {
                "total": f"{total / (1024**3):.1f} GiB",
                "used": f"{used / (1024**3):.1f} GiB",
                "free": f"{free / (1024**3):.1f} GiB",
                "pct": f"{int((used / total) * 100)}%" if total else "0%",
            }
        else:
            disks[mount] = {"total": "-", "used": "-", "free": "unmounted", "pct": "-"}
    return disks


def _mount_label(path_str: str) -> str:
    if path_str.startswith("/mnt/dev/"):
        return "/mnt/dev"
    if path_str.startswith("/mnt/media/"):
        return "/mnt/media"
    return "/home"


def scan_inventory(root: Path = ROOT, scan_roots: list[Path] = SCAN_ROOTS) -> dict[str, object]:
    swap_models = load_llama_swap_models(root=root)
    load_media_runtimes(root=root)

    served_paths: set[str] = set()
    served_shard_keys: set[tuple[str, str]] = set()
    for m_cfg in swap_models.values():
        cmd = str(m_cfg.get("cmd", ""))
        for part in cmd.split():
            if part.endswith(tuple(MODEL_EXTENSIONS)) or "mmproj" in part:
                p = part.replace("${env.L3MS_ROOT}", str(root))
                if os.path.exists(p):
                    raw_norm = os.path.abspath(p)
                    real_norm = str(Path(p).resolve())
                    served_paths.add(raw_norm)
                    served_paths.add(real_norm)
                    for candidate in (raw_norm, real_norm):
                        key = shard_group_key(candidate)
                        if key is not None:
                            served_shard_keys.add(key)

    comfy_paths: set[str] = set()
    for rel in (Path("vendor/ComfyUI/models"), Path("toolkits/l2m2/vendor/ComfyUI/models")):
        comfy_models_dir = root / rel
        if comfy_models_dir.exists():
            for dirpath, _, filenames in os.walk(comfy_models_dir):
                for fname in filenames:
                    cp = Path(dirpath) / fname
                    try:
                        comfy_paths.add(str(cp.resolve()))
                    except Exception:
                        pass

    warnings: list[str] = []
    unique_files: dict[str, dict[str, object]] = {}
    for scan_root in scan_roots:
        if not scan_root.exists():
            warnings.append(f"Scan root missing or unmounted: {scan_root}")
            continue
        for dirpath, _, filenames in os.walk(scan_root, followlinks=False):
            for fname in filenames:
                p = Path(dirpath) / fname
                if p.suffix.lower() not in MODEL_EXTENSIONS:
                    continue
                try:
                    sz = p.stat().st_size
                    raw = os.path.abspath(str(p))
                    real = str(p.resolve())
                    is_link = p.is_symlink()

                    is_served_llm = is_path_served(raw, real, served_paths, served_shard_keys)
                    is_media = (
                        "/models/media/" in real
                        or real in comfy_paths
                    )

                    if is_served_llm:
                        status = "ACTIVE (llama-swap)"
                        action = "Keep (Primary Serving Tier)"
                    elif is_media:
                        status = "MEDIA (ComfyUI / L2M2)"
                        action = "Keep (Media Runtime Model)"
                    else:
                        status = "UNREFERENCED"
                        action = "Candidate for cleanup"

                    if real not in unique_files:
                        unique_files[real] = {
                            "path": real,
                            "name": fname,
                            "size": sz,
                            "mount": _mount_label(real),
                            "symlinks": [],
                            "status": status,
                            "action": action,
                            "is_served": is_served_llm or is_media,
                        }
                    elif is_served_llm and not unique_files[real]["is_served"]:
                        unique_files[real]["status"] = status
                        unique_files[real]["action"] = action
                        unique_files[real]["is_served"] = True
                    if is_link:
                        unique_files[real]["symlinks"].append(str(p))
                except Exception:
                    pass

    items = sorted(unique_files.values(), key=lambda x: x["size"], reverse=True)
    return {
        "disks": get_disk_usage(),
        "warnings": warnings,
        "total_files": len(items),
        "total_bytes": sum(i["size"] for i in items),
        "items": items,
        "served_count": sum(1 for i in items if i["is_served"]),
        "unserved_count": sum(1 for i in items if not i["is_served"]),
    }


def format_report_md(inv: dict[str, object]) -> str:
    lines = ["# Model Storage & Cleanup Audit Report\n"]
    if inv.get("warnings"):
        lines.append("### Warnings")
        for w in inv["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    lines.append("### Filesystem Mounts")
    lines.append("| Mount | Total | Used | Available | Use% |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for mnt, d in inv["disks"].items():
        lines.append(f"| `{mnt}` | {d['total']} | {d['used']} | **{d['free']}** | {d['pct']} |")

    total_gib = inv["total_bytes"] / (1024**3)
    lines.append(f"\n**Total Tracked Model Footprint:** {total_gib:.2f} GiB across {inv['total_files']} files.\n")

    lines.append("### Model Inventory & Serving Status")
    lines.append("| Status | Size | Model File / Directory | Mount | Symlinks | Action |")
    lines.append("| :---: | :--- | :--- | :--- | :---: | :--- |")

    for item in inv["items"]:
        sz_gib = item["size"] / (1024**3)
        sym_count = len(item["symlinks"])
        lines.append(
            f"| `{item['status']}` | {sz_gib:6.2f} GiB | `{item['name']}` | `{item['mount']}` | {sym_count} | {item['action']} |"
        )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="L3MS Model Inventory & Cleanup Tool")
    parser.add_argument("--json", action="store_true", help="Output raw JSON data")
    parser.add_argument("--delete", metavar="PATH", help="Safely delete a model path after verifying it is not an active tier")
    args = parser.parse_args()

    inv = scan_inventory()

    if args.delete:
        target = Path(args.delete).resolve()
        target_str = str(target)
        if not target.exists():
            print(f"Error: Target path does not exist: {target}", file=sys.stderr)
            return 1

        matching = [i for i in inv["items"] if i["path"] == target_str or target_str in i["path"]]
        if any(m["is_served"] for m in matching):
            print(f"Refusing to delete: {target} is actively served in llama-swap.yaml!", file=sys.stderr)
            return 1

        sz_del = sum(m["size"] for m in matching) if matching else target.stat().st_size
        print(f"Deleting unserved model: {target} ({sz_del / (1024**3):.2f} GiB)...")
        if target.is_dir():
            import shutil
            shutil.rmtree(target)
        else:
            target.unlink()
        print("Done. Reclaimed storage.")
        return 0

    if args.json:
        print(json.dumps(inv, indent=2))
    else:
        print(format_report_md(inv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
