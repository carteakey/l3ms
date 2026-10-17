#!/usr/bin/env python3
"""L3MS Model Inventory & Cleanup Auditor.

Scans all homelab model storage mounts (/home, /mnt/dev, /run/media/kchauhan/Games),
cross-references with llama-swap.yaml and media-runtimes.json, and produces a
structured inventory report highlighting active, retired, and candidate orphan models
for safe cleanup.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]

SCAN_ROOTS = [
    Path("/home/kchauhan/models"),
    Path("/mnt/dev/models"),
    Path("/run/media/kchauhan/Games/models"),
]

MODEL_EXTENSIONS = {".gguf", ".safetensors", ".bin", ".pt", ".pth"}


def load_llama_swap_models() -> dict[str, dict[str, object]]:
    swap_file = ROOT / "llama-swap.yaml"
    if not swap_file.exists():
        return {}
    models = {}
    macros = {}
    current_model = None
    try:
        with open(swap_file, "r", encoding="utf-8") as f:
            for line in f:
                # Capture macro definitions under variables:
                macro_match = re.match(r'^\s{2}"([^"]+)":\s*"([^"]+)"', line)
                if macro_match:
                    macros[macro_match.group(1)] = macro_match.group(2)

                # Match model id: "qwen38-flash-next":
                m_match = re.match(r'^\s{2}"([^"]+)":', line)
                if m_match and "cmd:" not in line and not line.strip().startswith("#"):
                    # Check if it's in models section (not variables)
                    m_name = m_match.group(1)
                    if m_name not in macros:
                        current_model = m_name
                        models[current_model] = {"cmd": ""}
                elif current_model and "cmd:" in line:
                    continue
                elif current_model and line.startswith("    "):
                    models[current_model]["cmd"] += " " + line.strip()
                elif line.strip() and not line.startswith(" "):
                    current_model = None
    except Exception as e:
        print(f"Warning: Failed to parse llama-swap.yaml: {e}", file=sys.stderr)

    # Resolve macros in commands and extract referenced config files
    for m_id, m_cfg in models.items():
        cmd = m_cfg.get("cmd", "")
        for k, v in macros.items():
            cmd = cmd.replace(f"${{{k}}}", v)
            cmd = cmd.replace(f"${k}", v)
        cmd = cmd.replace("${env.L3MS_ROOT}", str(ROOT))
        m_cfg["cmd"] = cmd

        # If command uses --config <json_path>, read the json and append args
        config_matches = re.findall(r'--config\s+([^\s]+)', cmd)
        for cfg_path_str in config_matches:
            cfg_path = Path(cfg_path_str)
            if cfg_path.exists():
                try:
                    with open(cfg_path, "r", encoding="utf-8") as jf:
                        cfg_json = json.load(jf)
                        # Extract strings from args or root
                        all_vals = []
                        if isinstance(cfg_json.get("args"), list):
                            all_vals.extend(cfg_json["args"])
                        for v in cfg_json.values():
                            if isinstance(v, str):
                                all_vals.append(v)
                        m_cfg["cmd"] += " " + " ".join(all_vals)
                except Exception:
                    pass

    return models


def load_media_runtimes() -> list[dict[str, object]]:
    media_file = ROOT / "toolkits" / "l2m2" / "media-runtimes.json"
    if not media_file.exists():
        return []
    try:
        with open(media_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("profiles", [])
    except Exception as e:
        print(f"Warning: Failed to parse media-runtimes.json: {e}", file=sys.stderr)
        return []


def get_disk_usage() -> dict[str, dict[str, str]]:
    disks = {}
    for mount in ["/home", "/mnt/dev", "/run/media/kchauhan/Games"]:
        if os.path.exists(mount):
            stat = os.statvfs(mount)
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bavail * stat.f_frsize
            used = total - free
            disks[mount] = {
                "total": f"{total / (1024**3):.1f} GiB",
                "used": f"{used / (1024**3):.1f} GiB",
                "free": f"{free / (1024**3):.1f} GiB",
                "pct": f"{int((used / total) * 100)}%",
            }
    return disks


def scan_inventory() -> dict[str, object]:
    swap_models = load_llama_swap_models()
    media_profiles = load_media_runtimes()

    # Collect served paths
    served_paths = set()
    for m_id, m_cfg in swap_models.items():
        cmd = str(m_cfg.get("cmd", ""))
        for part in cmd.split():
            if part.endswith(tuple(MODEL_EXTENSIONS)) or "mmproj" in part:
                p = part.replace("${env.L3MS_ROOT}", str(ROOT))
                if os.path.exists(p):
                    served_paths.add(str(Path(p).resolve()))

    # Check ComfyUI symlinks
    comfy_models_dir = ROOT / "toolkits" / "l2m2" / "vendor" / "ComfyUI" / "models"
    comfy_paths = set()
    if comfy_models_dir.exists():
        for dirpath, _, filenames in os.walk(comfy_models_dir):
            for fname in filenames:
                cp = Path(dirpath) / fname
                try:
                    comfy_paths.add(str(cp.resolve()))
                except Exception:
                    pass

    # Scan filesystem
    unique_files: dict[str, dict[str, object]] = {}
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for dirpath, _, filenames in os.walk(root, followlinks=False):
            for fname in filenames:
                p = Path(dirpath) / fname
                ext = p.suffix.lower()
                if ext in MODEL_EXTENSIONS:
                    try:
                        sz = p.stat().st_size
                        real = str(p.resolve())
                        is_link = p.is_symlink()

                        # Determine category & status
                        is_served_llm = real in served_paths or any(real in sp for sp in served_paths)
                        # Shard 2 auto-detection: if Shard 1 is served, Shard 2 is served
                        if not is_served_llm and "-00002-of-00002.gguf" in fname:
                            shard1_equivalent = real.replace("-00002-of-00002.gguf", "-00001-of-00002.gguf")
                            if shard1_equivalent in served_paths or any(shard1_equivalent in sp for sp in served_paths):
                                is_served_llm = True

                        is_media = (
                            "/media/" in real
                            or real in comfy_paths
                            or any(real in cp for cp in comfy_paths)
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
                                "mount": "/mnt/dev" if "/mnt/dev" in real else ("/run/media/kchauhan/Games" if "Games" in real else "/home"),
                                "symlinks": [],
                                "status": status,
                                "action": action,
                                "is_served": is_served_llm or is_media,
                            }
                        if is_link:
                            unique_files[real]["symlinks"].append(str(p))
                    except Exception:
                        pass

    # Group into models / directories
    items = sorted(unique_files.values(), key=lambda x: x["size"], reverse=True)
    return {
        "disks": get_disk_usage(),
        "total_files": len(items),
        "total_bytes": sum(i["size"] for i in items),
        "items": items,
        "served_count": sum(1 for i in items if i["is_served"]),
        "unserved_count": sum(1 for i in items if not i["is_served"]),
    }


def format_report_md(inv: dict[str, object]) -> str:
    lines = []
    lines.append("# Model Storage & Cleanup Audit Report\n")
    lines.append("### Filesystem Mounts")
    lines.append("| Mount | Total | Used | Available | Use% |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for mnt, d in inv["disks"].items():
        lines.append(f"| `{mnt}` | {d['total']} | {d['used']} | **{d['free']}** | {d['pct']} |")

    total_gib = inv["total_bytes"] / (1024**3)
    lines.append(f"\n**Total Tracked Model Footprint:** {total_gib:.2f} GiB across {inv['total_files']} files.\n")

    lines.append("### Model Inventory & Serving Status")
    lines.append("| Status | Size | Model File / Directory | Mount | Symlinks | Action |")
    lines.append("| :---: | :--- | :--- | :--- | :---: | :---: |")

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

        # Check if active
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
