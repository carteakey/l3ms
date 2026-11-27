"""Server-side llama-swap client, YAML catalog parser, engine inspector, log tailer, and bench reader.

Stdlib-only module. Security contract:
- LLAMA_SWAP_API_KEY is read in-process from env, systemd unit Environment, or
  repo `.env` and NEVER returned in JSON payloads, HTML, or logs.
- All log lines and upstream error strings pass through `scrub_secrets()`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request


VALID_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._:@/-]{1,128}$")
BEARER_RE = re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._\-+/=]{4,}")
L3MS_KEY_RE = re.compile(r"\bl3ms-[A-Za-z0-9._\-]{6,}\b")
ENV_SECRET_RE = re.compile(
    r"(?i)\b(LLAMA_SWAP_API_KEY|API_KEY|X-API-KEY|AUTHORIZATION)\s*([:=])\s*(['\"]?)([^\s'\",;]+)\3"
)


def scrub_secrets(text: str, extra_secrets: list[str] | None = None) -> str:
    """Defensively redact API keys and Bearer tokens from text before returning to browsers."""
    if not text:
        return text
    out = text
    if extra_secrets:
        for secret in extra_secrets:
            if secret and len(secret) >= 4:
                out = out.replace(secret, "[REDACTED]")
    out = BEARER_RE.sub(r"\1[REDACTED]", out)
    out = L3MS_KEY_RE.sub("[REDACTED_KEY]", out)
    out = ENV_SECRET_RE.sub(r"\1\2[REDACTED]", out)
    return out


def parse_systemd_env_for_key(env_output: str, var_name: str = "LLAMA_SWAP_API_KEY") -> str:
    """Extract an environment variable from `systemctl --user show ... -p Environment --value`."""
    prefix = f"{var_name}="
    try:
        tokens = shlex.split(env_output)
    except ValueError:
        tokens = env_output.split()
    for token in tokens:
        if token.startswith(prefix):
            return token[len(prefix) :].strip().strip("'\"")
    return ""


def load_api_key(
    repo_root: Path | None = None,
    env: dict[str, str] | None = None,
    systemd_reader: Callable[[], str] | None = None,
) -> str:
    """Load LLAMA_SWAP_API_KEY server-side matching `maintenance/l3ms-ctl.sh`."""
    env_map = os.environ if env is None else env
    direct = env_map.get("LLAMA_SWAP_API_KEY", "").strip()
    if direct:
        return direct

    if systemd_reader is None:
        def _read_systemd() -> str:
            try:
                proc = subprocess.run(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        "llama-swap.service",
                        "--property=Environment",
                        "--value",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2.5,
                    check=False,
                )
                return proc.stdout if proc.returncode == 0 else ""
            except (OSError, subprocess.TimeoutExpired):
                return ""
        systemd_reader = _read_systemd

    from_systemd = parse_systemd_env_for_key(systemd_reader())
    if from_systemd:
        return from_systemd

    if repo_root is not None:
        dotenv = repo_root / ".env"
        if dotenv.is_file():
            try:
                for line in dotenv.read_text(encoding="utf-8", errors="replace").splitlines():
                    stripped = line.strip()
                    if stripped.startswith("LLAMA_SWAP_API_KEY="):
                        return stripped.split("=", 1)[1].strip().strip("'\"")
            except OSError:
                pass
    return ""


def _unquote(value: str) -> str:
    s = value.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "yes", "on", "1"}


def _parse_inline_list(raw: str) -> list[str]:
    inner = raw.strip()
    if not (inner.startswith("[") and inner.endswith("]")):
        return []
    body = inner[1:-1].strip()
    if not body:
        return []
    return [_unquote(item) for item in body.split(",") if item.strip()]


def _expand_macros(text: str, macros: dict[str, str], repo_root: Path | None = None) -> str:
    """Resolve `${macro}` and `${env.L3MS_ROOT}` / `${env.HOME}` placeholders."""
    out = text
    root_str = str(repo_root) if repo_root else os.environ.get("L3MS_ROOT", "")
    home_str = str(Path.home())

    for _ in range(3):
        changed = False
        if "${env.L3MS_ROOT}" in out and root_str:
            out = out.replace("${env.L3MS_ROOT}", root_str)
            changed = True
        if "${env.HOME}" in out and home_str:
            out = out.replace("${env.HOME}", home_str)
            changed = True
        for key, val in macros.items():
            token = "${" + key + "}"
            if token in out:
                out = out.replace(token, val)
                changed = True
        if not changed:
            break
    return out


def _detect_engine(model_id: str, cmd_raw: str, cmd_resolved: str) -> str:
    hay = f"{cmd_raw} {cmd_resolved}".lower()
    if "polystrata" in hay or "strata-poly" in hay:
        return "polystrata"
    if "strata_server" in hay or "strata-lean" in hay or "--engine strata" in hay:
        return "strata"
    if "comfyui" in hay or model_id == "comfyui_auto":
        return "comfyui"
    if "audiocpp" in hay or model_id == "audiocpp_auto":
        return "audio.cpp"
    if "acestep" in hay or "ace-server" in hay or "ace-step" in model_id:
        return "acestep.cpp"
    if "speech_cpp_server" in hay or "nemotron-3.5-transcribe" in model_id:
        return "speech.cpp"
    if "d3_server" in hay or "llama.cpp-d3" in hay:
        return "llama.cpp-d3"
    if "inkling2_server" in hay or "llama.cpp-inkling2" in hay:
        return "llama.cpp-inkling2"
    if "llama-server" in hay or "llama_server" in hay or "master_server" in hay:
        return "llama.cpp"
    return "custom"


def _classify_category(model_id: str, engine: str, metadata: dict[str, Any], group_name: str | None) -> str:
    task = str(metadata.get("task", "")).lower()
    if task == "embeddings" or group_name in {"embeddings", "co-resident"}:
        return "Embeddings & Co-Resident"
    if task == "decision" or engine == "llama.cpp-d3":
        return "Decisions"
    if task in {"music", "tts", "audio", "image", "video"} or engine in {
        "comfyui",
        "audio.cpp",
        "speech.cpp",
        "acestep.cpp",
    }:
        return "Media & Audio"
    if engine in {"strata", "polystrata"}:
        return "Primary Tiers"
    if "inkling" in model_id or "qwen38" in model_id:
        return "Fallback & Variants"
    return "Specialized LLMs"


def _extract_config_path(cmd_resolved: str) -> str | None:
    m = re.search(r"--config\s+([^\s]+)", cmd_resolved)
    if not m:
        return None
    return _unquote(m.group(1))


def parse_llama_swap_yaml(text: str, repo_root: Path | None = None) -> dict[str, Any]:
    """Stdlib-only parser for L3MS `llama-swap.yaml`.

    Supports `macros:`, `models:` (including `aliases:`, `unlisted:`, `ttl:`,
    `env:`, `metadata:`, `cmd: |` and `cmd: >-`), `groups:`, and
    `hooks.on_startup.preload`. Ignores commented retired blocks.
    """
    global_ttl = 600
    macros: dict[str, str] = {}
    models: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {}
    preload: list[str] = []

    section: str | None = None
    current_model: str | None = None
    current_group: str | None = None
    subblock: str | None = None  # "cmd", "env", "aliases", "metadata", "members", "preload"
    cmd_style: str = "|"
    cmd_indent: int = 0
    in_hooks_startup = False

    lines = text.splitlines()
    for raw_line in lines:
        stripped = raw_line.strip()
        indent = len(raw_line) - len(raw_line.lstrip(" "))

        if not stripped or stripped.startswith("#"):
            continue

        # Continuation of multiline cmd block
        if section == "models" and current_model and subblock == "cmd":
            if indent > cmd_indent:
                models[current_model]["_cmd_lines"].append(stripped)
                continue
            subblock = None

        # Top-level keys (indent == 0)
        if indent == 0:
            current_model = None
            current_group = None
            subblock = None
            in_hooks_startup = False
            m_top = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", stripped)
            if not m_top:
                continue
            key, val = m_top.group(1), m_top.group(2).strip()
            if key == "globalTTL" and val.isdigit():
                global_ttl = int(val)
                section = None
            elif key in {"macros", "models", "groups", "hooks"}:
                section = key
            else:
                section = None
            continue

        if section == "macros":
            m_mac = re.match(r'^"([^"]+)":\s*(.+)$', stripped) or re.match(
                r"^([A-Za-z0-9_.-]+):\s*(.+)$", stripped
            )
            if m_mac:
                macros[m_mac.group(1)] = _unquote(m_mac.group(2))
            continue

        if section == "models":
            # New model entry at indent == 2
            if indent == 2:
                m_id = re.match(r'^"([^"]+)":\s*$', stripped) or re.match(
                    r"^([A-Za-z0-9._:@/-]+):\s*$", stripped
                )
                if m_id:
                    current_model = m_id.group(1)
                    subblock = None
                    models[current_model] = {
                        "id": current_model,
                        "name": current_model,
                        "description": "",
                        "unlisted": False,
                        "ttl": None,
                        "proxy": "",
                        "checkEndpoint": "",
                        "useModelName": "",
                        "aliases": [],
                        "env": [],
                        "metadata": {},
                        "_cmd_lines": [],
                        "_cmd_style": "|",
                    }
                continue

            if not current_model:
                continue

            entry = models[current_model]

            # Model property at indent == 4
            if indent == 4:
                subblock = None
                m_prop = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", stripped)
                if not m_prop:
                    continue
                pkey, pval = m_prop.group(1), m_prop.group(2).strip()
                if pkey == "name":
                    entry["name"] = _unquote(pval)
                elif pkey == "description":
                    entry["description"] = _unquote(pval)
                elif pkey == "unlisted":
                    entry["unlisted"] = _parse_bool(pval)
                elif pkey == "ttl":
                    try:
                        entry["ttl"] = int(pval)
                    except ValueError:
                        pass
                elif pkey == "proxy":
                    entry["proxy"] = _unquote(pval)
                elif pkey == "checkEndpoint":
                    entry["checkEndpoint"] = _unquote(pval)
                elif pkey == "useModelName":
                    entry["useModelName"] = _unquote(pval)
                elif pkey == "cmd":
                    if pval in {"|", ">-", ">", "|-"}:
                        subblock = "cmd"
                        cmd_style = pval
                        entry["_cmd_style"] = pval
                        cmd_indent = indent
                    elif pval:
                        entry["_cmd_lines"].append(_unquote(pval))
                elif pkey == "env":
                    if pval.startswith("["):
                        entry["env"].extend(_parse_inline_list(pval))
                    else:
                        subblock = "env"
                elif pkey == "aliases":
                    if pval.startswith("["):
                        entry["aliases"].extend(_parse_inline_list(pval))
                    else:
                        subblock = "aliases"
                elif pkey == "metadata":
                    subblock = "metadata"
                continue

            # Nested list/map under model property (indent >= 6)
            if indent >= 6 and subblock in {"env", "aliases"}:
                if stripped.startswith("- "):
                    entry[subblock].append(_unquote(stripped[2:]))
                continue

            if indent >= 6 and subblock == "metadata":
                m_meta = re.match(r"^([A-Za-z0-9_.-]+):\s*(.*)$", stripped)
                if m_meta:
                    entry["metadata"][m_meta.group(1)] = _unquote(m_meta.group(2))
                continue

        if section == "groups":
            if indent == 2:
                m_grp = re.match(r'^"?([A-Za-z0-9_.-]+)"?:\s*$', stripped)
                if m_grp:
                    current_group = m_grp.group(1)
                    subblock = None
                    groups[current_group] = {
                        "name": current_group,
                        "swap": True,
                        "exclusive": True,
                        "persistent": False,
                        "members": [],
                    }
                continue
            if not current_group:
                continue
            grp = groups[current_group]
            if indent == 4:
                subblock = None
                m_gp = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", stripped)
                if not m_gp:
                    continue
                gkey, gval = m_gp.group(1), m_gp.group(2).strip()
                if gkey in {"swap", "exclusive", "persistent"}:
                    grp[gkey] = _parse_bool(gval)
                elif gkey == "members":
                    if gval.startswith("["):
                        grp["members"].extend(_parse_inline_list(gval))
                    else:
                        subblock = "members"
                continue
            if indent >= 6 and subblock == "members" and stripped.startswith("- "):
                grp["members"].append(_unquote(stripped[2:]))
                continue

        if section == "hooks":
            if indent == 2 and stripped.startswith("on_startup:"):
                in_hooks_startup = True
                subblock = None
                continue
            if in_hooks_startup and indent == 4 and stripped.startswith("preload:"):
                rest = stripped.split(":", 1)[1].strip()
                if rest.startswith("["):
                    preload.extend(_parse_inline_list(rest))
                else:
                    subblock = "preload"
                continue
            if in_hooks_startup and indent >= 6 and subblock == "preload" and stripped.startswith("- "):
                preload.append(_unquote(stripped[2:]))
                continue

    # Build member -> group lookup
    member_to_group: dict[str, dict[str, Any]] = {}
    for gname, ginfo in groups.items():
        for member in ginfo["members"]:
            member_to_group[member] = ginfo

    preload_set = set(preload)
    catalog: list[dict[str, Any]] = []

    for model_id, entry in models.items():
        cmd_raw = " ".join(entry.pop("_cmd_lines"))
        entry.pop("_cmd_style", None)
        cmd_resolved = _expand_macros(cmd_raw, macros, repo_root=repo_root)
        engine = _detect_engine(model_id, cmd_raw, cmd_resolved)
        grp_info = member_to_group.get(model_id)
        group_name = grp_info["name"] if grp_info else "default"
        exclusive = grp_info["exclusive"] if grp_info else True
        swap = grp_info["swap"] if grp_info else True

        config_path = _extract_config_path(cmd_resolved)
        has_vision = (
            "--mmproj" in cmd_resolved
            or "vision" in model_id.lower()
            or "inkling-2-mini" == model_id
            or "nemotron-3.5-bolt" == model_id
        )
        is_reasoning = any(
            kw in model_id.lower() for kw in ("qwen38", "qwen3-8", "inkling", "gemma", "nemotron", "strata", "plat", "abliterated")
        )
        task = str(entry["metadata"].get("task", "")).lower()
        if task in {"embeddings", "decision", "music", "tts", "audio", "image", "video"} or engine in {
            "comfyui",
            "audio.cpp",
            "speech.cpp",
            "acestep.cpp",
        }:
            is_reasoning = False

        entry.update(
            {
                "ttl": entry["ttl"] if entry["ttl"] is not None else global_ttl,
                "engine": engine,
                "group": group_name,
                "exclusive": exclusive,
                "swap": swap,
                "preload": model_id in preload_set,
                "category": _classify_category(model_id, engine, entry["metadata"], grp_info["name"] if grp_info else None),
                "has_vision": has_vision,
                "is_reasoning": is_reasoning,
                "config_path": config_path,
                "proxy_resolved": _expand_macros(entry["proxy"], macros, repo_root=repo_root),
                "cmd_summary": scrub_secrets(cmd_resolved),
                "env": [scrub_secrets(e) for e in entry["env"]],
            }
        )
        catalog.append(entry)

    return {
        "global_ttl": global_ttl,
        "macros": macros,
        "groups": groups,
        "preload": preload,
        "models": catalog,
        "models_by_id": {m["id"]: m for m in catalog},
    }


def inspect_engine_config(
    model_entry: dict[str, Any] | None,
    repo_root: Path,
    extra_secrets: list[str] | None = None,
) -> dict[str, Any] | None:
    """Inspect active Strata/PolyStrata JSON config or summarize llama.cpp CLI flags."""
    if not model_entry:
        return None

    engine = model_entry.get("engine", "custom")
    config_path_raw = model_entry.get("config_path")

    if engine in {"strata", "polystrata"} and config_path_raw:
        cfg_path = Path(config_path_raw)
        if not cfg_path.is_file():
            # Also try relative to repo_root if path pointed at another worktree
            rel_idx = config_path_raw.find("vendor/")
            if rel_idx != -1:
                candidate = repo_root / config_path_raw[rel_idx:]
                if candidate.is_file():
                    cfg_path = candidate

        if cfg_path.is_file():
            try:
                raw_text = cfg_path.read_text(encoding="utf-8", errors="replace")
                data = json.loads(raw_text)
            except (OSError, json.JSONDecodeError) as exc:
                return {
                    "engine": engine,
                    "config_path": str(cfg_path),
                    "error": f"failed to parse engine config: {exc}",
                }

            env_map = {
                str(k): scrub_secrets(str(v), extra_secrets)
                for k, v in (data.get("env") or {}).items()
            }
            args_list = [scrub_secrets(str(a), extra_secrets) for a in (data.get("args") or [])]
            vision = data.get("vision")
            if isinstance(vision, dict):
                vision = {k: scrub_secrets(str(v), extra_secrets) if isinstance(v, str) else v for k, v in vision.items()}

            rel_display = str(cfg_path)
            try:
                rel_display = str(cfg_path.relative_to(repo_root))
            except ValueError:
                pass

            return {
                "engine": engine,
                "config_path": rel_display,
                "exe": data.get("exe"),
                "model_name": data.get("model_name"),
                "display_name": data.get("display_name"),
                "port": data.get("port"),
                "log": data.get("log"),
                "fit_max_tokens": data.get("fit_max_tokens"),
                "reasoning_budget_tokens": data.get("reasoning_budget_tokens"),
                "args": args_list,
                "env": env_map,
                "strata_env": {k: v for k, v in env_map.items() if k.startswith("STRATA_")},
                "vision": vision,
                "sampling": data.get("sampling"),
                "allowed_hosts": data.get("allowed_hosts") or [],
            }

    # Non-Strata engine summary from parsed command line
    cmd = model_entry.get("cmd_summary", "")
    env_dict: dict[str, str] = {}
    for item in model_entry.get("env") or []:
        if "=" in item:
            k, v = item.split("=", 1)
            env_dict[k] = scrub_secrets(v, extra_secrets)

    flags: list[str] = []
    try:
        tokens = shlex.split(cmd)
        flags = [t for t in tokens if t.startswith("-") or t in {"on", "off", "q8_0", "q5_0", "q4_1", "f16"}]
    except ValueError:
        pass

    return {
        "engine": engine,
        "config_path": None,
        "exe": None,
        "model_name": model_entry.get("id"),
        "log": None,
        "args": flags[:40],
        "env": env_dict,
        "strata_env": {k: v for k, v in env_dict.items() if k.startswith("STRATA_")},
        "vision": {"enabled": True} if model_entry.get("has_vision") else None,
        "sampling": None,
    }


def parse_bench_jsonl(filename: str, text: str) -> list[dict[str, Any]]:
    """Normalize rows from `logs/results/*.jsonl` (Strata tuning, Qwen3.8 A/B, Inkling A/B, diag)."""
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue

        label = obj.get("arm") or obj.get("label") or f"row-{line_no}"
        ts = str(obj.get("ts") or "")

        # Decode throughput (prefer steady-state mean)
        tg_steady = obj.get("tg_steady")
        if tg_steady is None:
            tg_steady = obj.get("tg_steady_mean")
        if tg_steady is None:
            tg_steady = obj.get("tg_mean")

        edit_tg = obj.get("edit_tg")

        # Draft acceptance average from tg_probes if present
        draft_acc = None
        probes = obj.get("tg_probes")
        if isinstance(probes, list):
            accs = [p.get("draft_acc") for p in probes if isinstance(p, dict) and p.get("draft_acc") is not None]
            if accs:
                draft_acc = round((sum(accs) / len(accs)) * 100.0, 1)

        # Prefill metrics
        pp2k = obj.get("pp2k")
        pp8k = obj.get("pp8k")
        pp_long = obj.get("pp20k")
        pp_long_label = "20k" if pp_long is not None else None
        pp_map = obj.get("pp")
        if isinstance(pp_map, dict):
            def _avg_pp(key: str) -> float | None:
                items = pp_map.get(key)
                if isinstance(items, list):
                    vals = [i.get("pp") for i in items if isinstance(i, dict) and i.get("pp") is not None]
                    if vals:
                        return round(sum(vals) / len(vals), 1)
                return None

            if pp2k is None:
                pp2k = _avg_pp("2048")
            if pp8k is None:
                pp8k = _avg_pp("8192")
            if pp_long is None:
                pp32k = _avg_pp("32768")
                if pp32k is not None:
                    pp_long = pp32k
                    pp_long_label = "32k"

        vram_mib = obj.get("vram_peak_mib")
        if vram_mib is None:
            vram_mib = obj.get("vram_mb")

        kind = "diag" if "hit_rate_pct" in obj and tg_steady is None else "ab"

        rows.append(
            {
                "file": filename,
                "kind": kind,
                "ts": ts,
                "label": str(label),
                "tg_steady": round(float(tg_steady), 2) if tg_steady is not None else None,
                "edit_tg": round(float(edit_tg), 2) if edit_tg is not None else None,
                "draft_acc_pct": draft_acc,
                "pp2k": round(float(pp2k), 1) if pp2k is not None else None,
                "pp8k": round(float(pp8k), 1) if pp8k is not None else None,
                "pp_long": round(float(pp_long), 1) if pp_long is not None else None,
                "pp_long_label": pp_long_label,
                "vram_mib": int(round(float(vram_mib))) if vram_mib is not None else None,
                "load_s": round(float(obj["load_s"]), 2) if obj.get("load_s") is not None else None,
                "needle_4k": obj.get("needle_4k"),
                "hit_rate_pct": obj.get("hit_rate_pct"),
                "io_mb_per_token": obj.get("io_mb_per_token"),
                "expert_slots": obj.get("expert_slots"),
                "env": {
                    str(k): scrub_secrets(str(v))
                    for k, v in (obj.get("env") or {}).items()
                    if isinstance(obj.get("env"), dict)
                },
            }
        )
    return rows


def load_bench_results(
    repo_root: Path,
    selected_file: str | None = None,
    limit: int = 60,
) -> dict[str, Any]:
    """Read and normalize `logs/results/*.jsonl` files."""
    results_dir = repo_root / "logs" / "results"
    if not results_dir.is_dir():
        return {"files": [], "runs": []}

    jsonl_files = sorted(p for p in results_dir.glob("*.jsonl") if p.is_file())
    file_names = [p.name for p in jsonl_files]
    runs: list[dict[str, Any]] = []

    for path in jsonl_files:
        if selected_file and path.name != selected_file:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        runs.extend(parse_bench_jsonl(path.name, text))

    # Sort newest first by ISO-like timestamp string
    runs.sort(key=lambda r: r.get("ts", ""), reverse=True)
    clamped = max(1, min(limit, 200))
    return {"files": file_names, "runs": runs[:clamped], "total": len(runs)}


def tail_file_lines(path: Path, max_lines: int = 100, max_bytes: int = 262144) -> list[str]:
    """Read the last `max_lines` from a potentially large log file in bounded memory."""
    if not path.is_file():
        return []
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            chunk = fh.read()
    except OSError:
        return []
    text = chunk.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return lines[-max_lines:]


class RouterClient:
    """Server-side client for llama-swap (`:8080`), YAML catalog, logs, and benches."""

    def __init__(
        self,
        repo_root: Path,
        swap_url: str = "http://127.0.0.1:8080",
        api_key: str | None = None,
        tailscale_host: str = "yeti-cachy.llama-alnair.ts.net",
        ui_port: int = 8089,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.swap_url = swap_url.rstrip("/")
        self.tailscale_host = tailscale_host
        self.ui_port = ui_port
        self._api_key = api_key if api_key is not None else load_api_key(self.repo_root)

    def _secrets_list(self) -> list[str]:
        return [self._api_key] if self._api_key else []

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        hdrs = {"Accept": "application/json"}
        if self._api_key:
            hdrs["Authorization"] = f"Bearer {self._api_key}"
        if extra:
            hdrs.update(extra)
        return hdrs

    def _http_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout: float = 5.0,
    ) -> tuple[int, str, str | None]:
        url = f"{self.swap_url}{path}"
        data = None
        extra_headers: dict[str, str] = {}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            extra_headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=self._headers(extra_headers), method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read(1024 * 1024).decode("utf-8", errors="replace")
                return int(resp.status), raw, None
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read(65536).decode("utf-8", errors="replace")
            except Exception:
                pass
            return int(exc.code), err_body, scrub_secrets(f"HTTP {exc.code}: {exc.reason}", self._secrets_list())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return 0, "", scrub_secrets(str(exc), self._secrets_list())

    def parse_catalog(self) -> dict[str, Any]:
        swap_yaml = self.repo_root / "llama-swap.yaml"
        if not swap_yaml.is_file():
            return {"global_ttl": 600, "macros": {}, "groups": {}, "preload": [], "models": [], "models_by_id": {}}
        text = swap_yaml.read_text(encoding="utf-8", errors="replace")
        return parse_llama_swap_yaml(text, repo_root=self.repo_root)

    def fetch_running(self, timeout: float = 3.0) -> dict[str, Any]:
        status, body, err = self._http_request("GET", "/running", timeout=timeout)
        if status == 200:
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                return {"reachable": True, "running": [], "error": "invalid JSON from /running"}
            raw_running = payload.get("running")
            items: list[dict[str, Any]] = []
            if isinstance(raw_running, list):
                for item in raw_running:
                    if isinstance(item, dict):
                        items.append(
                            {
                                "model": str(item.get("model", "")),
                                "state": str(item.get("state", "unknown")),
                            }
                        )
                    elif isinstance(item, str):
                        items.append({"model": item, "state": "ready"})
            return {"reachable": True, "running": items, "error": None}
        return {
            "reachable": status > 0,
            "running": [],
            "error": scrub_secrets(err or f"status {status}", self._secrets_list()),
        }

    def fetch_v1_models(self, timeout: float = 3.0) -> tuple[bool, list[str], str | None]:
        status, body, err = self._http_request("GET", "/v1/models", timeout=timeout)
        if status == 200:
            try:
                payload = json.loads(body)
                ids = [str(d["id"]) for d in payload.get("data", []) if isinstance(d, dict) and "id" in d]
                return True, ids, None
            except (json.JSONDecodeError, KeyError):
                return True, [], "invalid JSON from /v1/models"
        return status > 0, [], scrub_secrets(err or f"status {status}", self._secrets_list())

    def get_router_state(self, include_unlisted: bool = False) -> dict[str, Any]:
        catalog = self.parse_catalog()
        running_info = self.fetch_running()
        v1_ok, v1_ids, v1_err = self.fetch_v1_models()

        models_by_id = catalog["models_by_id"]
        # Map aliases back to canonical model entry too
        alias_to_canonical: dict[str, dict[str, Any]] = {}
        for m in catalog["models"]:
            for alias in m.get("aliases") or []:
                alias_to_canonical[alias] = m

        running_enriched: list[dict[str, Any]] = []
        active_configs: list[dict[str, Any]] = []

        for item in running_info["running"]:
            mid = item["model"]
            meta = models_by_id.get(mid) or alias_to_canonical.get(mid)
            cfg = inspect_engine_config(meta, self.repo_root, extra_secrets=self._secrets_list()) if meta else None
            enriched = {
                "model": mid,
                "state": item["state"],
                "name": meta["name"] if meta else mid,
                "engine": meta["engine"] if meta else "unknown",
                "group": meta["group"] if meta else "default",
                "exclusive": meta["exclusive"] if meta else True,
                "ttl": meta["ttl"] if meta else catalog["global_ttl"],
                "has_vision": meta["has_vision"] if meta else False,
                "config": cfg,
            }
            running_enriched.append(enriched)
            if cfg:
                active_configs.append({"model": mid, **cfg})

        running_ids = {r["model"] for r in running_enriched}
        visible_models = []
        unlisted_count = 0
        for m in catalog["models"]:
            if m["unlisted"]:
                unlisted_count += 1
                if not include_unlisted:
                    continue
            row = dict(m)
            row["running"] = (m["id"] in running_ids) or any(a in running_ids for a in m.get("aliases") or [])
            visible_models.append(row)

        return {
            "reachable": running_info["reachable"] or v1_ok,
            "auth_configured": bool(self._api_key),
            "running": running_enriched,
            "v1_models": v1_ids,
            "active_engine_config": active_configs[0] if active_configs else None,
            "active_engine_configs": active_configs,
            "catalog": visible_models,
            "catalog_total": len(catalog["models"]),
            "unlisted_count": unlisted_count,
            "groups": catalog["groups"],
            "preload": catalog["preload"],
            "global_ttl": catalog["global_ttl"],
            "endpoints": {
                "tailscale_host": self.tailscale_host,
                "ui_port": self.ui_port,
                "swap_port": urllib.parse.urlparse(self.swap_url).port or 8080,
            },
            "error": running_info["error"] or v1_err,
        }

    def wake_model(self, model_id: str, timeout: float = 120.0) -> dict[str, Any]:
        if not VALID_MODEL_ID_RE.match(model_id):
            return {"ok": False, "model_id": model_id, "error": "invalid model id"}

        before = self.fetch_running(timeout=2.0)
        catalog = self.parse_catalog()
        target_meta = catalog["models_by_id"].get(model_id)
        target_exclusive = target_meta["exclusive"] if target_meta else True

        evicted_candidates = [
            r["model"]
            for r in before["running"]
            if r["model"] != model_id and target_exclusive
        ]

        encoded = urllib.parse.quote(model_id, safe="")
        started = time.monotonic()
        status, _body, err = self._http_request("GET", f"/upstream/{encoded}/", timeout=timeout)
        elapsed_s = round(time.monotonic() - started, 2)
        ok = 200 <= status < 400
        after = self.fetch_running(timeout=3.0)

        return {
            "ok": ok,
            "model_id": model_id,
            "http_status": status,
            "elapsed_s": elapsed_s,
            "evicted_candidates": evicted_candidates,
            "running": after["running"],
            "error": None if ok else scrub_secrets(err or f"upstream returned HTTP {status}", self._secrets_list()),
        }

    def unload_models(self, model_id: str | None = None, timeout: float = 15.0) -> dict[str, Any]:
        started = time.monotonic()
        if model_id:
            if not VALID_MODEL_ID_RE.match(model_id):
                return {"ok": False, "model_id": model_id, "error": "invalid model id"}
            encoded = urllib.parse.quote(model_id, safe="")
            status, body, err = self._http_request("POST", f"/api/models/unload/{encoded}", timeout=timeout)
        else:
            # Call llama-swap /unload to free VRAM; fall back to POST /api/models/unload if 404/405.
            status, body, err = self._http_request("GET", "/unload", timeout=timeout)
            if status in {404, 405}:
                status, body, err = self._http_request("POST", "/api/models/unload", timeout=timeout)
        elapsed_s = round(time.monotonic() - started, 2)
        ok = 200 <= status < 300
        after = self.fetch_running(timeout=3.0)
        return {
            "ok": ok,
            "model_id": model_id,
            "http_status": status,
            "elapsed_s": elapsed_s,
            "message": scrub_secrets(body.strip()[:200], self._secrets_list()),
            "running": after["running"],
            "error": None if ok else scrub_secrets(err or f"unload returned HTTP {status}", self._secrets_list()),
        }

    def smoke_model(self, model_id: str, timeout: float = 90.0) -> dict[str, Any]:
        """Run a bounded smoke probe (`max_tokens=64` for LLM/reasoning models)."""
        if not VALID_MODEL_ID_RE.match(model_id):
            return {"ok": False, "model_id": model_id, "error": "invalid model id"}

        catalog = self.parse_catalog()
        meta = catalog["models_by_id"].get(model_id)
        engine = meta["engine"] if meta else "llama.cpp"
        task = str((meta.get("metadata") or {}).get("task", "")).lower() if meta else ""

        started = time.monotonic()
        encoded = urllib.parse.quote(model_id, safe="")

        if engine == "comfyui":
            status, body, err = self._http_request("GET", f"/upstream/{encoded}/system_stats", timeout=timeout)
            elapsed_s = round(time.monotonic() - started, 2)
            ok = 200 <= status < 300
            return {
                "ok": ok,
                "model_id": model_id,
                "probe_type": "system_stats",
                "http_status": status,
                "elapsed_s": elapsed_s,
                "snippet": scrub_secrets(body.strip()[:200], self._secrets_list()),
                "has_reasoning": False,
                "error": None if ok else scrub_secrets(err or f"HTTP {status}", self._secrets_list()),
            }

        if engine in {"audio.cpp", "speech.cpp", "acestep.cpp"} or task == "decision":
            status, body, err = self._http_request("GET", f"/upstream/{encoded}/health", timeout=timeout)
            elapsed_s = round(time.monotonic() - started, 2)
            ok = 200 <= status < 300
            return {
                "ok": ok,
                "model_id": model_id,
                "probe_type": "health",
                "http_status": status,
                "elapsed_s": elapsed_s,
                "snippet": scrub_secrets(body.strip()[:200], self._secrets_list()),
                "has_reasoning": False,
                "error": None if ok else scrub_secrets(err or f"HTTP {status}", self._secrets_list()),
            }

        if task == "embeddings":
            status, body, err = self._http_request(
                "POST",
                "/v1/embeddings",
                body={"model": model_id, "input": "L3MS health check"},
                timeout=timeout,
            )
            elapsed_s = round(time.monotonic() - started, 2)
            ok = 200 <= status < 300
            dims = None
            if ok:
                try:
                    data = json.loads(body)
                    emb = data["data"][0]["embedding"]
                    dims = len(emb)
                except Exception:
                    pass
            return {
                "ok": ok,
                "model_id": model_id,
                "probe_type": "embeddings",
                "http_status": status,
                "elapsed_s": elapsed_s,
                "snippet": f"{dims}-dim vector OK" if dims is not None else scrub_secrets(body.strip()[:160], self._secrets_list()),
                "has_reasoning": False,
                "error": None if ok else scrub_secrets(err or f"HTTP {status}", self._secrets_list()),
            }

        # Standard chat / reasoning probe: ALWAYS max_tokens >= 64 per AGENTS.md.
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with OK"}],
            "max_tokens": 64,
        }
        status, body, err = self._http_request("POST", "/v1/chat/completions", body=payload, timeout=timeout)
        elapsed_s = round(time.monotonic() - started, 2)
        if not (200 <= status < 300):
            return {
                "ok": False,
                "model_id": model_id,
                "probe_type": "chat",
                "http_status": status,
                "elapsed_s": elapsed_s,
                "snippet": "",
                "has_reasoning": False,
                "error": scrub_secrets(err or f"HTTP {status}", self._secrets_list()),
            }

        snippet = ""
        has_reasoning = False
        usage = None
        try:
            parsed = json.loads(body)
            msg = parsed["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            reasoning = (msg.get("reasoning_content") or "").strip()
            has_reasoning = bool(reasoning) or "<think>" in content
            snippet = content or (f"[reasoning] {reasoning[:140]}" if reasoning else "(empty content)")
            usage = parsed.get("usage")
        except Exception:
            snippet = body.strip()[:160]

        return {
            "ok": True,
            "model_id": model_id,
            "probe_type": "chat",
            "http_status": status,
            "elapsed_s": elapsed_s,
            "snippet": scrub_secrets(snippet[:240], self._secrets_list()),
            "has_reasoning": has_reasoning,
            "usage": usage,
            "error": None,
        }

    def get_logs(
        self,
        source: str = "llama-swap",
        lines: int = 100,
        journal_runner: Callable[[int], tuple[int, str, str]] | None = None,
    ) -> dict[str, Any]:
        """Return bounded, secret-scrubbed log lines for `llama-swap` or `engine`."""
        n = max(1, min(int(lines), 500))
        secrets = self._secrets_list()

        if source == "engine":
            log_path: Path | None = None
            # Prefer log file of currently running Strata / PolyStrata engine
            running_state = self.get_router_state(include_unlisted=True)
            active_cfg = running_state.get("active_engine_config")
            if active_cfg and active_cfg.get("log"):
                cand = Path(active_cfg["log"])
                if cand.is_file():
                    log_path = cand
            if log_path is None:
                candidates: list[Path] = []
                for glob_pat in ("vendor/strata-lean/*.log", "vendor/polystrata/*.log"):
                    candidates.extend(p for p in self.repo_root.glob(glob_pat) if p.is_file())
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    log_path = candidates[0]

            if log_path is None:
                return {
                    "source": "engine",
                    "path": None,
                    "lines": [],
                    "count": 0,
                    "note": "No active Strata/PolyStrata log file found; llama.cpp and media engines log to the llama-swap journal.",
                }
            raw_lines = tail_file_lines(log_path, max_lines=n)
            scrubbed = [scrub_secrets(line, secrets) for line in raw_lines]
            return {
                "source": "engine",
                "path": str(log_path),
                "lines": scrubbed,
                "count": len(scrubbed),
                "note": None,
            }

        # Default: llama-swap systemd user journal
        if journal_runner is None:
            def _run_journal(count: int) -> tuple[int, str, str]:
                try:
                    proc = subprocess.run(
                        [
                            "journalctl",
                            "--user",
                            "-u",
                            "llama-swap.service",
                            "-n",
                            str(count),
                            "--no-pager",
                            "-o",
                            "short-iso",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=3.0,
                        check=False,
                    )
                    return proc.returncode, proc.stdout, proc.stderr
                except (OSError, subprocess.TimeoutExpired) as exc:
                    return -1, "", str(exc)
            journal_runner = _run_journal

        rc, out, err = journal_runner(n)
        if rc != 0:
            return {
                "source": "llama-swap",
                "path": "journalctl --user -u llama-swap.service",
                "lines": [],
                "count": 0,
                "note": scrub_secrets(err.strip() or f"journalctl exited {rc}", secrets),
            }
        scrubbed = [scrub_secrets(line, secrets) for line in out.splitlines()[-n:]]
        return {
            "source": "llama-swap",
            "path": "journalctl --user -u llama-swap.service",
            "lines": scrubbed,
            "count": len(scrubbed),
            "note": None,
        }

    def get_benches(self, selected_file: str | None = None, limit: int = 60) -> dict[str, Any]:
        return load_bench_results(self.repo_root, selected_file=selected_file, limit=limit)
