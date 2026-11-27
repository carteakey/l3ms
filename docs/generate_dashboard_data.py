#!/usr/bin/env python3
"""Generate the public dashboard payload from llama-swap configuration."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "llama-swap.yaml"
META = ROOT / "docs" / "dashboard-meta.json"
ENGINES_META = ROOT / "docs" / "engines-meta.json"
COMMUNITY_RUNS = ROOT / "docs" / "community-runs.json"
MEDIA_MANIFEST = ROOT / "media-runtimes.json"
DECISION_MANIFEST = ROOT / "decision-runtimes.json"
STACK_META = ROOT / "docs" / "stack-meta.json"
OUTPUT = ROOT / "docs" / "generated-models.js"
SOURCE_BASE = "https://github.com/carteakey/l3ms/blob/main/"

EVIDENCE_FIELDS = (
    "pp",
    "tg",
    "testedContext",
    "cacheState",
    "draftAcceptance",
    "llamaCppCommit",
    "benchmarkCommand",
)

ALLOWED_STACK_META_KEYS = {
    "guide",
    "experiment",
    "modes",
    "tier",
    "role",
    "vramGb",
    "ramGb",
}

STRATA_PATH_FLAGS = {
    "--pack": "/path/to/strata-pack",
    "--native": "/path/to/model.gguf",
    "--ple-gguf": "/path/to/ple-q8.gguf",
    "--expert-profile": "/path/to/expert-profile.bin",
    "--mtp": "/path/to/mtp-draft-dir",
    "--model": "/path/to/model.gguf",
    "--draft-model": "/path/to/draft-model.gguf",
    "--mmproj": "/path/to/mmproj.gguf",
}


def parse_scalar(value: str):
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith('"'):
        return json.loads(value)
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part) for part in inner.split(",")]
    return value


def parse_macros(text: str) -> dict[str, str]:
    lines = text.splitlines()
    macros: dict[str, str] = {}
    in_macros = False
    for line in lines:
        if line == "macros:":
            in_macros = True
            continue
        if in_macros and line and not line.startswith((" ", "#")):
            break
        if not in_macros:
            continue
        match = re.match(r'^\s+"([^"]+)":\s*(.+?)\s*$', line)
        if not match:
            match = re.match(r"^\s+([A-Za-z0-9_.-]+):\s*(.+?)\s*$", line)
        if match:
            key, raw = match.groups()
            val = parse_scalar(raw)
            if isinstance(val, str):
                macros[key] = val
    return macros


def parse_preload(text: str) -> list[str]:
    lines = text.splitlines()
    preloaded: list[str] = []
    in_preload = False
    for line in lines:
        if re.match(r"^\s+preload:\s*$", line):
            in_preload = True
            continue
        if in_preload:
            item_match = re.match(r'^\s+-\s+"([^"]+)"\s*$', line)
            if not item_match:
                item_match = re.match(r"^\s+-\s+([A-Za-z0-9_.-]+)\s*$", line)
            if item_match:
                preloaded.append(item_match.group(1))
            elif line.strip() and not line.strip().startswith("#"):
                break
    return preloaded


def parse_groups(text: str) -> dict[str, dict]:
    lines = text.splitlines()
    groups: dict[str, dict] = {}
    in_groups = False
    current_group: str | None = None
    in_members = False
    for line in lines:
        if line == "groups:":
            in_groups = True
            continue
        if in_groups and line and not line.startswith((" ", "#")):
            break
        if not in_groups:
            continue
        group_match = re.match(r'^  "?([A-Za-z0-9_.-]+)"?:\s*$', line)
        if group_match:
            current_group = group_match.group(1)
            groups[current_group] = {"members": [], "exclusive": True, "swap": True}
            in_members = False
            continue
        if not current_group:
            continue
        if re.match(r"^    members:\s*$", line):
            in_members = True
            continue
        if in_members:
            member_match = re.match(r'^      -\s+"?([^"\s]+)"?\s*$', line)
            if member_match:
                groups[current_group]["members"].append(member_match.group(1))
                continue
            if line.strip() and not line.strip().startswith("#"):
                in_members = False
        field_match = re.match(r"^    ([a-zA-Z_]+):\s*(.*)$", line)
        if field_match:
            field, raw = field_match.groups()
            if raw:
                groups[current_group][field] = parse_scalar(raw)
    return groups


def parse_models(text: str) -> dict[str, dict]:
    lines = text.splitlines()
    models: dict[str, dict] = {}
    in_models = False
    current_id: str | None = None
    index = 0

    while index < len(lines):
        line = lines[index]
        if line == "models:":
            in_models = True
            index += 1
            continue
        if in_models and line and not line.startswith((" ", "#")):
            break

        model_match = re.match(r'^  "([^"]+)":\s*$', line) if in_models else None
        if not model_match and in_models:
            model_match = re.match(r"^  ([A-Za-z0-9_.-]+):\s*$", line)
        if model_match:
            current_id = model_match.group(1)
            models[current_id] = {"id": current_id, "env": [], "aliases": []}
            index += 1
            continue

        if current_id:
            field_match = re.match(r"^    ([a-zA-Z_]+):\s*(.*)$", line)
            if field_match:
                field, raw_value = field_match.groups()
                raw_stripped = raw_value.strip()
                if field == "cmd" and raw_stripped in {"|", "|-", ">", ">-"}:
                    command_lines = []
                    index += 1
                    while index < len(lines):
                        command_line = lines[index]
                        if command_line and not command_line.startswith("      "):
                            break
                        if command_line.startswith("      "):
                            command_lines.append(command_line[6:])
                        index += 1
                    models[current_id]["cmd"] = "\n".join(command_lines).strip()
                    continue
                if field == "env":
                    index += 1
                    while index < len(lines):
                        env_match = re.match(r'^      -\s+"(.*)"\s*$', lines[index])
                        if not env_match:
                            env_match = re.match(r"^      -\s+([^\s#]+)\s*$", lines[index])
                        if not env_match:
                            break
                        models[current_id]["env"].append(env_match.group(1))
                        index += 1
                    continue
                if field == "aliases":
                    if raw_stripped.startswith("["):
                        parsed_aliases = parse_scalar(raw_stripped)
                        if isinstance(parsed_aliases, list):
                            models[current_id]["aliases"].extend(str(a) for a in parsed_aliases)
                        index += 1
                        continue
                    index += 1
                    while index < len(lines):
                        alias_match = re.match(r'^      -\s+"([^"]+)"\s*$', lines[index])
                        if not alias_match:
                            alias_match = re.match(r"^      -\s+([A-Za-z0-9_.:-]+)\s*$", lines[index])
                        if not alias_match:
                            break
                        models[current_id]["aliases"].append(alias_match.group(1))
                        index += 1
                    continue
                if raw_stripped:
                    models[current_id][field] = parse_scalar(raw_stripped)
        index += 1

    return models


def context_from_command(command: str) -> str:
    match = re.search(r"(?:--ctx-size|-c)\s+(\d+)", command)
    if not match:
        return "Custom"
    size = int(match.group(1))
    return f"{round(size / 1024)}k" if size >= 1024 else str(size)


def benchmark_command(path_value: str | None) -> str | None:
    if not path_value:
        return None
    if Path(path_value).name.startswith("run-"):
        return None
    if path_value.endswith(".py"):
        return f"python3 {path_value}"
    if path_value.endswith(".sh"):
        return f"bash {path_value}"
    return path_value or None


def profile_evidence(display: dict) -> dict:
    evidence = {
        "scope": "local",
        "pp": display.get("pp"),
        "tg": display["tps"],
        "testedContext": display.get("testedContext"),
        "cacheState": display.get("cacheState"),
        "draftAcceptance": display.get("draftAcceptance"),
        "llamaCppCommit": display.get("llamaCppCommit"),
        "benchmarkCommand": display.get(
            "benchmarkCommand", benchmark_command(display.get("benchmark"))
        ),
    }
    missing = [field for field in EVIDENCE_FIELDS if field not in evidence]
    if missing:
        raise SystemExit(f"Profile evidence is missing fields: {missing}")
    return evidence


def validate_community_runs(data: dict) -> list[dict]:
    runs = data.get("runs")
    if not isinstance(runs, list):
        raise SystemExit("community-runs.json must contain a runs array")

    required = {
        "id",
        "submittedAt",
        "sourceUrl",
        "hardware",
        "software",
        "model",
        "benchmark",
        "metrics",
    }
    seen_ids = set()
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise SystemExit(f"Community run {index} must be an object")
        missing = sorted(required - set(run))
        if missing:
            raise SystemExit(f"Community run {index} is missing: {', '.join(missing)}")
        if run["id"] in seen_ids:
            raise SystemExit(f"Duplicate community run id: {run['id']}")
        seen_ids.add(run["id"])
        if not str(run["sourceUrl"]).startswith("https://"):
            raise SystemExit(f"Community run {run['id']} needs an HTTPS sourceUrl")
        if run["benchmark"].get("command") in {None, ""}:
            raise SystemExit(f"Community run {run['id']} needs an exact benchmark command")
        if run["software"].get("llamaCppCommit") in {None, ""}:
            raise SystemExit(f"Community run {run['id']} needs a llama.cpp commit")
        if run["metrics"].get("tg") is None:
            raise SystemExit(f"Community run {run['id']} needs a TG result")
    return runs


def detect_recipe_kind(model: dict) -> str:
    cmd = model.get("cmd", "")
    if "--engine strata" in cmd:
        if "polystrata" in cmd.lower():
            return "polystrata"
        return "strata"
    return "llama.cpp"


def recipe_label_for(kind: str) -> str:
    if kind in {"strata", "polystrata"}:
        return "Portable Strata / PolyStrata Recipe"
    return "Portable llama.cpp command"


def _sanitize_path_value(value: str, fallback: str = "/path/to/asset") -> str:
    if any(marker in value for marker in ("/home/", "/mnt/", "/opt/", ".ts.net", "/root/")):
        return fallback
    return value


def sanitize_strata_config(raw: dict, kind: str = "strata") -> dict:
    """Return a public-safe, host-path-free excerpt of a Strata/PolyStrata JSON config."""
    exe_default = "./vendor/polystrata/build/strata-poly" if kind == "polystrata" else "./vendor/strata-lean/build/strata"
    raw_args = raw.get("args", [])
    clean_args: list[str] = []
    idx = 0
    while idx < len(raw_args):
        token = str(raw_args[idx])
        clean_args.append(token)
        if token in STRATA_PATH_FLAGS and idx + 1 < len(raw_args):
            clean_args.append(STRATA_PATH_FLAGS[token])
            idx += 2
            continue
        idx += 1
    clean_args = [
        _sanitize_path_value(arg, "/path/to/file") if arg not in STRATA_PATH_FLAGS else arg
        for arg in clean_args
    ]

    excerpt: dict = {
        "exe": exe_default,
        "args": clean_args,
    }
    if isinstance(raw.get("env"), dict) and raw["env"]:
        excerpt["env"] = dict(raw["env"])
    if isinstance(raw.get("sampling"), dict) and raw["sampling"]:
        excerpt["sampling"] = dict(raw["sampling"])
    if isinstance(raw.get("vision"), dict):
        v = raw["vision"]
        clean_vision: dict = {
            "exe": "./build-vision/bin/strata-vision" if kind == "polystrata" else "./vendor/strata/engine/strata-vision",
            "mmproj": "/path/to/mmproj-F16.gguf",
            "model": "/path/to/model.gguf",
        }
        for key in ("gpu", "threads", "max_tokens"):
            if key in v:
                clean_vision[key] = v[key]
        excerpt["vision"] = clean_vision
    for key in (
        "fit_max_tokens",
        "reasoning_loop_recovery",
        "reasoning_close_retry",
        "reasoning_budget_tokens",
        "repeat_stop_tokens",
    ):
        if key in raw:
            excerpt[key] = raw[key]
    return excerpt


def portable_command(model: dict) -> str:
    command = model.get("cmd", "")
    command = re.sub(r"(?m)^\s*#.*\n?", "", command).strip()
    replacements = {
        "${cpu_range}": '"$CPU_RANGE"',
        "${ecores}": '"$ECORES"',
        "${llama_server}": '"$LLAMA_SERVER"',
        "${ik_server}": '"$IK_LLAMA_SERVER"',
        "${qwen_mtp_server}": '"$QWEN_MTP_SERVER"',
        "${sarvam_server}": '"$SARVAM_SERVER"',
        "${PORT}": '"$PORT"',
        "${inference_port}": '"$PORT"',
        "${chat_template}": '"$CHAT_TEMPLATE_PATH"',
    }
    for source, target in replacements.items():
        command = command.replace(source, target)

    command = re.sub(r"(?m)(\s-m\s+)\S+", r'\1"$MODEL_PATH"', command, count=1)
    command = re.sub(r"(?m)(--model\s+)\S+", r'\1"$MODEL_PATH"', command, count=1)
    command = re.sub(r"(?m)(--spec-draft-model\s+)\S+", r'\1"$DRAFT_MODEL_PATH"', command)
    command = re.sub(r"(?m)(--mmproj\s+)\S+", r'\1"$MMPROJ_PATH"', command)
    command = re.sub(r"(?m)(--config\s+)/(?:home|mnt)/\S+", r'\1"$ENGINE_CONFIG"', command)
    command = command.replace("--host 0.0.0.0", "--host 127.0.0.1")

    # New engine macros must not leak into shell snippets as unset lowercase
    # router variables. Require operator-supplied paths, not inferred binaries.
    extra_variables = set()
    if "${env.L3MS_ROOT}" in command:
        command = command.replace("${env.L3MS_ROOT}", "${L3MS_ROOT}")
        extra_variables.add("L3MS_ROOT")
    if "${env.HOME}" in command:
        command = command.replace("${env.HOME}", "${HOME}")

    def portable_macro(match):
        name = match.group(1)
        if name in {"L3MS_ROOT", "HOME"}:
            return match.group(0)
        variable = name.upper()
        extra_variables.add(variable)
        return f'"${variable}"'

    command = re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}", portable_macro, command)

    lines = [line.strip() for line in command.splitlines() if line.strip()]
    shell_command = " \\\n  ".join(lines)
    preamble = []
    if "$MODEL_PATH" in shell_command or "--engine strata" not in shell_command:
        preamble.append('MODEL_PATH="/path/to/model.gguf"')
    if "$CPU_RANGE" in shell_command or "--engine strata" not in shell_command:
        preamble.append('CPU_RANGE="${CPU_RANGE:-0-11}"')
    if "$ECORES" in shell_command:
        preamble.append('ECORES="${ECORES:-12-15}"')
    preamble.append('PORT="${PORT:-8080}"')
    preamble.extend(
        f'{variable}="/path/to/{variable.lower()}"'
        for variable in sorted(extra_variables)
    )
    if "$LLAMA_SERVER" in shell_command:
        preamble.append('LLAMA_SERVER="${LLAMA_SERVER:-./vendor/llama.cpp/build/bin/llama-server}"')
    if "$IK_LLAMA_SERVER" in shell_command:
        preamble.append('IK_LLAMA_SERVER="${IK_LLAMA_SERVER:-./vendor/ik_llama.cpp/build/bin/llama-server}"')
    if "$QWEN_MTP_SERVER" in shell_command:
        preamble.append('QWEN_MTP_SERVER="${QWEN_MTP_SERVER:-./vendor/llama.cpp/build/bin/llama-server}"')
    if "$SARVAM_SERVER" in shell_command:
        preamble.append('SARVAM_SERVER="${SARVAM_SERVER:-./vendor/llama.cpp-pr-test-20275/build/bin/llama-server}"')
    if "$DRAFT_MODEL_PATH" in shell_command:
        preamble.append('DRAFT_MODEL_PATH="/path/to/draft-model.gguf"')
    if "$MMPROJ_PATH" in shell_command:
        preamble.append('MMPROJ_PATH="/path/to/mmproj.gguf"')
    if "$CHAT_TEMPLATE_PATH" in shell_command:
        preamble.append('CHAT_TEMPLATE_PATH="./chat-template.jinja"')
    for env_value in model.get("env", []):
        preamble.append(f"export {env_value}")
    return "\n".join(preamble) + "\n\n" + shell_command


def portable_recipe(
    model: dict,
    macros: dict[str, str] | None = None,
    root: Path = ROOT,
) -> tuple[str, str, str, dict | None]:
    """Return (recipe_kind, recipe_label, command_text, engine_config_excerpt)."""
    kind = detect_recipe_kind(model)
    label = recipe_label_for(kind)
    base_command = portable_command(model)
    if kind not in {"strata", "polystrata"}:
        return kind, label, base_command, None

    config_excerpt = None
    cmd_raw = model.get("cmd", "")
    config_match = re.search(r"--config\s+\$\{([A-Za-z_][A-Za-z_0-9]*)\}", cmd_raw)
    if config_match and macros:
        macro_name = config_match.group(1)
        macro_val = macros.get(macro_name, "")
        if macro_val:
            resolved_str = macro_val.replace("${env.L3MS_ROOT}", str(root))
            config_path = Path(resolved_str)
            if config_path.is_file():
                try:
                    raw_cfg = json.loads(config_path.read_text())
                    config_excerpt = sanitize_strata_config(raw_cfg, kind=kind)
                except (OSError, json.JSONDecodeError):
                    config_excerpt = None

    if not config_excerpt:
        return kind, label, base_command, None

    env_lines = [
        f"export {k}={json.dumps(str(v))}"
        for k, v in sorted(config_excerpt.get("env", {}).items())
    ]
    parts = []
    if env_lines:
        parts.append("# Active engine runtime switches\n" + "\n".join(env_lines))
    parts.append("# Launch server\n" + base_command)
    parts.append(
        "# Portable engine JSON config excerpt\n"
        + json.dumps(config_excerpt, indent=2)
    )
    return kind, label, "\n\n".join(parts), config_excerpt


def public_doc_url(relative: str) -> str:
    if not isinstance(relative, str) or not relative:
        raise SystemExit("Stack guide/evidence path must be a non-empty string")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".md":
        raise SystemExit(f"Unsafe stack guide/evidence path: {relative}")
    if not (ROOT / path).is_file():
        raise SystemExit(f"Stack guide/evidence does not exist: {relative}")
    return SOURCE_BASE + path.as_posix()


def profile_source_url(manifest: Path, profile_id: str) -> str:
    marker = f'"id": {json.dumps(profile_id)}'
    lines = manifest.read_text().splitlines()
    matches = [number for number, line in enumerate(lines, 1) if marker in line]
    if len(matches) != 1:
        raise SystemExit(f"Expected one {profile_id} source line in {manifest}")
    return SOURCE_BASE + manifest.relative_to(ROOT).as_posix() + f"#L{matches[0]}"


def checked_stack_metadata(profiles: list[dict], metadata: dict, lane: str) -> None:
    ids = [profile.get("id") for profile in profiles]
    if any(not isinstance(profile_id, str) or not profile_id for profile_id in ids):
        raise SystemExit(f"{lane} manifest has an empty/invalid profile ID")
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{lane} manifest has duplicate profile IDs")
    if set(ids) != set(metadata):
        missing = sorted(set(ids) - set(metadata))
        stale = sorted(set(metadata) - set(ids))
        raise SystemExit(
            f"{lane} display metadata mismatch; missing={missing or 'none'}, "
            f"stale={stale or 'none'}"
        )
    for profile_id, display in metadata.items():
        if not isinstance(display, dict) or set(display) - ALLOWED_STACK_META_KEYS:
            raise SystemExit(f"Invalid {lane} display metadata for {profile_id}")
        public_doc_url(display.get("guide"))
        if display.get("experiment") is not None:
            public_doc_url(display["experiment"])
        if "modes" in display and (
            lane != "media"
            or not isinstance(display["modes"], list)
            or not all(isinstance(mode, str) and mode for mode in display["modes"])
        ):
            raise SystemExit(f"Invalid {lane} modes for {profile_id}")


def media_public_profiles(manifest: dict, metadata: dict) -> list[dict]:
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("profiles"), list):
        raise SystemExit("Unsupported media manifest schema")
    profiles = manifest["profiles"]
    checked_stack_metadata(profiles, metadata, "media")
    public = []
    for profile in profiles:
        profile_id = profile["id"]
        if not all(isinstance(profile.get(key), str) and profile[key] for key in ("name", "runtime", "status", "script")):
            raise SystemExit(f"Incomplete media profile {profile_id}")
        if not isinstance(profile.get("tasks"), list) or not all(isinstance(task, str) and task for task in profile["tasks"]):
            raise SystemExit(f"Incomplete media tasks for {profile_id}")
        script = Path(profile["script"])
        if script.is_absolute() or ".." in script.parts or not (MEDIA_MANIFEST.parent / script).is_file():
            raise SystemExit(f"Invalid media script for {profile_id}")
        display = metadata[profile_id]
        public.append({
            "id": profile_id,
            "name": profile["name"],
            "runtime": profile["runtime"],
            "device": profile.get("device", "cuda"),
            "variant": profile.get("variant", ""),
            "status": profile["status"],
            "tasks": profile["tasks"],
            "modes": display.get("modes", []),
            "tier": display.get("tier", "burst"),
            "role": display.get("role", ""),
            "vramGb": display.get("vramGb"),
            "ramGb": display.get("ramGb"),
            "sourceUrl": profile_source_url(MEDIA_MANIFEST, profile_id),
            "guideUrl": public_doc_url(display["guide"]),
            "experimentUrl": public_doc_url(display["experiment"]) if display.get("experiment") else None,
        })
    return public


def decision_public_profiles(manifest: dict, metadata: dict) -> list[dict]:
    if not manifest.get("version") or not isinstance(manifest.get("profiles"), list):
        raise SystemExit("Unsupported decision manifest schema")
    profiles = manifest["profiles"]
    checked_stack_metadata(profiles, metadata, "decisions")
    public = []
    for profile in profiles:
        profile_id = profile["id"]
        if not all(isinstance(profile.get(key), str) and profile[key] for key in ("name", "engine", "type", "entrypoint")):
            raise SystemExit(f"Incomplete decision profile {profile_id}")
        if profile["type"] not in {"local", "api"}:
            raise SystemExit(f"Unknown decision type for {profile_id}")
        entrypoint = Path(profile["entrypoint"])
        if entrypoint.is_absolute() or ".." in entrypoint.parts or not (DECISION_MANIFEST.parent / entrypoint).is_file():
            raise SystemExit(f"Invalid decision entrypoint for {profile_id}")
        display = metadata[profile_id]
        kind = "hosted" if profile["type"] == "api" else "local"
        public.append({
            "id": profile_id,
            "name": profile["name"],
            "engine": profile["engine"],
            "backbone": profile.get("backbone", ""),
            "kind": kind,
            "device": profile.get("device", "unspecified"),
            "tier": display.get("tier", "remote" if kind == "hosted" else "burst"),
            "role": display.get("role", ""),
            "vramGb": profile.get("vram_gb"),
            "ramGb": 0.0 if kind == "hosted" else (1.0 if profile.get("vram_gb", 0) else 1.5),
            "declaredLatencyMs": profile.get("expected_latency_ms"),
            "sourceUrl": profile_source_url(DECISION_MANIFEST, profile_id),
            "guideUrl": public_doc_url(display["guide"]),
            "experimentUrl": public_doc_url(display["experiment"]) if display.get("experiment") else None,
        })
    return public


def embedding_public_profiles(parsed: dict[str, dict], groups: dict[str, dict]) -> list[dict]:
    co_resident_ids = set()
    for group in groups.values():
        if group.get("exclusive") is False or group.get("swap") is False:
            co_resident_ids.update(group.get("members", []))
    public = []
    for model_id, source in parsed.items():
        cmd = source.get("cmd", "")
        if not cmd or source.get("unlisted", False) or "--embedding" not in cmd:
            continue
        is_coresident = model_id in co_resident_ids or "--n-gpu-layers 0" in cmd
        public.append({
            "id": model_id,
            "name": source.get("name", model_id),
            "description": source.get("description", ""),
            "engine": "llama.cpp-master",
            "tier": "co-resident" if is_coresident else "burst",
            "device": "cpu" if "--n-gpu-layers 0" in cmd else "cuda",
            "context": context_from_command(cmd),
            "vramGb": 0.0 if "--n-gpu-layers 0" in cmd else 1.0,
            "ramGb": 0.6,
            "recipeKind": "llama.cpp",
            "recipeLabel": "Portable llama.cpp command",
            "command": portable_command(source),
            "sourceUrl": SOURCE_BASE + "llama-swap.yaml",
        })
    return public


def main() -> None:
    config_text = CONFIG.read_text()
    parsed = parse_models(config_text)
    macros = parse_macros(config_text)
    groups = parse_groups(config_text)
    preloaded = parse_preload(config_text)

    meta = json.loads(META.read_text())
    engines_meta = json.loads(ENGINES_META.read_text()) if ENGINES_META.is_file() else {}
    community_runs = validate_community_runs(json.loads(COMMUNITY_RUNS.read_text()))
    stack_meta = json.loads(STACK_META.read_text())
    media = media_public_profiles(json.loads(MEDIA_MANIFEST.read_text()), stack_meta["media"])
    decision_manifest = json.loads(DECISION_MANIFEST.read_text())
    decisions = decision_public_profiles(decision_manifest, stack_meta["decisions"])
    embeddings = embedding_public_profiles(parsed, groups)

    # Router declarations for native decisions and media belong in their own
    # lanes, never in the token-generation ranking.
    native_decision_ids = {
        profile["model"] for profile in decision_manifest["profiles"]
        if profile.get("engine") == "llamacpp"
    }
    native_media_ids = {profile["id"] for profile in media}
    public_models = []
    active_ids = {
        model_id
        for model_id, model in parsed.items()
        if model.get("cmd")
        and not model.get("unlisted", False)
        # Utility embedding profiles remain discoverable in the embeddings lane
        # and never enter the token-generation ranking.
        and "--embedding" not in model.get("cmd", "")
        and model_id not in native_decision_ids
        and model_id not in native_media_ids
        # llama-swap-managed media engines (reserved *_auto ids) are not LLMs.
        and not model_id.endswith("_auto")
    }
    metadata_ids = set(meta["models"])
    if active_ids != metadata_ids:
        missing = sorted(active_ids - metadata_ids)
        stale = sorted(metadata_ids - active_ids)
        raise SystemExit(
            f"Dashboard metadata mismatch; missing={missing or 'none'}, stale={stale or 'none'}"
        )

    # Declared profiles with no measured throughput stay out of the ranking and
    # are listed separately; a missing measurement is never filled in.
    unmeasured = []
    for model_id, display in meta["models"].items():
        if model_id not in parsed:
            raise SystemExit(f"Dashboard model missing from llama-swap.yaml: {model_id}")
        source = parsed[model_id]
        if source.get("unlisted"):
            raise SystemExit(f"Dashboard model is marked unlisted: {model_id}")
        context = context_from_command(source.get("cmd", ""))
        if context == "Custom" and display.get("testedContext"):
            context = display["testedContext"]
        recipe_kind, recipe_label, command_text, config_excerpt = portable_recipe(
            source, macros=macros, root=ROOT
        )
        aliases = source.get("aliases", [])
        is_preload = bool(display.get("preload")) or model_id in preloaded

        if display.get("tps") is None:
            unmeasured.append(
                {
                    "id": model_id,
                    "name": display.get("displayName", source.get("name", model_id)),
                    "description": display.get("summary", source.get("description", "")),
                    "role": display.get("role", ""),
                    "tier": display.get("tier", "burst"),
                    "engine": display.get("engine", recipe_kind),
                    "modalities": display.get("modalities", ["Language"]),
                    "vramGb": display.get("vramGb"),
                    "ramGb": display.get("ramGb"),
                    "diskGb": display.get("diskGb"),
                    "nvmeStream": display.get("nvmeStream"),
                    "context": context,
                    "quant": display.get("quant"),
                    "tags": display.get("tags", []),
                    "aliases": aliases,
                    "recipeKind": recipe_kind,
                    "recipeLabel": recipe_label,
                    "command": command_text,
                    "engineConfig": config_excerpt,
                    "sourceUrl": SOURCE_BASE + "llama-swap.yaml",
                }
            )
            continue

        public_models.append(
            {
                **display,
                "id": model_id,
                "name": display.get("displayName", source.get("name", model_id)),
                "description": display.get("summary", source.get("description", "")),
                "preload": is_preload,
                "aliases": aliases,
                "context": context,
                "evidence": profile_evidence(display),
                "recipeKind": recipe_kind,
                "recipeLabel": recipe_label,
                "command": command_text,
                "engineConfig": config_excerpt,
                "sourceUrl": SOURCE_BASE + "llama-swap.yaml",
                "benchmarkUrl": f"{SOURCE_BASE}{display['benchmark']}",
            }
        )

    public_models.sort(key=lambda model: model["tps"], reverse=True)
    payload = {
        "generatedAt": meta["updated"],
        "system": meta["system"],
        "methodology": meta["methodology"],
        "preloadedModels": preloaded,
        "spotlight": meta.get("spotlight", []),
        "models": public_models,
        "unmeasured": unmeasured,
        "embeddings": embeddings,
        "media": media,
        "decisions": decisions,
        "engines": engines_meta,
        "roofline": meta.get("roofline", {}),
        "colabBurst": meta.get("colabBurst", {}),
        "hardwareTuning": meta.get("hardwareTuning", {}),
        "benchmarks": meta["benchmarks"],
        "progression": meta.get("progression", []),
        "inklingProgression": meta.get("inklingProgression", []),
        # Deliberately separate from models: community hardware must never
        # enter the local RTX 4070 ranking or its sort order.
        "communityRuns": community_runs,
        "archived": meta["archived"],
    }
    OUTPUT.write_text(
        "// Generated by docs/generate_dashboard_data.py. Do not edit by hand.\n"
        f"window.L3MS_DASHBOARD = {json.dumps(payload, indent=2)};\n"
    )
    print(
        f"Generated {OUTPUT.relative_to(ROOT)} with {len(public_models)} measured, "
        f"{len(unmeasured)} unmeasured, {len(embeddings)} embedding, "
        f"{len(media)} media, and {len(decisions)} decision profiles"
    )


if __name__ == "__main__":
    main()
