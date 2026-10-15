#!/usr/bin/env python3
"""Generate the public dashboard payload from llama-swap configuration."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "llama-swap.yaml"
META = ROOT / "docs" / "dashboard-meta.json"
COMMUNITY_RUNS = ROOT / "docs" / "community-runs.json"
MEDIA_MANIFEST = ROOT / "toolkits" / "l2m2" / "media-runtimes.json"
DECISION_MANIFEST = ROOT / "toolkits" / "l3ds" / "decision-runtimes.json"
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


def parse_scalar(value: str):
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith('"'):
        return json.loads(value)
    return value


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
        if model_match:
            current_id = model_match.group(1)
            models[current_id] = {"id": current_id, "env": []}
            index += 1
            continue

        if current_id:
            field_match = re.match(r"^    ([a-zA-Z_]+):\s*(.*)$", line)
            if field_match:
                field, raw_value = field_match.groups()
                if field == "cmd" and raw_value == "|":
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
                        env_match = re.match(r'^      - "(.*)"\s*$', lines[index])
                        if not env_match:
                            break
                        models[current_id]["env"].append(env_match.group(1))
                        index += 1
                    continue
                if raw_value:
                    models[current_id][field] = parse_scalar(raw_value)
        index += 1

    return models


def context_from_command(command: str) -> str:
    match = re.search(r"--ctx-size\s+(\d+)", command)
    if not match:
        return "Custom"
    size = int(match.group(1))
    return f"{round(size / 1024)}k" if size >= 1024 else str(size)


def benchmark_command(path_value: str) -> str | None:
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
            "benchmarkCommand", benchmark_command(display["benchmark"])
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


def portable_command(model: dict) -> str:
    command = model.get("cmd", "")
    command = re.sub(r"(?m)^\s*#.*\n?", "", command).strip()
    replacements = {
        "${cpu_range}": '"$CPU_RANGE"',
        "${llama_server}": '"$LLAMA_SERVER"',
        "${ik_server}": '"$IK_LLAMA_SERVER"',
        "${qwen_mtp_server}": '"$QWEN_MTP_SERVER"',
        "${sarvam_server}": '"$SARVAM_SERVER"',
        "${PORT}": '"$PORT"',
        "${chat_template}": '"$CHAT_TEMPLATE_PATH"',
    }
    for source, target in replacements.items():
        command = command.replace(source, target)

    command = re.sub(r"(?m)(\s-m\s+)\S+", r'\1"$MODEL_PATH"', command, count=1)
    command = re.sub(r"(?m)(--model\s+)\S+", r'\1"$MODEL_PATH"', command, count=1)
    command = re.sub(r"(?m)(--spec-draft-model\s+)\S+", r'\1"$DRAFT_MODEL_PATH"', command)
    command = re.sub(r"(?m)(--mmproj\s+)\S+", r'\1"$MMPROJ_PATH"', command)
    command = command.replace("--host 0.0.0.0", "--host 127.0.0.1")

    # New engine macros must not leak into shell snippets as unset lowercase
    # router variables. Require operator-supplied paths, not inferred binaries.
    extra_variables = set()
    if "${env.L3MS_ROOT}" in command:
        command = command.replace("${env.L3MS_ROOT}", "${L3MS_ROOT}")
        extra_variables.add("L3MS_ROOT")

    def portable_macro(match):
        name = match.group(1)
        if name == "L3MS_ROOT":
            return match.group(0)
        variable = name.upper()
        extra_variables.add(variable)
        return f'"${variable}"'

    command = re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}", portable_macro, command)

    lines = [line.strip() for line in command.splitlines() if line.strip()]
    shell_command = " \\\n  ".join(lines)
    preamble = [
        'MODEL_PATH="/path/to/model.gguf"',
        'CPU_RANGE="${CPU_RANGE:-0-11}"',
        'PORT="${PORT:-8080}"',
    ]
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
        if not isinstance(display, dict) or set(display) - {"guide", "experiment", "modes"}:
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
            "status": profile["status"],
            "tasks": profile["tasks"],
            "modes": display.get("modes", []),
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
        public.append({
            "id": profile_id,
            "name": profile["name"],
            "engine": profile["engine"],
            "kind": "hosted" if profile["type"] == "api" else "local",
            "device": profile.get("device", "unspecified"),
            "declaredLatencyMs": profile.get("expected_latency_ms"),
            "sourceUrl": profile_source_url(DECISION_MANIFEST, profile_id),
            "guideUrl": public_doc_url(display["guide"]),
            "experimentUrl": public_doc_url(display["experiment"]) if display.get("experiment") else None,
        })
    return public


def main() -> None:
    parsed = parse_models(CONFIG.read_text())
    meta = json.loads(META.read_text())
    community_runs = validate_community_runs(json.loads(COMMUNITY_RUNS.read_text()))
    stack_meta = json.loads(STACK_META.read_text())
    media = media_public_profiles(json.loads(MEDIA_MANIFEST.read_text()), stack_meta["media"])
    decision_manifest = json.loads(DECISION_MANIFEST.read_text())
    decisions = decision_public_profiles(decision_manifest, stack_meta["decisions"])
    # Router declarations for native decisions belong in the Decisions view,
    # never in the token-generation ranking.
    native_decision_ids = {
        profile["model"] for profile in decision_manifest["profiles"]
        if profile.get("engine") == "llamacpp"
    }
    public_models = []
    active_ids = {
        model_id
        for model_id, model in parsed.items()
        if model.get("cmd")
        and not model.get("unlisted", False)
        # The public leaderboard measures token generation. Utility embedding
        # profiles remain discoverable through llama-swap but do not have TPS.
        and "--embedding" not in model.get("cmd", "")
        and model_id not in native_decision_ids
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
        if display.get("tps") is None:
            unmeasured.append(
                {
                    "id": model_id,
                    "name": display.get("displayName", source.get("name", model_id)),
                    "description": display.get("summary", source.get("description", "")),
                    "context": context,
                    "quant": display.get("quant"),
                    "tags": display.get("tags", []),
                    "command": portable_command(source),
                    "sourceUrl": "https://github.com/carteakey/l3ms/blob/main/llama-swap.yaml",
                }
            )
            continue
        public_models.append(
            {
                **display,
                "id": model_id,
                "name": display.get("displayName", source.get("name", model_id)),
                "description": display.get("summary", source.get("description", "")),
                "context": context,
                "evidence": profile_evidence(display),
                "command": portable_command(source),
                "sourceUrl": "https://github.com/carteakey/l3ms/blob/main/llama-swap.yaml",
                "benchmarkUrl": f"https://github.com/carteakey/l3ms/blob/main/{display['benchmark']}",
            }
        )

    public_models.sort(key=lambda model: model["tps"], reverse=True)
    payload = {
        "generatedAt": meta["updated"],
        "system": meta["system"],
        "methodology": meta["methodology"],
        "models": public_models,
        "unmeasured": unmeasured,
        "media": media,
        "decisions": decisions,
        "benchmarks": meta["benchmarks"],
        "progression": meta.get("progression", []),
        # Deliberately separate from models: community hardware must never
        # enter the local RTX 4070 ranking or its sort order.
        "communityRuns": community_runs,
        "archived": meta["archived"],
    }
    OUTPUT.write_text(
        "// Generated by docs/generate_dashboard_data.py. Do not edit by hand.\n"
        f"window.L3MS_DASHBOARD = {json.dumps(payload, indent=2)};\n"
    )
    print(f"Generated {OUTPUT.relative_to(ROOT)} with {len(public_models)} measured and {len(unmeasured)} unmeasured served profiles")


if __name__ == "__main__":
    main()
