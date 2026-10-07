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
OUTPUT = ROOT / "docs" / "generated-models.js"

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
        "${strata_python}": '"$STRATA_PYTHON"',
        "${strata_server}": '"$STRATA_SERVER"',
        "${strata_iq3_config}": '"$STRATA_CONFIG"',
        "${strata_vision_config}": '"$STRATA_CONFIG"',
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

    lines = [line.strip() for line in command.splitlines() if line.strip()]
    shell_command = " \\\n  ".join(lines)
    preamble = [
        'MODEL_PATH="/path/to/model.gguf"',
        'CPU_RANGE="${CPU_RANGE:-0-11}"',
        'PORT="${PORT:-8080}"',
    ]
    if "$LLAMA_SERVER" in shell_command:
        preamble.append('LLAMA_SERVER="${LLAMA_SERVER:-./vendor/llama.cpp/build/bin/llama-server}"')
    if "$IK_LLAMA_SERVER" in shell_command:
        preamble.append('IK_LLAMA_SERVER="${IK_LLAMA_SERVER:-./vendor/ik_llama.cpp/build/bin/llama-server}"')
    if "$QWEN_MTP_SERVER" in shell_command:
        preamble.append('QWEN_MTP_SERVER="${QWEN_MTP_SERVER:-./vendor/llama.cpp/build/bin/llama-server}"')
    if "$SARVAM_SERVER" in shell_command:
        preamble.append('SARVAM_SERVER="${SARVAM_SERVER:-./vendor/llama.cpp-pr-test-20275/build/bin/llama-server}"')
    if "$STRATA_SERVER" in shell_command:
        preamble.append('STRATA_PYTHON="${STRATA_PYTHON:-./vendor/strata/.venv/bin/python}"')
        preamble.append('STRATA_SERVER="${STRATA_SERVER:-./vendor/strata/serve/server.py}"')
        preamble.append('STRATA_CONFIG="${STRATA_CONFIG:-./vendor/strata/strata-iq3_xxs.json}"')
    if "$DRAFT_MODEL_PATH" in shell_command:
        preamble.append('DRAFT_MODEL_PATH="/path/to/draft-model.gguf"')
    if "$MMPROJ_PATH" in shell_command:
        preamble.append('MMPROJ_PATH="/path/to/mmproj.gguf"')
    if "$CHAT_TEMPLATE_PATH" in shell_command:
        preamble.append('CHAT_TEMPLATE_PATH="./chat-template.jinja"')
    for env_value in model.get("env", []):
        preamble.append(f"export {env_value}")
    return "\n".join(preamble) + "\n\n" + shell_command


def main() -> None:
    parsed = parse_models(CONFIG.read_text())
    meta = json.loads(META.read_text())
    community_runs = validate_community_runs(json.loads(COMMUNITY_RUNS.read_text()))
    public_models = []
    active_ids = {
        model_id
        for model_id, model in parsed.items()
        if model.get("cmd")
        and not model.get("unlisted", False)
        # The public leaderboard measures token generation. Utility embedding
        # profiles remain discoverable through llama-swap but do not have TPS.
        and "--embedding" not in model.get("cmd", "")
    }
    metadata_ids = set(meta["models"])
    if active_ids != metadata_ids:
        missing = sorted(active_ids - metadata_ids)
        stale = sorted(metadata_ids - active_ids)
        raise SystemExit(
            f"Dashboard metadata mismatch; missing={missing or 'none'}, stale={stale or 'none'}"
        )

    for model_id, display in meta["models"].items():
        if model_id not in parsed:
            raise SystemExit(f"Dashboard model missing from llama-swap.yaml: {model_id}")
        source = parsed[model_id]
        if source.get("unlisted"):
            raise SystemExit(f"Dashboard model is marked unlisted: {model_id}")
        context = context_from_command(source.get("cmd", ""))
        if context == "Custom" and display.get("testedContext"):
            context = display["testedContext"]
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
    print(f"Generated {OUTPUT.relative_to(ROOT)} with {len(public_models)} served profiles")


if __name__ == "__main__":
    main()
