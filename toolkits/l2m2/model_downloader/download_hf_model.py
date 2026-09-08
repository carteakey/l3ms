#!/usr/bin/env python3
"""
HuggingFace Model Downloader

Downloads models from HuggingFace Hub with configurable patterns and concurrency.
"""

import argparse
import json
import logging
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env from repo root (two levels up from this file)
_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

try:
    from huggingface_hub import snapshot_download
except ImportError as error:
    snapshot_download = None
    HUB_IMPORT_ERROR = error
else:
    HUB_IMPORT_ERROR = None


ESTIMATE_SCHEMA_VERSION = 1
MAX_ESTIMATE_MODELS = 256
MAX_ESTIMATE_FILES = 10_000
MAX_ESTIMATE_ERROR_CHARS = 1_000
MAX_U64 = (1 << 64) - 1


class EstimateError(ValueError):
    """An estimator input or result violated the machine-output contract."""


# Detect active download backend and report it.
# hf_xet is the current fast path (chunk-based deduplication via Xet storage).
# hf_transfer is deprecated and must NOT be force-enabled alongside hf_xet.
def _report_download_backend() -> None:
    try:
        import importlib.metadata as _meta

        xet_ver = _meta.version("hf_xet")
        print(f"  download backend: hf_xet {xet_ver} (Xet storage, chunk dedup)")
        return
    except (_meta.PackageNotFoundError, ValueError):
        pass
    # Warn if someone still has HF_HUB_ENABLE_HF_TRANSFER set in environment
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        print(
            "  download backend: hf_transfer (DEPRECATED — unset HF_HUB_ENABLE_HF_TRANSFER and install hf_xet)"
        )
        return
    print(
        "  download backend: huggingface_hub default (install hf_xet for faster downloads)"
    )


def has_local_files(local_dir: str) -> bool:
    """Return True if the local directory contains at least one non-hidden file."""
    local = Path(local_dir)
    return local.exists() and any(
        f for f in local.rglob("*") if f.is_file() and not f.name.startswith(".")
    )


def download_model(
    repo_id: str,
    local_dir: str,
    allow_patterns: List[str] = None,
    ignore_patterns: List[str] = None,
    revision: str = None,
    force_download: bool = False,
    max_workers: Optional[int] = None,
    update_only: bool = False,
) -> str:
    if update_only:
        if not has_local_files(local_dir):
            print(
                f"  No local files found — skipping (run without --update to download fresh)"
            )
            return local_dir
        print(f"  ↻ Syncing {repo_id} — pulling only changed/new files")

    print(f"Downloading {repo_id} to {local_dir}")
    if allow_patterns:
        print(f"  Including patterns: {allow_patterns}")
    if ignore_patterns:
        print(f"  Excluding patterns: {ignore_patterns}")
    if max_workers is not None:
        print(f"  Max download workers: {max_workers}")

    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            revision=revision or "main",
            force_download=force_download,
            max_workers=max_workers,
        )
        print(f"✓ Successfully downloaded to: {downloaded_path}")
        return downloaded_path
    except (OSError, RuntimeError, ValueError) as hub_err:
        local = Path(local_dir)
        if local.exists() and any(f for f in local.iterdir() if f.is_file()):
            print(f"  Hub unreachable, keeping existing files in {local_dir}")
            print(f"  (Reason: {hub_err})")
            return local_dir
        print(f"✗ Error downloading {repo_id}: {hub_err}")
        raise


def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        return {}


def resolve_local_dir(repo_id: str, base_models_dir: str, local_dir: str = None) -> str:
    if local_dir:
        return local_dir
    parts = repo_id.split("/")
    if len(parts) == 2:
        org, model = parts
        return os.path.join(base_models_dir, org, model)
    return os.path.join(base_models_dir, repo_id.replace("/", "_"))


def resolve_max_workers(
    model_max_workers: Optional[int],
    cli_max_workers: Optional[int],
    slow: bool,
) -> Optional[int]:
    """Resolve concurrency with explicit runtime controls taking precedence."""
    if cli_max_workers is not None:
        return cli_max_workers
    if model_max_workers is not None:
        return model_max_workers
    return 4 if slow else None


def load_config_for_estimate(config_path: str) -> Dict[str, Any]:
    """Load a config without writing human diagnostics to stdout."""
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError as error:
        raise EstimateError(f"config file not found: {config_path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EstimateError(f"could not read config {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise EstimateError("config root must be a JSON object")
    return config


def _checked_nonnegative_size(value: Any, repo_id: str, filename: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        label = str(filename or "<unknown>")[:200]
        raise EstimateError(
            f"dry-run returned invalid file_size for {repo_id}/{label}: "
            "expected a nonnegative integer"
        )
    if value > MAX_U64:
        raise EstimateError(f"dry-run file_size for {repo_id} exceeds uint64")
    return value


def _checked_add(left: int, right: int, label: str) -> int:
    if right > MAX_U64 - left:
        raise EstimateError(f"{label} exceeds uint64")
    return left + right


def estimate_model(
    repo_id: str,
    local_dir: str,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    revision: Optional[str] = None,
    force_download: bool = False,
    remaining_files: int = MAX_ESTIMATE_FILES,
) -> Dict[str, Any]:
    """Return a bounded aggregate of Hugging Face's filtered dry-run results."""
    if snapshot_download is None:
        raise EstimateError(
            'huggingface_hub is not installed; install it with: pip install -U "huggingface_hub"'
        )
    resolved_revision = revision or "main"
    try:
        results = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            revision=resolved_revision,
            force_download=force_download,
            dry_run=True,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise EstimateError(f"dry-run failed for {repo_id}: {error}") from error

    if isinstance(results, (str, bytes)):
        raise EstimateError(
            "dry-run returned a download path instead of file metadata; "
            "upgrade huggingface_hub"
        )
    try:
        iterator = iter(results)
    except TypeError as error:
        raise EstimateError(
            f"dry-run returned invalid metadata for {repo_id}: expected a file list"
        ) from error

    matched_files = 0
    total_bytes = 0
    download_bytes = 0
    cached_bytes = 0
    for info in iterator:
        matched_files += 1
        if matched_files > remaining_files:
            raise EstimateError(
                f"dry-run matched more than {MAX_ESTIMATE_FILES} files across all models"
            )

        filename = getattr(info, "filename", "<unknown>")
        size = _checked_nonnegative_size(
            getattr(info, "file_size", None), repo_id, filename
        )
        is_cached = getattr(info, "is_cached", None)
        will_download = getattr(info, "will_download", None)
        if not isinstance(is_cached, bool) or not isinstance(will_download, bool):
            raise EstimateError(
                f"dry-run returned invalid cache flags for {repo_id}/{str(filename)[:200]}"
            )

        total_bytes = _checked_add(total_bytes, size, "model total_bytes")
        if will_download:
            download_bytes = _checked_add(
                download_bytes, size, "model download_bytes"
            )
        elif is_cached:
            cached_bytes = _checked_add(cached_bytes, size, "model cached_bytes")

    return {
        "repo_id": repo_id,
        "revision": resolved_revision,
        "matched_files": matched_files,
        "total_bytes": total_bytes,
        "download_bytes": download_bytes,
        "cached_bytes": cached_bytes,
    }


def _validated_optional_string(value: Any, field: str, repo_id: str) -> Optional[str]:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise EstimateError(f"{field} for {repo_id} must be a string or null")
    return value


def _validated_patterns(value: Any, field: str, repo_id: str) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise EstimateError(
            f"{field} for {repo_id} must be a string, array of strings, or null"
        )
    return value


def _aggregate_estimates(models: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = {
        "models": len(models),
        "matched_files": 0,
        "total_bytes": 0,
        "download_bytes": 0,
        "cached_bytes": 0,
    }
    for model in models:
        totals["matched_files"] += model["matched_files"]
        if totals["matched_files"] > MAX_ESTIMATE_FILES:
            raise EstimateError(
                f"dry-run matched more than {MAX_ESTIMATE_FILES} files across all models"
            )
        for field in ("total_bytes", "download_bytes", "cached_bytes"):
            totals[field] = _checked_add(totals[field], model[field], f"total {field}")
    return {
        "schema_version": ESTIMATE_SCHEMA_VERSION,
        "models": models,
        "totals": totals,
    }


def build_estimate(args: argparse.Namespace, base_models_dir: str) -> Dict[str, Any]:
    """Build the machine estimate without printing or mutating local model files."""
    estimates: List[Dict[str, Any]] = []
    matched_files = 0

    if args.config:
        config = load_config_for_estimate(args.config)
        models = config.get("models", [])
        if not isinstance(models, list):
            raise EstimateError("config models must be an array")
        config_base = config.get("base_models_dir", base_models_dir)
        if not isinstance(config_base, str):
            raise EstimateError("config base_models_dir must be a string")

        eligible_models = []
        for model in models:
            if not isinstance(model, dict) or not model.get("enabled", True):
                continue
            repo_id = model.get("repo_id")
            if not isinstance(repo_id, str) or not repo_id.strip():
                continue
            eligible_models.append((model, repo_id.strip()))

        if len(eligible_models) > MAX_ESTIMATE_MODELS:
            raise EstimateError(
                f"config contains {len(eligible_models)} eligible models; "
                f"limit is {MAX_ESTIMATE_MODELS}"
            )

        for model, repo_id in eligible_models:
            local_dir = _validated_optional_string(
                model.get("local_dir"), "local_dir", repo_id
            )
            revision = _validated_optional_string(
                model.get("revision"), "revision", repo_id
            )
            estimate = estimate_model(
                repo_id=repo_id,
                local_dir=resolve_local_dir(repo_id, config_base, local_dir),
                allow_patterns=_validated_patterns(
                    model.get("allow_patterns"), "allow_patterns", repo_id
                ),
                ignore_patterns=_validated_patterns(
                    model.get("ignore_patterns"), "ignore_patterns", repo_id
                ),
                revision=revision,
                force_download=bool(model.get("force_download", False)),
                remaining_files=MAX_ESTIMATE_FILES - matched_files,
            )
            matched_files += estimate["matched_files"]
            estimates.append(estimate)
    elif args.repo_id:
        estimate = estimate_model(
            repo_id=args.repo_id,
            local_dir=resolve_local_dir(
                args.repo_id, base_models_dir, args.local_dir
            ),
            allow_patterns=args.allow_patterns,
            ignore_patterns=args.ignore_patterns,
            revision=args.revision,
            force_download=args.force_download,
        )
        estimates.append(estimate)
    else:
        raise EstimateError("either --repo-id or --config is required")

    return _aggregate_estimates(estimates)


def _bounded_error(error: Exception) -> str:
    message = " ".join(str(error).splitlines()).strip() or error.__class__.__name__
    return message[:MAX_ESTIMATE_ERROR_CHARS]


def emit_estimate_json(args: argparse.Namespace, base_models_dir: str) -> int:
    """Emit exactly one JSON object on success and only stderr on failure."""
    try:
        previous_logging_disable = logging.root.manager.disable
        try:
            logging.disable(max(previous_logging_disable, logging.CRITICAL))
            with open(os.devnull, "w", encoding="utf-8") as sink:
                with redirect_stdout(sink), redirect_stderr(sink):
                    estimate = build_estimate(args, base_models_dir)
                    output = json.dumps(estimate, separators=(",", ":"))
        finally:
            logging.disable(previous_logging_disable)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"estimate failed: {_bounded_error(error)}", file=sys.stderr)
        return 1

    sys.stdout.write(output + "\n")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download models from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./download_hf_model.py --repo-id Qwen/Qwen3-32B-GGUF --allow-patterns "*Q6_K*"
  ./download_hf_model.py --repo-id Qwen/Qwen3-32B-GGUF --max-workers 2
  ./download_hf_model.py --config models_config.json
  ./download_hf_model.py --config models_config.json --slow
  ./download_hf_model.py --config models_config.json --update
        """,
    )

    parser.add_argument("--config", "-c", help="Path to JSON configuration file")
    parser.add_argument("--repo-id", "-r", help="Repository ID to download")
    parser.add_argument("--local-dir", "-d", help="Local directory to save the model")
    parser.add_argument(
        "--allow-patterns", "-a", nargs="+", help="File patterns to include"
    )
    parser.add_argument(
        "--ignore-patterns", "-i", nargs="+", help="File patterns to exclude"
    )
    parser.add_argument("--revision", help="Specific revision/branch to download")
    parser.add_argument(
        "--force-download", action="store_true", help="Re-download existing files"
    )
    parser.add_argument(
        "--max-workers", type=int, default=None, help="Max concurrent download workers"
    )
    parser.add_argument(
        "--slow", action="store_true", help="Slow preset: max_workers=4"
    )
    parser.add_argument(
        "--update",
        "-u",
        action="store_true",
        help="Sync updates for models already on disk; skip models with no local files",
    )
    parser.add_argument("--base-models-dir", help="Base directory for all models")
    parser.add_argument(
        "--estimate-json",
        action="store_true",
        help="Emit one machine-readable dry-run size estimate and download nothing",
    )

    args = parser.parse_args(argv)
    effective_max_workers = resolve_max_workers(None, args.max_workers, args.slow)
    base_models_dir = args.base_models_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models"
    )

    if args.estimate_json:
        return emit_estimate_json(args, base_models_dir)

    if HUB_IMPORT_ERROR is not None:
        print("Error: huggingface_hub is not installed.")
        print('Please install it with: pip install -U "huggingface_hub"')
        return 1

    _report_download_backend()

    if args.config:
        config = load_config(args.config)
        if not config:
            return 0

        base_models_dir = config.get("base_models_dir", base_models_dir)
        models = config.get("models", [])
        enabled_models = [m for m in models if m.get("enabled", True)]
        disabled_count = len(models) - len(enabled_models)

        print(
            f"Found {len(models)} models in configuration ({len(enabled_models)} enabled, {disabled_count} disabled)"
        )

        if not enabled_models:
            print("No enabled models to download")
            return 0

        for i, m in enumerate(enabled_models, 1):
            repo_id = m.get("repo_id")
            if not repo_id:
                print(f"Skipping model {i}: no repo_id specified")
                continue

            print(f"\n[{i}/{len(enabled_models)}] Processing {repo_id}")
            if m.get("description"):
                print(f"  Description: {m['description']}")

            local_dir = resolve_local_dir(repo_id, base_models_dir, m.get("local_dir"))

            try:
                download_model(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    allow_patterns=m.get("allow_patterns"),
                    ignore_patterns=m.get("ignore_patterns"),
                    revision=m.get("revision"),
                    force_download=m.get("force_download", False),
                    max_workers=resolve_max_workers(
                        m.get("max_workers"), args.max_workers, args.slow
                    ),
                    update_only=args.update,
                )
            except (OSError, RuntimeError, ValueError) as e:
                print(f"Failed to download {repo_id}: {e}")
                continue

    elif args.repo_id:
        local_dir = resolve_local_dir(args.repo_id, base_models_dir, args.local_dir)
        download_model(
            repo_id=args.repo_id,
            local_dir=local_dir,
            allow_patterns=args.allow_patterns,
            ignore_patterns=args.ignore_patterns,
            revision=args.revision,
            force_download=args.force_download,
            max_workers=effective_max_workers,
            update_only=args.update,
        )

    else:
        parser.print_help()
        print("\nError: Either --repo-id or --config must be specified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
