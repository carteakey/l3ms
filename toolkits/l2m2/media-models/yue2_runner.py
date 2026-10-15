#!/usr/bin/env python3
"""Run YuE2 music generation with safe defaults and parameter validation."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUDGET_GIB = 12.0
DEFAULT_QUANTIZATION = "fp8"
DEFAULT_COT = "full"
DEFAULT_SEED = 42


def resolve_runtime_paths() -> dict[str, Path]:
    """Resolve repository root, python binary, model directories, and output folder."""
    yue_root_env = os.environ.get("L3MS_YUE_ROOT")
    if yue_root_env:
        yue_root = Path(yue_root_env).resolve()
    else:
        repo_candidate = Path.home() / "repos" / "YuE"
        vendor_candidate = ROOT / "vendor" / "YuE"
        if repo_candidate.is_dir():
            yue_root = repo_candidate.resolve()
        elif vendor_candidate.is_dir():
            yue_root = vendor_candidate.resolve()
        else:
            yue_root = repo_candidate

    python_env = os.environ.get("L2M2_PYTHON") or os.environ.get("YUE_PYTHON")
    if python_env:
        python_bin = Path(python_env).resolve()
    else:
        venv_python = yue_root / ".venv" / "bin" / "python"
        if venv_python.is_file():
            python_bin = venv_python
        else:
            python_bin = Path(sys.executable)

    media_root = Path(os.environ.get("L3MS_MEDIA_ROOT", Path.home() / "models" / "media")).resolve()
    model_root_env = os.environ.get("L3MS_YUE_MODEL_DIR")
    if model_root_env:
        model_root = Path(model_root_env).resolve()
    else:
        model_root = media_root / "YuE2"

    output_dir = Path(os.environ.get("L3MS_MEDIA_OUTPUT_DIR", Path.home() / "media-output")).resolve()

    return {
        "yue_root": yue_root,
        "python_bin": python_bin,
        "model_root": model_root,
        "media_root": media_root,
        "output_dir": output_dir,
    }


def validate_settings(
    style: str | None,
    lyrics: str | None,
    cot: str,
    quantization: str,
    budget: float,
    cfg_scale: float | None = None,
    abc_file: str | Path | None = None,
) -> None:
    """Validate request parameters before execution."""
    if cot not in {"full", "melody", "off"}:
        raise ValueError(f"Invalid cot mode: {cot!r}. Allowed: full, melody, off")

    if quantization not in {"none", "fp8"}:
        raise ValueError(f"Invalid quantization: {quantization!r}. Allowed: none, fp8")

    if budget <= 0:
        raise ValueError(f"Memory budget must be positive, got: {budget}")

    if cfg_scale is not None and cfg_scale <= 0:
        raise ValueError(f"cfg_scale must be positive, got: {cfg_scale}")

    if abc_file is not None:
        abc_path = Path(abc_file)
        if not abc_path.is_file():
            raise FileNotFoundError(f"ABC file not found: {abc_path}")
        if cot == "off":
            raise ValueError("Supplying an ABC score requires cot='full' or 'melody'")

    if not style and not abc_file:
        raise ValueError("Must provide either --style/--prompt or an --abc-file")


def prepare_request(
    style: str | None,
    lyrics: str | None,
    cot: str = DEFAULT_COT,
    seed: int = DEFAULT_SEED,
    cfg_scale: float | None = None,
    req_id: str | None = None,
    abc_content: str | None = None,
) -> dict[str, object]:
    """Assemble a standard YuE2 song request dictionary."""
    request: dict[str, object] = {
        "cot": cot,
        "seed": seed,
    }
    if req_id:
        request["id"] = req_id
    if style:
        request["style"] = style
    if lyrics:
        request["lyrics"] = lyrics
    if cfg_scale is not None:
        request["cfg_scale"] = cfg_scale
    if abc_content is not None:
        request["abc"] = abc_content
    return request


def build_cli_args(
    command: str = "generate",
    model: str | None = None,
    vae: str | None = None,
    quantization: str = DEFAULT_QUANTIZATION,
    budget: float = DEFAULT_BUDGET_GIB,
    backend: str = "torch",
    offload_ar: bool = False,
    cot: str | None = None,
    style: str | None = None,
    lyrics: str | None = None,
    lyrics_file: Path | None = None,
    abc_file: Path | None = None,
    request_file: Path | None = None,
    output_path: Path | None = None,
    seed: int | None = None,
    cfg_scale: float | None = None,
    stage: str | None = None,
    quiet: bool = False,
) -> list[str]:
    """Build the argument list for the underlying yue2 CLI command."""
    args = [command]

    if model:
        args.extend(["--model", str(model)])
    if vae:
        args.extend(["--vae", str(vae)])

    args.extend(["--quantization", quantization])
    args.extend(["--budget", str(budget)])
    args.extend(["--backend", backend])

    if offload_ar:
        args.append("--offload-ar")

    if command == "doctor":
        if output_path:
            args.extend(["--output", str(output_path)])
        return args

    if cot:
        args.extend(["--cot", cot])
    if request_file:
        args.extend(["--request", str(request_file)])
    if style:
        args.extend(["--style", style])
    if lyrics:
        args.extend(["--lyrics", lyrics])
    if lyrics_file:
        args.extend(["--lyrics-file", str(lyrics_file)])
    if abc_file:
        args.extend(["--abc-file", str(abc_file)])
    if seed is not None:
        args.extend(["--seed", str(seed)])
    if cfg_scale is not None:
        args.extend(["--cfg-scale", str(cfg_scale)])
    if stage:
        args.extend(["--stage", stage])
    if output_path:
        args.extend(["--output", str(output_path)])
    if quiet:
        args.append("--quiet")

    return args


def run_yue2(
    cli_args: list[str],
    python_bin: Path,
    yue_root: Path,
) -> int:
    """Execute yue2 command via python -m yue2.cli."""
    cmd = [str(python_bin), "-m", "yue2.cli"] + cli_args
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{yue_root / 'src'}:{env.get('PYTHONPATH', '')}"
    env["YUE2_KIT"] = str(yue_root)

    print(f"-> Running YuE2: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=str(yue_root), env=env)
    return result.returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doctor", "--check", action="store_true", help="Run environment readiness check")
    parser.add_argument("--prompt", "--text", "--style", dest="style", help="Style description and musical prompt")
    parser.add_argument("--prompt-file", dest="prompt_file", type=Path, help="Read style prompt from text file")
    parser.add_argument("--tags", dest="tags", help="Style tags (alias for style)")
    parser.add_argument("--lyrics", dest="lyrics", help="Song lyrics to arrange")
    parser.add_argument("--lyrics-file", dest="lyrics_file", type=Path, help="Read song lyrics from file")
    parser.add_argument("--abc-file", dest="abc_file", type=Path, help="Supplied ABC score file for exact-score conditioning")
    parser.add_argument("--request", dest="request", type=Path, help="Path to request JSON file")
    parser.add_argument("--cot", choices=("full", "melody", "off"), default=DEFAULT_COT, help="Chain-of-thought planning mode")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed")
    parser.add_argument("--cfg-scale", type=float, help="Classifier-free guidance scale")
    parser.add_argument("--stage", choices=("plan", "audio"), default="audio", help="Stop after planning score or synthesize audio")
    parser.add_argument("--quantization", choices=("none", "fp8"), default=DEFAULT_QUANTIZATION, help="Quantization mode (fp8 recommended on 12GB)")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_GIB, help="VRAM budget in GiB (default: 12)")
    parser.add_argument("--offload-ar", action="store_true", help="Offload AR model to CPU when NAR runs")
    parser.add_argument("--backend", choices=("torch", "torch-eager", "vllm"), default="torch", help="Execution backend")
    parser.add_argument("--model", help="YuE2 model path or HF repo ID")
    parser.add_argument("--vae", help="YuE2 VAE path or HF repo ID")
    parser.add_argument("--output", "--output-dir", dest="output", help="Output directory for generated song artifacts")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_runtime_paths()

    if args.doctor:
        cli_args = build_cli_args(
            command="doctor",
            model=args.model,
            vae=args.vae,
            quantization=args.quantization,
            budget=args.budget,
            backend=args.backend,
        )
        return run_yue2(cli_args, paths["python_bin"], paths["yue_root"])

    style = args.style or args.tags
    if args.prompt_file:
        if not args.prompt_file.is_file():
            print(f"Error: Prompt file not found: {args.prompt_file}", file=sys.stderr)
            return 1
        style = args.prompt_file.read_text(encoding="utf-8").strip()

    lyrics = args.lyrics
    if args.lyrics_file:
        if not args.lyrics_file.is_file():
            print(f"Error: Lyrics file not found: {args.lyrics_file}", file=sys.stderr)
            return 1
        lyrics = args.lyrics_file.read_text(encoding="utf-8").strip()

    if not args.request:
        if not lyrics and not args.abc_file:
            print("Error: Either --lyrics/--lyrics-file or --request or --abc-file must be provided", file=sys.stderr)
            return 1
        try:
            validate_settings(
                style=style,
                lyrics=lyrics,
                cot=args.cot,
                quantization=args.quantization,
                budget=args.budget,
                cfg_scale=args.cfg_scale,
                abc_file=args.abc_file,
            )
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    output = args.output
    if not output:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = paths["output_dir"] / f"yue2-{stamp}"
    else:
        output = Path(output).resolve()

    model = args.model
    vae = args.vae
    if not model and (paths["model_root"] / "YuE2-3B").is_dir():
        model = str(paths["model_root"] / "YuE2-3B")
    if not vae and (paths["model_root"] / "YuE2-Vae").is_dir():
        vae = str(paths["model_root"] / "YuE2-Vae")

    cli_args = build_cli_args(
        command="generate",
        model=model,
        vae=vae,
        quantization=args.quantization,
        budget=args.budget,
        backend=args.backend,
        offload_ar=args.offload_ar,
        cot=args.cot,
        style=style,
        lyrics=lyrics,
        abc_file=args.abc_file,
        request_file=args.request,
        output_path=output,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        stage=args.stage,
        quiet=args.quiet,
    )

    return run_yue2(cli_args, paths["python_bin"], paths["yue_root"])


if __name__ == "__main__":
    sys.exit(main())
