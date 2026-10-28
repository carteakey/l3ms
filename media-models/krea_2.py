#!/usr/bin/env python3
"""Generate Krea 2 text-to-image through a local ComfyUI API."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = (
    "diffusion_models/krea2_raw_int8_convrot.safetensors",
    "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
    "vae/qwen_image_vae.safetensors",
    "loras/krea2_turbo_distill_r128.safetensors",
)
REQUIRED_NODES = (
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "LoraLoaderModelOnly",
    "CLIPTextEncode",
    "ConditioningZeroOut",
    "EmptyLatentImage",
    "KSampler",
    "VAEDecode",
    "SaveImage",
)
MAX_PIXELS = 1024 * 1536  # 1,572,864 pixels maximum supported resolution


def build_workflow(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    cfg: float,
    lora_strength: float,
    lora_name: str,
    filename_prefix: str,
    refusal_bypass: bool = False,
    refusal_bypass_lora: str = "Krea2_TextFusion_Refusal_Reduction.safetensors",
    refusal_bypass_strength: float = 1.0,
    extra_loras: tuple[tuple[str, float], ...] = (),
) -> dict[str, dict[str, object]]:
    """Return the API-format graph for Krea 2 Raw int8 with Turbo distill LoRA."""
    workflow: dict[str, dict[str, object]] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "krea2_raw_int8_convrot.safetensors",
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_4b_fp8_scaled.safetensors",
                "type": "krea2",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"},
        },
    }

    model_source: list[object] = ["1", 0]
    if lora_name and lora_name.lower() != "none" and lora_strength > 0:
        workflow["4"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": lora_name,
                "strength_model": lora_strength,
            },
        }
        model_source = ["4", 0]

    if refusal_bypass and refusal_bypass_strength > 0:
        workflow["11"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_source,
                "lora_name": refusal_bypass_lora,
                "strength_model": refusal_bypass_strength,
            },
        }
        model_source = ["11", 0]

    for index, (extra_name, extra_strength) in enumerate(extra_loras):
        node_id = str(12 + index)
        workflow[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_source,
                "lora_name": extra_name,
                "strength_model": extra_strength,
            },
        }
        model_source = [node_id, 0]

    workflow["5"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["2", 0],
            "text": prompt,
        },
    }

    if negative_prompt and negative_prompt.strip():
        workflow["6"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": negative_prompt.strip(),
            },
        }
    else:
        workflow["6"] = {
            "class_type": "ConditioningZeroOut",
            "inputs": {
                "conditioning": ["5", 0],
            },
        }

    workflow["7"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }

    workflow["8"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_source,
            "positive": ["5", 0],
            "negative": ["6", 0],
            "latent_image": ["7", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        },
    }

    workflow["9"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["8", 0], "vae": ["3", 0]},
    }

    workflow["10"] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["9", 0],
            "filename_prefix": filename_prefix,
        },
    }

    return workflow


def parse_extra_loras(specs: list[str]) -> list[tuple[str, float]]:
    """Parse NAME or NAME:STRENGTH specs into (filename, strength) pairs."""
    parsed: list[tuple[str, float]] = []
    for spec in specs:
        name, separator, raw = spec.rpartition(":")
        if separator:
            try:
                strength = float(raw)
            except ValueError as error:
                raise ValueError(f"--extra-lora strength is not a number: {spec}") from error
        else:
            name, strength = raw, 1.0
        if not name.endswith(".safetensors") or name.startswith("/") or ".." in name:
            raise ValueError("--extra-lora must be a .safetensors file inside ComfyUI's loras/")
        if strength < 0 or strength > 10:
            raise ValueError("--extra-lora strength must be between 0 and 10")
        parsed.append((name, strength))
    return parsed


def validate_settings(
    width: int,
    height: int,
    steps: int,
    cfg: float,
    lora_strength: float,
    timeout: float,
) -> None:
    for label, value in (("width", width), ("height", height)):
        if value < 32 or value % 32:
            raise ValueError(f"{label} must be a multiple of 32 (got {value})")
        if value > 1536:
            raise ValueError(f"{label} cannot exceed 1536 (got {value})")
    pixels = width * height
    if pixels > MAX_PIXELS:
        raise ValueError(
            f"Resolution ({width}x{height} = {pixels} px) exceeds maximum supported "
            f"1024x1536 ({MAX_PIXELS} px)"
        )
    if steps < 1 or steps > 100:
        raise ValueError("steps must be between 1 and 100")
    if cfg <= 0 or cfg > 20:
        raise ValueError("CFG must be greater than 0 and at most 20")
    if lora_strength < 0 or lora_strength > 5:
        raise ValueError("lora-strength must be between 0 and 5")
    if timeout <= 0:
        raise ValueError("timeout must be positive")


def api_url() -> str:
    # Direct ComfyUI only. This lane has no router fallback and no Authorization header, so
    # reaching ComfyUI through llama-swap needs L2M2_COMFY_URL set to the routed URL. The shared
    # rule (used by qwen_image_21.py and minimax_h3_video.py) is in comfy_endpoint.py.
    return os.environ.get("L2M2_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")


def request_bytes(path: str, payload: dict[str, object] | None = None, timeout: float = 30) -> bytes:
    url = api_url() + path
    headers: dict[str, str] = {}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except HTTPError as error:
        detail = error.read(1000).decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(
            f"Cannot reach ComfyUI at {api_url()}. Start it with "
            "maintenance/comfyui-service.sh start-gpu after reserving the GPU."
        ) from error


def request_json(path: str, payload: dict[str, object] | None = None, timeout: float = 30) -> object:
    return json.loads(request_bytes(path, payload, timeout).decode("utf-8"))


def check_models() -> None:
    model_dir = ROOT / "vendor" / "ComfyUI" / "models"
    missing = [str(model_dir / relative) for relative in MODEL_FILES if not (model_dir / relative).is_file()]
    if missing:
        raise RuntimeError(
            "Missing Krea 2 model files; run maintenance/setup-krea-2-comfyui.sh:\n"
            + "\n".join(missing)
        )


def check_api() -> None:
    request_json("/system_stats")
    info = request_json("/object_info")
    if not isinstance(info, dict):
        raise RuntimeError("ComfyUI returned an invalid /object_info response")
    missing = [name for name in REQUIRED_NODES if name not in info]
    if missing:
        raise RuntimeError(
            "ComfyUI is missing required Krea 2 workflow nodes: "
            + ", ".join(missing)
            + ". Run maintenance/setup-comfyui.sh."
        )


def check_gpu_exclusive() -> None:
    if sys.platform.startswith("linux"):
        systemctl = shutil.which("systemctl")
        if systemctl:
            result = subprocess.run(
                [systemctl, "--user", "is-active", "--quiet", "llama-swap.service"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                raise RuntimeError(
                    "llama-swap.service is active. Stop it before Krea 2 generation; "
                    "the L2M2 ComfyUI profile will not interrupt it."
                )
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        # ComfyUI itself holds the checkpoint resident between jobs, and it is the workload
        # we are about to submit to, so its own PID must not count as a foreign workload.
        comfy_pids: set[int] = set()
        systemctl = shutil.which("systemctl")
        if systemctl:
            res = subprocess.run(
                [systemctl, "--user", "show", "-p", "MainPID", "--value", "l2m2-comfyui.service"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip().isdigit():
                pid = int(res.stdout.strip())
                if pid > 0:
                    comfy_pids.add(pid)

        result = subprocess.run(
            [nvidia_smi, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.splitlines():
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    pid, mem = int(parts[0]), int(parts[1])
                    if pid in comfy_pids:
                        continue
                    if mem >= 1500:
                        raise RuntimeError(
                            f"GPU is busy (PID {pid} using {mem} MiB). Wait for other workloads to finish."
                        )
        else:
            result_total = subprocess.run(
                [nvidia_smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result_total.returncode == 0:
                try:
                    used_mib = int(result_total.stdout.splitlines()[0].strip())
                    if used_mib >= 1500 and not comfy_pids:
                        raise RuntimeError(
                            f"GPU is busy ({used_mib} MiB used). Wait for other workloads to finish."
                        )
                except (IndexError, ValueError):
                    pass


def wait_for_image(prompt_id: str, timeout: float) -> dict[str, str]:
    started = time.monotonic()
    last_notice = started
    while time.monotonic() - started < timeout:
        history = request_json("/history/" + quote(prompt_id), timeout=30)
        if isinstance(history, dict):
            record = history.get(prompt_id)
            if isinstance(record, dict):
                status = record.get("status", {})
                status_text = status.get("status_str", "") if isinstance(status, dict) else ""
                if status_text == "error":
                    raise RuntimeError("ComfyUI reports that this generation failed; inspect its logs.")
                outputs = record.get("outputs", {})
                if isinstance(outputs, dict):
                    saved = outputs.get("10", {})
                    images = saved.get("images", []) if isinstance(saved, dict) else []
                    if images:
                        return images[0]
                if isinstance(status, dict) and status.get("completed"):
                    raise RuntimeError("ComfyUI completed the job but returned no saved image.")
        if time.monotonic() - last_notice >= 30:
            print("Krea 2 generation is still running in ComfyUI...")
            last_notice = time.monotonic()
        time.sleep(2)
    raise RuntimeError(
        f"Generation exceeded the {timeout:g}s timeout; it may still be running in ComfyUI."
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Generate Krea 2 images through local ComfyUI.")
    result.add_argument("--prompt", help="Text prompt")
    result.add_argument("--prompt-file", type=Path, help="Read the prompt from a UTF-8 file")
    result.add_argument("--negative-prompt", default="", help="Optional negative prompt")
    result.add_argument("--output", type=Path, help="Output PNG path")
    result.add_argument("--width", type=int, default=1024, help="Image width (multiple of 32, max 1536)")
    result.add_argument("--height", type=int, default=1024, help="Image height (multiple of 32, max 1536)")
    result.add_argument("--steps", type=int, default=12, help="Sampling steps (default: 12)")
    result.add_argument("--seed", type=int, default=42, help="RNG seed")
    result.add_argument("--cfg", type=float, default=1.5, help="Classifier-free guidance scale (default: 1.5)")
    result.add_argument(
        "--lora-strength",
        type=float,
        default=0.60,
        help="Turbo distill LoRA strength (default: 0.60)",
    )
    result.add_argument(
        "--lora-name",
        default="krea2_turbo_distill_r128.safetensors",
        help="Turbo LoRA filename in ComfyUI loras/ (set 'none' to disable)",
    )
    result.add_argument(
        "--refusal-bypass",
        "--bypass",
        action="store_true",
        help="Enable Krea2 TextFusion refusal-reduction bypass LoRA",
    )
    result.add_argument(
        "--refusal-bypass-lora",
        default="Krea2_TextFusion_Refusal_Reduction.safetensors",
        help="Refusal reduction bypass LoRA filename in ComfyUI loras/",
    )
    result.add_argument(
        "--refusal-bypass-strength",
        type=float,
        default=1.0,
        help="Refusal reduction bypass LoRA strength (default: 1.0)",
    )
    result.add_argument(
        "--extra-lora",
        action="append",
        default=[],
        metavar="NAME[:STRENGTH]",
        help="Additional LoRA/LoKR in ComfyUI loras/ chained after the bypass; "
        "NAME:STRENGTH or bare NAME (repeatable, default strength 1.0)",
    )
    result.add_argument("--timeout", type=float, default=3600, help="Job timeout in seconds")
    result.add_argument("--check", action="store_true", help="Check model files and local ComfyUI nodes")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        check_models()
        if args.check:
            check_api()
            print("Krea 2 model files and ComfyUI nodes are ready.")
            return 0
        if args.prompt and args.prompt_file:
            raise ValueError("--prompt and --prompt-file are mutually exclusive")
        prompt = args.prompt
        if args.prompt_file:
            prompt = args.prompt_file.read_text(encoding="utf-8")
        if not prompt or not prompt.strip():
            raise ValueError("Krea 2 requires --prompt or --prompt-file")
        validate_settings(args.width, args.height, args.steps, args.cfg, args.lora_strength, args.timeout)
        extra_loras = tuple(parse_extra_loras(args.extra_lora))
        lora_dir = ROOT / "vendor" / "ComfyUI" / "models" / "loras"
        for extra_name, _ in extra_loras:
            if not (lora_dir / extra_name).is_file():
                raise RuntimeError(f"Missing extra LoRA: {lora_dir / extra_name}")
        check_gpu_exclusive()
        check_api()

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename_prefix = "l2m2-krea-2-" + timestamp
        workflow = build_workflow(
            prompt,
            args.negative_prompt,
            args.width,
            args.height,
            args.steps,
            args.seed,
            args.cfg,
            args.lora_strength,
            args.lora_name,
            filename_prefix,
            refusal_bypass=args.refusal_bypass,
            refusal_bypass_lora=args.refusal_bypass_lora,
            refusal_bypass_strength=args.refusal_bypass_strength,
            extra_loras=extra_loras,
        )
        response = request_json("/prompt", {"prompt": workflow}, timeout=60)
        if not isinstance(response, dict) or not response.get("prompt_id"):
            raise RuntimeError(f"ComfyUI did not accept the workflow: {response}")
        prompt_id = str(response["prompt_id"])
        extra_note = ""
        if extra_loras:
            pairs = ", ".join(f"{name}@{strength:g}" for name, strength in extra_loras)
            extra_note = f", extra LoRA {pairs}"
        print(
            f"Generating {args.width}x{args.height}, {args.steps} steps, CFG {args.cfg}, "
            f"LoRA strength {args.lora_strength}, seed {args.seed}{extra_note}; "
            f"ComfyUI job {prompt_id}."
        )
        image = wait_for_image(prompt_id, args.timeout)
        output_dir = Path(os.environ.get("L3MS_MEDIA_OUTPUT_DIR", str(Path.home() / "media-output")))
        destination = args.output or output_dir / ("krea-2-" + timestamp + ".png")
        destination = destination.expanduser()
        if not destination.is_absolute():
            destination = Path.cwd() / destination
        if destination.suffix.lower() != ".png":
            raise ValueError("Krea 2 output must use a .png path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"Refusing to overwrite existing output: {destination}")
        query = urlencode(
            {
                "filename": image["filename"],
                "subfolder": image.get("subfolder", ""),
                "type": image.get("type", "output"),
            }
        )
        destination.write_bytes(request_bytes("/view?" + query, timeout=120))
        print(f"Saved {destination}")
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Krea 2: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
