#!/usr/bin/env python3
"""Generate Qwen Image 2.1 Turbo text-to-image through a local ComfyUI API."""

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
    "diffusion_models/qwen-image-2.1-turbo-int8-convrot.safetensors",
    "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
    "vae/qwen_image_2.1_vae_bf16.safetensors",
)
REQUIRED_NODES = (
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "TextEncodeQwenImage21",
    "EmptyLatentImage",
    "KSamplerSelect",
    "ManualSigmas",
    "SamplerCustom",
    "VAEDecode",
    "SaveImage",
)
DEFAULT_TURBO_SIGMAS = (
    "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0"
)


def build_workflow(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    cfg: float,
    sigmas: str,
    filename_prefix: str,
    image_filename: str | None = None,
    use_image_latent: bool = False,
) -> dict[str, dict[str, object]]:
    """Return the API-format graph for Qwen Image 2.1 Turbo with custom 8-step sigmas."""
    workflow: dict[str, dict[str, object]] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "qwen-image-2.1-turbo-int8-convrot.safetensors",
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_8b_int8_convrot.safetensors",
                "type": "qwen_image",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"},
        },
        "4": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "clip": ["2", 0],
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "resolution": max(width, height),
            },
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": "euler",
            },
        },
        "7": {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": sigmas,
            },
        },
        "8": {
            "class_type": "SamplerCustom",
            "inputs": {
                "model": ["1", 0],
                "add_noise": True,
                "noise_seed": seed,
                "cfg": cfg,
                "positive": ["4", 0],
                "negative": ["4", 1],
                "sampler": ["6", 0],
                "sigmas": ["7", 0],
                "latent_image": ["4", 2] if use_image_latent else ["5", 0],
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["3", 0]},
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["9", 0],
                "filename_prefix": filename_prefix,
            },
        },
    }
    if image_filename:
        workflow["11"] = {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        }
        workflow["4"]["inputs"]["vae"] = ["3", 0]
        workflow["4"]["inputs"]["images.image_1"] = ["11", 0]
    return workflow


def validate_settings(width: int, height: int, cfg: float, timeout: float) -> None:
    for label, value in (("width", width), ("height", height)):
        if value < 32 or value > 2048 or value % 32:
            raise ValueError(f"{label} must be a multiple of 32 between 32 and 2048")
    if cfg <= 0 or cfg > 20:
        raise ValueError("CFG must be greater than 0 and at most 20")
    if timeout <= 0:
        raise ValueError("timeout must be positive")


def api_url() -> str:
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
            "Missing Qwen Turbo model files; run maintenance/setup-qwen-image-2.1-turbo-comfyui.sh:\n"
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
            "ComfyUI is missing required Qwen Turbo workflow nodes: "
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
                    "llama-swap.service is active. Stop it before Qwen generation; "
                    "the L2M2 ComfyUI profile will not interrupt it."
                )

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
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
            # Fallback to total used memory if compute-apps is unsupported
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
                    for node_output in outputs.values():
                        if isinstance(node_output, dict):
                            images = node_output.get("images", [])
                            if images:
                                return images[0]
                if isinstance(status, dict) and status.get("completed"):
                    raise RuntimeError("ComfyUI completed the job but returned no saved image.")
        if time.monotonic() - last_notice >= 30:
            print("Qwen Turbo generation is still running in ComfyUI...")
            last_notice = time.monotonic()
        time.sleep(2)
    raise RuntimeError(
        f"Generation exceeded the {timeout:g}s timeout; it may still be running in ComfyUI."
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Generate Qwen Image 2.1 Turbo through local ComfyUI.")
    result.add_argument("--prompt", help="Text prompt")
    result.add_argument("--prompt-file", type=Path, help="Read the prompt from a UTF-8 file")
    result.add_argument("--negative-prompt", default="", help="Optional negative prompt")
    result.add_argument("--output", type=Path, help="Output PNG path")
    result.add_argument("--image", type=Path, help="Input reference image for image editing / conditioning")
    result.add_argument("--width", type=int, default=1024)
    result.add_argument("--height", type=int, default=1024)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--cfg", type=float, default=1.0)
    result.add_argument(
        "--sigmas",
        default=DEFAULT_TURBO_SIGMAS,
        help="Comma-separated noise schedule sigmas (default: official 8-step schedule)",
    )
    result.add_argument("--timeout", type=float, default=3600)
    result.add_argument("--check", action="store_true", help="Check model files and local ComfyUI nodes")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        check_models()
        if args.check:
            check_api()
            print("Qwen Image 2.1 Turbo model files and ComfyUI nodes are ready.")
            return 0
        if args.prompt and args.prompt_file:
            raise ValueError("--prompt and --prompt-file are mutually exclusive")
        prompt = args.prompt
        if args.prompt_file:
            prompt = args.prompt_file.read_text(encoding="utf-8")
        if not prompt or not prompt.strip():
            raise ValueError("Qwen Image 2.1 Turbo requires --prompt or --prompt-file")
        validate_settings(args.width, args.height, args.cfg, args.timeout)
        check_gpu_exclusive()
        check_api()

        image_filename = None
        use_image_latent = False
        if args.image:
            image_path = args.image.expanduser()
            if not image_path.is_file():
                raise ValueError(f"Input image not found: {image_path}")
            comfy_input = ROOT / "vendor" / "ComfyUI" / "input"
            comfy_input.mkdir(parents=True, exist_ok=True)
            target_copy = comfy_input / image_path.name
            if not target_copy.exists():
                shutil.copy2(image_path, target_copy)
            image_filename = image_path.name
            use_image_latent = (args.width == 1024 and args.height == 1024)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename_prefix = "l2m2-qwen-image-2.1-turbo-" + timestamp
        workflow = build_workflow(
            prompt,
            args.negative_prompt,
            args.width,
            args.height,
            args.seed,
            args.cfg,
            args.sigmas,
            filename_prefix,
            image_filename=image_filename,
            use_image_latent=use_image_latent,
        )
        response = request_json("/prompt", {"prompt": workflow}, timeout=60)
        if not isinstance(response, dict) or not response.get("prompt_id"):
            raise RuntimeError(f"ComfyUI did not accept the workflow: {response}")
        prompt_id = str(response["prompt_id"])
        print(
            f"Generating Turbo {args.width}x{args.height}, 8 steps, seed {args.seed}; "
            f"ComfyUI job {prompt_id}."
        )
        image = wait_for_image(prompt_id, args.timeout)
        output_dir = Path(os.environ.get("L3MS_MEDIA_OUTPUT_DIR", str(Path.home() / "media-output")))
        destination = args.output or output_dir / ("qwen-image-2.1-turbo-" + timestamp + ".png")
        destination = destination.expanduser()
        if not destination.is_absolute():
            destination = Path.cwd() / destination
        if destination.suffix.lower() != ".png":
            raise ValueError("Qwen Image 2.1 Turbo output must use a .png path")
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
        print(f"Qwen Image 2.1 Turbo: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
