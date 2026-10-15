#!/usr/bin/env python3
"""Generate Qwen Image 2.1 text-to-image through a local ComfyUI API."""

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
    "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
    "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
    "vae/qwen_image_2.1_vae_bf16.safetensors",
)
DEFAULT_UNET = "qwen_image_2.1_int8_convrot.safetensors"
REQUIRED_NODES = (
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "TextEncodeQwenImage21",
    "EmptyLatentImage",
    "KSampler",
    "VAEDecode",
    "SaveImage",
)


def build_workflow(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    cfg: float,
    filename_prefix: str,
    unet_name: str = DEFAULT_UNET,
) -> dict[str, dict[str, object]]:
    """Return the API-format graph matching Comfy-Org's Qwen 2.1 workflow."""
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": unet_name,
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
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["4", 1],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["6", 0], "vae": ["3", 0]},
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["7", 0],
                "filename_prefix": filename_prefix,
            },
        },
    }


def validate_settings(
    width: int,
    height: int,
    steps: int,
    cfg: float,
    timeout: float,
    unet_name: str = DEFAULT_UNET,
) -> None:
    for label, value in (("width", width), ("height", height)):
        if value < 32 or value > 2048 or value % 32:
            raise ValueError(f"{label} must be a multiple of 32 between 32 and 2048")
    if steps < 1 or steps > 100:
        raise ValueError("steps must be between 1 and 100")
    if cfg <= 0 or cfg > 20:
        raise ValueError("CFG must be greater than 0 and at most 20")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if not unet_name.endswith(".safetensors") or unet_name.startswith("/") or ".." in unet_name:
        raise ValueError("--model must be a .safetensors file inside ComfyUI's diffusion_models/")


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


def check_models(unet_name: str = DEFAULT_UNET) -> None:
    model_dir = ROOT / "vendor" / "ComfyUI" / "models"
    relatives = (f"diffusion_models/{unet_name}", *MODEL_FILES[1:])
    missing = [str(model_dir / relative) for relative in relatives if not (model_dir / relative).is_file()]
    if missing:
        raise RuntimeError(
            "Missing Qwen model files; run maintenance/setup-qwen-image-2.1-comfyui.sh:\n"
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
            "ComfyUI is missing required Qwen workflow nodes: "
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
                    saved = outputs.get("8", {})
                    images = saved.get("images", []) if isinstance(saved, dict) else []
                    if images:
                        return images[0]
                if isinstance(status, dict) and status.get("completed"):
                    raise RuntimeError("ComfyUI completed the job but returned no saved image.")
        if time.monotonic() - last_notice >= 30:
            print("Qwen generation is still running in ComfyUI...")
            last_notice = time.monotonic()
        time.sleep(2)
    raise RuntimeError(
        f"Generation exceeded the {timeout:g}s timeout; it may still be running in ComfyUI."
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Generate Qwen Image 2.1 through local ComfyUI.")
    result.add_argument("--prompt", help="Text prompt")
    result.add_argument("--prompt-file", type=Path, help="Read the prompt from a UTF-8 file")
    result.add_argument("--negative-prompt", default="", help="Optional negative prompt")
    result.add_argument("--output", type=Path, help="Output PNG path")
    result.add_argument("--width", type=int, default=1024)
    result.add_argument("--height", type=int, default=1024)
    result.add_argument("--steps", type=int, default=25)
    result.add_argument("--seed", type=int, default=42)
    # 2.5, not 1.0: at cfg 1 ComfyUI skips the negative-conditioning pass entirely, so the
    # per-row negative prompts in prompts/*.tsv would be inert. Measured on NSFW probes:
    # cfg 1.0 was the worst regime for multi-person anatomy, cfg 3.5/5.0 over-sharpen.
    result.add_argument("--cfg", type=float, default=2.5)
    result.add_argument(
        "--model",
        default=DEFAULT_UNET,
        help=f"Diffusion model filename in ComfyUI diffusion_models/ (default: {DEFAULT_UNET})",
    )
    result.add_argument("--timeout", type=float, default=3600)
    result.add_argument("--check", action="store_true", help="Check model files and local ComfyUI nodes")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        check_models(args.model)
        if args.check:
            check_api()
            print("Qwen Image 2.1 model files and ComfyUI nodes are ready.")
            return 0
        if args.prompt and args.prompt_file:
            raise ValueError("--prompt and --prompt-file are mutually exclusive")
        prompt = args.prompt
        if args.prompt_file:
            prompt = args.prompt_file.read_text(encoding="utf-8")
        if not prompt or not prompt.strip():
            raise ValueError("Qwen Image 2.1 requires --prompt or --prompt-file")
        validate_settings(args.width, args.height, args.steps, args.cfg, args.timeout, unet_name=args.model)
        check_gpu_exclusive()
        check_api()

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename_prefix = "l2m2-qwen-image-2.1-" + timestamp
        workflow = build_workflow(
            prompt,
            args.negative_prompt,
            args.width,
            args.height,
            args.steps,
            args.seed,
            args.cfg,
            filename_prefix,
            unet_name=args.model,
        )
        response = request_json("/prompt", {"prompt": workflow}, timeout=60)
        if not isinstance(response, dict) or not response.get("prompt_id"):
            raise RuntimeError(f"ComfyUI did not accept the workflow: {response}")
        prompt_id = str(response["prompt_id"])
        print(
            f"Generating {args.width}x{args.height} with {args.model}, {args.steps} steps, "
            f"seed {args.seed}; "
            f"ComfyUI job {prompt_id}."
        )
        image = wait_for_image(prompt_id, args.timeout)
        output_dir = Path(os.environ.get("L3MS_MEDIA_OUTPUT_DIR", str(Path.home() / "media-output")))
        destination = args.output or output_dir / ("qwen-image-2.1-" + timestamp + ".png")
        destination = destination.expanduser()
        if not destination.is_absolute():
            destination = Path.cwd() / destination
        if destination.suffix.lower() != ".png":
            raise ValueError("Qwen Image 2.1 output must use a .png path")
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
        print(f"Qwen Image 2.1: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
