#!/usr/bin/env python3
"""Generate FLUX.3 Image Klein text-to-image through a local ComfyUI API."""

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
    "diffusion_models/flux3_image_klein_int8_convrot.safetensors",
    "text_encoders/qwen3vl_4b_int8_convrot.safetensors",
    "vae/flux2-vae.safetensors",
)
DEFAULT_UNET = "flux3_image_klein_int8_convrot.safetensors"
REQUIRED_NODES = (
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "CLIPTextEncode",
    "EmptyFlux2LatentImage",
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
    clip_name: str = "qwen3vl_4b_int8_convrot.safetensors",
    vae_name: str = "flux2-vae.safetensors",
    boxes: list[dict[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    """Return the API-format graph matching FLUX.3 Klein ComfyUI pipeline."""
    final_prompt = prompt
    if boxes:
        rows = []
        for i, el in enumerate(boxes):
            desc = str(el.get("desc", ""))
            if el.get("type") == "text" and el.get("text"):
                desc = f'{desc} "{el["text"]}"'.strip()
            rows.append({"id": f"obj_{i + 1}", "bbox": el.get("bbox", []), "desc": desc})
        if rows:
            final_prompt = f"{prompt.rstrip()} {json.dumps(rows, ensure_ascii=False)}"

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
                "clip_name": clip_name,
                "type": "flux3",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": vae_name,
            },
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": final_prompt,
            },
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": negative_prompt,
            },
        },
        "6": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["7", 0],
                "vae": ["3", 0],
            },
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["8", 0],
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
        if value < 32 or value > 4096 or value % 16:
            raise ValueError(f"{label} must be a multiple of 16 between 32 and 4096")
    if steps < 1 or steps > 100:
        raise ValueError("steps must be between 1 and 100")
    if cfg <= 0 or cfg > 20:
        raise ValueError("CFG must be greater than 0 and at most 20")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if not unet_name.endswith(".safetensors") or unet_name.startswith("/") or ".." in unet_name:
        raise ValueError("--model must be a .safetensors file inside ComfyUI's diffusion_models/")


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


def check_models(unet_name: str = DEFAULT_UNET) -> None:
    model_dir = ROOT / "vendor" / "ComfyUI" / "models"
    relatives = (f"diffusion_models/{unet_name}", *MODEL_FILES[1:])
    missing = [str(model_dir / relative) for relative in relatives if not (model_dir / relative).is_file()]
    if missing:
        raise RuntimeError(
            "Missing FLUX.3 Klein model files; run maintenance/setup-flux-3-klein-comfyui.sh:\n"
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
            "ComfyUI is missing required FLUX.3 Klein workflow nodes: "
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
                    "llama-swap.service is active. Stop it before FLUX.3 Klein generation; "
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
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            foreign = []
            for line in lines:
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 2 and parts[0].isdigit():
                    pid = int(parts[0])
                    if pid not in comfy_pids:
                        foreign.append(f"PID {pid} ({parts[1]} MiB)")
            if foreign:
                raise RuntimeError(
                    "GPU memory is claimed by non-ComfyUI processes: "
                    + ", ".join(foreign)
                    + ". Free the GPU before running FLUX.3 Klein."
                )


def wait_for_prompt(prompt_id: str, timeout: float) -> dict[str, object]:
    start = time.monotonic()
    while True:
        history = request_json(f"/history/{quote(prompt_id)}")
        if isinstance(history, dict) and prompt_id in history:
            return history[prompt_id]
        if time.monotonic() - start > timeout:
            raise TimeoutError(f"Timed out waiting {timeout}s for prompt {prompt_id}")
        time.sleep(0.5)


def extract_output_images(entry: dict[str, object]) -> list[dict[str, str]]:
    outputs = entry.get("outputs")
    if not isinstance(outputs, dict):
        return []
    images: list[dict[str, str]] = []
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        items = node_output.get("images")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and "filename" in item:
                    images.append(
                        {
                            "filename": str(item["filename"]),
                            "subfolder": str(item.get("subfolder", "")),
                            "type": str(item.get("type", "output")),
                        }
                    )
    return images


def save_image_bytes(image_ref: dict[str, str], destination: Path) -> None:
    query = urlencode(image_ref)
    data = request_bytes(f"/view?{query}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text-to-image with FLUX.3 Image Klein via the local ComfyUI API."
    )
    parser.add_argument("prompt", nargs="?", default="", help="Text prompt describing the desired image.")
    parser.add_argument("--negative-prompt", default="", help="Optional negative prompt (default: none).")
    parser.add_argument("--width", type=int, default=1024, help="Width in pixels (multiple of 16, default: 1024).")
    parser.add_argument("--height", type=int, default=1024, help="Height in pixels (multiple of 16, default: 1024).")
    parser.add_argument("--steps", type=int, default=16, help="Denoising steps (distilled default: 16).")
    parser.add_argument("--cfg", type=float, default=1.0, help="Classifier-free guidance scale (distilled default: 1.0).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--output", "-o", default=None, help="Output image file path.")
    parser.add_argument("--model", default=DEFAULT_UNET, help=f"Diffusion model filename (default: {DEFAULT_UNET}).")
    parser.add_argument("--boxes", default=None, help="Optional path to boxes JSON or inline JSON string for layout control.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Generation timeout in seconds (default: 300).")
    parser.add_argument("--check", action="store_true", help="Perform preflight checks without generating.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_settings(args.width, args.height, args.steps, args.cfg, args.timeout, args.model)

    boxes_data = None
    if args.boxes:
        if os.path.exists(args.boxes):
            with open(args.boxes, "r", encoding="utf-8") as bf:
                boxes_data = json.load(bf)
        else:
            boxes_data = json.loads(args.boxes)

    if args.check:
        check_models(args.model)
        check_api()
        check_gpu_exclusive()
        print("FLUX.3 Klein ComfyUI environment and models are verified and ready.")
        return 0

    if not args.prompt:
        print("Error: prompt is required unless --check is specified.", file=sys.stderr)
        return 2

    check_models(args.model)
    check_api()
    check_gpu_exclusive()

    filename_prefix = f"flux3_klein_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    workflow = build_workflow(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
        cfg=args.cfg,
        filename_prefix=filename_prefix,
        unet_name=args.model,
        boxes=boxes_data,
    )

    response = request_json("/prompt", {"prompt": workflow})
    if not isinstance(response, dict) or "prompt_id" not in response:
        raise RuntimeError(f"Unexpected /prompt response from ComfyUI: {response}")

    prompt_id = str(response["prompt_id"])
    history_entry = wait_for_prompt(prompt_id, args.timeout)
    images = extract_output_images(history_entry)
    if not images:
        raise RuntimeError(f"No output images found for prompt {prompt_id}")

    output_path = Path(args.output) if args.output else Path(f"{filename_prefix}.png")
    save_image_bytes(images[0], output_path)
    print(f"Generated image saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
