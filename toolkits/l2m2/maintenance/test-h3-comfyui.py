#!/usr/bin/env python3
"""Run MiniMax-H3 Ref2VA baseline experiment through local ComfyUI."""

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
from urllib.parse import quote
from urllib.request import Request, urlopen

COMFY_URL = os.environ.get("COMFY_LOCAL_URL", "http://127.0.0.1:8188")
ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_TEMPLATE = ROOT / "workflows/minimax-h3/video_minimax_h3_r2v-20step-no-lora.json"


def request_json(path: str, data: dict[str, object] | None = None, timeout: float = 30) -> object:
    url = COMFY_URL.rstrip("/") + path
    body = None
    headers = {}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=body, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as err:
        err_msg = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI HTTP {err.code} from {path}: {err_msg}") from err
    except URLError as err:
        raise RuntimeError(f"Cannot reach ComfyUI at {COMFY_URL}: {err.reason}") from err


def check_server() -> None:
    try:
        stats = request_json("/system_stats", timeout=5)
        if not isinstance(stats, dict) or "system" not in stats:
            raise RuntimeError("Invalid system_stats from ComfyUI")
    except Exception as err:
        raise RuntimeError(f"ComfyUI server check failed: {err}") from err


def build_prompt_payload(
    ref_image_1: str,
    ref_image_2: str,
    prompt_text: str,
    megapixels: float,
    steps: int,
    seed: int,
    duration: float,
) -> dict[str, dict[str, object]]:
    # Converted API graph matching video_minimax_h3_r2v-20step-no-lora.json
    return {
        "92": {
            "inputs": {
                "filename_prefix": "video/MiniMax_H3",
                "format": "auto",
                "video": ["130", 0],
                "format.codec": "auto",
            },
            "class_type": "SaveVideo",
            "_meta": {"title": "Save Video"},
        },
        "115": {
            "inputs": {
                "aspect_ratio": "16:9 (Widescreen)",
                "megapixels": megapixels,
                "multiple": 32,
            },
            "class_type": "ResolutionSelector",
            "_meta": {"title": "Resolution Selector (Size)"},
        },
        "119": {
            "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
            "class_type": "VAELoader",
            "_meta": {"title": "Load VAE"},
        },
        "120": {
            "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
            "class_type": "VAELoader",
            "_meta": {"title": "Load VAE"},
        },
        "121": {
            "inputs": {"samples": ["125", 0], "vae": ["120", 0]},
            "class_type": "VAEDecodeAudio",
            "_meta": {"title": "VAE Decode Audio"},
        },
        "122": {
            "inputs": {"samples": ["125", 0], "vae": ["119", 0]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
        },
        "123": {
            "inputs": {"sampler_name": "res_multistep"},
            "class_type": "KSamplerSelect",
            "_meta": {"title": "KSamplerSelect"},
        },
        "124": {
            "inputs": {
                "scheduler": "simple",
                "denoise": 1,
                "model": ["127", 0],
                "steps": ["142", 0],
            },
            "class_type": "BasicScheduler",
            "_meta": {"title": "BasicScheduler"},
        },
        "125": {
            "inputs": {
                "noise": ["129", 0],
                "guider": ["126", 0],
                "sampler": ["123", 0],
                "sigmas": ["124", 0],
                "latent_image": ["136", 1],
            },
            "class_type": "SamplerCustomAdvanced",
            "_meta": {"title": "SamplerCustomAdvanced"},
        },
        "126": {
            "inputs": {"model": ["141", 0], "conditioning": ["136", 0]},
            "class_type": "BasicGuider",
            "_meta": {"title": "Basic Guider"},
        },
        "127": {
            "inputs": {
                "unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                "weight_dtype": "default",
            },
            "class_type": "UNETLoader",
            "_meta": {"title": "Load Diffusion Model"},
        },
        "128": {
            "inputs": {
                "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "type": "minimax",
                "device": "default",
            },
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load CLIP"},
        },
        "129": {
            "inputs": {"noise_seed": seed},
            "class_type": "RandomNoise",
            "_meta": {"title": "RandomNoise"},
        },
        "130": {
            "inputs": {
                "fps": 24,
                "bit_depth": 8,
                "color_space": "sRGB",
                "codec": "none",
                "images": ["122", 0],
                "audio": ["121", 0],
            },
            "class_type": "CreateVideo",
            "_meta": {"title": "Create Video"},
        },
        "131": {
            "inputs": {
                "expression": "max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17",
                "values.a": ["132", 0],
            },
            "class_type": "ComfyMathExpression",
            "_meta": {"title": "Math Expression"},
        },
        "132": {
            "inputs": {"value": duration},
            "class_type": "PrimitiveFloat",
            "_meta": {"title": "Float (Duration)"},
        },
        "136": {
            "inputs": {
                "ref_image_size": "match",
                "clip": ["128", 0],
                "prompt": ["138", 0],
                "width": ["115", 0],
                "height": ["115", 1],
                "length": ["131", 1],
                "vae": ["119", 0],
                "audio_vae": ["120", 0],
                "ref_images.ref_image_0": ["137", 0],
                "ref_images.ref_image_1": ["139", 0],
            },
            "class_type": "MiniMaxH3ReferenceToVideo",
            "_meta": {"title": "MiniMax H3 Reference to Video"},
        },
        "137": {
            "inputs": {"image": ref_image_1},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"},
        },
        "138": {
            "inputs": {"value": prompt_text},
            "class_type": "PrimitiveStringMultiline",
            "_meta": {"title": "Input Text (Prompt)"},
        },
        "139": {
            "inputs": {"image": ref_image_2},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"},
        },
        "141": {
            "inputs": {"switch": ["146", 0], "on_false": ["127", 0], "on_true": ["127", 0]},
            "class_type": "ComfySwitchNode",
            "_meta": {"title": "If/Else Switch (model)"},
        },
        "142": {
            "inputs": {"switch": ["146", 0], "on_false": ["143", 0], "on_true": ["144", 0]},
            "class_type": "ComfySwitchNode",
            "_meta": {"title": "If/Else Switch (Steps)"},
        },
        "143": {
            "inputs": {"value": steps},
            "class_type": "PrimitiveInt",
            "_meta": {"title": "Int (Full)"},
        },
        "144": {
            "inputs": {"value": 4},
            "class_type": "PrimitiveInt",
            "_meta": {"title": "Int (Lightning LoRA)"},
        },
        "146": {
            "inputs": {"value": False},
            "class_type": "PrimitiveBoolean",
            "_meta": {"title": "Boolean (Enable Lightning LoRA)"},
        },
    }


def wait_for_video(prompt_id: str, timeout: float = 3600) -> dict[str, object]:
    started = time.monotonic()
    last_notice = started
    while time.monotonic() - started < timeout:
        history = request_json("/history/" + quote(prompt_id), timeout=30)
        if isinstance(history, dict) and prompt_id in history:
            record = history[prompt_id]
            if isinstance(record, dict):
                status = record.get("status", {})
                if isinstance(status, dict):
                    status_text = status.get("status_str", "")
                    if status_text == "error":
                        messages = status.get("messages", [])
                        raise RuntimeError(f"ComfyUI reports execution error: {messages}")
                    if status.get("completed"):
                        outputs = record.get("outputs", {})
                        if isinstance(outputs, dict):
                            saved = outputs.get("92", {})
                            for key in ("videos", "gifs", "images"):
                                items = saved.get(key, [])
                                if items:
                                    return items[0]
                        raise RuntimeError("Job completed but no video artifact was saved in outputs.")
        if time.monotonic() - last_notice >= 30:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] MiniMax-H3 ComfyUI generation is running...")
            last_notice = time.monotonic()
        time.sleep(3)
    raise RuntimeError(f"Job exceeded timeout of {timeout}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test MiniMax-H3 on ComfyUI.")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    parser.add_argument("--megapixels", type=float, default=0.25, help="Megapixels (default: 0.25)")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds (default: 5.0)")
    parser.add_argument("--ref-image-1", default="ref_vashti_chair.png", help="Reference image 1 in input/")
    parser.add_argument("--ref-image-2", default="ref_vashti_standing.png", help="Reference image 2 in input/")
    parser.add_argument("--prompt", help="Prompt text")
    parser.add_argument("--out-dir", type=Path, help="Experiment record directory")
    args = parser.parse_args()

    check_server()

    default_prompt = (
        "A quiet indoor scene. A woman seated in a chair under soft warm window lighting, "
        "turning her head slowly to look thoughtfully towards the camera. "
        "<Picture 1> and <Picture 2> show character and costume styling. "
        "Consistent room tone, realistic natural lighting, no music."
    )
    prompt_text = args.prompt or default_prompt

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or (Path.home() / f"media-output/h3-comfyui-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment output directory: {out_dir}")
    print(f"Configuration: steps={args.steps}, seed={args.seed}, megapixels={args.megapixels}, duration={args.duration}s")
    print(f"References: <Picture 1>={args.ref_image_1}, <Picture 2>={args.ref_image_2}")

    prompt_payload = build_prompt_payload(
        ref_image_1=args.ref_image_1,
        ref_image_2=args.ref_image_2,
        prompt_text=prompt_text,
        megapixels=args.megapixels,
        steps=args.steps,
        seed=args.seed,
        duration=args.duration,
    )

    with open(out_dir / "prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt_text + "\n")
    with open(out_dir / "workflow-api.json", "w", encoding="utf-8") as f:
        json.dump(prompt_payload, f, indent=2)

    start_time = time.monotonic()
    print("Submitting prompt to ComfyUI...")
    submit_response = request_json("/prompt", {"prompt": prompt_payload})
    if not isinstance(submit_response, dict) or "prompt_id" not in submit_response:
        raise RuntimeError(f"Unexpected response from /prompt: {submit_response}")

    prompt_id = submit_response["prompt_id"]
    print(f"Queued in ComfyUI as prompt_id: {prompt_id}")

    try:
        video_info = wait_for_video(prompt_id)
        wall_seconds = round(time.monotonic() - start_time, 2)
        print(f"Generation finished in {wall_seconds}s! Output info: {video_info}")

        filename = video_info.get("filename", "")
        subfolder = video_info.get("subfolder", "")
        comfy_out_dir = Path.home() / "media-output/comfyui"
        source_path = comfy_out_dir / subfolder / filename if subfolder else comfy_out_dir / filename

        target_path = out_dir / filename
        if source_path.exists():
            shutil.copy2(source_path, target_path)
            print(f"Copied video to {target_path}")

            # Probe with ffprobe
            probe_cmd = [
                "ffprobe", "-v", "error", "-show_format", "-show_streams",
                "-of", "json", str(target_path)
            ]
            probe_proc = subprocess.run(probe_cmd, capture_output=True, text=True)
            if probe_proc.returncode == 0:
                with open(out_dir / "probe.json", "w") as f:
                    f.write(probe_proc.stdout)

        with open(out_dir / "result.txt", "w") as f:
            f.write(f"exit_code=0\nwall_seconds={wall_seconds}\nprompt_id={prompt_id}\n")

        print("Experiment run completed successfully.")
        return 0

    except Exception as err:
        wall_seconds = round(time.monotonic() - start_time, 2)
        with open(out_dir / "result.txt", "w") as f:
            f.write(f"exit_code=1\nwall_seconds={wall_seconds}\nerror={err}\n")
        print(f"Experiment failed: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
