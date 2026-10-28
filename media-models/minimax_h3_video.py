#!/usr/bin/env python3
"""Generate a MiniMax-H3 reference-to-video clip through local ComfyUI.

MiniMax-H3 is MiniMax's omni-modal model: the graph samples video and audio latents
jointly (one sampler output, decoded twice - video VAE for frames, audio VAE for the
soundtrack), and the reference conditioning node takes up to nine images whose tokens
ride through every sampling step.

This builder reproduces the official Comfy-Org reference-to-video graph
(workflows/minimax-h3/video_minimax_h3_r2v-official.json and
video_minimax_h3_r2v-20step-no-lora.json, which are the same 29-node graph) minus the
UI-only plumbing:

- the LoRA/no-LoRA and 4-step/20-step ComfySwitchNodes become a python branch on
  --turbo-lora (the shipped switch defaults to False, i.e. no LoRA and 20 steps);
- ResolutionSelector becomes --width/--height (or --megapixels/--aspect, whose
  ceil-to-32 rule reproduces the graph's own megapixel table: 0.2 -> 608x352,
  0.3 -> 736x416, 0.4 -> 864x480);
- the ComfyMathExpression length node becomes video_frames_for_seconds().

Wire facts verified against this ComfyUI build on 2026-10-27 (/object_info):

- MiniMaxH3ReferenceToVideo outputs are ['positive', 'LATENT'] and BasicGuider takes
  model + conditioning with no negative input, so **this graph cannot express a negative
  prompt**. Wording, the reference images and the text encoder are the only levers.
- ref_images is an autogrow set (template prefix "ref_image_", min 0, max 9), so the API
  wire key is the dotted finalized id "ref_images.ref_image_0" - same rule as Qwen Image
  2.1's "images.image_1". A bare ref_image_0 is an unexpected kwarg and a nested dict
  arrives empty.
- ref_image_size 'match' scales each reference down to the generation's pixel area; 'max'
  uses the 2048 px short edge, and the engine's own tooltip warns that 'max' can be
  several times slower because reference tokens ride every step.
- length is a 24 fps frame count with min 5, step 17, max 3600 (124 = about 5 s, the
  trained range is about 124-362), so valid lengths are 5 + 17k.
- The official graph does not use MiniMaxH3SigmaShift; --sigma-shift opts in.
- Keyframes are not in the official graph either. --first-frame, --last-frame and
  --guide-image PATH[:FRAME_IDX] chain MiniMaxH3AddGuide nodes onto the reference conditioning,
  each anchoring one image at a frame index (negative counts from the end of the clip). That node
  is chosen over MiniMaxH3ImageToVideo on purpose: both produce (conditioning, latent) and only
  one can feed BasicGuider, so the keyframe node would *replace* the nine-reference conditioning
  rather than add to it, and AddGuide is the only one that takes an arbitrary frame index.
  AddGuide cover-crops its image to the output size, unlike the keyframe node's plain stretch of
  frame 0.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from comfy_endpoint import resolve_endpoint

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "vendor" / "ComfyUI" / "models"
DEFAULT_UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
# On this box the stock name is a symlink to the heretic (uncensored) encoder file, so the
# default is the uncensored encoder whichever of the two names you pass. See
# docs/media-models.md in the L3MS tree and docs/plan-identity-locking.md in the NSFW tree.
DEFAULT_TE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
HERETIC_TE = "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"
DEFAULT_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
DEFAULT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

MAX_REF_IMAGES = 9
MAX_GUIDES = 9
MIN_FRAMES = 5
FRAME_STEP = 17
MAX_FRAMES = 3600
FPS = 24.0
DIM_MULTIPLE = 32
SCHEDULERS = (
    "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform",
    "beta", "normal", "linear_quadratic", "kl_optimal",
)
REF_IMAGE_SIZES = ("match", "max")
MODEL_SOURCE = ("127", 0)  # UNETLoader output; loras/sigma-shift rehook this
REQUIRED_NODES = (
    "UNETLoader", "CLIPLoader", "VAELoader", "MiniMaxH3ReferenceToVideo",
    "BasicGuider", "BasicScheduler", "KSamplerSelect", "RandomNoise",
    "SamplerCustomAdvanced", "VAEDecode", "VAEDecodeAudio", "CreateVideo", "SaveVideo",
)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
API: dict[str, str] = {"base": "", "key": ""}


def video_frames_for_seconds(seconds: float) -> int:
    """The official graph's length node, inlined.

    max(5, round(a * 24)) + (5 - (n % 17)) % 17 rounds the frame count up to the next
    legal 5 + 17k value, which is what the video VAE's temporal compression needs.
    """
    base = max(MIN_FRAMES, int(round(seconds * FPS)))
    return base + (MIN_FRAMES - (base % FRAME_STEP)) % FRAME_STEP


def normalize_frames(count: int) -> int:
    """Round a requested frame count up to the next 5 + 17k value."""
    if count < MIN_FRAMES:
        raise ValueError(f"frame count must be at least {MIN_FRAMES}")
    if count > MAX_FRAMES:
        raise ValueError(f"frame count must be at most {MAX_FRAMES}")
    if (count - MIN_FRAMES) % FRAME_STEP:
        steps = (count - MIN_FRAMES) // FRAME_STEP + 1
        adjusted = MIN_FRAMES + steps * FRAME_STEP
        if adjusted > MAX_FRAMES:
            raise ValueError(
                f"{count} frames rounds up past the {MAX_FRAMES} frame ceiling; "
                f"the largest legal length is {max_legal_frames()}"
            )
        return adjusted
    return count


def max_legal_frames() -> int:
    """Largest 5 + 17k frame count the engine accepts (3592 for a 3600 ceiling)."""
    return MAX_FRAMES - ((MAX_FRAMES - MIN_FRAMES) % FRAME_STEP)


def ceil_to_multiple(value: float, multiple: int = DIM_MULTIPLE) -> int:
    return int(math.ceil(value / multiple) * multiple)


def dims_for_megapixels(megapixels: float, aspect: float = 16 / 9, multiple: int = DIM_MULTIPLE) -> tuple[int, int]:
    """Reproduce the official graph's megapixel table for an aspect ratio.

    Rounding each side up to the next multiple of 32 from the exact sqrt dimensions
    reproduces the shipped table exactly (0.2 -> 608x352, 0.3 -> 736x416, 0.4 -> 864x480).
    """
    if megapixels <= 0 or aspect <= 0:
        raise ValueError("--megapixels and --aspect must both be positive")
    width = ceil_to_multiple(math.sqrt(megapixels * 1_000_000 * aspect), multiple)
    height = ceil_to_multiple(math.sqrt(megapixels * 1_000_000 / aspect), multiple)
    return width, height


def validate_asset_name(label: str, name: str) -> None:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"{label} must be a bare filename in ComfyUI's model dirs: {name!r}")


def validate_settings(
    width: int,
    height: int,
    frames: int,
    steps: int,
    seed: int,
    scheduler: str,
    sampler: str,
    ref_image_size: str,
    ref_count: int,
    timeout: float,
) -> None:
    for label, value in (("--width", width), ("--height", height)):
        if value < DIM_MULTIPLE or value % DIM_MULTIPLE:
            raise ValueError(f"{label} must be a multiple of {DIM_MULTIPLE} (got {value})")
    if frames < MIN_FRAMES or frames > MAX_FRAMES:
        raise ValueError(f"frame count must be between {MIN_FRAMES} and {MAX_FRAMES}")
    if (frames - MIN_FRAMES) % FRAME_STEP:
        raise ValueError(f"frame count must be {MIN_FRAMES} + {FRAME_STEP}k (got {frames})")
    if steps < 1:
        raise ValueError("--steps must be at least 1")
    if seed < 0:
        raise ValueError("--seed must be 0 or greater")
    if scheduler not in SCHEDULERS:
        raise ValueError(f"--scheduler must be one of {', '.join(SCHEDULERS)}")
    if not sampler:
        raise ValueError("--sampler must not be empty")
    if ref_image_size not in REF_IMAGE_SIZES:
        raise ValueError(f"--ref-image-size must be one of {', '.join(REF_IMAGE_SIZES)}")
    if ref_count > MAX_REF_IMAGES:
        raise ValueError(f"at most {MAX_REF_IMAGES} --ref-image values are allowed")
    if timeout <= 0:
        raise ValueError("--timeout must be positive")


def configure_api(url: str | None = None) -> str:
    """Resolve the ComfyUI endpoint through the shared rule in comfy_endpoint.py.

    ComfyUI is a llama-swap engine (comfyui_auto, routed at /comfyui/), so pointing at the
    router is the normal path: the router swaps ComfyUI in and evicts whatever model was
    resident, including the model this agent is running on. A bare LLAMA_SWAP_API_KEY with no
    LLAMA_SWAP_URL means the house router on :8080.
    """
    base, key = resolve_endpoint(url)
    API["base"] = base
    API["key"] = key
    return API["base"]


def via_router() -> bool:
    return "/comfyui" in API["base"]


def api_url() -> str:
    return API["base"] or configure_api()


def request_bytes(path: str, payload: dict[str, object] | None = None, timeout: float = 30) -> bytes:
    url = api_url() + path
    headers: dict[str, str] = {}
    if API["key"]:
        headers["Authorization"] = f"Bearer {API['key']}"
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
            f"Cannot reach ComfyUI at {api_url()}. Through llama-swap, check "
            "LLAMA_SWAP_URL/LLAMA_SWAP_API_KEY and that comfyui_auto is declared; "
            "directly, start it with maintenance/comfyui-service.sh start-gpu."
        ) from error


def request_json(path: str, payload: dict[str, object] | None = None, timeout: float = 30) -> object:
    return json.loads(request_bytes(path, payload, timeout).decode("utf-8"))


def check_models(unet: str, te: str, video_vae: str, audio_vae: str, want_audio: bool = True) -> None:
    relatives = [
        f"diffusion_models/{unet}",
        f"text_encoders/{te}",
        f"vae/{video_vae}",
    ]
    if want_audio:
        relatives.append(f"vae/{audio_vae}")
    missing = [str(MODEL_DIR / relative) for relative in relatives if not (MODEL_DIR / relative).is_file()]
    if missing:
        raise RuntimeError(
            "Missing MiniMax-H3 model files; the manifest is models/minimax-h3-comfyui-files.txt:\n"
            + "\n".join(missing)
        )


def check_api(extra_nodes: tuple[str, ...] = (), timeout: float = 600.0,
              scheduler: str | None = None, sampler: str | None = None) -> None:
    """Verify the engine answers, the nodes exist, and any requested knobs are offered.

    --check calls this with no scheduler/sampler (nothing has been chosen yet), so the knob
    checks are guarded on the requested value, not on a hard-coded list.
    """
    # The router holds this first request open while it swaps the resident model out and
    # cold-boots ComfyUI (246s measured on the image lane), so the budget is a boot, not a
    # request. Same for the video lane, plus the router must then load ~36.6 GiB of weights.
    request_json("/system_stats", timeout=timeout)
    info = request_json("/object_info", timeout=timeout)
    if not isinstance(info, dict):
        raise RuntimeError("ComfyUI returned an invalid /object_info response")
    missing = [name for name in (*REQUIRED_NODES, *extra_nodes) if name not in info]
    if missing:
        raise RuntimeError(
            "ComfyUI is missing required MiniMax-H3 nodes: " + ", ".join(missing)
            + ". Run maintenance/setup-comfyui.sh and check the custom-node packs."
        )
    for label, node, widget, value in (
        ("--scheduler", "BasicScheduler", "scheduler", scheduler),
        ("--sampler", "KSamplerSelect", "sampler_name", sampler),
    ):
        if not value:
            continue
        spec = info[node]["input"]["required"][widget][1]
        options = spec.get("options") or spec.get("values") or []
        values = [o.get("key") if isinstance(o, dict) else o for o in options]
        if value not in values:
            raise RuntimeError(
                f"{label} {value!r} is not offered by this ComfyUI "
                f"(offered: {', '.join(map(str, values[:8]))})"
            )


def check_gpu_exclusive() -> None:
    """Skip the host checks when llama-swap owns the card; it arbitrates the swap itself."""
    if via_router():
        return
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return
    result = subprocess.run(
        [nvidia_smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True,
    )
    if result.returncode == 0:
        try:
            used = int(result.stdout.splitlines()[0].strip())
        except (IndexError, ValueError):
            return
        if used >= 1500:
            raise RuntimeError(f"GPU is busy ({used} MiB used). Wait for other workloads to finish.")


def error_detail(status: dict) -> str:
    """Pull ComfyUI's own exception text out of a failed history record."""
    bits: list[str] = []
    for entry in status.get("messages") or []:
        for item in entry if isinstance(entry, list) else [entry]:
            if not isinstance(item, dict):
                continue
            text = item.get("exception_message") or item.get("reason") or item.get("error")
            if not text:
                continue
            node = item.get("node_id") or item.get("node") or ""
            kind = item.get("exception_type") or item.get("class_type") or ""
            bits.append(f"node {node} ({kind}): {text}".strip())
    return "; ".join(bits[:4]) or "history carried no message"


def find_saved_video(record: dict) -> dict:
    """Locate SaveVideo's output entry without hard-coding its history key.

    SaveImage files land under outputs[node]["images"]; the video node's key is
    container-shaped instead, and it varies by ComfyUI version, so scan every node's
    outputs for a dict carrying a filename rather than guessing the key.
    """
    candidates: list[dict] = []
    for node_out in (record.get("outputs") or {}).values():
        if not isinstance(node_out, dict):
            continue
        for value in node_out.values():
            items = value if isinstance(value, list) else [value]
            for item in items:
                if isinstance(item, dict) and item.get("filename"):
                    candidates.append(item)
    if not candidates:
        raise LookupError("no saved file in this job's outputs")
    videos = [c for c in candidates if str(c["filename"]).lower().endswith((".mp4", ".webm", ".mkv", ".mov"))]
    return (videos or candidates)[-1]


def wait_for_video(prompt_id: str, timeout: float) -> dict:
    started = time.monotonic()
    last_notice = started
    while time.monotonic() - started < timeout:
        try:
            # While the router is loading ~36.6 GiB of weights a poll can block for minutes;
            # one slow round-trip is not a failure, the deadline is.
            history = request_json("/history/" + quote(prompt_id), timeout=60)
        except OSError:
            time.sleep(5)
            continue
        if isinstance(history, dict):
            record = history.get(prompt_id)
            if isinstance(record, dict):
                status = record.get("status", {})
                if isinstance(status, dict) and status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI job failed: {error_detail(status)}")
                try:
                    return find_saved_video(record)
                except LookupError:
                    pass
                if isinstance(status, dict) and status.get("completed"):
                    raise RuntimeError("ComfyUI completed the job but saved no video file.")
        if time.monotonic() - last_notice >= 60:
            print("MiniMax-H3 video is still sampling in ComfyUI...")
            last_notice = time.monotonic()
        time.sleep(3)
    raise RuntimeError(f"Generation exceeded the {timeout:g}s timeout; it may still be running in ComfyUI.")


def stage_images(paths: list[str], prefix: str = "l2m2-h3-ref", label: str = "--ref-image") -> tuple[str, ...]:
    """Copy images into ComfyUI's input dir under a content-hashed name.

    LoadImage resolves bare filenames inside the server's base input directory, which for
    this build is vendor/ComfyUI/input (not models/input), so reference and guide frames have
    to be staged there; the content hash keeps different anchors apart and identical ones from
    copying twice.
    """
    if len(paths) > MAX_REF_IMAGES:
        raise ValueError(f"MiniMax-H3 takes at most {MAX_REF_IMAGES} {label} values")
    input_dir = ROOT / "vendor" / "ComfyUI" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    import hashlib

    names: list[str] = []
    for index, raw in enumerate(paths):
        source = Path(raw).expanduser()
        if not source.is_file():
            raise ValueError(f"{label} is not a file: {source}")
        suffix = source.suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError(f"{label} must be one of {' '.join(IMAGE_SUFFIXES)}: {source}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        name = f"{prefix}-{index + 1}-{digest}{suffix}"
        destination = input_dir / name
        if not destination.is_file():
            shutil.copyfile(source, destination)
        names.append(name)
    return tuple(names)


def stage_reference_images(paths: list[str]) -> tuple[str, ...]:
    """Stage the --ref-image values; see stage_images."""
    return stage_images(paths)


GUIDE_INDEX = re.compile(":(-?[0-9]+)$")


def parse_guide_spec(raw: str) -> tuple[str, int]:
    """Split PATH[:FRAME_IDX]. A trailing :INTEGER is always read as the frame index.

    The colon goes to the index whenever the string ends in it, so a file genuinely named like
    x:12.png has to reach a guide through --first-frame or --last-frame, which take a bare path
    with no parsing.
    """
    match = GUIDE_INDEX.search(raw)
    if match:
        path = raw[:match.start()]
        if not path:
            raise ValueError(f"--guide-image needs a path before the frame index, got {raw!r}")
        return path, int(match.group(1))
    if not raw.strip():
        raise ValueError("--guide-image needs a path")
    return raw, 0


def parse_guides(values: list[str], first_frame: str | None = None,
                 last_frame: str | None = None, frames: int | None = None) -> tuple[tuple[str, int], ...]:
    """Resolve the keyframe flags into (path, frame_idx) pairs, in anchor order.

    Order is --first-frame, then --guide-image in the order given, then --last-frame: AddGuide
    nodes chain, and a fixed order keeps the graph byte-stable across runs.
    """
    guides: list[tuple[str, int]] = []
    if first_frame:      # frame 0
        guides.append((first_frame, 0))
    guides.extend(parse_guide_spec(raw) for raw in values)
    if last_frame:       # frame -1, the clip's last frame
        guides.append((last_frame, -1))
    if len(guides) > MAX_GUIDES:
        raise ValueError(f"MiniMax-H3 takes at most {MAX_GUIDES} guide frames, got {len(guides)}")
    if frames is not None:
        for path, index in guides:
            resolved = index if index >= 0 else frames + index
            if not 0 <= resolved < frames:
                raise ValueError(
                    f"guide frame index {index} for {path!r} lands on frame {resolved}, outside "
                    f"the clip's {frames} frames (0..{frames - 1}, or negative to count from the end)"
                )
    return tuple(guides)


def parse_loras(values: list[str]) -> tuple[tuple[str, float], ...]:
    result = []
    for raw in values:
        name, _, strength = raw.partition(":")
        name = name.strip()
        if not name:
            raise ValueError("--turbo-lora needs a filename")
        validate_asset_name("--turbo-lora", name)
        result.append((name, float(strength) if strength else 1.0))
    return tuple(result)


def build_workflow(
    prompt: str,
    width: int,
    height: int,
    frames: int,
    steps: int,
    seed: int,
    filename_prefix: str,
    scheduler: str = "simple",
    sampler: str = "res_multistep",
    unet_name: str = DEFAULT_UNET,
    te_name: str = DEFAULT_TE,
    video_vae_name: str = DEFAULT_VIDEO_VAE,
    audio_vae_name: str = DEFAULT_AUDIO_VAE,
    ref_names: tuple[str, ...] = (),
    guide_names: tuple[tuple[str, int], ...] = (),
    ref_image_size: str = "match",
    turbo_lora: tuple[tuple[str, float], ...] = (),
    sigma_shift: tuple[float, float] | None = None,
    want_audio: bool = True,
    fps: float = FPS,
) -> dict:
    """API-format graph, using the official node ids so logs line up with the workflow JSON."""
    graph: dict[str, dict] = {
        "127": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "128": {"class_type": "CLIPLoader", "inputs": {"clip_name": te_name, "type": "minimax"}},
        "119": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae_name}},
        "120": {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae_name}},
    }

    model_from = MODEL_SOURCE
    if turbo_lora:
        previous = "127"
        for index, (name, strength) in enumerate(turbo_lora):
            node = f"14{5 + index}"
            graph[node] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"model": (previous, 0), "lora_name": name, "strength_model": strength},
            }
            previous = node
        model_from = (previous, 0)
    if sigma_shift:
        graph["147"] = {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {"model": model_from, "shift_video": sigma_shift[0], "shift_audio": sigma_shift[1]},
        }
        model_from = ("147", 0)

    ref_inputs: dict[str, object] = {}
    for index, name in enumerate(ref_names):
        node = f"13{7 + index}" if index < 3 else f"15{index - 3}"
        graph[node] = {"class_type": "LoadImage", "inputs": {"image": name}}
        # Autogrow slots are addressed by their dotted finalized id, never bare or nested.
        ref_inputs[f"ref_images.ref_image_{index}"] = (node, 0)

    reference: dict[str, object] = {
        "clip": ("128", 0),
        "prompt": prompt,
        "width": width,
        "height": height,
        "length": frames,
        "ref_image_size": ref_image_size,
        "vae": ("119", 0),
        "audio_vae": ("120", 0),
    }
    reference.update(ref_inputs)
    graph["136"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": reference}

    # Guide frames chain onto the reference conditioning: each AddGuide takes the previous guide's
    # conditioning plus the same AV latent from the reference node and returns conditioning only,
    # so the sampler still draws from the latent the reference node allocated. LoadImage ids for
    # guides start at 180 because reference slots already use 137-139 and 150-155.
    if len(guide_names) > MAX_GUIDES:
        raise ValueError(f"MiniMax-H3 takes at most {MAX_GUIDES} guide frames, got {len(guide_names)}")
    conditioning_from = ("136", 0)
    for index, (name, frame_idx) in enumerate(guide_names):
        load_id = f"18{index}"
        graph[load_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        node = f"16{index}"
        graph[node] = {
            "class_type": "MiniMaxH3AddGuide",
            "inputs": {
                "positive": conditioning_from,
                "latent": ("136", 1),
                "vae": ("119", 0),
                "image": (load_id, 0),
                "frame_idx": frame_idx,
            },
        }
        conditioning_from = (node, 0)
    graph["126"] = {"class_type": "BasicGuider",
                    "inputs": {"model": model_from, "conditioning": conditioning_from}}
    graph["124"] = {
        "class_type": "BasicScheduler",
        "inputs": {"model": model_from, "scheduler": scheduler, "steps": steps, "denoise": 1.0},
    }
    graph["123"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}}
    graph["129"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    graph["125"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ("129", 0),
            "guider": ("126", 0),
            "sampler": ("123", 0),
            "sigmas": ("124", 0),
            "latent_image": ("136", 1),
        },
    }
    graph["122"] = {"class_type": "VAEDecode", "inputs": {"samples": ("125", 0), "vae": ("119", 0)}}
    create: dict[str, object] = {"images": ("122", 0), "fps": fps}
    if want_audio:
        graph["121"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ("125", 0), "vae": ("120", 0)}}
        create["audio"] = ("121", 0)
    graph["130"] = {"class_type": "CreateVideo", "inputs": create}
    graph["92"] = {
        "class_type": "SaveVideo",
        "inputs": {"video": ("130", 0), "filename_prefix": filename_prefix, "format": "auto", "codec": "auto"},
    }
    return graph


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Generate a MiniMax-H3 reference-to-video clip through local ComfyUI.")
    result.add_argument("--prompt", help="Text prompt (this graph has no negative prompt; say what you want)")
    result.add_argument("--prompt-file", type=Path, help="Read the prompt from a UTF-8 file")
    result.add_argument("--output", type=Path, help="Output .mp4 path")
    result.add_argument("--width", type=int, help="Frame width, multiple of 32 (default from --megapixels/--aspect)")
    result.add_argument("--height", type=int, help="Frame height, multiple of 32")
    result.add_argument("--megapixels", type=float, default=0.2, help="Target megapixels when --width/--height are absent (default 0.2, the low end of the recipe's 0.2-0.3 range)")
    result.add_argument("--aspect", type=float, default=16 / 9, help="Aspect ratio used with --megapixels (default 16:9)")
    result.add_argument("--frames", type=int, help=f"Frame count; must be {MIN_FRAMES} + {FRAME_STEP}k at 24 fps, rounded up if not")
    result.add_argument("--seconds", type=float, help="Clip length in seconds; converted with the official graph's own formula")
    result.add_argument("--steps", type=int, default=20, help="Sampling steps (20 without a Turbo LoRA, 4 with the shipped one)")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--scheduler", default="simple", help=f"BasicScheduler scheduler (default simple; options: {', '.join(SCHEDULERS)})")
    result.add_argument("--sampler", default="res_multistep", help="KSamplerSelect sampler name (official graph uses res_multistep)")
    result.add_argument("--model", default=DEFAULT_UNET, help=f"DiT filename in ComfyUI diffusion_models/ (default: {DEFAULT_UNET})")
    result.add_argument("--te", default=DEFAULT_TE, help=f"Text encoder filename in ComfyUI text_encoders/ (default: {DEFAULT_TE}; on this box both minimax TE names resolve to the uncensored heretic file)")
    result.add_argument("--video-vae", default=DEFAULT_VIDEO_VAE)
    result.add_argument("--audio-vae", default=DEFAULT_AUDIO_VAE)
    result.add_argument("--ref-image", action="append", default=[], metavar="PATH",
                        help=f"Reference image whose tokens ride every step, up to {MAX_REF_IMAGES}, repeatable. "
                             "Use one nude-view matte per character you want to stay recognizable.")
    result.add_argument("--ref-image-size", default="match", choices=REF_IMAGE_SIZES,
                        help="match scales refs to the output's pixel area; max uses the 2048px short edge and is several times slower")
    result.add_argument("--first-frame", metavar="PATH",
                        help="image anchored at frame 0 by MiniMaxH3AddGuide (cover-cropped to the output size). "
                             "Keeps --ref-image conditioning, which MiniMaxH3ImageToVideo would replace")
    result.add_argument("--last-frame", metavar="PATH",
                        help="image anchored at the clip's final frame (AddGuide frame index -1)")
    result.add_argument("--guide-image", action="append", default=[], metavar="PATH[:FRAME_IDX]",
                        help=f"image anchored at an arbitrary frame, repeatable, chained in the order given; a "
                             f"negative index counts from the end, up to {MAX_GUIDES} anchors in total. "
                             "--first-frame and --last-frame are the frame 0 and -1 cases")
    result.add_argument("--turbo-lora", action="append", default=[], metavar="NAME[:STRENGTH]",
                        help="LoRA filename in ComfyUI loras/ applied to the DiT only; repeatable, chained in order")
    result.add_argument("--sigma-shift", metavar="VIDEO:AUDIO",
                        help="Insert MiniMaxH3SigmaShift with these shifts (the official graph does not use the node)")
    result.add_argument("--no-audio", action="store_true", help="Skip audio decoding (video-only mp4)")
    result.add_argument("--fps", type=float, default=FPS, help="Frame rate of the saved video (the DiT samples at 24 fps)")
    result.add_argument("--url", help="ComfyUI base URL, used verbatim. Otherwise $L2M2_COMFY_URL verbatim, "
                                      "then $LLAMA_SWAP_URL/comfyui, then the router implied by a bare "
                                      "$LLAMA_SWAP_API_KEY, then http://127.0.0.1:8188.")
    result.add_argument("--timeout", type=float, default=3600, help="How long to wait for the clip to finish")
    result.add_argument("--warm-timeout", type=float, default=600.0,
                        help="How long --check and the submit wait for the router to boot ComfyUI")
    result.add_argument("--check", action="store_true", help="Check model files and ComfyUI nodes, then exit")
    result.add_argument("--dry-run", action="store_true", help="Print the API-format workflow JSON and exit (no engine call)")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        configure_api(args.url)
        want_audio = not args.no_audio
        check_models(args.model, args.te, args.video_vae, args.audio_vae, want_audio)
        if args.dry_run:
            args.check = False
        if args.check:
            check_api(timeout=args.warm_timeout)
            print("MiniMax-H3 model files and ComfyUI nodes are ready.")
            return 0
        if args.prompt and args.prompt_file:
            raise ValueError("--prompt and --prompt-file are mutually exclusive")
        prompt = args.prompt
        if args.prompt_file:
            prompt = args.prompt_file.read_text(encoding="utf-8")
        if not prompt or not prompt.strip():
            raise ValueError("MiniMax-H3 requires --prompt or --prompt-file")

        if args.width or args.height:
            if not (args.width and args.height):
                raise ValueError("--width and --height must be given together (or give neither and use --megapixels/--aspect)")
            width, height = args.width, args.height
        else:
            width, height = dims_for_megapixels(args.megapixels, args.aspect)

        if args.frames and args.seconds:
            raise ValueError("--frames and --seconds are mutually exclusive")
        frames = normalize_frames(args.frames) if args.frames else video_frames_for_seconds(args.seconds if args.seconds else 5.0)
        if args.frames and frames != args.frames:
            print(f"Adjusting {args.frames} frames to {frames}: lengths must be {MIN_FRAMES} + {FRAME_STEP}k at {FPS:g} fps.")

        loras = parse_loras(args.turbo_lora)
        shift = None
        if args.sigma_shift:
            video_shift, _, audio_shift = args.sigma_shift.partition(":")
            shift = (float(video_shift), float(audio_shift) if audio_shift else 3.0)
        refs = list(args.ref_image)
        guides = parse_guides(args.guide_image, args.first_frame, args.last_frame, frames)
        validate_settings(
            width, height, frames, args.steps, args.seed, args.scheduler, args.sampler,
            args.ref_image_size, len(refs), args.timeout,
        )
        validate_asset_name("--model", args.model)
        validate_asset_name("--te", args.te)
        validate_asset_name("--video-vae", args.video_vae)
        if want_audio:
            validate_asset_name("--audio-vae", args.audio_vae)

        staged_guides: tuple[tuple[str, int], ...] = ()
        if guides:
            staged_names = stage_images([path for path, _ in guides], prefix="l2m2-h3-guide",
                                        label="--guide frame")
            staged_guides = tuple(zip(staged_names, (index for _, index in guides)))

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        graph = build_workflow(
            prompt, width, height, frames, args.steps, args.seed,
            f"h3/minimax-h3-{timestamp}", scheduler=args.scheduler, sampler=args.sampler,
            unet_name=args.model, te_name=args.te, video_vae_name=args.video_vae,
            audio_vae_name=args.audio_vae, ref_names=tuple(stage_reference_images(refs)) if refs else (),
            guide_names=staged_guides, ref_image_size=args.ref_image_size,
            turbo_lora=loras, sigma_shift=shift,
            want_audio=want_audio, fps=args.fps,
        )
        if args.dry_run:
            print(json.dumps(graph, indent=1, sort_keys=True))
            print(f"# {width}x{height}, {frames} frames at {args.fps:g} fps "
                  f"({frames / args.fps:.2f}s), {args.steps} steps, {args.scheduler}/{args.sampler}, "
                  f"seed {args.seed}, {len(refs)} reference(s), {len(staged_guides)} anchored frame(s) "
                  f"{[idx for _, idx in staged_guides]}, {width * height * frames / 1e6:.2f} megapixel-frames",
                  file=sys.stderr)
            return 0

        check_gpu_exclusive()
        extra_nodes: list[str] = []
        if refs or staged_guides:
            extra_nodes.append("LoadImage")
        if staged_guides:
            extra_nodes.append("MiniMaxH3AddGuide")
        if loras:
            extra_nodes.append("LoraLoaderModelOnly")
        if shift:
            extra_nodes.append("MiniMaxH3SigmaShift")
        check_api(tuple(extra_nodes), timeout=args.warm_timeout, scheduler=args.scheduler, sampler=args.sampler)

        response = request_json("/prompt", {"prompt": graph}, timeout=max(60.0, args.warm_timeout))
        if not isinstance(response, dict) or not response.get("prompt_id"):
            raise RuntimeError(f"ComfyUI did not accept the workflow: {response}")
        prompt_id = str(response["prompt_id"])
        print(
            f"Generating {width}x{height} x {frames} frames ({frames / args.fps:.2f}s) with {args.model}, "
            f"{args.steps} steps, {args.scheduler}/{args.sampler}, seed {args.seed}, "
            f"TE {args.te}, refs {len(refs)} at {args.ref_image_size}, "
            f"anchored frames {[idx for _, idx in staged_guides] or 'none'}, "
            f"audio {'on' if want_audio else 'off'}; ComfyUI job {prompt_id} at {api_url()}."
        )
        saved = wait_for_video(prompt_id, args.timeout)
        output_dir = Path(os.environ.get("L3MS_MEDIA_OUTPUT_DIR", str(Path.home() / "media-output")))
        destination = args.output or output_dir / ("minimax-h3-" + timestamp + ".mp4")
        destination = destination.expanduser()
        if not destination.is_absolute():
            destination = Path.cwd() / destination
        if destination.suffix.lower() not in (".mp4", ".webm", ".mkv", ".mov"):
            raise ValueError("MiniMax-H3 output must be a video path (.mp4, .webm, .mkv, .mov)")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"Refusing to overwrite existing output: {destination}")
        query = urlencode({
            "filename": saved["filename"],
            "subfolder": saved.get("subfolder", ""),
            "type": saved.get("type", "output"),
        })
        destination.write_bytes(request_bytes("/view?" + query, timeout=300))
        print(f"Saved {destination} (ComfyUI file {saved.get('subfolder', '')}/{saved['filename']})")
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"MiniMax-H3 video: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
