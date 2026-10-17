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
DEFAULT_TE = "qwen3vl_8b_int8_convrot.safetensors"
MAX_REF_IMAGES = 16
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
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
    te_name: str = DEFAULT_TE,
    ref_names: tuple[str, ...] = (),
    loras: tuple[tuple[str, float], ...] = (),
    ref_resolution: int = 1024,
    edit_name: str = "",
    denoise: float = 1.0,
) -> dict[str, dict[str, object]]:
    """Return the API-format graph matching Comfy-Org's Qwen 2.1 workflow.

    With no references and no LoRAs the graph is the stock one. References enter
    through TextEncodeQwenImage21's image_1..16 slots: the text encoder sees them
    and their VAE latents are spliced into the token sequence. With
    ref_resolution == 0 the references keep their native size and the node's own
    latent output (sized to the first reference) replaces EmptyLatentImage; with an
    explicit resolution the references are downscaled to that cap (fewer spliced
    tokens, less VRAM) and EmptyLatentImage keeps the requested width/height.
    """
    graph = {
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
                "clip_name": te_name,
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

    te_inputs = graph["4"]["inputs"]
    latent_source: list[object] = ["5", 0]
    if edit_name:
        # An edit keeps the source render's composition by sampling from its own latents;
        # references then steer identity while denoise decides how much may change.
        graph["24"] = {"class_type": "LoadImage", "inputs": {"image": edit_name}}
        graph["25"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["24", 0], "vae": ["3", 0]}}
        latent_source = ["25", 0]
    if ref_names:
        te_inputs["vae"] = ["3", 0]
        te_inputs["resolution"] = ref_resolution
        for index, image_name in enumerate(ref_names):
            node = str(20 + index)
            graph[node] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
            # Autogrow slots are addressed by their finalized (dotted) input id: the server
            # registers "images.image_1" as the live input and folds it back into the
            # "images" dict before execute(). Bare "image_1" arrives as an unexpected kwarg.
            te_inputs[f"images.image_{index + 1}"] = [node, 0]
        if not edit_name and ref_resolution == 0:
            latent_source = ["4", 2]  # frame follows the first reference at native size

    model_source: list[object] = ["1", 0]
    for offset, (lora_name, strength) in enumerate(loras):
        node = str(12 + offset)
        graph[node] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model_source, "lora_name": lora_name, "strength_model": strength},
        }
        model_source = [node, 0]

    graph["6"]["inputs"]["model"] = model_source
    graph["6"]["inputs"]["latent_image"] = latent_source
    graph["6"]["inputs"]["denoise"] = denoise
    return graph


def validate_asset_name(label: str, name: str) -> None:
    if not name.endswith(".safetensors") or name.startswith("/") or ".." in name or "/" in name:
        raise ValueError(f"{label} must be a .safetensors filename inside ComfyUI's model directories")


def parse_loras(values: list[str]) -> tuple[tuple[str, float], ...]:
    """Parse NAME[:STRENGTH] entries in the order given."""
    parsed: list[tuple[str, float]] = []
    for value in values:
        name, _, strength = value.partition(":")
        name = name.strip()
        if not name:
            raise ValueError("--extra-lora needs a LoRA filename")
        validate_asset_name("--extra-lora", name)
        try:
            weight = float(strength) if strength else 1.0
        except ValueError:
            raise ValueError(f"--extra-lora strength is not a number: {value}") from None
        if weight < 0 or weight > 10:
            raise ValueError("--extra-lora strength must be between 0 and 10")
        parsed.append((name, weight))
    return tuple(parsed)


def validate_settings(
    width: int,
    height: int,
    steps: int,
    cfg: float,
    timeout: float,
    unet_name: str = DEFAULT_UNET,
    te_name: str = DEFAULT_TE,
    ref_resolution: int = 1024,
    edit_name_present: bool = False,
    denoise: float = 1.0,
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
    if ref_resolution < 0 or ref_resolution > 4096:
        raise ValueError("--ref-resolution must be between 0 and 4096")
    if edit_name_present and not (0 < denoise <= 1):
        raise ValueError("--denoise must be greater than 0 and at most 1 when editing an image")
    validate_asset_name("--model", unet_name)
    validate_asset_name("--te", te_name)


def configure_api(url: str | None = None) -> str:
    """Resolve the ComfyUI endpoint: explicit flag, then direct URL, then llama-swap.

    ComfyUI is a llama-swap engine (comfyui_auto, routed at /comfyui/), so pointing at
    the router is the normal path: the router swaps ComfyUI in and evicts whatever
    model was resident. Set LLAMA_SWAP_URL + LLAMA_SWAP_API_KEY and nothing else.
    """
    base = (url or os.environ.get("L2M2_COMFY_URL") or "").strip()
    key = ""
    if not base:
        router = (os.environ.get("LLAMA_SWAP_URL") or "").strip()
        if router:
            base = router.rstrip("/") + "/comfyui"
            key = (os.environ.get("LLAMA_SWAP_API_KEY") or "").strip()
    if not base:
        base = "http://127.0.0.1:8188"
    API["base"] = base.rstrip("/")
    API["key"] = key
    return API["base"]


def via_router() -> bool:
    return "/comfyui" in API["base"]


API: dict[str, str] = {"base": "", "key": ""}


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


def check_models(unet_name: str = DEFAULT_UNET, te_name: str = DEFAULT_TE) -> None:
    model_dir = ROOT / "vendor" / "ComfyUI" / "models"
    relatives = (f"diffusion_models/{unet_name}", f"text_encoders/{te_name}", MODEL_FILES[2])
    missing = [str(model_dir / relative) for relative in relatives if not (model_dir / relative).is_file()]
    if missing:
        raise RuntimeError(
            "Missing Qwen model files; run maintenance/setup-qwen-image-2.1-comfyui.sh:\n"
            + "\n".join(missing)
        )


def check_api(extra_nodes: tuple[str, ...] = (), timeout: float = 300.0) -> None:
    # The router holds this first request open while it swaps the resident model out and
    # cold-boots ComfyUI, which takes minutes, not seconds; a 30s socket timeout would
    # report a healthy engine as dead.
    request_json("/system_stats", timeout=timeout)
    info = request_json("/object_info", timeout=timeout)
    if not isinstance(info, dict):
        raise RuntimeError("ComfyUI returned an invalid /object_info response")
    missing = [name for name in (*REQUIRED_NODES, *extra_nodes) if name not in info]
    if missing:
        raise RuntimeError(
            "ComfyUI is missing required Qwen workflow nodes: "
            + ", ".join(missing)
            + ". Run maintenance/setup-comfyui.sh."
        )


def check_gpu_exclusive() -> None:
    """Skip the host checks when llama-swap owns the card: it arbitrates the swap itself.

    Direct ComfyUI runs still refuse a busy card, whitelisting the toolkit's own unit.
    """
    if via_router():
        return

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


def error_detail(status: dict) -> str:
    """Pull ComfyUI's own exception text out of a failed history record."""
    bits: list[str] = []
    messages = status.get("messages") or []
    if isinstance(messages, list):
        for entry in messages:
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


def wait_for_image(prompt_id: str, timeout: float) -> dict[str, str]:
    started = time.monotonic()
    last_notice = started
    while time.monotonic() - started < timeout:
        try:
            # While the router is still loading the image model ComfyUI can sit on a request
            # for minutes; one slow poll is not a failure, the deadline is.
            history = request_json("/history/" + quote(prompt_id), timeout=30)
        except OSError:
            time.sleep(5)
            continue
        if isinstance(history, dict):
            record = history.get(prompt_id)
            if isinstance(record, dict):
                status = record.get("status", {})
                status_text = status.get("status_str", "") if isinstance(status, dict) else ""
                if status_text == "error":
                    raise RuntimeError(f"ComfyUI job failed: {error_detail(status)}")
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


def stage_reference_images(paths: list[str], ) -> tuple[str, ...]:
    """Copy references into ComfyUI's input dir under a content-hashed name.

    LoadImage resolves bare filenames inside ComfyUI's base input directory, which for
    this build is vendor/ComfyUI/input (the server log resolves LoadImage there, not
    models/input), so the files have to be staged there; hashing the name keeps
    different anchors apart and identical ones from being copied twice.
    """
    if len(paths) > MAX_REF_IMAGES:
        raise ValueError(f"Qwen Image 2.1 takes at most {MAX_REF_IMAGES} --ref-image values")
    input_dir = ROOT / "vendor" / "ComfyUI" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    import hashlib

    names: list[str] = []
    for index, raw in enumerate(paths):
        source = Path(raw).expanduser()
        if not source.is_file():
            raise ValueError(f"--ref-image is not a file: {source}")
        suffix = source.suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError(f"--ref-image must be one of {' '.join(IMAGE_SUFFIXES)}: {source}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        name = f"l2m2-qwen-ref-{index + 1}-{digest}{suffix}"
        destination = input_dir / name
        if not destination.is_file():
            shutil.copyfile(source, destination)
        names.append(name)
    return tuple(names)


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
    result.add_argument(
        "--te",
        default=DEFAULT_TE,
        help=f"Text encoder filename in ComfyUI text_encoders/ (default: {DEFAULT_TE})",
    )
    result.add_argument(
        "--ref-image",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            f"Reference image for identity/composition conditioning, up to {MAX_REF_IMAGES}, "
            "repeatable. The text encoder sees it and its VAE latents ride through every "
            "sampling step. Output size follows the first reference image, so size the "
            "anchor to the frame you want."
        ),
    )
    result.add_argument(
        "--ref-resolution",
        type=int,
        default=1024,
        help="References are resized to about this many pixels per side (0 keeps their own size and lets the first reference set the output frame; any other value keeps the output at --width/--height).",
    )
    result.add_argument(
        "--extra-lora",
        action="append",
        default=[],
        metavar="NAME[:STRENGTH]",
        help="LoRA filename in ComfyUI loras/, applied after the UNet; repeatable, chained in order.",
    )
    result.add_argument(
        "--edit-image",
        metavar="PATH",
        help=(
            "Source render to edit: it is VAE-encoded and sampled from at --denoise, so the "
            "composition survives while --ref-image steers identity. Output size follows this image."
        ),
    )
    result.add_argument(
        "--denoise",
        type=float,
        default=1.0,
        help="How much of the --edit-image to repaint (0.3 keeps framing, 1.0 replaces it).",
    )
    result.add_argument(
        "--url",
        help=(
            "ComfyUI base URL. Defaults to $L2M2_COMFY_URL, then $LLAMA_SWAP_URL/comfyui "
            "(llama-swap's route, key from $LLAMA_SWAP_API_KEY), then http://127.0.0.1:8188."
        ),
    )
    result.add_argument("--timeout", type=float, default=3600)
    result.add_argument(
        "--warm-timeout",
        type=float,
        default=300.0,
        help="How long --check waits for ComfyUI to answer; the router holds that request open while it swaps or cold-boots the engine.",
    )
    result.add_argument("--check", action="store_true", help="Check model files and local ComfyUI nodes")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        configure_api(args.url)
        check_models(args.model, args.te)
        if args.check:
            check_api(timeout=args.warm_timeout)
            print("Qwen Image 2.1 model files and ComfyUI nodes are ready.")
            return 0
        if args.prompt and args.prompt_file:
            raise ValueError("--prompt and --prompt-file are mutually exclusive")
        prompt = args.prompt
        if args.prompt_file:
            prompt = args.prompt_file.read_text(encoding="utf-8")
        if not prompt or not prompt.strip():
            raise ValueError("Qwen Image 2.1 requires --prompt or --prompt-file")
        loras = parse_loras(args.extra_lora)
        refs = list(args.ref_image)
        if len(refs) > MAX_REF_IMAGES:
            raise ValueError(f"at most {MAX_REF_IMAGES} --ref-image values are allowed")
        validate_settings(
            args.width,
            args.height,
            args.steps,
            args.cfg,
            args.timeout,
            unet_name=args.model,
            te_name=args.te,
            ref_resolution=args.ref_resolution,
            edit_name_present=bool(args.edit_image),
            denoise=args.denoise,
        )
        check_gpu_exclusive()
        extra_nodes: list[str] = []
        if refs:
            extra_nodes.append("LoadImage")
        if args.edit_image:
            extra_nodes.append("VAEEncode")
        if loras:
            extra_nodes.append("LoraLoaderModelOnly")
        check_api(tuple(extra_nodes))

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename_prefix = "l2m2-qwen-image-2.1-" + timestamp
        ref_names = stage_reference_images(refs) if refs else ()
        edit_name = stage_reference_images([args.edit_image])[0] if args.edit_image else ""
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
            te_name=args.te,
            ref_names=ref_names,
            loras=loras,
            ref_resolution=args.ref_resolution,
            edit_name=edit_name,
            denoise=args.denoise,
        )
        # The router holds this open while it evicts the resident model and loads the image
        # model (measured cold: 246s before /system_stats answered), so the accept-wait is
        # budgeted like a boot rather than like a request.
        response = request_json("/prompt", {"prompt": workflow}, timeout=max(60.0, args.warm_timeout))
        if not isinstance(response, dict) or not response.get("prompt_id"):
            raise RuntimeError(f"ComfyUI did not accept the workflow: {response}")
        prompt_id = str(response["prompt_id"])
        print(
            f"Generating {args.width}x{args.height} with {args.model}, {args.steps} steps, "
            f"seed {args.seed}; "
            + (f"{len(ref_names)} reference image(s); " if ref_names else "")
            + (f"editing {args.edit_image} at denoise {args.denoise:g}; " if args.edit_image else "")
            + (f"{len(loras)} LoRA(s); " if loras else "")
            + f"ComfyUI job {prompt_id} at {api_url()}."
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
