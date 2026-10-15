#!/usr/bin/env python3
"""Grade media output through the L3MS llama-swap vision tier.

The agent driving media tests is often a text-only model, so it cannot look at
its own renders. This posts downscaled thumbnails plus a rubric to the router's
vision-capable tier and prints the verdict. Requesting the vision tier swaps
whatever model the router holds, so treat the call as a short self-pause.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen

ROUTER = os.environ.get("L3MS_LLAMA_SWAP_URL", "http://127.0.0.1:8080").rstrip("/")
VISION_MODEL = os.environ.get("L3MS_VISION_MODEL", "qwen38-flash-next-plat-vision")
UNIT_DROPINS = Path.home() / ".config/systemd/user/llama-swap.service.d"

DEFAULT_RUBRIC = (
    "You are grading generated images against the text prompt that produced them. "
    "For each image, in the order given, output one block:\n"
    "ID: <id>\n"
    "MATCH: <does the image contain exactly what the prompt asked, including the "
    "requested nudity/exposure level — yes/partial/no, one line>\n"
    "SANITIZED: <did the model add clothing, hands/arms covering, pasties, cropping, "
    "or other modesty that the prompt did not ask for — none/what>\n"
    "ANATOMY: <hands, fingers, arms, legs, breasts, genitals, torso — correct or "
    "specific defect>\n"
    "ARTIFACTS: <warped background, merged bodies, garbled text, plastic skin — none/list>\n"
    "VERDICT: pass|partial|fail\n"
    "Be blunt and specific. Do not moralize about the content."
)


def api_key() -> str:
    key = os.environ.get("LLAMA_SWAP_API_KEY", "")
    if key:
        return key.strip().strip('"')
    if UNIT_DROPINS.is_dir():
        for dropin in sorted(UNIT_DROPINS.glob("*.conf")):
            match = re.search(r"^LLAMA_SWAP_API_KEY=(.+)$", dropin.read_text(), re.M)
            if match:
                return match.group(1).strip().strip('"')
    raise SystemExit("No LLAMA_SWAP_API_KEY in env or llama-swap service drop-ins.")


def thumbnail(path: Path, max_side: int, quality: int) -> str:
    from PIL import Image

    with Image.open(path) as handle:
        handle = handle.convert("RGB")
        handle.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        handle.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="PNG/JPEG artifacts")
    parser.add_argument("--prompt-file", type=Path, help="TSV prompt pack (id<TAB>prompt<TAB>negative)")
    parser.add_argument("--rubric", default=DEFAULT_RUBRIC, help="Grading rubric sent first")
    parser.add_argument("--model", default=VISION_MODEL, help=f"Router model id (default {VISION_MODEL})")
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--reasoning-log", type=Path, help="Write the model's reasoning stream here (default: stderr)")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    prompts: dict[str, str] = {}
    if args.prompt_file:
        for line in args.prompt_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or "\t" not in line:
                continue
            row = line.split("\t")
            prompts[row[0]] = row[1]

    content: list[dict[str, object]] = [{"type": "text", "text": args.rubric}]
    for image in args.images:
        stem = image.name.rsplit("_s", 1)[0].replace(".png", "")
        asked = prompts.get(stem) or prompts.get(stem.split("__", 1)[0]) or "(prompt not supplied)"
        content.append({"type": "text", "text": f"--- {stem} | prompt: {asked}"})
        content.append({"type": "image_url", "image_url": {"url": thumbnail(image, args.max_side, args.quality)}})

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
        "max_tokens": args.max_tokens,
    }
    request = Request(
        ROUTER + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key()},
        method="POST",
    )
    print(f"POST {ROUTER}/v1/chat/completions model={args.model} images={len(args.images)}", flush=True)
    with urlopen(request, timeout=args.timeout) as response:
        body = json.load(response)
    choice = body.get("choices", [{}])[0]
    message = choice.get("message", {})
    reasoning = (message.get("reasoning_content") or "").strip()
    content_text = (message.get("content") or "").strip()
    if reasoning:
        if args.reasoning_log:
            args.reasoning_log.write_text(reasoning + "\n", encoding="utf-8")
        else:
            print(f"--- reasoning ---\n{reasoning}\n--- answer ---", file=sys.stderr)
    print(content_text or reasoning or f"(empty response: {json.dumps(body)[:1200]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
