#!/usr/bin/env python3
"""Export the exact ComfyUI API workflows a Qwen Image probe pack will run.

Reads a TSV prompt pack (id<TAB>prompt<TAB>negative) and writes one API-format
workflow JSON per row, so a probe run can be replayed with
`curl -X POST http://127.0.0.1:8188/prompt -d @<file>` without the Python runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "media-models"))

from qwen_image_21 import build_workflow  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True, help="TSV prompt pack")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for workflow JSON files")
    parser.add_argument("--prefix", default="l2m2-qwen-image-2.1", help="SaveImage filename_prefix")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for line in args.pack.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "\t" not in line:
            continue
        row = line.split("\t")
        probe_id, prompt = row[0], row[1]
        negative = row[2] if len(row) > 2 else ""
        workflow = build_workflow(prompt, negative, args.width, args.height, args.steps, args.seed, args.cfg, f"{args.prefix}-{probe_id}")
        path = args.out_dir / f"{probe_id}.json"
        path.write_text(json.dumps({"prompt": workflow}, indent=2) + "\n", encoding="utf-8")
        written.append(path)

    manifest = {
        "pack": str(args.pack),
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "cfg": args.cfg,
        "seed": args.seed,
        "sampler": "euler/simple, denoise 1.0",
        "weights": [
            "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
            "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
            "vae/qwen_image_2.1_vae_bf16.safetensors",
        ],
        "replay": "curl -s -X POST http://127.0.0.1:8188/prompt -H 'Content-Type: application/json' -d @<file>",
        "workflows": [p.name for p in written],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(written)} workflows + manifest.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
