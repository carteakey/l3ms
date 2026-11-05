#!/usr/bin/env python3
"""Multimodal Vision evaluation script for Qwen3.8-Flash-Next-Plat-Vision (Strata v0.1.45)."""
import argparse
import base64
import json
import os
import sys
import time
import urllib.request

DEFAULT_PROMPT = (
    "Perform a rigorous, exhaustive visual analysis of this screenshot. "
    "1. Read and transcribe all visible telemetry metrics (GPU, VRAM, speed, power, temperature, context fill). "
    "2. Transcribe and summarize the activity displayed in the right-hand terminal. "
    "3. Analyze the overall UI layout and operational state of the system. "
    "Think carefully and structure your final response with Markdown headings."
)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def post_json_stream(url, key, payload, timeout=3600):
    payload["stream"] = True
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {key}"} if key else {})
        },
    )
    content_chunks = []
    reasoning_chunks = []
    timings = {}
    usage = {}
    tokens_received = 0
    t_first = None

    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw_line in r:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                if "timings" in chunk and chunk["timings"]:
                    timings = chunk["timings"]
                if "usage" in chunk and chunk["usage"]:
                    usage = chunk["usage"]
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    r_text = delta.get("reasoning_content")
                    if r_text:
                        reasoning_chunks.append(r_text)
                        tokens_received += 1
                    c_text = delta.get("content")
                    if c_text:
                        content_chunks.append(c_text)
                        tokens_received += 1
                        if t_first is None:
                            t_first = time.time()
                        if tokens_received % 500 == 0:
                            print(f"  ... generated {tokens_received} tokens ...", flush=True)
            except Exception:
                pass

    content = "".join(content_chunks)
    reasoning = "".join(reasoning_chunks)
    return content, reasoning, timings, usage, tokens_received

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--key", default=os.environ.get("LLAMA_SWAP_API_KEY", ""))
    ap.add_argument("--model", default="qwen38-flash-next-plat-vision")
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-tokens", type=int, default=-1)
    ap.add_argument("--reasoning-effort", default="low", choices=["none", "low", "medium", "high", "xhigh"])
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()

    print(f"[{a.model}] Encoding {a.image}...")
    b64_data = encode_image(a.image)

    ext = os.path.splitext(a.image)[1].lower().replace(".", "")
    mime = f"image/{ext}" if ext in ["png", "jpeg", "jpg", "webp"] else "image/png"
    if ext == "jpg":
        mime = "image/jpeg"

    payload = {
        "model": a.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": a.prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64_data}"
                        }
                    }
                ]
            }
        ],
    }
    if a.max_tokens and a.max_tokens > 0:
        payload["max_tokens"] = a.max_tokens
    else:
        payload["max_tokens"] = -1

    if a.reasoning_effort:
        payload["reasoning_effort"] = a.reasoning_effort

    max_display = a.max_tokens if a.max_tokens and a.max_tokens > 0 else "UNCAPPED (-1 / 128k ctx)"
    print(f"[{a.model}] Sending multimodal request to {a.url} (max_tokens={max_display}, effort={a.reasoning_effort})...")
    t0 = time.time()
    content, reasoning, tm, usage, tokens_received = post_json_stream(a.url, a.key, payload)
    wall = time.time() - t0

    tps = tm.get("predicted_per_second")
    if not tps and wall > 0 and tokens_received > 0:
        tps = round(tokens_received / wall, 1)

    result = {
        "model": a.model,
        "image": a.image,
        "wall_sec": round(wall, 2),
        "prompt": a.prompt,
        "timings": tm,
        "usage": usage,
        "completion_tokens": tokens_received,
        "predicted_per_second": tps,
        "reasoning_len": len(reasoning),
        "content_len": len(content),
        "reasoning": reasoning,
        "content": content
    }

    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Extract HTML if present
    html_content = ""
    if "```html" in content:
        html_content = content.split("```html", 1)[1].split("```", 1)[0].strip()
    elif "<!DOCTYPE html>" in content:
        html_content = content[content.find("<!DOCTYPE html>"):].split("```", 1)[0].strip()
    elif "<html>" in content:
        html_content = content[content.find("<html>"):].split("```", 1)[0].strip()
    elif "```" in content:
        html_content = content.split("```", 1)[1].split("```", 1)[0].strip()

    out_html = a.out_json.replace(".json", ".html")
    if html_content:
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[{a.model}] Saved extracted HTML ({len(html_content.splitlines())} lines) to {out_html}")

    print(f"[{a.model}] Completed in {wall:.1f}s. Output saved to {a.out_json}")
    print(f"[{a.model}] Decode throughput: {tm.get('predicted_per_second')} t/s | Completion tokens: {usage.get('completion_tokens')}")
    print("\n--- Model Response ---")
    print(content[:600] + ("..." if len(content) > 600 else ""))

if __name__ == "__main__":
    main()
