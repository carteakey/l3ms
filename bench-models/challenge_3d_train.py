#!/usr/bin/env python3
"""Challenge 3D Train Scene evaluation script for Qwen3.8-Flash-Next PLE comparisons."""
import argparse
import json
import re
import sys
import time
import urllib.request

PROMPT = (
    "Build a beautiful 3D scene of a train driving through a futuristic cyberpunk city in HTML. "
    "dystopian vibes, detailed models, try not to use too many boxes in the models. "
    "Be creative and add some user interactivity like camera and weather controls. "
    "Ensure the HTML file is completely self-contained, executable, and fully closed."
)

def post(url, key, payload, timeout=3600):
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {key}"} if key else {})
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def extract_html(text):
    m = re.search(r"```html\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(<!DOCTYPE html.*</html>)", text, re.DOTALL | re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return text.strip()

def analyze_html(html):
    return {
        "bytes": len(html),
        "has_threejs": "three.js" in html.lower() or "three.module.js" in html.lower() or "three.min.js" in html.lower(),
        "has_camera_controls": "orbitcontrols" in html.lower() or "controls" in html.lower(),
        "has_weather_controls": "rain" in html.lower() or "fog" in html.lower() or "weather" in html.lower(),
        "has_train": "train" in html.lower() or "locomotive" in html.lower(),
        "has_lighting": "ambientlight" in html.lower() or "pointlight" in html.lower() or "directionallight" in html.lower(),
        "line_count": len(html.splitlines()),
        "is_complete": "</html>" in html.lower(),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--key", default="")
    ap.add_argument("--model", default="qwen38-flash-next-plat")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-html", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--reasoning-budget", type=int, default=0, help="0 = completely open / unbounded reasoning")
    ap.add_argument("--reasoning-effort", default=None, choices=["none", "low", "medium", "high", "xhigh"])
    a = ap.parse_args()

    effort_str = f", effort={a.reasoning_effort}" if a.reasoning_effort else ""
    budget_str = f"reasoning_budget={a.reasoning_budget}" if a.reasoning_budget > 0 else "reasoning_budget=OPEN (unbounded)"
    print(f"[{a.label}] Sending 3D cyberpunk train challenge to {a.url} (model={a.model}, max_tokens={a.max_tokens}, {budget_str}{effort_str})...")
    t0 = time.time()
    payload = {
        "model": a.model,
        "messages": [
            {"role": "user", "content": PROMPT}
        ],
        "max_tokens": a.max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
    }
    if a.reasoning_effort:
        payload["reasoning_effort"] = a.reasoning_effort
    if a.reasoning_budget > 0:
        payload["reasoning_budget_tokens"] = a.reasoning_budget
    resp = post(a.url, a.key, payload)
    wall = time.time() - t0

    choice = resp["choices"][0]
    content = choice["message"].get("content") or ""
    reasoning = choice["message"].get("reasoning_content") or ""
    tm = resp.get("timings", {})
    usage = resp.get("usage", {})

    html = extract_html(content)
    with open(a.out_html, "w", encoding="utf-8") as f:
        f.write(html)

    stats = {
        "label": a.label,
        "wall_sec": round(wall, 2),
        "timings": tm,
        "usage": usage,
        "predicted_per_second": tm.get("predicted_per_second"),
        "analysis": analyze_html(html),
        "reasoning_len": len(reasoning),
        "content_len": len(content),
    }

    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"[{a.label}] Completed in {wall:.1f}s. Saved HTML to {a.out_html}")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
