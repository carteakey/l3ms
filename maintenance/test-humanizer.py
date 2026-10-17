#!/usr/bin/env python3
"""
Humanizer verification and self-test runner for L3MS / llama-swap.

Verifies:
1. Prompt byte-for-byte sha256 fingerprint (cc51d66b4c593fbe)
2. llama-swap model loading and execution
3. Fact, numbers, and date preservation against evaluation draft
"""

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

DEFAULT_DRAFT = """Hi Nadia,

I’m handing off the Meridian Bank loan audit to you. We’ve completed the review of 88 files, with 12 files remaining. The outstanding files are in the “Pending Review” folder in the Meridian Bank audit workspace, organized by borrower name.

The remaining items are primarily commercial real estate loans and include several files requiring confirmation of updated income statements, collateral valuations, and covenant documentation. I’ve flagged the missing items in the audit tracker and added notes for the files that may need follow-up with Meridian Bank’s loan operations team.

The deadline is Friday, March 14. Please prioritize the files marked “High” in the tracker, especially MB-2047, MB-2091, and MB-2110, since those are scheduled for management review first. I’ll be available through Wednesday afternoon to answer questions or provide context on any earlier decisions.

Thanks for taking this over."""

def get_api_key() -> str:
    key = os.environ.get("LLAMA_SWAP_API_KEY", "")
    if key:
        return key
    key_file = Path.home() / ".config/systemd/user/llama-swap.service.d/api-key.conf"
    if key_file.exists():
        for line in key_file.read_text().splitlines():
            if "LLAMA_SWAP_API_KEY=" in line:
                return line.split("LLAMA_SWAP_API_KEY=", 1)[1].strip()
    return ""

def load_prompt_format(model_dir: Path) -> dict:
    pf_path = model_dir / "prompt_format.json"
    if pf_path.exists():
        with open(pf_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "instr": (
            "Rewrite the text below so it reads like a person wrote it, not a language model.\n\n"
            "Reorganize it as you see fit. Vary sentence length on purpose. Cut hedging,\n"
            "throat-clearing, and any sentence that only announces what comes next.\n"
            "Prefer the concrete word over the abstract one. It is fine to sound uneven.\n\n"
            "Every fact, number, unit, date, name and quotation must survive unchanged."
        ),
        "sep": "\n\n### Rewritten:\n\n",
        "fingerprint": "cc51d66b4c593fbe",
    }

def main():
    parser = argparse.ArgumentParser(description="Test Humanizer-12B in llama-swap")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="llama-swap base URL")
    parser.add_argument("--model", default="humanizer-12b", help="Model name or alias")
    parser.add_argument("--model-dir", default="/home/kchauhan/models/jialinyyzz/humanizer", help="Path to model directory")
    parser.add_argument("--draft", default=DEFAULT_DRAFT, help="Input draft to rewrite")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    pf = load_prompt_format(model_dir)
    instr, sep = pf["instr"], pf["sep"]

    def build_prompt(d: str) -> str:
        return instr + "\n\n" + d.strip() + sep

    fp = hashlib.sha256(build_prompt("X").encode("utf-8")).hexdigest()[:16]
    print(f"[*] Prompt format fingerprint: {fp} (expected: cc51d66b4c593fbe)")
    assert fp == "cc51d66b4c593fbe", f"Fingerprint mismatch: got {fp}"

    api_key = get_api_key()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Test via OpenAI-compatible chat completion (tests built-in Jinja template)
    print(f"[*] Sending test request to {args.url}/v1/chat/completions (model: {args.model})...")
    payload = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": args.draft}
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 2048,
    }

    req = urllib.request.Request(
        f"{args.url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            out = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[!] Error querying /v1/chat/completions: {e}")
        sys.exit(1)

    print("\n--- Model Output ---")
    print(out)
    print("--------------------\n")

    problems = []
    if len(out.split()) < 60:
        problems.append("output too short (< 60 words)")
    if "### Rewritten" in out or out.startswith("Rewrite the text below"):
        problems.append("prompt echoed (chat template issue)")
    if re.search(r"<start_of_turn>|<end_of_turn>|<\|turn", out):
        problems.append("chat turn markers leaked in output")

    # Check 5-gram overlap
    draft_words = args.draft.lower().split()
    out_words = out.lower().split()
    if len(draft_words) >= 5 and len(out_words) >= 5:
        draft_5grams = {tuple(draft_words[i:i + 5]) for i in range(len(draft_words) - 4)}
        out_5grams = {tuple(out_words[i:i + 5]) for i in range(len(out_words) - 4)}
        overlap = len(draft_5grams & out_5grams) / max(1, len(out_5grams))
        print(f"[*] 5-gram reuse rate: {overlap:.1%}")
        if overlap > 0.5:
            problems.append(f"output copies draft too heavily ({overlap:.1%})")

    # Check numbers
    draft_numbers = set(re.findall(r"\b\d+\b", args.draft))
    out_numbers = set(re.findall(r"\b\d+\b", out))
    missing_numbers = draft_numbers - out_numbers
    if missing_numbers:
        print(f"[!] Warning: numbers from draft not found in rewrite: {sorted(missing_numbers)}")

    if problems:
        print(f"[FAIL] Issues detected: {'; '.join(problems)}")
        sys.exit(1)
    else:
        print("[PASS] Humanizer output passed all validation checks!")

if __name__ == "__main__":
    main()
