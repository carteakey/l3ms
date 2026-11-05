#!/usr/bin/env python3
"""Identity gate for a Strata tuning arm.

Upstream's own acceptance rule for a switch it calls "Exact" (docs/DETAILS.md of the
v0.2.7 tag, line 1031): "the 10-prompt and 4K/20K identity gate with the switch on equals
the gate with it off". This is that gate, run through the same HTTP surface the serving
tier uses.

Greedy (temperature 0) on a fixed prompt set, no timestamps, no cache reuse. A switch that
does not touch model math must produce the same completion text and the same token counts.
A switch that re-routes experts, changes rounding, or changes the accept rule will not.

Writes one JSON object per arm to --out and prints it. Compare two arms by hand or with
--compare <reference.json>; the harness stores the sha in the diag row so the sweep table
carries the quality column next to the speed column.
"""

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.request

FILLER_WORDS = (
    "the cache holds a row of expert weights that the router reached twice in a row while "
    "the disk kept the rest on a spinning plane and the scheduler waited for a page that "
    "never came back because another process had already paged it out of the working set "
    "and the kernel decided that the reclaim thread should run first on the smallest slab"
).split()


def filler_tokens(approx):
    """Deterministic prose of about `approx` tokens, seeded so every arm gets the same bytes."""
    rng = random.Random(20261028)
    out, n = [], 0
    while n < approx:
        start = rng.randrange(len(FILLER_WORDS))
        clause = " ".join(FILLER_WORDS[start:] if start + 12 >= len(FILLER_WORDS)
                          else FILLER_WORDS[start:start + rng.randrange(8, 13)])
        out.append(clause[0].upper() + clause[1:] + ".")
        n += len(clause.split())
    return " ".join(out)


def doc(approx_tokens, marker_index):
    """A long deterministic document with one distinctive marker sentence at a fixed index."""
    body = filler_tokens(approx_tokens)
    sentences = body.split(". ")
    sentences[marker_index % len(sentences)] = (
        sentences[marker_index % len(sentences)].rstrip(".")
        + ". The marker for this document is lantern-42-kestrel."
    )
    return ". ".join(sentences)


BUGGY = '''Fix the bug in this function. It is called from two threads.

def append_note(path, note):
    with open(path, "r") as f:
        data = f.read()
    with open(path, "w") as f:
        f.write(data + note)
'''


def prompts():
    return [
        ("code_small",
         "Write a Python function atomic_write_text(path: str, text: str) -> None that writes "
         "a file atomically on POSIX. No comments, no explanation, code only.", 128),
        ("code_fix", BUGGY + "Return the fixed function, code only.", 192),
        ("json_strict",
         'Return exactly one JSON object, nothing else, with keys "total" (integer), "items" '
         '(array of three strings: "alpha", "beta", "gamma") and "ok" (boolean true).', 96),
        ("explain_short",
         "In at most two sentences, explain why a memory-mapped read is faster on first access "
         "after the page cache is warm than on the access that follows a page reclaim.", 96),
        ("code_long",
         "Write a Python class RingBuffer with a fixed capacity, push, pop, and __len__, plus a "
         "drain() that returns the items oldest first. No comments, code only.", 384),
        ("pp4k", "Reading: " + doc(3600, 7) + "\nWhat is the marker for this document? Answer in one line.", 48),
        ("pp20k", "Reading: " + doc(18000, 31) + "\nWhat is the marker for this document? Answer in one line.", 48),
    ]


def post(url, key, payload, timeout=900):
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": "Bearer " + key} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def run_one(url, key, name, content, max_tokens):
    t = time.time()
    r = post(url, key, {"messages": [{"role": "user", "content": content}],
                        "max_tokens": max_tokens, "temperature": 0, "top_k": 1,
                        "seed": 42, "cache_prompt": False})
    ch = (r.get("choices") or [{}])[0]
    m = ch.get("message") or {}
    text = (m.get("content") or "") + "\u241f" + (m.get("reasoning_content") or "")
    u = r.get("usage") or {}
    # How deep the agreement goes matters more than whether it is identical: a flip at token 300 of
    # a 384-token generation is a near-tie in the logits, a flip at token 5 is a math change.
    def head(n):
        return hashlib.sha256(text[:n].encode()).hexdigest()[:12]
    return {"name": name,
            "prompt_tokens": u.get("prompt_tokens") or (r.get("timings") or {}).get("prompt_n"),
            "completion_tokens": u.get("completion_tokens") or (r.get("timings") or {}).get("predicted_n"),
            "finish": ch.get("finish_reason"),
            "sha": hashlib.sha256(text.encode()).hexdigest()[:16],
            "head64": head(240), "head256": head(960),
            "wall": round(time.time() - t, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--compare", default="", help="reference identity json to diff against")
    a = ap.parse_args()

    parts = [run_one(a.url, a.key, n, c, m) for (n, c, m) in prompts()]
    blob = "".join(f"{p['name']}|{p['prompt_tokens']}|{p['completion_tokens']}|{p['sha']}\n" for p in parts)
    res = {"label": a.label, "ts": time.strftime("%F %T"),
           "identity_sha": hashlib.sha256(blob.encode()).hexdigest()[:16],
           "prompts": parts}

    if a.compare:
        try:
            ref = json.load(open(a.compare))
            mine = {p["name"]: p for p in parts}
            diffs = []
            for rp in ref.get("prompts", []):
                p = mine.get(rp["name"])
                if not p:
                    diffs.append(f"{rp['name']}: missing")
                elif p["sha"] != rp["sha"] or p["completion_tokens"] != rp["completion_tokens"]:
                    # "diverged at" scores how far the two runs agreed: 0 prompts agree fully, and the
                    # head levels show where they parted. Same-token-count diffs are argmax flips.
                    # parted <where the two texts first differ>; "past the tail" cannot happen
                    # (the full sha already differed), so three levels are enough.
                    depth = ("within the first 240 chars" if p["head64"] != rp.get("head64") else
                             "after 240, before 960 chars" if p["head256"] != rp.get("head256") else
                             "past 960 chars")
                    diffs.append(f"{rp['name']}: sha {rp['sha']} -> {p['sha']}, "
                                 f"tokens {rp['completion_tokens']} -> {p['completion_tokens']}, parted {depth}")
            res["identity_prompts"] = len(ref.get("prompts", []))
            res["identity_diff_n"] = len(diffs)
            res["identity_vs"] = ref.get("label")
            res["identity_pass"] = not diffs
            res["identity_diffs"] = diffs
        except (OSError, ValueError) as e:
            res["identity_pass"] = None
            res["identity_diffs"] = [f"reference unreadable: {e}"]

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "prompts"}, indent=1))
    for p in parts:
        print(f"   {p['name']:14s} in {p['prompt_tokens']} out {p['completion_tokens']} "
              f"{p['finish']} {p['sha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
