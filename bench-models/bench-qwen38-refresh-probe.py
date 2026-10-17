#!/usr/bin/env python3
"""Engine-agnostic probe for the qwen4exp refresh A/B (llama-server and Strata).

Talks to any OpenAI-compatible server that returns llama-style `timings`.
Per arm: warmup decodes (discarded), N steady decode probes on code prompts,
unique-prefix prefill probes (cache_prompt=false) at two sizes, and a needle
check. Appends one JSON line to --out. System frontmatter comes from
bench-env.sh, captured by the orchestrator (bench-qwen38-refresh-ab.sh).
"""
import argparse, json, subprocess, sys, time, urllib.request

CODE_PROMPTS = [
    "Write a Rust function that parses a CSV line with quoted fields and escaped quotes into Vec<String>, with unit tests.",
    "Implement an LRU cache in Python with O(1) get/put using OrderedDict-free doubly linked list, with type hints and docstrings.",
    "Write a C++17 thread-safe bounded MPMC queue using std::mutex and condition_variable, with a small usage example.",
    "Write a bash script that rotates logs in a directory, compresses files older than 7 days, and deletes archives older than 30 days.",
]
FILLER = ("The quick brown fox jumps over the lazy dog while engineers profile memory "
          "bandwidth on hybrid MoE inference with speculative decoding. ")


def post(url, key, payload, timeout=1200):
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def vram_mb():
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
    except Exception:
        return None


def decode_probe(url, key, prompt, n):
    t = time.time()
    r = post(url, key, {"messages": [{"role": "user", "content": f"[{time.time_ns()}] {prompt}"}],
                        "max_tokens": n, "temperature": 0, "cache_prompt": False})
    tm = r.get("timings", {})
    u = r.get("usage", {})
    dn, da = tm.get("draft_n", 0), tm.get("draft_n_accepted", 0)
    return {"tg": tm.get("predicted_per_second"), "n": tm.get("predicted_n", u.get("completion_tokens")),
            "draft_acc": (da / dn) if dn else None, "wall": time.time() - t}


def prefill_probe(url, key, approx_tokens):
    reps = max(1, approx_tokens // 28)
    r = post(url, key, {"messages": [{"role": "user", "content": f"[{time.time_ns()}] Summarize in one sentence: " + FILLER * reps}],
                        "max_tokens": 4, "temperature": 0, "cache_prompt": False})
    tm = r.get("timings", {})
    return {"pp": tm.get("prompt_per_second"), "n": tm.get("prompt_n", r.get("usage", {}).get("prompt_tokens"))}


def needle_probe(url, key, approx_tokens):
    code = f"OBSIDIAN-{time.time_ns() % 100000}"
    reps = max(2, approx_tokens // 28)
    body = (FILLER * (reps // 2)) + f" The secret token is {code}. " + (FILLER * (reps // 2))
    r = post(url, key, {"messages": [{"role": "user", "content": body + "\nWhat is the secret token? Answer with the token only."}],
                        "max_tokens": 256, "temperature": 0, "cache_prompt": False})
    m = r["choices"][0]["message"]
    return code in ((m.get("content") or "") + (m.get("reasoning_content") or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--tg-probes", type=int, default=4)
    ap.add_argument("--tg-tokens", type=int, default=256)
    ap.add_argument("--pp-sizes", default="2048,8192")
    a = ap.parse_args()

    for i in range(a.warmups):
        decode_probe(a.url, a.key, CODE_PROMPTS[i % len(CODE_PROMPTS)], a.tg_tokens)
    tg = [decode_probe(a.url, a.key, CODE_PROMPTS[i % len(CODE_PROMPTS)], a.tg_tokens) for i in range(a.tg_probes)]
    pp = {}
    for s in [int(x) for x in a.pp_sizes.split(",") if x]:
        pp[s] = [prefill_probe(a.url, a.key, s) for _ in range(2)]
    needle = needle_probe(a.url, a.key, 4096)
    tgs = [p["tg"] for p in tg if p["tg"]]
    res = {"label": a.label, "ts": time.strftime("%F %T"), "tg_probes": tg,
           "tg_mean": sum(tgs) / len(tgs) if tgs else None,
           "tg_min": min(tgs) if tgs else None, "tg_max": max(tgs) if tgs else None,
           "pp": pp, "needle_4k": needle, "vram_mb": vram_mb()}
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "tg_probes"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
