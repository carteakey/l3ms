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


def deep_probe(url, key, approx_tokens, n):
    """Decode speed at a deep context, not prefill speed at one.

    A --kv-grow style elastic K/V spends expert slots while a long context is live, so the number that
    matters is decode throughput after a 98K-token prompt is sitting in the cache, not pp on the way in.
    The prompt is unique per call (timestamp prefix) so the KV is genuinely re-prefilled.
    """
    reps = max(1, approx_tokens // 28)
    r = post(url, key, {"messages": [{"role": "user",
                                      "content": f"[{time.time_ns()}] Summarize in one sentence, then continue: "
                                                 + FILLER * reps}],
                        "max_tokens": n, "temperature": 0, "cache_prompt": False})
    tm = r.get("timings", {})
    dn, da = tm.get("draft_n", 0), tm.get("draft_n_accepted", 0)
    return {"tg": tm.get("predicted_per_second"), "pp": tm.get("prompt_per_second"),
            "n": tm.get("prompt_n"), "out": tm.get("predicted_n"),
            "draft_acc": (da / dn) if dn else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--warmups", type=int, default=6)
    ap.add_argument("--tg-probes", type=int, default=4)
    ap.add_argument("--tg-tokens", type=int, default=256)
    ap.add_argument("--pp-sizes", default="2048,8192")
    ap.add_argument("--deep-sizes", default="", help="deep-context decode probes, e.g. 98304 (off by default)")
    ap.add_argument("--meta", default="", help="JSON object merged into the row as config (engine, args, env, commit)")
    a = ap.parse_args()

    for i in range(a.warmups):
        decode_probe(a.url, a.key, CODE_PROMPTS[i % len(CODE_PROMPTS)], a.tg_tokens)
    tg = [decode_probe(a.url, a.key, CODE_PROMPTS[i % len(CODE_PROMPTS)], a.tg_tokens) for i in range(a.tg_probes)]
    pp = {}
    for s in [int(x) for x in a.pp_sizes.split(",") if x]:
        pp[s] = [prefill_probe(a.url, a.key, s) for _ in range(2)]
    deep = {}
    for s in [int(x) for x in a.deep_sizes.split(",") if x]:
        deep[s] = [deep_probe(a.url, a.key, s, 256) for _ in range(2)]
    needle = needle_probe(a.url, a.key, 4096)
    tgs = [p["tg"] for p in tg if p["tg"]]
    # Steady state is the last probes, not the average of all of them. On 2026-10-28 a Strata arm
    # still climbing at its third measurement probe (70.0, 76.1, 85.1 t/s, draft acceptance climbing
    # with it: 0.680, 0.715, 0.828) out-read an identical arm 30 minutes later (67.2, 68.7, 72.4),
    # so tg_mean on a short probe run measures how far along the warm-up each arm happened to be.
    steady = tgs[-3:] if len(tgs) >= 3 else tgs
    res = {"label": a.label, "ts": time.strftime("%F %T"), "config": json.loads(a.meta) if a.meta else {}, "tg_probes": tg,
           "tg_mean": sum(tgs) / len(tgs) if tgs else None,
           "tg_steady_mean": sum(steady) / len(steady) if steady else None,
           "tg_steady_spread": (max(steady) - min(steady)) if steady else None,
           "tg_min": min(tgs) if tgs else None, "tg_max": max(tgs) if tgs else None,
           "pp": pp, "deep": deep, "needle_4k": needle, "vram_mb": vram_mb()}
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "tg_probes"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
