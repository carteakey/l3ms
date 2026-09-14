#!/usr/bin/env python3
"""
Prompt Processing (pp / prefill) Benchmark Matrix for Qwen3.8-Flash-Next
Compares Old Gold vs Refreshed Master, layer offload (-ncmoe 46 vs 45 vs fit),
ubatch sizes, and batch thread scaling across varied prompt lengths (512, 1024, 2048 tokens).
"""

import os
import sys
import time
import json
import subprocess
import urllib.request
import urllib.error

REPO = "/home/kchauhan/repos/l3ms"
MODEL = "/home/kchauhan/models/qwen38-flash-next/AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf"
PORT = 8019

BIN_GOLD = f"{REPO}/vendor/llama.cpp-master/build/bin/llama-server-gold-9d817213a"
BIN_MASTER = f"{REPO}/vendor/llama.cpp-master/build/bin/llama-server"
BIN_PR28243 = f"{REPO}/vendor/llama.cpp-pr-test-28243/build/bin/llama-server"

# Generate non-repetitive prompt templates of ~512, ~1024, and ~2048 tokens
BASE_TEXT = (
    "In the domain of high performance local large language model inference, efficient prefill "
    "and prompt processing bandwidth are determined by the interaction between CPU memory throughput, "
    "PCIe bus utilization, GPU compute core saturation, and kernel arithmetic intensity. When mixture "
    "of experts architectures are partially offloaded across heterogeneous memory tiers, every micro-batch "
    "must amortize the memory bus bandwidth consumed by transferring active expert weights into cache. "
)

PROMPTS = {
    "pp-512": BASE_TEXT * 9,    # ~500-550 tokens
    "pp-1024": BASE_TEXT * 18,  # ~1000-1100 tokens
    "pp-2048": BASE_TEXT * 36,  # ~2000-2200 tokens
}

def get_vram_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            universal_newlines=True
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0

def wait_for_server(proc, timeout=120):
    start = time.time()
    url = f"http://127.0.0.1:{PORT}/health"
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return False
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False

def query_pp(prompt_text):
    # Unique nonce to defeat KV prefix cache
    unique_content = f"[{time.time_ns()}] Summarize the technical implications in one bullet point: {prompt_text}"
    payload = {
        "messages": [{"role": "user", "content": unique_content}],
        "max_tokens": 16,
        "temperature": 0.0
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=data_bytes,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    timings = data.get("timings", {})
    prompt_n = timings.get("prompt_n", 0)
    prompt_ms = timings.get("prompt_ms", 1)
    prompt_per_sec = timings.get("prompt_per_second", 0.0)
    if prompt_per_sec == 0 and prompt_ms > 0:
        prompt_per_sec = (prompt_n / prompt_ms) * 1000.0
    return prompt_n, prompt_ms, prompt_per_sec

def run_arm(label, binary, extra_args):
    print(f"\n=======================================================")
    print(f"Starting Arm: {label}")
    print(f"Binary: {os.path.basename(binary)}")
    print(f"Extra args: {' '.join(extra_args)}")
    print(f"=======================================================")

    cmd = [
        "taskset", "-c", "0-11",
        binary,
        "-m", MODEL,
        "-c", "16384",
        "--parallel", "1",
        "-fa", "on",
        "--jinja",
        "-ctk", "q8_0", "-ctv", "q8_0",
        "--lazy-mode", "on",
        "--no-warmup",
        "--host", "127.0.0.1",
        "--port", str(PORT),
    ] + extra_args

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ok = wait_for_server(proc)
        if not ok:
            print(f"ERROR: Server failed to start for {label}!")
            return None

        # Warmup prompt (to fault in initial pages)
        print("Warming up page cache with ~512 tok prefill...")
        w_n, w_ms, w_pps = query_pp(PROMPTS["pp-512"])
        print(f"  Warmup: {w_pps:.1f} t/s ({w_n} tok in {w_ms:.1f} ms)")

        results = {}
        for p_name in ["pp-512", "pp-1024", "pp-2048"]:
            # Run 2 trials per length to check consistency
            n1, ms1, pps1 = query_pp(PROMPTS[p_name])
            n2, ms2, pps2 = query_pp(PROMPTS[p_name])
            avg_pps = (pps1 + pps2) / 2.0
            results[p_name] = {
                "tok_count": n2,
                "trial1_pps": pps1,
                "trial2_pps": pps2,
                "avg_pps": avg_pps,
            }
            print(f"  {p_name} ({n2} tokens): T1={pps1:.1f} t/s, T2={pps2:.1f} t/s -> Avg: {avg_pps:.1f} t/s")

        vram = get_vram_mib()
        results["vram_mib"] = vram
        print(f"VRAM Used: {vram} MiB")
        return results

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(3)

def main():
    arms = [
        (
            "1. Gold (9d817213a) -ncmoe 46, ub 1024, tb 12",
            BIN_GOLD,
            ["-ngl", "99", "-ncmoe", "46", "-b", "4096", "-ub", "1024", "-t", "10", "--threads-batch", "12", "--prio", "2"]
        ),
        (
            "2. Master (b78a39a2f) -ncmoe 46, ub 1024, tb 12",
            BIN_MASTER,
            ["-ngl", "99", "-ncmoe", "46", "-b", "4096", "-ub", "1024", "-t", "10", "--threads-batch", "12", "--prio", "2"]
        ),
        (
            "3. Master -ncmoe 45 (+1 MoE on GPU), ub 1024, tb 12",
            BIN_MASTER,
            ["-ngl", "99", "-ncmoe", "45", "-b", "4096", "-ub", "1024", "-t", "10", "--threads-batch", "12", "--prio", "2"]
        ),
        (
            "4. Master --fit on --fit-target 512, ub 1024, tb 12",
            BIN_MASTER,
            ["--fit", "on", "--fit-target", "512", "-b", "4096", "-ub", "1024", "-t", "10", "--threads-batch", "12", "--prio", "2"]
        ),
        (
            "5. Master -ncmoe 45, ub 2048, tb 12",
            BIN_MASTER,
            ["-ngl", "99", "-ncmoe", "45", "-b", "4096", "-ub", "2048", "-t", "10", "--threads-batch", "12", "--prio", "2"]
        ),
        (
            "6. Master -ncmoe 45, ub 1024, tb 16 (all logical cores)",
            BIN_MASTER,
            ["-ngl", "99", "-ncmoe", "45", "-b", "4096", "-ub", "1024", "-t", "10", "--threads-batch", "16", "--prio", "2"]
        ),
    ]

    all_results = {}
    for label, binary, args in arms:
        res = run_arm(label, binary, args)
        if res:
            all_results[label] = res

    # Output JSON and Markdown
    out_json = f"{REPO}/bench-models/logs/results/pp_sweep_results.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\n=======================================================")
    print("FINAL PROMPT PROCESSING (PP) BENCHMARK SUMMARY")
    print("=======================================================\n")
    print("| Configuration | ~512 tok (t/s) | ~1024 tok (t/s) | ~2048 tok (t/s) | Mean PP (t/s) | VRAM (MiB) |")
    print("|---|---:|---:|---:|---:|---:|")
    for label, res in all_results.items():
        p512 = res["pp-512"]["avg_pps"]
        p1024 = res["pp-1024"]["avg_pps"]
        p2048 = res["pp-2048"]["avg_pps"]
        mean_pp = (p512 + p1024 + p2048) / 3.0
        vram = res["vram_mib"]
        print(f"| {label} | {p512:.1f} | {p1024:.1f} | {p2048:.1f} | **{mean_pp:.1f}** | {vram} |")

if __name__ == "__main__":
    main()
