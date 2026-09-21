#!/usr/bin/env python3
"""
Direct Reads Benchmark Matrix for Qwen3.8-Flash-Next
Compares Upstream Master (--lazy-mode on) vs PR #29030 (--lazy-mode on vs --lazy-mode on-direct).
Measures cold vs warm prefill throughput, decode throughput, disk read_bytes, and output bit-identity.
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

BIN_MASTER = f"{REPO}/vendor/llama.cpp-master/build/bin/llama-server"
BIN_PR29030 = f"{REPO}/vendor/llama.cpp-pr-test-29030/build/bin/llama-server"

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

def get_process_io_read_bytes(pid):
    try:
        with open(f"/proc/{pid}/io", "r") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0

def wait_for_server(proc, timeout=180):
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

def query_chat(prompt_text, max_tokens=16, temperature=0.0, unique=True):
    content = f"[{time.time_ns()}] Summarize: {prompt_text}" if unique else prompt_text
    payload = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": 42
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=data_bytes,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    text_content = message.get("content", "")
    reasoning_content = message.get("reasoning_content", "")

    timings = data.get("timings", {})
    prompt_n = timings.get("prompt_n", 0)
    prompt_ms = timings.get("prompt_ms", 1)
    prompt_per_sec = timings.get("prompt_per_second", 0.0)
    if prompt_per_sec == 0 and prompt_ms > 0:
        prompt_per_sec = (prompt_n / prompt_ms) * 1000.0

    predicted_n = timings.get("predicted_n", 0)
    predicted_ms = timings.get("predicted_ms", 1)
    predicted_per_sec = timings.get("predicted_per_second", 0.0)
    if predicted_per_sec == 0 and predicted_ms > 0:
        predicted_per_sec = (predicted_n / predicted_ms) * 1000.0

    return {
        "prompt_n": prompt_n,
        "prompt_ms": prompt_ms,
        "prompt_per_sec": prompt_per_sec,
        "predicted_n": predicted_n,
        "predicted_ms": predicted_ms,
        "predicted_per_sec": predicted_per_sec,
        "text": text_content,
        "reasoning": reasoning_content
    }

def run_arm(label, binary, lazy_mode, ubatch=2048):
    print(f"\n=======================================================")
    print(f"Starting Arm: {label}")
    print(f"Binary: {os.path.basename(binary)}")
    print(f"Lazy Mode: {lazy_mode} | UBatch: {ubatch}")
    print(f"=======================================================")

    cmd = [
        "taskset", "-c", "0-11",
        binary,
        "-m", MODEL,
        "-c", "16384",
        "--parallel", "1",
        "--fit", "on", "--fit-target", "512",
        "-b", "4096", "-ub", str(ubatch),
        "-fa", "on", "--jinja",
        "-ctk", "q8_0", "-ctv", "q8_0",
        "-t", "10", "--threads-batch", "12", "--prio", "2",
        "--lazy-mode", lazy_mode,
        "--no-warmup",
        "--host", "127.0.0.1",
        "--port", str(PORT),
    ]

    log_f = open("/tmp/direct-reads-srv.log", "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    pid = proc.pid
    try:
        ok = wait_for_server(proc)
        if not ok:
            print(f"ERROR: Server failed to start for {label}!")
            with open("/tmp/direct-reads-srv.log", "r") as f:
                print(f.read()[-1000:])
            return None

        # 1. Cold Probe (First prompt immediately after load)
        io_before = get_process_io_read_bytes(pid)
        cold_res = query_chat(PROMPTS["pp-512"], max_tokens=16, temperature=0.0)
        io_after = get_process_io_read_bytes(pid)
        cold_io_mb = (io_after - io_before) / (1024 * 1024)
        print(f"  [Cold] pp-512 ({cold_res['prompt_n']} tok): {cold_res['prompt_per_sec']:.1f} t/s | Disk Read: {cold_io_mb:.2f} MB")

        # 2. Sweep across prompt sizes (Warm / Steady-state)
        results = {
            "cold_512_pps": cold_res["prompt_per_sec"],
            "cold_512_io_mb": cold_io_mb,
            "prompts": {}
        }

        for p_name in ["pp-512", "pp-1024", "pp-2048"]:
            io_b = get_process_io_read_bytes(pid)
            r1 = query_chat(PROMPTS[p_name], max_tokens=16, temperature=0.0)
            r2 = query_chat(PROMPTS[p_name], max_tokens=16, temperature=0.0)
            io_a = get_process_io_read_bytes(pid)
            io_mb = (io_a - io_b) / (1024 * 1024)

            avg_pps = (r1["prompt_per_sec"] + r2["prompt_per_sec"]) / 2.0
            results["prompts"][p_name] = {
                "tok_count": r2["prompt_n"],
                "t1_pps": r1["prompt_per_sec"],
                "t2_pps": r2["prompt_per_sec"],
                "avg_pps": avg_pps,
                "read_mb": io_mb
            }
            print(f"  [Warm] {p_name} ({r2['prompt_n']} tok): T1={r1['prompt_per_sec']:.1f} t/s, T2={r2['prompt_per_sec']:.1f} t/s -> Avg: {avg_pps:.1f} t/s (I/O: {io_mb:.2f} MB)")

        # 3. Decode & Output Bit-Identity Verification (64 tokens)
        fixed_prompt = "Write a fast Python function to compute Fibonacci numbers."
        dec_res = query_chat(fixed_prompt, max_tokens=64, temperature=0.0, unique=False)
        results["decode_tg"] = dec_res["predicted_per_sec"]
        results["output_sample"] = (dec_res["text"] + dec_res["reasoning"])[:100]
        print(f"  [Decode] 64-tok gen: {dec_res['predicted_per_sec']:.2f} t/s")

        vram = get_vram_mib()
        results["vram_mib"] = vram
        print(f"VRAM Used: {vram} MiB")
        return results

    except Exception as e:
        print(f"Exception during arm {label}: {e}")
        with open("/tmp/direct-reads-srv.log", "r") as f:
            print("--- Server log tail ---")
            print("".join(f.readlines()[-30:]))
        return None
    finally:
        proc.terminate()
        log_f.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(3)

def main():
    arms = [
        ("1. Master b78a39a2f (--lazy-mode on, ub 2048)", BIN_MASTER, "on", 2048),
        ("2. PR #29030 (--lazy-mode on, ub 2048)", BIN_PR29030, "on", 2048),
        ("3. PR #29030 (--lazy-mode on-direct, ub 2048)", BIN_PR29030, "on-direct", 2048),
        ("4. PR #29030 (--lazy-mode on-direct, ub 1024)", BIN_PR29030, "on-direct", 1024),
    ]

    all_results = {}
    for label, binary, lazy_mode, ubatch in arms:
        res = run_arm(label, binary, lazy_mode, ubatch)
        if res:
            all_results[label] = res

    # Output JSON and Markdown
    out_json = f"{REPO}/bench-models/logs/results/direct_reads_sweep.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\n=======================================================")
    print("DIRECT READS (--lazy-mode on-direct) BENCHMARK SUMMARY")
    print("=======================================================\n")
    print("| Configuration | Cold ~512 (t/s) | Warm ~512 | Warm ~1024 | Warm ~2048 | Decode tg (t/s) | VRAM (MiB) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for label, res in all_results.items():
        cold = res["cold_512_pps"]
        p512 = res["prompts"]["pp-512"]["avg_pps"]
        p1024 = res["prompts"]["pp-1024"]["avg_pps"]
        p2048 = res["prompts"]["pp-2048"]["avg_pps"]
        dec = res["decode_tg"]
        vram = res["vram_mib"]
        print(f"| {label} | {cold:.1f} | {p512:.1f} | {p1024:.1f} | {p2048:.1f} | {dec:.2f} | {vram} |")

if __name__ == "__main__":
    main()
