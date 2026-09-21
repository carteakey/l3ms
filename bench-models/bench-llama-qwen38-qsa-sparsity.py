#!/usr/bin/env python3
"""
QSA Sparsity Stack Benchmark Matrix for Qwen3.8-Flash-Next
Compares Baseline Master vs QSA Sparsity Stack (#28770 + #28699 + #28213)
across context depths (1k, 4k, 8k, 12k tokens).
Measures prefill throughput (t/s), decode throughput (t/s), and memory usage.
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
BIN_QSA_SPARSE = f"{REPO}/vendor/llama.cpp-pr-test-28770-28699-28213/build/bin/llama-server"

# Build prompt text scaled to varied context depths
BASE_BLOCK = (
    "In large scale language models utilizing quasi-sparse attention architectures, "
    "context processing is optimized by decoupling attention key scoring from value retrieval. "
    "The indexer compresses sequence history into summary blocks, projecting routing vectors "
    "that dynamically select sparse subsets of keys during decoding. By maintaining an incremental "
    "pooled key cache and avoiding full quadratic attention matrices, throughput is preserved across "
    "extended context lengths without losing long range dependency resolution. "
)

PROMPT_DEPTHS = {
    "1k-depth": BASE_BLOCK * 16,    # ~1000 tokens
    "4k-depth": BASE_BLOCK * 64,    # ~4000 tokens
    "8k-depth": BASE_BLOCK * 128,   # ~8000 tokens
    "12k-depth": BASE_BLOCK * 192,  # ~12000 tokens
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

def query_chat(prompt_text, max_tokens=64, temperature=0.0):
    content = f"[{time.time_ns()}] Instruction: Read the following technical context and explain the primary scaling bottleneck in one concise paragraph.\n\nContext:\n{prompt_text}"
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

def run_arm(label, binary, extra_env=None):
    print(f"\n=======================================================")
    print(f"Starting Arm: {label}")
    print(f"Binary: {os.path.basename(binary)}")
    if extra_env:
        print(f"Extra Env: {extra_env}")
    print(f"=======================================================")

    cmd = [
        "taskset", "-c", "0-11",
        binary,
        "-m", MODEL,
        "-c", "16384",
        "--parallel", "1",
        "--fit", "on", "--fit-target", "512",
        "-b", "4096", "-ub", "1024",
        "-fa", "on", "--jinja",
        "-ctk", "q8_0", "-ctv", "q8_0",
        "-t", "10", "--threads-batch", "12", "--prio", "2",
        "--lazy-mode", "on",
        "--no-warmup",
        "--host", "127.0.0.1",
        "--port", str(PORT),
    ]

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    log_f = open("/tmp/qsa-sparsity-srv.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    try:
        ok = wait_for_server(proc)
        if not ok:
            print(f"ERROR: Server failed to start for {label}!")
            with open("/tmp/qsa-sparsity-srv.log", "r") as f:
                print(f.read()[-1000:])
            return None

        # Warmup probe
        print("Warmup probe (~500 tokens)...")
        w = query_chat(PROMPT_DEPTHS["1k-depth"][:500], max_tokens=16)
        print(f"  Warmup prefill: {w['prompt_per_sec']:.1f} t/s, decode: {w['predicted_per_sec']:.2f} t/s")

        results = {"depths": {}}
        for depth_name in ["1k-depth", "4k-depth", "8k-depth", "12k-depth"]:
            prompt = PROMPT_DEPTHS[depth_name]
            r = query_chat(prompt, max_tokens=64)
            results["depths"][depth_name] = {
                "prompt_tokens": r["prompt_n"],
                "pp_pps": r["prompt_per_sec"],
                "gen_tokens": r["predicted_n"],
                "tg_pps": r["predicted_per_sec"]
            }
            print(f"  {depth_name} ({r['prompt_n']} tok context): prefill={r['prompt_per_sec']:.1f} t/s | decode={r['predicted_per_sec']:.2f} t/s ({r['predicted_n']} gen tok)")

        vram = get_vram_mib()
        results["vram_mib"] = vram
        print(f"VRAM Used: {vram} MiB")
        return results

    except Exception as e:
        print(f"Exception during arm {label}: {e}")
        with open("/tmp/qsa-sparsity-srv.log", "r") as f:
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
        ("1. Master b78a39a2f (Dense attention baseline)", BIN_MASTER, None),
        ("2. QSA Sparsity Stack (#28770 + #28699 + #28213)", BIN_QSA_SPARSE, None),
        ("3. QSA Sparsity Stack (Pooled-Cache disabled A/B)", BIN_QSA_SPARSE, {"LLAMA_QSA_NO_POOLED_CACHE": "1"}),
    ]

    all_results = {}
    for label, binary, extra_env in arms:
        res = run_arm(label, binary, extra_env)
        if res:
            all_results[label] = res

    # Output JSON and Markdown
    out_json = f"{REPO}/bench-models/logs/results/qsa_sparsity_sweep.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\n=========================================================================================")
    print("QSA SPARSITY STACK BENCHMARK SUMMARY (CONTEXT SCALING)")
    print("=========================================================================================\n")
    print("| Configuration | 1k pp (t/s) | 1k tg (t/s) | 4k pp | 4k tg | 8k pp | 8k tg | 12k pp | 12k tg | VRAM |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, res in all_results.items():
        d = res["depths"]
        p1 = d["1k-depth"]["pp_pps"]; g1 = d["1k-depth"]["tg_pps"]
        p4 = d["4k-depth"]["pp_pps"]; g4 = d["4k-depth"]["tg_pps"]
        p8 = d["8k-depth"]["pp_pps"]; g8 = d["8k-depth"]["tg_pps"]
        p12 = d["12k-depth"]["pp_pps"]; g12 = d["12k-depth"]["tg_pps"]
        vram = res["vram_mib"]
        print(f"| {label} | {p1:.1f} | {g1:.2f} | {p4:.1f} | {g4:.2f} | {p8:.1f} | {g8:.2f} | {p12:.1f} | {g12:.2f} | {vram} |")

if __name__ == "__main__":
    main()
