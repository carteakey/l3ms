#!/usr/bin/env python3
"""
MTP + QSA Sparsity Benchmark Matrix for Qwen3.8-Flash-Next
Compares Production MTP (PR #28243 baseline) vs Unified QSA Sparsity + MTP Stack
(#28770 + #28699 + #28213 + #28243) across the standardized 6-prompt unique corpus.
"""

import os
import re
import sys
import time
import json
import subprocess
import urllib.request
import urllib.error

REPO = "/home/kchauhan/repos/l3ms"
MODEL = "/home/kchauhan/models/qwen38-flash-next/AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf"
HEAD = "/home/kchauhan/models/unsloth/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf"
PROMPTS_FILE = f"{REPO}/bench-models/prompts/qwen38-flash-next-unique.jsonl"
PORT = 8027

BIN_PROD_MTP = f"{REPO}/vendor/llama.cpp-pr-test-28243/build/bin/llama-server"
BIN_QSA_MTP = f"{REPO}/vendor/llama.cpp-pr-test-28770-28699-28213/build/bin/llama-server"

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

def query_chat(prompt_text, max_tokens=192, temperature=0.0):
    payload = {
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": 123
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=data_bytes,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))

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
        "predicted_per_sec": predicted_per_sec
    }

def run_arm(label, binary, ncmoe=45, n_max=2, p_min=0.7):
    print(f"\n=======================================================")
    print(f"Starting Arm: {label}")
    print(f"Binary: {os.path.basename(binary)}")
    print(f"Flags: ncmoe={ncmoe}, n_max={n_max}, p_min={p_min}")
    print(f"=======================================================")

    cmd = [
        "taskset", "-c", "0-11",
        binary,
        "-m", MODEL,
        "--spec-draft-model", HEAD,
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", str(n_max),
        "--spec-draft-p-min", str(p_min),
        "--spec-draft-ngl", "99",
        "-ngl", "99", "-ncmoe", str(ncmoe),
        "-c", "16384",
        "--parallel", "1",
        "-b", "2048", "-ub", "512",
        "-fa", "on", "--jinja",
        "-ctk", "q8_0", "-ctv", "q8_0",
        "-t", "10", "--threads-batch", "12", "--prio", "2",
        "--lazy-mode", "on",
        "--no-warmup",
        "--host", "127.0.0.1",
        "--port", str(PORT),
    ]

    env = os.environ.copy()
    env["GGML_CUDA_NO_PINNED"] = "1"
    env["GGML_CUDA_GRAPH_OPT"] = "1"

    log_path = f"/tmp/mtp-qsa-{label.replace(' ', '_')}.log"
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    try:
        ok = wait_for_server(proc)
        if not ok:
            print(f"ERROR: Server failed to start for {label}!")
            with open(log_path, "r") as f:
                print(f.read()[-1000:])
            return None

        vram = get_vram_mib()
        print(f"Server is UP. Baseline VRAM: {vram} MiB")

        # 2 warmups
        print("Warming up expert residency & MTP cache (2 probes)...")
        w1 = query_chat("Count from 1 to 40, one number per line.", max_tokens=64)
        w2 = query_chat("Count from 1 to 40, one number per line.", max_tokens=64)
        print(f"  Warmup 1: {w1['predicted_per_sec']:.2f} t/s | Warmup 2: {w2['predicted_per_sec']:.2f} t/s")

        # Read the 6 tasks
        tasks = []
        with open(PROMPTS_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))

        task_results = {}
        total_tokens = 0
        total_ms = 0.0

        for t in tasks:
            tname = t["name"]
            prompt = t["prompt"]
            max_tok = t.get("max_tokens", 192)

            res = query_chat(prompt, max_tokens=max_tok)
            task_results[tname] = {
                "prompt_tokens": res["prompt_n"],
                "pp_tps": res["prompt_per_sec"],
                "gen_tokens": res["predicted_n"],
                "dec_tps": res["predicted_per_sec"],
                "dec_ms": res["predicted_ms"]
            }
            total_tokens += res["predicted_n"]
            total_ms += res["predicted_ms"]
            print(f"  [{tname}]: dec={res['predicted_per_sec']:5.2f} t/s ({res['predicted_n']} tok in {res['predicted_ms']/1000:.2f}s) | pp={res['prompt_per_sec']:5.1f} t/s")

        agg_tps = (total_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0
        peak_vram = get_vram_mib()
        print(f"\n==> {label} AGGREGATE DECODE: {agg_tps:.2f} t/s across {total_tokens} tokens ({total_ms/1000:.2f}s) | Peak VRAM: {peak_vram} MiB")

        # Parse acceptance from log
        acceptance_info = []
        with open(log_path, "r") as f:
            for line in f:
                if "draft acceptance" in line:
                    acceptance_info.append(line.strip())
        last_acc = acceptance_info[-1] if acceptance_info else "N/A"
        print(f"Draft acceptance summary: {last_acc}")

        return {
            "tasks": task_results,
            "aggregate_tps": agg_tps,
            "total_tokens": total_tokens,
            "total_ms": total_ms,
            "peak_vram": peak_vram,
            "last_acceptance": last_acc
        }

    except Exception as e:
        print(f"Exception during {label}: {e}")
        with open(log_path, "r") as f:
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
        time.sleep(4)

def main():
    arms = [
        ("1. Production MTP Baseline (PR #28243)", BIN_PROD_MTP, 45, 2, 0.7),
        ("2. Unified QSA Sparsity + MTP Stack", BIN_QSA_MTP, 45, 2, 0.7),
    ]

    all_results = {}
    for label, binary, ncmoe, n_max, p_min in arms:
        res = run_arm(label, binary, ncmoe, n_max, p_min)
        if res:
            all_results[label] = res

    # Write results JSON
    out_json = f"{REPO}/bench-models/logs/results/mtp_qsa_comparison.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\n=========================================================================================================")
    print("MTP + QSA SPARSITY STACK PERFORMANCE COMPARISON (6-PROMPT UNIQUE CORPUS)")
    print("=========================================================================================================\n")
    header = f"| Task | {'Production MTP (#28243)':<25} | {'Unified QSA + MTP Stack':<25} | Delta |"
    print(header)
    print("|---|---:|---:|---:|")

    task_keys = ["code-pathlib", "code-rust-lru", "code-sql-batch", "story-radio", "story-library", "story-orchard"]
    
    prod_data = all_results.get("1. Production MTP Baseline (PR #28243)", {}).get("tasks", {})
    qsa_data = all_results.get("2. Unified QSA Sparsity + MTP Stack", {}).get("tasks", {})

    for tk in task_keys:
        pt = prod_data.get(tk, {}).get("dec_tps", 0.0)
        qt = qsa_data.get(tk, {}).get("dec_tps", 0.0)
        delta_pct = ((qt - pt) / pt * 100.0) if pt > 0 else 0.0
        delta_str = f"{delta_pct:+.1f}%"
        print(f"| `{tk}` | {pt:5.2f} t/s | {qt:5.2f} t/s | **{delta_str}** |")

    prod_agg = all_results.get("1. Production MTP Baseline (PR #28243)", {}).get("aggregate_tps", 0.0)
    qsa_agg = all_results.get("2. Unified QSA Sparsity + MTP Stack", {}).get("aggregate_tps", 0.0)
    agg_delta = ((qsa_agg - prod_agg) / prod_agg * 100.0) if prod_agg > 0 else 0.0
    print(f"| **AGGREGATE** | **{prod_agg:5.2f} t/s** | **{qsa_agg:5.2f} t/s** | **{agg_delta:+.1f}%** |")
    
    prod_vram = all_results.get("1. Production MTP Baseline (PR #28243)", {}).get("peak_vram", 0)
    qsa_vram = all_results.get("2. Unified QSA Sparsity + MTP Stack", {}).get("peak_vram", 0)
    print(f"| **PEAK VRAM** | {prod_vram} MiB | {qsa_vram} MiB | {qsa_vram - prod_vram:+d} MiB |")

if __name__ == "__main__":
    main()
