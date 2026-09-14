#!/usr/bin/env python3
"""
Test feasibility of ub=2048 across:
1. Gold 64k (--fit on --fit-target 512, -b 4096 -ub 2048)
2. MTP 16k (-ncmoe 45, -b 4096 -ub 2048, shared-Q4_K_M head)
3. Vision 16k (-ncmoe 45, -b 2048 -ub 1024 or 2048, mmproj-F16)
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
MTP_HEAD = "/home/kchauhan/models/unsloth/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf"
MMPROJ = "/home/kchauhan/models/unsloth/Qwen3.8-Flash-Next-GGUF/mmproj-F16.gguf"
PORT = 8019

BIN_MASTER = f"{REPO}/vendor/llama.cpp-master/build/bin/llama-server"
BIN_PR28243 = f"{REPO}/vendor/llama.cpp-pr-test-28243/build/bin/llama-server"

def get_vram_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            universal_newlines=True
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0

def test_config(name, cmd):
    print(f"\n==================================================")
    print(f"Testing: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"==================================================")
    log_file = f"/tmp/test_{name.replace(' ', '_').lower()}.log"
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    try:
        ok = False
        for _ in range(60):
            if proc.poll() is not None:
                print(f"Process terminated with code {proc.returncode}!")
                break
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health")
                with urllib.request.urlopen(req, timeout=1) as resp:
                    if resp.status == 200:
                        ok = True
                        break
            except Exception:
                pass
            time.sleep(1)

        if not ok:
            print("FAILED TO START. Log tail:")
            with open(log_file) as f:
                lines = f.readlines()
                print("".join(lines[-15:]))
            return False, 0, 0, 0

        vram = get_vram_mib()
        print(f"Successfully loaded! VRAM: {vram} MiB (Headroom: {12282 - vram} MiB)")

        # Run a test prompt: ~500 tokens in, 32 tokens out
        content = f"[{time.time_ns()}] Write a technical paragraph analyzing memory bandwidth in MoE inference."
        req_data = {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 32,
            "temperature": 0.0
        }
        data_bytes = json.dumps(req_data).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            data=data_bytes,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
        timings = resp_data.get("timings", {})
        pps = timings.get("prompt_per_second", 0.0)
        tps = timings.get("predicted_per_second", 0.0)
        print(f"Query Result: PP={pps:.1f} t/s, TG={tps:.1f} t/s")
        return True, vram, pps, tps

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(2)

def main():
    cases = [
        (
            "1. Gold 64k Fit-On ub=2048",
            [
                "taskset", "-c", "0-11", BIN_MASTER,
                "-m", MODEL,
                "--fit", "on", "--fit-target", "512",
                "-c", "65536", "--parallel", "1",
                "-b", "4096", "-ub", "2048",
                "-fa", "on", "--jinja",
                "-ctk", "q8_0", "-ctv", "q8_0",
                "-t", "10", "--threads-batch", "12", "--prio", "2",
                "--lazy-mode", "on", "--no-warmup",
                "--host", "127.0.0.1", "--port", str(PORT)
            ]
        ),
        (
            "2. MTP 16k ncmoe=45 ub=2048",
            [
                "taskset", "-c", "0-11", BIN_PR28243,
                "-m", MODEL,
                "--spec-draft-model", MTP_HEAD,
                "-ngl", "99", "-ncmoe", "45",
                "-c", "16384", "--parallel", "1",
                "-b", "4096", "-ub", "2048",
                "-fa", "on", "--jinja",
                "-ctk", "q8_0", "-ctv", "q8_0",
                "-t", "10", "--threads-batch", "12", "--prio", "2",
                "--lazy-mode", "on",
                "--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-p-min", "0.7",
                "--spec-draft-ngl", "99",
                "--no-warmup",
                "--host", "127.0.0.1", "--port", str(PORT)
            ]
        ),
        (
            "3. MTP 16k ncmoe=45 ub=1024",
            [
                "taskset", "-c", "0-11", BIN_PR28243,
                "-m", MODEL,
                "--spec-draft-model", MTP_HEAD,
                "-ngl", "99", "-ncmoe", "45",
                "-c", "16384", "--parallel", "1",
                "-b", "4096", "-ub", "1024",
                "-fa", "on", "--jinja",
                "-ctk", "q8_0", "-ctv", "q8_0",
                "-t", "10", "--threads-batch", "12", "--prio", "2",
                "--lazy-mode", "on",
                "--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-p-min", "0.7",
                "--spec-draft-ngl", "99",
                "--no-warmup",
                "--host", "127.0.0.1", "--port", str(PORT)
            ]
        ),
        (
            "4. Vision 16k ncmoe=45 ub=1024",
            [
                "taskset", "-c", "0-11", BIN_MASTER,
                "-m", MODEL,
                "--mmproj", MMPROJ,
                "-ngl", "99", "-ncmoe", "45",
                "-c", "16384", "--parallel", "1",
                "-b", "2048", "-ub", "1024",
                "-fa", "on", "--jinja",
                "-ctk", "q8_0", "-ctv", "q8_0",
                "-t", "10", "--threads-batch", "12", "--prio", "2",
                "--lazy-mode", "on", "--no-warmup",
                "--host", "127.0.0.1", "--port", str(PORT)
            ]
        ),
    ]

    for name, cmd in cases:
        test_config(name, cmd)

if __name__ == "__main__":
    main()
