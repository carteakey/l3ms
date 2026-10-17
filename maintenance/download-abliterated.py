#!/usr/bin/env python3
"""
Resumable HTTP Range Downloader & Pack Setup for:
SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF (IQ3_XXS)
"""
import os
import sys
import time
import subprocess
import urllib.request
import urllib.error

DEST_DIR = "/home/kchauhan/models/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/IQ3_XXS"
STRATA_RT_DIR = "/home/kchauhan/models/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/strata/rt"
SOURCE_SHARD2 = "/home/kchauhan/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"
TARGET_SHARD2 = os.path.join(DEST_DIR, "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00002-of-00002.gguf")

BASE_HF_URL = "https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/resolve/main"

DOWNLOAD_ITEMS = [
    {
        "name": "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf",
        "url": f"{BASE_HF_URL}/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf",
        "expected_size": 47342144896,
        "dest_dir": DEST_DIR,
    },
    {
        "name": "dense.bin",
        "url": f"{BASE_HF_URL}/strata/rt/dense.bin",
        "expected_size": 116099072,
        "dest_dir": STRATA_RT_DIR,
    },
    {
        "name": "dense.txt",
        "url": f"{BASE_HF_URL}/strata/rt/dense.txt",
        "expected_size": 1880,
        "dest_dir": STRATA_RT_DIR,
    },
    {
        "name": "draft_vocab.bin",
        "url": f"{BASE_HF_URL}/strata/rt/draft_vocab.bin",
        "expected_size": 425196,
        "dest_dir": STRATA_RT_DIR,
    },
    {
        "name": "experts.bin",
        "url": f"{BASE_HF_URL}/strata/rt/experts.bin",
        "expected_size": 707788800,
        "dest_dir": STRATA_RT_DIR,
    },
]

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB chunks


def ensure_link():
    os.makedirs(DEST_DIR, exist_ok=True)
    os.makedirs(STRATA_RT_DIR, exist_ok=True)
    if not os.path.exists(TARGET_SHARD2):
        if os.path.exists(SOURCE_SHARD2):
            try:
                os.link(SOURCE_SHARD2, TARGET_SHARD2)
                print(f"[LINK] Hardlinked PLE Shard 2 from:\n  {SOURCE_SHARD2}\n  -> {TARGET_SHARD2}")
            except OSError as err:
                print(f"[LINK] Hardlink failed ({err}); falling back to symlink...")
                os.symlink(SOURCE_SHARD2, TARGET_SHARD2)
                print(f"[LINK] Symlinked PLE Shard 2 from:\n  {SOURCE_SHARD2}\n  -> {TARGET_SHARD2}")
            print("[LINK] Successfully linked Shard 2 (saved 28.8 GB disk/transit).")
        else:
            print(f"[WARN] Source Shard 2 not found at {SOURCE_SHARD2}!")
    else:
        print(f"[INFO] Shard 2 already exists: {TARGET_SHARD2}")


def download_file(item):
    filename = item["name"]
    url = item["url"]
    expected_size = item["expected_size"]
    dest_dir = item["dest_dir"]
    filepath = os.path.join(dest_dir, filename)

    while True:
        current_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0

        if current_size >= expected_size:
            print(f"[COMPLETE] {filename} is fully downloaded ({current_size:,} bytes).")
            return filepath

        print(f"\n[DOWNLOAD] {filename}")
        print(f"  Target:        {filepath}")
        print(f"  Current size:  {current_size:,} bytes ({current_size / (1024**3):.2f} GB)")
        print(f"  Expected size: {expected_size:,} bytes ({expected_size / (1024**3):.2f} GB)")
        print(f"  Remaining:     {expected_size - current_size:,} bytes ({(expected_size - current_size) / (1024**3):.2f} GB)")

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
            "Range": f"bytes={current_size}-{expected_size - 1}",
        }

        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                status = resp.status
                if status not in (200, 206):
                    raise RuntimeError(f"Unexpected HTTP status {status}")

                start_time = time.time()
                last_report = start_time
                bytes_downloaded_this_session = 0

                with open(filepath, "ab") as f:
                    while True:
                        chunk = resp.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        f.flush()
                        bytes_downloaded_this_session += len(chunk)
                        now = time.time()

                        if now - last_report >= 5:
                            elapsed = now - start_time
                            speed = (bytes_downloaded_this_session / elapsed) / (1024 * 1024) if elapsed > 0 else 0
                            current_total = current_size + bytes_downloaded_this_session
                            pct = (current_total / expected_size) * 100
                            eta_sec = (expected_size - current_total) / max(speed * 1024 * 1024, 1)
                            eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s"
                            print(
                                f"  [{pct:6.2f}%] {current_total / (1024**3):.2f}/{expected_size / (1024**3):.2f} GB | "
                                f"Speed: {speed:5.1f} MB/s | "
                                f"ETA: {eta_str}",
                                end="\r",
                                flush=True,
                            )
                            last_report = now

            final_size = os.path.getsize(filepath)
            if final_size >= expected_size:
                print(f"\n[DONE] Finished {filename} ({final_size:,} bytes)")
                return filepath
            else:
                print(f"\n[INFO] Connection closed at {final_size:,} bytes. Reconnecting...")
                time.sleep(2)

        except Exception as e:
            print(f"\n[WARN] Connection interrupted: {e}. Retrying in 5 seconds...")
            time.sleep(5)


def build_strata_pack():
    shard1 = os.path.join(DEST_DIR, "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf")
    out_pack = "/home/kchauhan/repos/l3ms/vendor/Strata-data/packs/sc117_iq3_xxs"
    python_bin = "/home/kchauhan/repos/l3ms/vendor/strata/.venv/bin/python"
    iq_pack_script = "/home/kchauhan/repos/l3ms/vendor/strata-latest/tools/iq_pack.py"

    if os.path.exists(out_pack) and os.path.exists(os.path.join(out_pack, "native_experts.txt")):
        print(f"[INFO] Strata pack already exists at {out_pack}.")
        return

    print(f"\n=== Generating Strata Native Pack ===")
    print(f"  Shard 1: {shard1}")
    print(f"  Output:  {out_pack}")
    os.makedirs(out_pack, exist_ok=True)
    cmd = [python_bin, iq_pack_script, "--gguf", shard1, "--out", out_pack]
    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("[SUCCESS] Strata native pack built successfully.")
    else:
        print(f"[ERROR] Failed to build Strata pack (code {res.returncode})")


def main():
    print("=== SC117 Qwen3.8-Flash-Next Abliterated IQ3_XXS Staging ===")
    ensure_link()

    for item in DOWNLOAD_ITEMS:
        download_file(item)

    print("\nAll downloads verified.")
    build_strata_pack()
    print("Staging complete!")


if __name__ == "__main__":
    main()
