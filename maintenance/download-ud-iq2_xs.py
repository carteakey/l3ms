#!/usr/bin/env python3
"""
Resumable HTTP Range Downloader for Qwen3.8-Flash-Next UD-IQ2_XS (Shards 1 & 2)
from unsloth/Qwen3.8-Flash-Next-GGUF.
Shard 3 (the 28.8 GB IQ4_NL PLE table) is skipped in favor of the existing Q8_0 PLE table.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

REPO = "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main"

FILES = [
    {
        "name": "Qwen3.8-Flash-Next-UD-IQ2_XS-00001-of-00003.gguf",
        "url": f"{REPO}/UD-IQ2_XS/Qwen3.8-Flash-Next-UD-IQ2_XS-00001-of-00003.gguf",
        "expected_size": 10946618,
    },
    {
        "name": "Qwen3.8-Flash-Next-UD-IQ2_XS-00002-of-00003.gguf",
        "url": f"{REPO}/UD-IQ2_XS/Qwen3.8-Flash-Next-UD-IQ2_XS-00002-of-00003.gguf",
        "expected_size": 47021008288,
    },
]

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB


def download_file(item: dict, target_dir: pathlib.Path) -> None:
    filename = item["name"]
    url = item["url"]
    expected_size = item["expected_size"]
    filepath = target_dir / filename

    while True:
        current_size = filepath.stat().st_size if filepath.exists() else 0
        if current_size >= expected_size:
            print(f"[COMPLETE] {filename} fully downloaded ({current_size:,} bytes).", flush=True)
            return

        print(f"\n[DOWNLOADING] {filename}", flush=True)
        print(f"  Current size:  {current_size:,} bytes ({current_size / (1024**3):.2f} GiB)")
        print(f"  Expected size: {expected_size:,} bytes ({expected_size / (1024**3):.2f} GiB)")
        print(f"  Remaining:     {expected_size - current_size:,} bytes ({(expected_size - current_size) / (1024**3):.2f} GiB)", flush=True)

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
            "Range": f"bytes={current_size}-{expected_size - 1}",
        }
        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status not in (200, 206):
                    raise RuntimeError(f"Unexpected HTTP status {resp.status}")

                start_time = time.time()
                last_report = start_time
                bytes_downloaded_session = 0

                with open(filepath, "ab") as f:
                    while True:
                        chunk = resp.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        f.flush()
                        bytes_downloaded_session += len(chunk)
                        now = time.time()

                        if now - last_report >= 5:
                            speed = bytes_downloaded_session / (now - start_time) / (1024 * 1024)
                            current_total = current_size + bytes_downloaded_session
                            pct = (current_total / expected_size) * 100
                            eta_sec = (expected_size - current_total) / max(speed * 1024 * 1024, 1)
                            eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s"
                            print(
                                f"  Progress: {pct:6.2f}% | "
                                f"{current_total / (1024**3):.2f}/{expected_size / (1024**3):.2f} GiB | "
                                f"Speed: {speed:5.1f} MB/s | "
                                f"ETA: {eta_str}",
                                end="\r",
                                flush=True,
                            )
                            last_report = now

            final_size = filepath.stat().st_size
            if final_size >= expected_size:
                print(f"\n[DONE] Finished {filename} ({final_size:,} bytes)", flush=True)
                return
            else:
                print(f"\n[INFO] Connection closed at {final_size:,} bytes. Reconnecting...", flush=True)
                time.sleep(2)

        except Exception as e:
            print(f"\n[WARN] Connection interrupted: {e}. Retrying in 5 seconds...", flush=True)
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="Download Qwen3.8-Flash-Next UD-IQ2_XS Shards 1 & 2")
    parser.add_argument("--dest", default="/mnt/dev/models/unsloth/UD-IQ2_XS", help="Target directory")
    args = parser.parse_args()

    dest = pathlib.Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"=== Downloading UD-IQ2_XS to {dest} ===\n")
    for item in FILES:
        download_file(item, dest)
    print("\n=== Shards 1 & 2 download complete! ===")


if __name__ == "__main__":
    main()
