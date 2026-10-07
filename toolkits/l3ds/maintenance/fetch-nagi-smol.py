#!/usr/bin/env python3
"""Explicit, revision-pinned Nagi-Smol acquisition; never fetch checkpoints.

Uses the existing Hub cache without replacing it. This is acquisition only,
not inference acceptance. Run with the isolated Nagi environment's Python.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


FILES = [
    (
        "nagisanzeninz/nagi-smol-v0",
        "6d842eed3cf2ba619a7ecade77e00eed16f22c57",
        [
            ("config.snapshot.yaml", None),
            ("model.pt", "ca9c530f1d4de61485d0537806254b74bb35ae1d673a7b88a47857cdea599c51"),
        ],
    ),
    (
        "answerdotai/ModernBERT-large",
        "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
        [
            ("config.json", None),
            ("tokenizer.json", None),
            ("tokenizer_config.json", None),
            ("special_tokens_map.json", None),
            ("model.safetensors", "44510fec5d3a81a1877f225637b869495f18e55f6f23a09abb9be0acc030295f"),
        ],
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Explicitly fetch about 3.5 GB")
    args = parser.parse_args()
    if not args.download:
        print(json.dumps({"download": False, "repositories": FILES}, indent=2))
        return

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.constants import HF_HUB_CACHE

    cache_parent = Path(HF_HUB_CACHE).expanduser()
    while not cache_parent.exists():
        cache_parent = cache_parent.parent
    if shutil.disk_usage(cache_parent).free < 8 * 1024**3:
        raise SystemExit("Acquisition refused: require 8 GiB free on the Hub cache filesystem")
    evidence = []
    for repository, revision, files in FILES:
        # The existing native loader uses default revisions internally. Seed
        # only the matching main reference, not an unpinned weight download.
        if HfApi().model_info(repository).sha != revision:
            raise SystemExit("Acquisition refused: upstream main changed; review the pinned revision")
        for filename, expected_sha256 in files:
            path = Path(hf_hub_download(repository, filename, revision=revision))
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if expected_sha256 is not None and digest != expected_sha256:
                raise SystemExit("Acquisition failed: weight checksum mismatch; cache preserved")
            evidence.append({
                "repository": repository,
                "revision": revision,
                "file": filename,
                "bytes": path.stat().st_size,
                "sha256": digest,
            })
        reference = Path(hf_hub_download(repository, files[0][0], revision="main"))
        if reference.parent.name != revision:
            raise SystemExit("Acquisition refused: upstream main raced acquisition; cache preserved")
    print(json.dumps({"acquired": evidence, "inference_verified": False}, indent=2))


if __name__ == "__main__":
    main()
