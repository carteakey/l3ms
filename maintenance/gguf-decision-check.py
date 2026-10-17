#!/usr/bin/env python3
"""Check whether a remote GGUF is a native llama.cpp decision model, without downloading it.

Reads only the KV metadata via HTTP range requests and prints the architecture,
`<arch>.decision.*` keys, causal-attention and MoE hints. A GGUF without
`<arch>.decision.type` is a plain LLM conversion (the decision head was lost) and
`/v1/systemone` will reject it with 501.

Usage:
  maintenance/gguf-decision-check.py <hf-repo> [<file.gguf>]
  maintenance/gguf-decision-check.py https://huggingface.co/<repo>/resolve/main/<file.gguf>
With only a repo, the smallest weights file (no mmproj/imatrix) is checked.
"""
import json, struct, sys, urllib.request

HF = "https://huggingface.co"
SIZES = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
FMTS = {0:"<B", 1:"<b", 2:"<H", 3:"<h", 4:"<I", 5:"<i", 6:"<f", 7:"<?", 10:"<Q", 11:"<q", 12:"<d"}
CHUNK = 4 << 20


class RangeReader:
    def __init__(self, url): self.url = url; self.buf = b""; self.pos = 0
    def take(self, n):
        while self.pos + n > len(self.buf):
            req = urllib.request.Request(self.url, headers={
                "Range": "bytes=%d-%d" % (len(self.buf), len(self.buf) + CHUNK - 1)})
            self.buf += urllib.request.urlopen(req, timeout=60).read()
        out = self.buf[self.pos:self.pos + n]; self.pos += n
        return out
    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def s(self): return self.take(self.u64()).decode("utf-8", "replace")
    def val(self, t):
        if t == 8: return self.s()
        if t == 9:
            et = self.u32(); n = self.u64()
            if et == 8:
                for _ in range(n): self.s()
            else:
                self.take(SIZES[et] * n)
            return f"<array[{n}]>"
        return struct.unpack(FMTS[t], self.take(SIZES[t]))[0]


def pick_file(repo):
    with urllib.request.urlopen(f"{HF}/api/models/{repo}/tree/main", timeout=60) as r:
        files = [f for f in json.load(r) if f.get("path", "").endswith(".gguf")
                 and "mmproj" not in f["path"].lower() and "imatrix" not in f["path"].lower()]
    if not files:
        sys.exit(f"no weights .gguf in {repo}")
    return min(files, key=lambda f: f["size"])["path"]


def main(argv):
    if not argv:
        sys.exit(__doc__)
    if argv[0].startswith("http"):
        url = argv[0]
    else:
        url = f"{HF}/{argv[0]}/resolve/main/{argv[1] if len(argv) > 1 else pick_file(argv[0])}"
    r = RangeReader(url)
    if r.take(4) != b"GGUF":
        sys.exit("not a GGUF file")
    r.u32(); r.u64(); n_kv = r.u64()
    meta = {}
    for _ in range(n_kv):
        k = r.s(); v = r.val(r.u32())
        if (k in ("general.architecture", "general.name") or ".decision." in k
                or k.endswith((".causal_attention", ".expert_count", ".expert_used_count"))):
            meta[k] = v
    arch = meta.get("general.architecture", "?")
    dtype = meta.get(f"{arch}.decision.type")
    print(url)
    print(json.dumps(meta, indent=2))
    print(f"decision model: {'yes (' + dtype + ')' if dtype else 'NO - plain LLM conversion'}")
    return 0 if dtype else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
