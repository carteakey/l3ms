#!/usr/bin/env python3
"""Minimal GGUF v3 header parser: print tensor types grouped by role."""
import struct, sys
from collections import Counter

TYPES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",
         10:"Q2_K",11:"Q3_K_S",12:"Q3_K_M",13:"Q3_K_L",14:"Q4_K_S",15:"Q4_K_M",
         16:"Q5_K_S",17:"Q5_K_M",18:"Q6_K",19:"IQ2_XXS",20:"IQ2_XS",21:"Q2_K_S",
         22:"IQ3_XS",23:"IQ3_XXS",24:"IQ1_S",25:"IQ4_NL",26:"IQ3_S",27:"IQ3_M",
         28:"IQ2_S",29:"IQ2_M",30:"IQ4_XS",31:"IQ1_M",32:"BF16",33:"Q4_0_4_4",
         34:"Q4_0_4_8",35:"Q4_0_8_8",36:"TQ1_0",37:"TQ2_0",38:"IQ4_NL"}

class R:
    def __init__(self, f): self.f = f
    def u8(self): return struct.unpack("<B", self.f.read(1))[0]
    def u32(self): return struct.unpack("<I", self.f.read(4))[0]
    def u64(self): return struct.unpack("<Q", self.f.read(8))[0]
    def i32(self): return struct.unpack("<i", self.f.read(4))[0]
    def f32(self): return struct.unpack("<f", self.f.read(4))[0]
    def f64(self): return struct.unpack("<d", self.f.read(8))[0]
    def s(self):
        n = self.u64(); return self.f.read(n).decode("utf-8", "replace")
    def val(self, t):
        if t == 0: return self.u8()
        if t == 1: return struct.unpack("<b", self.f.read(1))[0]
        if t == 2: return struct.unpack("<H", self.f.read(2))[0]
        if t == 3: return struct.unpack("<h", self.f.read(2))[0]
        if t == 4: return self.u32()
        if t == 5: return self.i32()
        if t == 6: return self.f32()
        if t == 7: return bool(self.u8())
        if t == 8: return self.s()
        if t == 9:
            et = self.u32(); n = self.u64()
            return [self.val(et) for _ in range(n)]
        if t == 10: return self.u64()
        if t == 11: return struct.unpack("<q", self.f.read(8))[0]
        if t == 12: return self.f64()
        raise ValueError(f"kv type {t}")

def main(path):
    with open(path, "rb") as fh:
        r = R(fh)
        assert fh.read(4) == b"GGUF", "not GGUF"
        ver = r.u32()
        n_tensors, n_kv = r.u64(), r.u64()
        for _ in range(n_kv):
            r.s(); r.val(r.u32())
        c = Counter(); widths = {}
        for _ in range(n_tensors):
            name = r.s()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            ty = r.u32(); r.u64()
            if "ffn" in name and "exps" in name: k = "routed-experts"
            elif "per_layer_token_embd" in name: k = "PLE-table"
            elif "nextn" in name or name.startswith("blk.48"): k = "MTP-head"
            else: k = "other"
            c[(k, TYPES.get(ty, f"t{ty}"))] += 1
            if k == "routed-experts" and dims:
                widths[dims[0]] = widths.get(dims[0], 0) + 1
        print(f"{path.split('/')[-1]}  (gguf v{ver}, {n_tensors} tensors)")
        for (k, ty), n in sorted(c.items()):
            print(f"  {k:15s} {ty:9s} x{n}")
        if widths:
            print("  expert ncols histogram:", dict(sorted(widths.items(), reverse=True)))

if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
