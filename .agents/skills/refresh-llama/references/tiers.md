# L3MS Llama Router Tiers & Build Reference

This document outlines the four router tiers configured in `llama-swap.yaml` for `Qwen3.8-Flash-Next` on the homelab host (`yeti-cachy`: RTX 4070 12 GB, 64 GB DDR5-5600, PCIe 4.0 NVMe).

---

## 1. Tier Specifications

| Tier ID | Role | Source Tree | Binary Macro | Key Optimizations | Peak VRAM |
|---|---|---|---|---|---|
| `qwen38-flash-next` | **Gold** (Production Base) | `vendor/llama.cpp-master` | `qwen38_master_server` | Upstream master (sparse Flash Attention #28770, kernel fusions #28896/#28901) | ~11.2 GB |
| `qwen38-flash-next-mtp` | **MTP** (Speculative) | `vendor/llama.cpp-pr-test-28770-28699-28213` | `qwen38_mtp_server` | Unified QSA Sparsity (#28770, #28699, #28213, #29166) + Daniel Han MTP (#28243) with compact `shared-Q4_K_M` head | ~11.8 GB |
| `qwen38-flash-next-exp` | **Exp** (Experimental) | `vendor/llama.cpp-pr-test-28770-28699-28213` | `qwen38_exp_server` | Unified QSA Sparsity Stack (#28770, #28699, #28213, #29166) without MTP | ~11.2 GB |
| `qwen38-flash-next-vision` | **Vision** (Multimodal) | `vendor/llama.cpp-master` | `qwen38_master_server` | Master + `mmproj-F16.gguf` projector, `-ncmoe 45` | ~10.4 GB |

---

## 2. Shared Serving & Hardware Constraints

1. **Hardware Bounds**:
   - Single discrete GPU with 12,282 MiB VRAM. Models claim ~11.2–11.8 GiB.
   - 64 GB DDR5-5600 host memory holds the ~46 GB routed MoE expert weights (`-ncmoe 45`).
   - The ~38 GB PLE (per-layer embedding) n-gram table stays on NVMe SSD via mmap (`--lazy-mode on`).
2. **Build Flags**:
   ```bash
   cmake -B build \
     -DGGML_CUDA=ON \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_CUDA_ARCHITECTURES=89 \
     -DGGML_CUDA_FA_ALL_QUANTS=ON
   ```
3. **Compilation Target Rule**:
   - Always build specific targets: `cmake --build build --config Release --target llama-server llama-bench -j$(nproc)`.
   - **Do not** build `all` or generic `cmake --build build`: GCC 16 LTO (`lto1`) crashes with an internal compiler error on `tests/test-fusion`.
4. **Smoke Testing Rule**:
   - Always pass `max_tokens >= 64` when testing reasoning models through `llama-swap`.
   - Small token budgets land completely inside `<think>` tags and return empty `content` (with text in `reasoning_content`), which is not an error.
