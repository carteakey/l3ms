# RTX 4070 12 GB Inference Roofline (`yeti-cachy`)

Measured hardware baselines and theoretical inference ceilings computed using `.agents/skills/inference-optimisation/scripts/ceiling.py` on `yeti-cachy` (NVIDIA RTX 4070 12 GB `sm_89`, Intel i5-12600K `performance` governor, 64 GB DDR5, CUDA 13.5, driver 620.73.04).

---

## 1. Measured Silicon Baselines (2026-11-24)

Measured with `llama-swap.service` stopped (idle GPU, `30 MiB` baseline VRAM) using a native `sm_89` CUDA + OpenMP microbenchmark:

| Hardware Path | Spec Sheet | Measured (`yeti-cachy`) | Notes |
| :--- | ---: | ---: | :--- |
| **VRAM Bandwidth (D2D copy, 512 MiB)** | 504.2 GB/s | **455.8 GB/s** | 90.4% of theoretical GDDR6X peak |
| **PCIe Gen4 x16 H2D Pinned Copy (256 MiB)** | ~28 GB/s | **25.0 GB/s** | Hard floor for host-to-GPU expert streaming |
| **CPU DDR5 Read Bandwidth (12 threads)** | ~76 GB/s | **60.0 GB/s** | Hard floor for AVX-VNNI CPU-offloaded expert misses |
| **FP16 Tensor Core GEMM (F32 accum)** | ~116 TFLOPS (sparse) / ~58 dense | **60.5 TFLOPS** | `4096^3`: 59.9 TFLOPS; `8192^3`: 60.5 TFLOPS |
| **GPU Geometry** | 46 SMs, 36 MiB L2 | `11.64 GiB` usable VRAM | `sm_89` (Ada Lovelace) |

---

## 2. Inkling 2 Mini (`UD-Q4_K_XL`, 16.4 GiB) Ceiling

Exact per-token weight traffic parsed from `/mnt/media/models/unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf` across all 48 layer visits (`6 + 3x12 + 6`, 8 of 128 experts active per visit):

- **Dense weights (48 visits):** `1.785 GB/tok` raw F32 controllers -> **`1.58 GB/tok`** with `--loop-bf16`
- **Active routed experts (48 visits, 8/128):** **`1.847 GB/tok`** (`14.71 GiB` total stored experts; `29.55 GB` visit-weighted)
- **LM head (`output.weight`, Q8_0):** **`0.492 GB/tok`**
- **Total active weight bytes/token (short ctx):** **`3.92 GB/tok`** (`4.125 GB/tok` without `--loop-bf16`)
- **KV cache (8 full-attention visits, FP8 E4M3):** **`16.5 KiB/tok`** (`4.43 GB` at 256K)

### `ceiling.py` Output (Assuming 100% All-in-VRAM, Zero Misses, Zero Launch Overhead)

| Context | All-in-VRAM Plain Ceiling (`drafts=0`) | All-in-VRAM + DSpark (`3 -> 2.8 acc`) | Practical FP16 Prefill Ceiling (50% MFU) |
| :--- | ---: | ---: | ---: |
| **4K** | 114 tok/s | 119 tok/s | 2,750 tok/s |
| **20K** | 107 tok/s | 116 tok/s | 2,562 tok/s |
| **64K** | 91 tok/s | 108 tok/s | 2,157 tok/s |
| **128K** | 74 tok/s | 99 tok/s | 1,754 tok/s |
| **256K** | 55 tok/s | 84 tok/s | 1,277 tok/s |

### Real 12 GB VRAM Ceiling vs Measured `PolyStrata` (`polystrata-4070`)

Because `16.4 GiB` of Q4 weights physically overflows `11.64 GiB` of VRAM, PolyStrata holds **1,809 of 3,072 experts in VRAM (8.1 GiB, ~95% hit rate)** and pins the remaining **1,263 experts (6.9 GiB)** in DDR5 RAM:

- **Real 12 GB plain-decode budget at 4K:**
  - VRAM reads (`3.83 GB` @ `455.8 GB/s`): `8.40 ms`
  - 5% offloaded expert misses (`92.4 MB` @ `60.0 GB/s` DDR5, partially overlapped): `~1.2-2.0 ms` net
  - Kernel launch + host-GPU doorbell floor: `~2.0-3.0 ms`
  - **Physical 12 GB plain-decode ceiling:** **~83-95 tok/s**
  - **Measured `PolyStrata` (`polystrata-4070`):** **74.65 tok/s** (peak **76.9 tok/s**, **~79-90% of real 12 GB ceiling**).
- **Real 12 GB prefill budget at 2K-20K:**
  - Every `--prompt-batch 2048` chunk must stream `6.9 GiB` of uncached experts over PCIe Gen4 (`7.41 GB / 25.0 GB/s = 296 ms` minimum transfer floor) alongside compute and FlashAttention.
  - **Default FP16 halves NVFP4:** `751 / 1,135 / 1,234 tok/s` (`2K / 8K / 20K`) - bottlenecked on `60.5 TFLOPS` FP16 math.
  - **With `STRATA_PREFILL_NVFP4=w4a8` + `STRATA_INK_FAST_CTRL=1` (Ada INT8 Tensor Cores):** **`1,483 / 2,036 / 2,153 tok/s`** (`2K / 8K / 20K`) - **84% of the 2,562 tok/s practical FP16 compute ceiling** (now faster on prefill than Qwen3.8-Flash-Next's 1,439-1,888 tok/s) and right near the joint PCIe `25.0 GB/s` + INT8 compute ceiling!

---

## 3. Why Qwen3.8-Flash-Next (~180B `UD-IQ2_XS`) Matches Inkling 2 Mini (26B `UD-Q4_K_XL`)

From `vendor/Strata-data/packs/ud-iq2_xs/`:
- `dense.bin`: **1.5 GB**
- Total experts (`48 layers x 512 experts`): **42.28 GB** (`~0.66-0.83 GB` active experts per token at 2-bit IQ2_XS)
- **Total active bytes/token:** only **`~2.2-2.3 GB/tok`** - **almost half of Inkling 2 Mini's `3.92 GB/tok`!**
- Even though Qwen3.8-Flash-Next has a higher expert miss rate (~28% misses vs Inkling's ~5% on 12 GB), its 2-bit experts transfer fewer bytes per miss AND native MTP speculative decoding amortizes each verify step across ~2.5-3.0 accepted tokens, landing at **73.0 tok/s** steady decode and **1,439-1,888 tok/s** `pp8k`.

---

## 4. What Is Actually Left in the Bag on RTX 4070 12 GB?

1. **Inkling 2 Mini `UD-Q4_K_XL` plain decode:** Nearly tapped out (~10-15% theoretical room left). Writing a custom engine from zero cannot escape the `455.8 GB/s` VRAM + `60 GB/s` DDR5 wall without changing the quant or adding lossless drafting that doesn't evict GPU experts.
2. **Breaking past 100 tok/s on Inkling 2 Mini (12 GB):** Only possible by dropping to a **~10 GiB IQ3/Q2 quant** so 100% of weights + DSpark fit in VRAM (eliminating PCIe/DDR5 entirely).
3. **Inkling 2 Mini long prefill:** Roughly **25-40% theoretical headroom** (~1,317 -> ~1,700-1,900 tok/s) if a future engine achieves true dual-stream PCIe H2D overlap without shrinking the resident expert cache.
4. **Inkling 2 Mini 512K-1M context:** Waiting on PolyStrata to extend `--kv-resident` / `--kv q4_0` (or `cudaMallocManaged` host KV) from `qwen35moe` to `inkling2`.
