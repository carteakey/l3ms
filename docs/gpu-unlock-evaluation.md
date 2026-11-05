# GPU-Unlock Evaluation: RTX 4070 (Ada Lovelace) & L3MS Stack

Evaluation of the `Zagitta/gpu-unlock` kernel-level firmware exploit for Yeti's NVIDIA GeForce RTX 4070 12 GB running under CachyOS Linux and the L3MS workbench.

Date: 2026-10-29  
Status: **Rejected / Not Recommended**

---

## 1. Executive Summary & Verdict

`gpu-unlock` is a legitimate, functioning firmware exploit that removes NVIDIA's artificial rate-limit on FP16, BF16, and TF32 tensor operations with FP32 accumulation on consumer GeForce hardware.

However, for our specific homelab environment and operational profile:
1. **Zero impact on primary LLM decode**: Strata (`qwen38-flash-next-plat`) autoregressive token generation is strictly bound by memory bandwidth, PCIe bus latency, and cache hit rates on quantized (`IQ3_XXS` / `Q8_0`) weights. It does not execute dense BF16 matrix-multiplication tensor routines.
2. **Negligible impact on LLM prefill**: Quantized model prompt processing uses integer/LUT dequantization kernels rather than dense BF16 MMA paths.
3. **Limited utility in ComfyUI**: While dense BF16 diffusion models (SDXL, DiT) could gain 20% to 35% in raw compute, our 12 GB VRAM envelope forces modern large diffusion models to run in FP8 or GGUF quants (where FP8 tensor operations are already unthrottled by NVIDIA). The card is also capped by a 165W - 200W Total Graphics Power (TGP) limit.
4. **Unacceptable operational risk**: Ada Lovelace support is classified as experimental. The tool executes an early-boot functional level reset (FLR) on the primary display adapter driving KDE Wayland (`kwin_wayland`), risking boot hangs, driver bind failures, and breakage on rolling CachyOS kernel updates.

**Verdict**: Do not install. The minor compute gains for isolated FP16 media workloads do not justify compromising host boot stability, kernel upgrade hygiene, or display reliability.

---

## 2. Technical Mechanism

Consumer GeForce GPUs (Turing, Ampere, Ada Lovelace) share underlying silicon with their enterprise and workstation counterparts (e.g., RTX 4000 Ada, RTX A-series), but NVIDIA artificially halves the issue rate of tensor instructions (`HMMA`, `MMA`) for:
- FP16 math with FP32 accumulation
- BF16 math with FP32 accumulation
- TF32 math with FP32 accumulation

The `gpu-unlock` tool bypasses this hardware rate limiter:
- **SEC2 Booter Exploit**: Uses a buffer overflow vulnerability in NVIDIA's signed SEC2 booter firmware (derived from CMP mining card reverse-engineering).
- **Early-Boot Kernel Module**: Injected into the `initramfs` before the proprietary `nvidia.ko` or `nvidia-drm.ko` modules load.
- **Speed Selector Registers**: Sets hardware registers (SS0 to `0x88888888` for full BF16 rate, SS1 to `0x8` for IMLA4).
- **Reset & Release**: Issues a PCI Functional Level Reset (FLR) to apply the register configuration, leaving the silicon unlocked before standard, unmodified NVIDIA drivers initialize and take control.
- **Volatile State**: No VBIOS flashing or driver patching is performed. The hardware resets to stock behavior upon full power-off.

---

## 3. Host Profile & Target Hardware

| Component | Host Specification |
| :--- | :--- |
| **System** | Yeti (`yeti-cachy`) |
| **OS / Kernel** | CachyOS Linux (Kernel `7.3.0-4-cachyos`, Arch-based rolling) |
| **GPU** | NVIDIA GeForce RTX 4070 (12 GB VRAM, AD104 silicon, `sm_89`, PCI ID `0000:01:00.0`) |
| **Power Target** | 165W stock cap (200W max configurable TGP) |
| **NVIDIA Driver** | `615.78.08` |
| **Display Role** | Primary display adapter powering KDE Wayland (`kwin_wayland`, PID 279378) |
| **Initramfs System** | `mkinitcpio` / CachyOS hooks |

---

## 4. Workload-by-Workload Impact Analysis

### A. LLM Token Generation / Decode (`qwen38-flash-next-plat`)
- **Impact**: **0% (No change)**
- **Reasoning & Community Findings**:
  - Autoregressive decode computes single-token vector-matrix products (GEMV, batch size = 1). This is entirely memory-bandwidth, RAM bandwidth, and PCIe transfer bound (expert streaming and NVMe PLE retrieval). Dense Tensor Core MMA instructions are not the bottleneck during single-token generation.
  - While some community commenters hypothesize that weight dequantization would benefit decode, real-world testing on Strata confirms this is false: community testers running Strata on unlocked GPUs reported zero change in decoding speed and only an isolated ~7% bump in prefill.

### B. LLM Prompt Processing / Prefill
- **Impact**: **+5% to +8%**
- **Reasoning**:
  - Prompt processing processes prompt tokens in parallel batches, converting the problem into a compute-bound GEMM.
  - However, our primary models run specialized quantized formats (`ISTA-DASLab GSQ-RCO IQ3_XXS` and `Q8_0` PLE) with custom integer/LUT dequantization and K8V4 KV cache.
  - Strata-specific community benchmarks verify approximately a 7% prefill improvement at best on unlocked setups.

### C. ComfyUI Media Generation (Images & Video)
- **Impact**: **+10% to +35% depending on model architecture and precision**
- **Reasoning & Architecture Nuance (Ampere vs. Ada)**:
  - **The Ampere vs. Ada distinction**: Much of the community enthusiasm (+30% to +55%) comes from RTX 30-series (Ampere) owners. Ampere lacks native FP8 tensor cores, forcing FP8/INT8 diffusion models to upcast intermediate activations into BF16/FP16 compute paths where the half-rate limit applied.
  - **Ada Lovelace (RTX 4070)**: 4th-generation Tensor Cores feature native FP8 execution (`E4M3`, `E5M2`) with FP32 accumulation that is not throttled in the same manner.
  - **Mixed-precision pipeline reality**: In diffusion pipelines (MiniMax H3, LTX-Video, SDXL), even when the core DiT/UNet weights are quantized (INT8/FP8), peripheral stages like cross-attention projections, text encoders (T5/CLIP), and VAE decoders frequently remain in BF16 or FP16. Community reports on quantized video/image diffusion (MiniMax H3, LTX 2.5) show practical end-to-end gains of ~10% on quantized models and up to ~30% on full BF16 models.
  - **The Power Limit Wall**: The RTX 4070 is constrained by a 165W stock TGP cap (200W max configurable). As community testing highlights, uncapping tensor core issue rates causes rapid power ramp. On power-constrained cards, thermal and power throttling quickly eat into theoretical throughput gains.

### D. Native Decision Models (`d3_server` / Clef / vLLM-D3)
- **Impact**: **0% (No change)**
- **Reasoning**: Decision models evaluate scoring logits across single ubatches (`-c = -b = -ub`) and do not execute sustained dense BF16 training passes.

### E. Model Fine-Tuning / LoRA Training
- **Impact**: **+25% to +45% (Theoretical)**
- **Reasoning**: Backward passes in BF16 LoRA training with FP32 accumulation directly leverage the unlocked GEMM lanes. However, Yeti functions strictly as an inference and media-serving workbench; local fine-tuning is not part of this host's daily workload.

---

## 5. Operational & Stability Risks

1. **Experimental Status on Ada**: The upstream repository classifies Ampere (RTX 30-series) as stable, but Ada Lovelace (RTX 40-series) is explicitly designated as **Experimental**.
2. **Display Server Collisions on Wayland**: Because Yeti's RTX 4070 acts as the active display adapter driving `kwin_wayland`, initiating a PCI FLR during boot between firmware initialization and `nvidia-drm` binding introduces a high probability of black screens, display timing issues, or boot hangs.
3. **Rolling Kernel Breakage**: CachyOS tracks bleeding-edge kernel releases (currently `7.3.0-4-cachyos`). Out-of-tree initramfs modules that hook early hardware initialization frequently require manual intervention across kernel upgrades.
4. **Conflict with Homelab Operational Guidelines**: `AGENTS.md` mandates deterministic script orchestration, high operational ergonomics, and stability for served models. Introducing an experimental, boot-time kernel exploit violates the stability requirements of the production host.

---

## 6. Re-evaluation Gate

This decision should only be revisited if:
1. Upstream `gpu-unlock` advances Ada Lovelace (RTX 40-series) to stable status with verified, problem-free display-card FLR on Wayland systems.
2. The host workload shifts significantly toward local BF16 fine-tuning or high-volume BF16 diffusion generation.
3. Headless server migration isolates the GPU entirely from desktop display server responsibilities.
