# Qwen3.8-Flash-Next (qwen4exp) Model Variant Ranking & Pitch

Internal reference and evaluation guide ranking 30 community and upstream variants
of `Qwen3.8-Flash-Next` against our specific hardware and serving architecture.

Author: Antigravity / L3MS  
Date: 2026-10-21  
Target Rig: RTX 4070 12 GB (Ada SM89) + i5-12600K (10c/16t AVX2) + 64 GB DDR5-5600 + Gen4 NVMe  
Serving Engines: Strata v0.1.45 (Platinum tier) & llama.cpp master / exp-stack-v2 (Gold / MTP tiers) via `llama-swap`  
Companion Docs:
- `docs/qwen38-flash-next-internal.md` (canonical serving baseline and MTP benchmarks)
- `docs/strata-evaluation-internal.md` (Strata dynamic expert caching and memory layout)
- `AGENTS.md` (operational rules and hardware thresholds)

---

## 1. System Hardware & Architectural Filter

Qwen3.8-Flash-Next is a 177B–180B parameter Mixture-of-Experts (MoE) architecture with
an active parameter footprint of ~3.8B per token, backed by a massive Per-Layer token
Embd (PLE) n-gram routing table (~28.8 GB in Q8/FP16).

On our host (`yeti-cachy`), hardware sets hard physical boundaries:
- **Fast memory ceiling**: 12 GB VRAM + ~60 GB usable host RAM (~72 GB total).
- **PLE isolation requirement**: The PLE routing table must reside on SSD via lazy mmap
  or remain pinned in page cache without claiming fast expert RAM. If a model attempts
  to load the full unquantized PLE into VRAM or forces non-PLE weights above ~55 GB, host
  memory exhausts, triggering page thrashing where decode drops from 60 t/s down to 3–5 t/s.
- **Strata acceleration**: Strata achieves **59–65 t/s steady-state decode** (pp=1,100–2,000 t/s)
  by dynamically caching 2,444–3,086 hot experts in 11.5 GB VRAM with native MTP speculative
  decoding and K8V4 KV cache @ 128k context. This requires GSQ-RCO 2-shard format.
- **Incompatible formats**: NVFP4 (requires Blackwell or multi-GPU Ada via TensorRT-LLM),
  pure FP8 (180 GB requires dual A100/H100), and unpruned BF16 (~360 GB) cannot run.

---

## 2. Tiered Model Ranking

### Tier S: Production Daily Drivers (Optimal for Strata)

These utilize modular sharding with fast host RAM expert residency, while the PLE routing table
(54.4 GB in unquantized Q8_0) streams from NVMe on demand.

| Rank | Model | Size / Format | Verified Throughput & Operational Role |
| :---: | :--- | :--- | :--- |
| **1** | `unsloth/Qwen3.8-Flash-Next-GGUF` (Dynamic V3.1) | 177B GGUF (`UD-IQ2_XS`, 75.8 GB) | **Primary daily driver (Active Platinum Tier).** Deployed 2026-11-05. Paired with unquantized Q8_0 PLE table. 39.4 GB host RAM working set (18+ GB free headroom) + 2,711 expert pairs in 4.33 GiB VRAM. Delivers **59.7 tok/s steady decode** (peaking at 62.3 tok/s) with **77.0%–88.3% MTP draft acceptance**. Scores +11.5 points on DeepSWE v1.1 (58.4 vs 46.9) over GSQ-RCO baseline. |
| **2** | `SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF` | 177B GGUF (`IQ3_XXS`) | **Primary uncensored driver.** Takes the GSQ quant and strips refusal vectors while preserving reasoning tracks. Paired with Q8_0 PLE table. Powers `qwen38-flash-next-plat-abliterated-vision` without refusal loops on edge-case code or multimodal inputs. |
| **3** | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | 177B GGUF (`IQ3_XXS`) | **Former baseline champion (Archived).** Delivered 70 tok/s steady decode at 43.8 GB fast RAM. Replaced 2026-11-05 by `UD-IQ2_XS` to gain +11.5 points on DeepSWE code reasoning and 4.4 GB more RAM headroom. |
| **4** | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` | 117B/177B GGUF (`IQ1_M`) | **Deep-context coding driver.** Shard 1 is compressed to 29.6 GB. Shard 2 (PLE) is 100% byte-identical to `IQ3_S` Shard 2 (hardlinkable on disk). Frees over 30 GB host RAM, enabling **full 262k context** without paging. |

---

### Tier A: Specialized Strata Builds & Fine-Tunes

Drop-in compatible with the Strata 2-shard pipeline, offering fine-tuned checkpoints
(Swift-1.5 multi-turn instruction following) or custom community repacks.

| Rank | Model | Size / Format | Operational Evaluation |
| :---: | :--- | :--- | :--- |
| **5** | `SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF` | 177B GGUF | Combines Alibaba's Swift-1.5 fine-tune (sharper multi-turn agent adherence) with SC117 abliteration and GSQ-RCO routing. Drop-in replacement for agent workloads. |
| **6** | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | 177B GGUF | Stock alignment version of Swift-1.5 GSQ-RCO. Strong instruction following, but retains standard Qwen alignment guardrails. |
| **7** | `jimmy31chen/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored-IQ3_S-Strata` | 177B GGUF (`IQ3_S`) | Pre-packaged community Strata build in `IQ3_S`. Slightly higher precision than `IQ3_XXS` (~47 GB Shard 1), fitting under our 60 GB RAM ceiling up to 32k context. |
| **8** | `WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3` | GGUF (`IQ3_S`) | Calibrated routing variant tuned for custom inference engines; follows standard Strata 2-shard conventions. |
| **9** | `alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF` | GGUF | Another repack of SC117 for Strata. Operationally identical to SC117 upstream. |

---

### Tier B: llama.cpp Workhorses (Gold, Vision & MTP Tiers)

These serve as our fallbacks when testing upstream PRs or running native `llama-server`.
They deliver **20.5–22.1 t/s steady decode** (plain) or **25.3–27.1 t/s** (with MTP speculative
decoding). They require `-ot "per_layer_token_embd\.weight=CPU"` with mmap to isolate the PLE.

| Rank | Model | Size / Format | Operational Evaluation |
| :---: | :--- | :--- | :--- |
| **10** | `unsloth/Qwen3.8-Flash-Next-GGUF` (Legacy V3) | 177B GGUF | **llama.cpp reference build.** Powers our Gold and Vision tiers (`qwen38-flash-next-vision`). We extract `mmproj-F16.gguf` from this repository for multimodal support. Decode runs at 20.9 t/s steady state. |
| **11** | `pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF` | 180B GGUF | Integrated MTP draft head repack. Works in llama.cpp MTP branches, though our Daniel Han 1.78 GB head setup (`mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`) has tighter VRAM margins (11,025 MiB). |
| **12** | `AtomicChat/Qwen3.8-Flash-Next-GGUF` | 177B GGUF (AD-4.27bpw) | **Former baseline champion.** 33 shards with dedicated Shard 33 for the PLE. Exceptional quantization quality, but at 54.5 GB RAM-side weights it thrashes if desktop apps compete for RAM. Archived to NAS in favor of Strata. |
| **13** | `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF` | 177B GGUF | Uncensored merge in standard GGUF layout. Works cleanly in llama.cpp, but lacks GSQ routing calibration. |
| **14** | `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF` | 177B GGUF | Clean abliteration, but packaged as monolithic shard files. Requires careful `-ot` mmap flags to avoid paging lockups. |
| **15** | `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF` (v1) | 177B GGUF | Older revision of the Huihui abliteration; superseded by v2. |
| **16** | `agentionai/Qwen3.8-Flash-Next-Gyro-GGUF` | 177B GGUF | Fine-tuned for agent tool calling and structured JSON outputs. Runs on master, but capped at 21 t/s without Strata kernel support. |
| **17** | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF` | 177B GGUF | Swift-1.5 without GSQ-RCO routing optimizations. Sub-optimal compared to ukisai's GSQ-RCO release. |

---

### Tier C: Sub-Optimal / Uncalibrated Community GGUFs

Viable to load in llama.cpp, but suffer from missing routing calibration, improper shard
boundaries, or broken speculative drafting.

| Rank | Model | Size / Format | Operational Evaluation |
| :---: | :--- | :--- | :--- |
| **17** | `bartowski/Qwen3.8-Flash-Next-GGUF` | 177B GGUF | Standard automated GGUF quant. Lacks GSQ routing calibration for MoE weights, leading to higher perplexity per bit. |
| **18** | `mradermacher/Qwen3.8-Flash-Next-Uncensored-i1-GGUF` | 177B GGUF | Automated imatrix conversion. Calibration dataset does not balance Flash-Next MoE routing gates evenly. |
| **19** | `mradermacher/Huihui-Qwen3.8-Flash-Next-abliterated-i1-GGUF` | 177B GGUF | Same imatrix limitations on Huihui weights. |
| **20** | `mradermacher/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF` | 177B GGUF | Mradermacher standard quant without imatrix; high quantization loss at sub-4-bit. |
| **21** | `antirez/qwen3.8-flash-next-gguf` | 188B GGUF | Early conversion by Salvatore Sanfilippo. Useful historical artifact, but lacks current GGUF metadata conventions. |
| **22** | `sh0wie/Qwen3.8-Flash-Next-REAP-288-GGUF` | 124B GGUF | Pruned MoE using REAP pruning. While smaller, pruning ruins attention geometry and breaks MTP draft acceptance rates. |

---

### Tier F: Incompatible / Hardware Mismatch (Cannot Run on 12 GB VRAM + 64 GB RAM)

These models target multi-GPU clusters, 80 GB enterprise accelerators, or Blackwell-native
hardware. Attempting to run them on our single RTX 4070 rig results in hard OOM or failure to initialize.

| Rank | Model | Format / Size | Hard Blocker on Our System |
| :---: | :--- | :--- | :--- |
| **23** | `Qwen/Qwen3.8-Flash-Next` (Base) | BF16 / 180B (~360 GB) | Exceeds fast memory by 5x. Cannot load. |
| **24** | `Qwen/Qwen3.8-Flash-Next-FP8` | FP8 / 180B (~180 GB) | Requires minimum dual 80 GB enterprise GPUs. Hard OOM. |
| **25** | `orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8` | FP8 / 180B (~180 GB) | Same FP8 memory exhaustion. |
| **26** | `nvidia/Qwen3.8-Flash-Next-NVFP4` | NVFP4 / 120B | Requires TensorRT-LLM and Blackwell-native FP4 tensor cores. Cannot offload via llama.cpp or Strata. |
| **27** | `RadixArk/Qwen3.8-Flash-Next-NVFP4` | NVFP4 / 120B | Same NVFP4 execution barrier. |
| **28** | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` | NVFP4 / 93B | Pruned NVFP4 checkpoint. Same architectural blocker. |
| **29** | `Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata` | 51B Hybrid | Experimental NVFP4/GPTQ format; unsupported kernels on desktop Ada via CPU-GPU offload. |
| **30** | `davetha/Qwen3.8-Flash-Next-DERISKED-W4A16-AWQ` | AWQ / 21B | Highly pruned AWQ built for pure GPU vLLM serving on 24 GB+ cards. Does not offload across host RAM. |

---

## 3. Storage & Operational Strategy for L3MS

1. **Keep Tier S as Default**: Maintain `ISTA-DASLab` (`IQ3_XXS`) as the primary boot model in `llama-swap.yaml` (`qwen38-flash-next-plat`), with `SC117` as the uncensored/vision driver (`qwen38-flash-next-plat-abliterated-vision`).
2. **Hardlink Shared PLE Shards**: When pulling `Coder` (`IQ1_M`), hardlink its Shard 2 to the existing `IQ3_S` Shard 2 (`/mnt/dev/models/ISTA-DASLab/...`):
   ```bash
   ln /mnt/dev/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
      /mnt/dev/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-00002-of-00002.gguf
   ```
   Both files are byte-identical (28,800,138,432 bytes), saving 28.8 GB of NVMe space.
3. **Avoid Pure FP8 / NVFP4 Downloads**: Filter automated scripts and downloader configs to reject any repo tagged `NVFP4`, `FP8`, or single-blob GGUF without PLE separation.
