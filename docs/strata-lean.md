# Strata `strata-lean` (`carteakey/strata-lean`)

This branch (`strata-lean`) is a streamlined, minimal-bloat fork of Strata tailored specifically for an **RTX 4070 12 GB (Ada Lovelace, `sm_89`, 46 SMs)** paired with 64 GB DDR5-5200 host RAM running **Qwen3.8-Flash-Next** (`qwen4exp`, ISTA-DASLab GSQ-RCO IQ3_XXS with unquantized Q8_0 PLE table).

Rather than tracking every upstream release or pulling in multi-vendor bloat, this fork cherry-picks only high-value inference kernels, speculative decoding improvements, and CUDA optimizations while pruning unused architectures.

---

## Performance Progression on RTX 4070 (12 GB)

Measured with `bench-models/bench-qwen38-refresh-ab.sh` on Linux 7.3.0 CachyOS, powersave governor, DDR5-5200 4x16 GB, serving `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/` (`IQ3_XXS`, unquantized Q8_0 PLE table on NVMe):

| Engine / Revision | Sustained Decode (t/s) | Peak Decode (t/s) | 8k Prefill (t/s) | 4k Needle Pass | VRAM Footprint |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Strata v0.1.45 (upstream tag)** | 63.83 | 67.7 | 1,926.1 | Yes (`true`) | 11,696 MiB |
| **Strata v0.2.6 (upstream tag)** | 65.53 | 70.5 | 1,890.7 | Yes (`true`) | 11,696 MiB |
| **`4070-main-lite` (initial pruned + HC fusions)** | 65.08 | 72.9 | 1,894.1 | Yes (`true`) | 11,698 MiB |
| **`4070-main-lite` + Ada `sm_89` MMVQ (PR #3453)** | **69.60** | **76.1** | **1,883.1** | Yes (`true`) | **11,696 MiB** |
| **`4070-main-lite` + `--kv-grow` elastic cache** | **73.53** | **78.9** | **1,888.0** | Yes (`true`) | **11,696 MiB** |

Net gain: **+9.70 t/s (+15.2%) over upstream v0.1.45**, with peak decode reaching **78.9 t/s** (3,277 expert slots on GPU, 62.1% hit rate).

Read that table with two caveats. A repeat run the same day came in at **67.88 t/s** (63.0 /
71.6), so run-to-run spread on this tier is about 2 t/s and anything under ~3 t/s needs a
paired control arm in the same boot. And 8k prefill has not improved: 1,883 here against
1,926 on plain v0.1.45 and 1,911 on v0.1.44.1. Those prefill cells are the higher probe of
each run's two pp8192 probes, and each pair spans 1,869 to 1,883 (fork + table), 1,911 to 1,926
(v0.1.45) and 1,908 to 1,914 (v0.1.44.1), i.e. means of 1,876 / 1,919 / 1,911. Decode is where
this fork won, prefill is where it has not kept up.

---

## Fork Modifications & Architecture Changes

### 1. Pruned Bloat & Build Streamlining
- **Removed Unused Backends (331 files / 172,853 LOC removed)**:
  - Deleted `src/kernels/metal/` (macOS Apple Silicon).
  - Deleted `src/kernels/sycl/` (Intel Arc).
  - Deleted AMD ROCm tuning scripts (`src/kernels/rocm/hipblaslt_tuning.hpp`, `src/kernels/rocm/rocblas_tuning.hpp`).
- **Gated Experimental Engines & Tests**:
  - Gated unreleased Qwen3.5 engine (`src/qwen35/`) behind `STRATA_ENABLE_QWEN35=OFF`.
  - Disabled 109 test binaries from standard build (`STRATA_BUILD_TESTS=OFF`).
  - Set default `CMAKE_CUDA_ARCHITECTURES=89` targeting Ada Lovelace.
- **Binary Footprint**:
  - `build/strata` executable reduced to **73.0 MB** (down from 77.5 MB on v0.1.45 and 78.7 MB on v0.2.6).

### 2. Tracked Inference Commits & Cherry-Picks
- **`8645328d` - Fused Hyper-Connections (`gr.cu`, `fused_gr.cu`)**:
  - Folded the hyper-connection inject and gate/mean operations into adjacent GEMV down grids (`fused_gm = hc <= 4`), eliminating two standalone kernel dispatches per verification step (6 launches reduced to 4).
- **`7ce60ad3` - `STRATA_GR_NORM_FAST=1` Register-Split Norm**:
  - Ported register-split norm kernel on CUDA producing bitwise-identical output while trimming register pressure.
- **`9aafcbf0` - Reasoning Loop Recovery (`serve/server.py`)**:
  - Added loop detection recovery: when `repeat_stop_tokens` fires inside `<think>`, the server wraps thinking into the final answer rather than failing the generation. Verified with 19/19 unit tests passing.
- **`0cef00fa` - Ada Lovelace `sm_89` MMVQ Interleave Table (Strata PR #3453)**:
  - Added hardware-calibrated warp row counts (`kIlRows_89[]`, `kIl1Rows_89[]`) for `IQ4_XS`, `Q4_K`, `Q5_K`, `Q6_K`, and `Q3_K` in `src/kernels/cuda/native_mmvq.cu`.
  - Yielded a direct +4.52 t/s (+7.0%) sustained decode improvement (65.08 -> 69.60 t/s).
- **`1bd03954` - QSA Single-Pass Fused RMSNorm + RoPE (`STRATA_QSA_FUSE_NR`)**:
  - Replaced 3 separate kernel launches (`split_q` + `rms_rows` + `rope`) for query, key, and indexer queries with a single CUDA pass (`rms_rope_rows`).
  - Cuts memory traffic from ~1.1 GB to ~0.4 GB per QSA layer at 8k tokens. Includes an assembly register fence (`asm volatile("" : "+v"(ss));`) to guarantee bitwise mathematical equivalence with separate kernels.
- **`3733343d` - Strict Mathematical Routing Preserved (`STRATA_ROUTE_TAIL_SKIP=0` Default)**:
  - Defaulted `STRATA_ROUTE_TAIL_SKIP` to OFF (0) in `strata-lean`, overriding upstream's default 7. Prevents dropping 7th and 8th ranked expert misses during verify windows, avoiding KL 0.003-0.006 output drift.

### 3. Evaluated & Rejected Experiments
- **`--mtp-q4 head` (On-the-fly Q4_0 draft head quantization)**:
  - Tested quantizing the MTP draft head from Q8_0 down to Q4_0 to reduce DRAM read traffic during drafting.
  - Result: Decode dropped from 69.60 t/s to 67.88 t/s. On-the-fly ALU dequantization in the draft step cost more latency than the memory bandwidth saved on an RTX 4070. Reverted to keep native Q8_0 draft head.

---

## Precision, Exactness & Quality-Speed Trade-Offs

This section documents where numerical precision or sampling policies are traded for speed, and which upstream shortcuts I explicitly rejected.

### Active Approximations & Trade-Offs

- **CPU Prefill Sharing (`STRATA_PREFILL_CPU_SHARE`) [Active by Default]**:
  - During prompt ingestion for chunks under 1,024 tokens, idle CPU threads evaluate un-cached expert rows in RAM (AVX2 FP32) concurrently while the GPU evaluates resident experts on tensor cores.
  - CPU AVX2 floating-point arithmetic rounds slightly differently from GPU Ada tensor cores, introducing a mean KL divergence of ~0.004 (up to 0.026 nats max) against an all-GPU prefill.
  - Setting `STRATA_PREFILL_CPU_SHARE=0` disables CPU sharing completely, forcing all expert evaluation onto the GPU for 100% bit-exact prefill.
- **KV Cache Quantization (`--kv k8v4`) [Active]**:
  - Stores K-cache in FP8 (`E4M3`) and V-cache in INT4 with randomized Hadamard rotation (`fwht256`).
  - Uses 1.32 GiB at 128k context (leaving 5.35 GiB of VRAM to hold 3,277 hot experts), compared to ~4.8 GiB for FP16 KV.
  - Hadamard rotation eliminates outlier distortion, keeping KL divergence small (~0.0008 vs FP16), but it remains a lossy quantization compared to full FP16 KV.
- **Model Card Sampling vs Greedy Decoding [Active, Greedy Avoided]**:
  - The runtime config (`strata-iq3_xxs-ple_q8.json`) enforces Qwen's official recommended sampling (`temperature: 1.0`, `top_p: 0.95`, `top_k: 20`, `min_p: 0.05`).
  - Pure greedy decoding (`temperature: 0` / argmax) is faster (no multinomial draw step), but small and quantized reasoning models fall into repetitive degenerate loops under greedy argmax.
- **Thinking Loop Recovery (`reasoning_loop_recovery: "recover"`) [Active]**:
  - If the model generates a repeating pattern longer than 48 tokens inside a `<think>` block, the server wraps thinking into the final response to prevent infinite loops.

### Rejected Upstream Shortcuts & Guarded Features

| Feature / Switch | Upstream Claim | Measured Result & Reason for Rejection |
| :--- | :--- | :--- |
| **`STRATA_ROUTE_TAIL_SKIP=7`** | +10-20% decode on 12-16 GB cards | **Disabled by default in `strata-lean`**. Upstream enabled this by default in 0.1.56, skipping rank >= 7 expert misses during verify windows. Causes KL ~0.003-0.006 output drift. |
| **`STRATA_ROUTE_PRIOR=1.5`** | Measured 102.23 t/s (93% cache hit rate) | **Rejected**. Re-weights routing probabilities towards cached experts. Upstream confirmed agent benchmark completion dropped from 5.6/10 down to 4.5/10. |
| **`STRATA_GR_NORM_FAST`** | Faster group norm on sm_70 | **Disabled**. Measured 64.85 t/s vs 71.13 base (-6.3 t/s slower on Ada) and lower precision. |
| **`STRATA_QSA_SPLIT_ATTN=1`** | +2.8-3.5% decode | **Kept Off**. Rounding-level KL drift of 0.0016 on INT8 MMA attention. |
| **`STRATA_HC_REQ8=1`** | +2% decode | **Kept Off**. INT8 quantized hyper-connections; introduces KL drift of 0.0018. |
| **Speculative Decoding (`--spec 4`)** | +60-80% decode speedup | **Mathematically Exact**. MTP draft tokens are strictly validated against target model logits via rejection sampling. Zero output quality loss. |

### 4. What the tier reports about itself at load

From `strata-iq3_xxs-ple_q8.log` (2026-10-28 boot), the numbers that decide where the next
win has to come from:

| line | value | reading |
| :--- | :--- | :--- |
| `expert cache 2583 slots, 4.24 GiB of VRAM` | 2,583 of 24,576 expert pairs | the shipped profile's head decides which 2,583 are hot |
| `the prompt path borrows 2323 CUDA0 cache slots` | 3.81 GiB | prompt reading and decode contend for the same VRAM |
| `serve: 410 MiB of VRAM free with everything loaded` | below the 512 MiB floor | the engine warns about driver paging to system RAM (`#2065`), and asks for `--vram-reserve-mib 832` |
| window captures | down to 282 MiB free | the CUDA graph windows are what push it under the floor |
| `decode expert cache hit rate: 58.6%` | 54,487 hits / 92,979 lookups | 41% of routed experts are computed off GPU |
| `3652 more read by the GPU over PCIe (3.8%)` | the rest is CPU pool work | decode on this box is partly a CPU job |
| `5 pool workers on logical processors 2,4,6,8,10` | mask `cpu_range: 0-11` | the 4 Alder Lake E-cores (12 to 15) sit outside the mask, idle |
| `MAP_HUGETLB unavailable (needed 20464 2 MiB pages, vm.nr_hugepages=0)` | arena on 4 KB pages | upstream prices ~750 TLB misses per 3 MB expert on this path |
| `mtp: draft head over 106299 tokens (178.4 MiB)` | shipped `rt` subset | `data/draft_vocab_en.bin` would take it to 40,525 tokens |

### 5. Candidate backlog (measured order, not a wish list)

Full reasoning, citations and the harness: `docs/strata-perf-next-steps-2026-10.md` in the
L3MS tree. Short version, in the order I would spend card time on them:

Measured on 2026-10-28 in the L3MS sweep harness (nine arms, 23 min, the production config run at
both ends as the paired control: **70.70 and 71.55 t/s**, pair mean 71.13). Numbers are decode mean
over four 256-token code probes; hit rate and PCIe share come from the engine's own lines.

1. **`STRATA_ROUTE_PRIOR=1.5`** (`src/core/verify.cpp:1849-1851`, `src/kernels/cuda/route_prior.cu`):
   residency-biased routing. Measured **102.23 t/s** (97.4 to 109.3), +31.1 over the pair mean, with the
   cache hit rate at **93.0%** against 57.4 and 57.7 in the two control arms and PCIe down to **1.3%**
   from 9.9%, with expert slots identical (2,583) and draft acceptance essentially at the control's level
   (accepted/proposed mean 0.831 against 0.794). Not served yet: the kernel and
   the call site both say it changes the output, and upstream's 0.2.7 release note flags agent-task
   completion (4.5 of 10 at 1.5 against 5.6 with it off, inside their spread). Upstream defaults 1.5 on
   a discrete NVIDIA card at or under 13 GiB and used 2.0 in its first candidates, which showed copy and
   code defects. This fork is on the 0.1.45 base, so nothing turns it on by itself here.
2. **VRAM headroom**: `--kv-grow` measured **73.53** (+2.40, inside the ~2 t/s spread) with slots 2,583 to
   **3,277** and hit rate 62.1%, but the run's worst disk traffic (0.0998 MB per generated token) and a
   32k prefill dip (1,928 against 1,950 and 1,948): needs a paired repeat before it is served.
   `--vram-reserve-mib 832` measured **68.75** (-2.38) while doing its job (410 to 544 MiB free, capture
   floor 276 to 410) at the cost of 77 slots: insurance, not speed. `--max-context 65536` unrun.
3. **A workload-shaped `--expert-profile`** built from my own coding-agent prompts:
   `tools/collect_profile_traces.py CONFIG.json OUTDIR --mode routing` (CUDA needs `--mode
   routing`; run it with the Strata venv, the tool imports `regex`), then `tools/make_profile.py
   OUTDIR/*.trace --reorder --out data/expert-profile-pi.bin` so the traces decide the head and
   the shipped base fills the tail. `--reorder` and `--no-base` belong to `make_profile.py`;
   `make_profile_mix.py` in this tree weights whole traces against each other and takes no base
   flags.
4. **`--mtp-draft-vocab data/draft_vocab_en.bin`**, measured and rejected: 40,525 tokens and 68.0 MiB
   against 106,299 and 178.4 MiB, 50 extra auto slots (2,003 against 1,953), and **5.35 t/s slower** on
   code with acceptance mean 0.768 against 0.794. A subset worth retrying is code-plus-English built with
   `tools/draft_vocab.py --corpus ... --coverage 0.98`, not the shipped English-only file.
5. **Speculative window**: `--spec 5` measured **61.00** (-10.13) with acceptance mean 0.763 against the
   control's 0.794, so the loss is the wider verification window itself and `--spec 4` stays. `STRATA_SPEC_PROB=1`, `--lookup-chain 5` and
   `--suffix-draft 0` are unrun.
6. **`--aux-cpus auto` with taskset `0-15`** so aux threads land on the idle E-cores (unrun; today's 5
   pool workers sit on logical processors 2, 4, 6, 8, 10).
7. **Prompt side** (`pp` has been flat for a month): `--prefill auto:16384` measured **dead neutral**
   (70.60 decode, 1,877 at 8k and 1,946 at 32k), so `--prefill auto` is not bounded by the 8,192 chunk
   ceiling on this card and raising it buys nothing. Still unrun: `--prefill-head-alias`,
   `STRATA_PROMPT_ATTN_IMMA=1`, `STRATA_HC_Q8=1`, `--ple-io mmap`.
8. **`vm.nr_hugepages` reservation** for the expert arena (needs sudo, `STRATA_NO_LARGEPAGES=1`
   is the same-boot A/B switch).

Closed items, do not re-run:

- **Ada `sm_89` MMVQ interleave table**: merged as our `0cef00fa`, and upstream carries it
  too, on by default for compute capability 8.9 in the 0.2.x line.
- **`STRATA_GR_NORM_FAST=1` on Ada**: measured **64.85 against the 71.13 pair mean, 6.3 t/s slower**, so
   it stays off on `sm_89`. Also a correction to what this README claimed before: our commit `7ce60ad3`
   is **not** in `v0.2.7`, which carries its own CUDA opt-in marked "measured on sm_70 only"
   (`src/kernels/cuda/fused_gr.cu:1272`). Nothing to re-apply in a re-slice.
- **`--mtp-q4 head`**: rejected, see item 3 above.
- **llama.cpp PR #30790**: named in an older version of this backlog; that PR does not exist.
  #30792 (native q8_0 KV in the flash-attn tile kernel) is real and open, and llama.cpp
  #31037 / #29143 (trimmed MTP draft head) are the same idea this engine already ships as
  `--mtp-draft-vocab` plus `tools/draft_vocab.py`. Reading them is a second opinion, not a
  port.

### 6. Upstream distance and the anti-bloat decision

Upstream moved 2,252 commits from the fork's merge base `477ac13c` (2026-10-16) to `v0.2.7.1`. However, the vast majority of that delta is multi-platform bloat (Apple Metal, AMD ROCm, Intel SYCL, experimental unreleased Qwen3.5 scaffolding, 100+ unit tests, and 218 undocumented switches). 

Re-slicing or rebasing the whole fork onto upstream would re-introduce all that dead code and defeat the purpose of `strata-lean`. This fork stays dedicated strictly to lean Ada `sm_89` execution on the RTX 4070. We do not swallow upstream churn wholesale; we selectively audit and cherry-pick isolated, high-leverage CUDA inference kernels and speculative decoding wins after before/after A/B bench verification.

---

- Fork repository: [carteakey/strata-lean](https://github.com/carteakey/strata-lean) on branch `strata-lean`
- Local vendor directory: [`vendor/strata-lean`](file:///home/kchauhan/repos/l3ms/vendor/strata-lean)
- Upstream base documentation: [vendor/strata-lean/UPSTREAM_README.md](file:///home/kchauhan/repos/l3ms/vendor/strata-lean/UPSTREAM_README.md)
