# Inkling 2 Mini — RTX 4070 12 GB Optimization Log

Empirical architecture notes, tensor placement math, GPU MoE LRU caching, DSpark speculative decoding, and A/B throughput measurements for **Thinking Machines Lab Inkling 2 Mini** (`unsloth/Inkling-2-Mini-GGUF`, `UD-Q4_K_XL` + `mmproj-F16` + `dspark-Q8_0`) on **RTX 4070 (12 GB VRAM) + i5-12600K + 64 GB DDR5**.

**Result**: **5.89 tok/s -> 65.80 tok/s steady decode** (11.2× over plain `llama.cpp` master `b12621`, 2.0× over Unsloth `b12651` `--fit on`) at 32k Q8_0 KV context inside **11,696 MiB peak VRAM**.

---

## 1. Architecture & GGUF Tensor Breakdown

Inkling 2 Mini (`general.architecture = inkling2_loop`) is a **26.4B total / ~3.4B raw (~5.4B effective) active MoE looped transformer** with 24 physical blocks (`blk.0`–`blk.23`), 128 experts per layer (8 active), 1M max context, and an integrated ViT vision projector (`mmproj-F16.gguf`, 610 MiB):

- **Prelude (`blk.0`–`blk.5`)**: 6 physical layers, executed **1× per token**.
- **Looped Core (`blk.6`–`blk.17`)**: 12 physical layers, executed **3× per token** (`loop.start = 6`, `loop.end = 18`, `loop.count = 3`). Each pass reuses the same physical weights with separate KV/conv slots.
- **Conclusion (`blk.18`–`blk.23`)**: 6 physical layers, executed **1× per token**.
- **Effective depth**: `6 + 3×12 + 6 = 48` logical layer passes per token.

### Exact GGUF Tensor Sizes (`Inkling-2-Mini-UD-Q4_K_XL.gguf`, 16.42 GiB)

| Group | Layers | Passes / Token | Expert Weights | Non-Expert (Attn/Loop) | Notes |
| :--- | :--- | :---: | ---: | ---: | :--- |
| Non-block | `token_embd` + `output` | — | — | 938.6 MiB | `token_embd.weight` (469.3 MiB, Q8_0) + `output.weight` (469.3 MiB, Q8_0) |
| Prelude | `blk.0`–`blk.5` | **1×** | 3,280.8 MiB (546.8 MiB/blk) | 172.0 MiB | 6 outer layers |
| **Looped Core** | **`blk.6`–`blk.17`** | **3×** | **6,561.0 MiB** (546.8 MiB/blk) | 464.4 MiB | **36 of 48 logical passes (75% of all MoE compute) in 6.41 GiB** |
| Tail (low-bit) | `blk.18`–`blk.19` | **1×** | 1,093.6 MiB (546.8 MiB/blk) | 57.2 MiB | 2 outer layers |
| Tail (high-bit) | `blk.20`–`blk.23` | **1×** | **4,131.2 MiB** (1,032.8 MiB/blk) | 114.8 MiB | Unsloth Dynamic keeps last 4 blocks at higher precision (~2× size/block) |
| **Total** | 24 physical (48 logical) | 48 | 15,066.6 MiB | 1,747.0 MiB | Plus `mmproj-F16.gguf` (610 MiB) + `dspark-Q8_0.gguf` (481 MiB) |

---

## 2. Engine & Format Findings

| Binary | Build / Commit | `inkling2_loop` | `dspark-Q8_0` | Notes |
| :--- | :--- | :---: | :---: | :--- |
| Unsloth `b12588-mix-c0c3de8` | `883286076` | No (`inkling2` only) | — | Rejected current HF GGUFs (`unknown model architecture: 'inkling2_loop'`) |
| `vendor/llama.cpp-master` | `b12621` (`e50c254a2`) | Yes (#32694 + #32704) | No (`joint_proj.weight` 5120×768 mismatch) | **5.89 tok/s** — lacks Unsloth's Inkling 2 CUDA fusions |
| **Unsloth `b12651-mix-c7059d6`** | `d426a3774` (CUDA 13) | **Yes** | **Yes** | **33.13 -> 65.80 tok/s**; carries banded FA (`#386`), SiTU/conv/depth-mix/Ada 4-row MoE CUDA fusions (`#401–#417`), loop-dedup MoE LRU cache (`#421`), and DSpark (`#405`) |

### Key CLI Pitfalls Found
- **Do not pair `-ngl 99` with `--fit on`**: `common_fit_params` aborts fitting (`n_gpu_layers already set by user to 99, abort`) and attempts a full 16.3 GiB CUDA allocation -> OOM.
- **`--no-mmap` was renamed to `--load-mode none`**: Required when offloading tensors to CPU to avoid mmap page-fault stalls (`llama_model_loader` warns otherwise).
- **`--ubatch-size 512` reserves 3,387 MiB compute pp buffer** on 48 logical slots; dropping to `-b 1024 -ub 256` reclaims ~1.5 GiB of VRAM.
- **Do not set `--spec-draft-p-min 0.5+` with DSpark**: DSpark's block drafter expects the default `p_min=0.0`; setting `0.5` or `0.6` silenced drafting completely (`acc=0/0`).
- **CUDA graph headroom**: Leave >= 500 MiB below 12,282 MiB at model load so `cudaGraphInstantiate` does not OOM mid-generation (seen at `1400 MiB` hybrid cache; safe at `11,494 -> 11,696 MiB`).

---

## 3. Full Tuning Sweep History (2026-11-23)

Host: RTX 4070 (12,282 MiB, driver `620.73.04`), i5-12600K (`governor=performance`, `taskset -c 0-11`, `llama-swap.service` stopped), 64 GB DDR5 (46 GiB available). Coding probes: 256 tokens greedy (`temperature=0.0`); warm-up probes discarded before computing `tg_steady`. Prefill probes (`pp2k`, `pp8k`): unique-prefix code prompts defeating KV cache reuse.

### Progression Summary (All Sweeps)

| Stage / Arm | Binary | Strategy & Key Flags | `tg_steady` (t/s) | `pp2k` (t/s) | `pp8k` (t/s) | DSpark Acc | VRAM (load -> peak) |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: | ---: |
| `S1: A_autofit_32k` | `master b12621` | `--fit on --fit-ctx 32768 --fit-target 512` | 5.89 | — | — | — | 11,284 MiB |
| `S2: 1_b12651_autofit` | `b12651` | `--fit on --fit-ctx 32768 --fit-target 512` | 33.13 | — | — | — | 11,298 MiB |
| `S2: 2_loopcore_nommproj` | `b12651` | `blk.6–17` (12/24) GPU, `token_embd` + outer CPU, `--no-mmproj-offload`, `-ub 256` | 37.75 | — | — | — | 9,510 MiB |
| `S2: 3_loopcore_plus2` | `b12651` | `blk.6–19` (14/24) GPU, `0–5,20–23` CPU | 40.19 | — | — | — | 10,600 MiB |
| `S2: 4_loopcore_dspark3` | `b12651` | `blk.6–17` GPU + DSpark `n=3` | 38.10 | — | — | 51.9% | 10,550 MiB |
| `S2: 5_loopcore_dspark5` | `b12651` | `blk.6–17` GPU + DSpark `n=5` | 37.88 | — | — | 36.2% | 10,556 MiB |
| `S3: 2_16gpu_t6_phys` | `b12651` | `blk.4–19` (16/24) GPU, physical P-cores `-t 6` (`0,2,4,6,8,10`) | 38.09 | — | — | — | 11,692 MiB |
| `S3/S4: 1_16gpu_q8kv` | `b12651` | `blk.4–19` (16/24 = 40/48 eff = 83.3%) GPU, `-t 10` | 43.41 | 546.3 | 541.6 | — | 11,692 MiB |
| `S4: 7_16gpu_t12` | `b12651` | `blk.4–19` (16/24) GPU, **`-t 12`** (best prefill arm) | 44.08 | **547.0** | **542.5** | — | 11,692 MiB |
| `S4: 4_hybrid_core_moec2200` | `b12651` | `blk.6–17` pinned + `--moe-cache-mib 2200` on outer 12 | 47.48 | 452.7 | 451.5 | — | 11,704 MiB |
| `S5: 1_hybrid_core_moec900_ds3` | `b12651` | `blk.6–17` pinned + `--moe-cache-mib 900` + DSpark `n=3` | 37.83 | 458.9 | 453.3 | 61.6% | 11,294 -> 11,488 MiB |
| `S4: 5_pure_moec_8200` | `b12651` | `-cmoe --moe-cache-mib 8200` (all 24 cached via `#421`), no draft | 52.04 | 180.5 | 181.7 | — | 11,156 MiB |
| `S4: 6_pure_moec_7400_dspark3` | `b12651` | `-cmoe --moe-cache-mib 7400` + DSpark `n=3` (`-t 10`) | 59.01 | 180.2 | 180.9 | 62.4% | 11,398 MiB |
| `S5: 2_pure_moec7500_ds3_t12` | `b12651` | `-cmoe --moe-cache-mib 7500` + DSpark `n=3` (`cpu-share` default) | 58.14 | 183.3 | 181.2 | 57.1% | 11,494 -> 11,688 MiB |
| `S5: 3_pure_moec7500_ds4_t12` | `b12651` | Same with DSpark `n=4` | 57.48 | 183.3 | 181.3 | 51.5% | 11,498 -> 11,760 MiB |
| `S5: 4_pure_moec7500_ds3_c00` | `b12651` | Same with `--moe-cache-cpu-share 0.0` | 58.22 | 183.5 | 181.2 | 57.1% | 11,494 -> 11,688 MiB |
| `S6: 1_32k_c035` | `b12651` | Same with `--moe-cache-cpu-share 0.35` | 62.52 | ~184 | ~181 | 61.6% | 11,494 -> 11,696 MiB |
| `S5: 5_pure_moec7500_ds3_c05` | `b12651` | Same with `--moe-cache-cpu-share 0.50` | 63.09 | 183.9 | 181.2 | 59.7% | 11,494 -> 11,710 MiB |
| `S6: 2_32k_c070` | `b12651` | `-cmoe --moe-cache-mib 7500 --moe-cache-cpu-share 0.70` + DSpark `n=3` (`-ub 256`) | 65.80 *(peak 69.8)* | ~184 | ~181 | 55.8% | 11,494 -> 11,696 MiB |
| `S6: 3_32k_c100` | `b12651` | Same with `--moe-cache-cpu-share 1.00` (zero PCIe miss fetch) | 40.98 | - | - | 59.7% | 11,494 -> 11,696 MiB |
| `S6: 4_64k_c050` | `b12651` | 64k context (`-c 65536 -ctk q8_0 -ctv q4_0`), `--moe-cache-mib 7000 --moe-cache-cpu-share 0.50` | 58.97 | 183.5 | 181.5 | 52.0% | 11,400 -> 11,640 MiB |
| `S7: 1_128k_q8q4_m5500` | `b12651` | **128k context** (`-c 131072 -ctk q8_0 -ctv q4_0`), `--moe-cache-mib 5500 --moe-cache-cpu-share 0.70` | 52.94 | 183.7 | - | 46.1% | 11,020 -> 11,246 MiB |
| `S7: 2_128k_q4q4_m6400` | `b12651` | **128k context** (`-c 131072 -ctk q4_0 -ctv q4_0`), `--moe-cache-mib 6400 --moe-cache-cpu-share 0.70` | 57.34 *(peak 60.8)* | 184.5 | - | 46.3% | 11,386 -> 11,640 MiB |
| `S7: 3_128k_q8q4_ub512` | `b12651` | **128k context** (`q8_0/q4_0`), `--moe-cache-mib 5200`, `-b 2048 -ub 512` | 52.80 | 333.9 *(+82%)* | - | 58.5% | 11,296 -> 11,502 MiB |
| `S8: strata-inkling v0.2.20` | `strata-inkling 0.1.66` | `--kv k8v4 --draft off` (32k & 128k Auto Host KV Offload; `1411 GPU / 1661 host` experts) | 25.7 *(32k)* / 25.8 *(128k)* | 665.5 *(32k)* / 656.9 *(128k)* | 685.6 | N/A *(GGUF DSpark name mismatch)* | ~10.8 GiB *(1.45 GiB free)* |
| `S9: 1_ub512_m6700` | `b12651` | `-b 2048 -ub 512`, `--moe-cache-mib 6700`, `cpu-share 0.70` + DSpark `n=3` | 55.89 | 331.4 | 324.2 | - | 11,502 MiB |
| `S9: 2_ub1024_m5400` | `b12651` | `-b 2048 -ub 1024`, `--moe-cache-mib 5400`, `cpu-share 0.70` + DSpark `n=3` | 52.74 | 550.2 | 528.8 | - | 11,164 MiB |
| `S9: 3_ub2048_m3500` | `b12651` | `-b 2048 -ub 2048`, `--moe-cache-mib 3500`, `cpu-share 0.70` + DSpark `n=3` | 41.24 | 891.9 | 746.3 | - | 11,744 MiB |
| `S10: 2_el_m7500_base` | `inkling2-lean` *(pre-#264)* | `LLAMA_MOE_CACHE_ELASTIC=1`, `-ub 2048`, `--moe-cache-mib 7500`, `--spec-draft-n-ubatch 64` | 65.59 | 796.0 | 578.2 | 49.7% | 10,948 -> 11,102 MiB |
| `S10: 3_el_m7500_vb1` | `inkling2-lean` *(pre-#264)* | Same + `LLAMA_MOE_CACHE_VISIT_BUDGET=1` | 65.73 | 790.0 | 576.8 | 60.7% | 11,362 -> 11,514 MiB |
| `S11: 2_32k_m7500_pr264` | `inkling2-lean` *(+PR #264)* | `ELASTIC=1` + `VISIT_BUDGET=1` + PR `#264`, `-ub 2048`, `m7500`, `--spec-draft-n-ubatch 256` | 66.35 | 817.7 | 725.3 | - | 11,608 -> 11,760 MiB |
| `S11: 1_32k_m7800_pr264` *(fallback `inkling-2-mini-llama`)* | `inkling2-lean` *(+PR #264)* | `ELASTIC=1` + `VISIT_BUDGET=1` + PR `#264`, `-b 2048 -ub 2048`, `m7800`, `cpu-share 0.70`, `--spec-draft-n-ubatch 256` | 68.66 *(peak 69.9)* | 819.6 *(c2: 822.5)* | 726.5 | 55-60% | 11,496 -> 11,650 MiB |
| `S11: 3_128k_m5400_pr264` | `inkling2-lean` *(+PR #264)* | `-c 131072 -ctk q8_0 -ctv q4_0`, `ELASTIC=1` + `VISIT_BUDGET=1` + PR `#264`, `-ub 2048`, `m5400`, `--spec-draft-n-ubatch 128` | 55.96 *(peak 58.9)* | 817.4 *(c2: 824.6)* | 730.6 | 50-55% | 11,276 -> 11,430 MiB |
| `S12: PolyStrata stock 32k` | `strata-poly` (`v0.2.20-poly.4`) | `--max-context 32768 --kv fp8 --vram-pool --loop-bf16 --prompt-batch 2048 --lock-experts` (no draft) | 73.1 *(peak 77.7)* | 1,120.0 | 1,298.9 | N/A | 11,566 -> 11,624 MiB |
| `S12: PolyStrata stock 128k` | `strata-poly` (`v0.2.20-poly.4`) | `--max-context 131072 --kv fp8 --vram-pool --loop-bf16 --prompt-batch 2048 --lock-experts` (no draft) | 72.1 *(peak 77.5)* | 1,131.4 | 1,299.7 | N/A | 11,566 -> 11,624 MiB |
| `S12: PolyStrata + DSpark Old` | `strata-poly` (`v0.2.20-poly.4`) | Same 128k + `--draft-model dspark-Inkling-2-Mini-Q8_0_Old.gguf` (-272 GPU experts) | 68.5 | 1,101.6 | 1,175.6 | adaptive | 11,568 -> 11,786 MiB |
| **`S13: PolyStrata 4070-PRs 128k` (SERVED)** | **`strata-poly` (`polystrata-4070`)** | **`--max-context 131072 --kv fp8 --vram-pool --loop-bf16 --prompt-batch 2048 --lock-experts` + PRs `#284`/`#288`/`#291` (`STRATA_INK_PRESTAGE=0`), 1,688 GPU experts (8.1 GiB), CPU `strata-vision`** | **74.65** *(peak 76.9, post-8k 75.7)* | **1,135.6** | **1,301.6** | **N/A (no draft)** | **11,566 -> 11,620 MiB** |

### Why PolyStrata (`vendor/polystrata`) Beat Both `llama.cpp-inkling2-lean` and Upstream `strata-inkling`
- **Basalt (`jesdga95/basalt`) audit**: Checked `CMakeLists.txt` directly — hard-aborts unless `CMAKE_CUDA_ARCHITECTURES=120` (Blackwell NVFP4 tensor cores + TMA); `PEER_ADA` only allows Ada as a secondary peer card alongside a Blackwell primary.
- **PolyStrata (`simplexai-labs/PolyStrata`, `strata-poly`)**: Native Ada `sm_89` + AVX-VNNI + cuBLASLt MXFP8 engine that reads `Inkling-2-Mini-UD-Q4_K_XL.gguf` (`inkling2_loop`) directly:
  1. **CUDA VMM `--vram-pool` + `--kv fp8` + `--loop-bf16`**: Claims 8,274 MiB in one VMM pool and grows 128k KV in 2 MiB pages on demand, keeping **1,688 of 3,072 experts resident in 8.1 GiB VRAM** (~95-96% cache hit rate!) while the remaining 1,384 experts stay pinned in 6.9 GiB RAM (`--lock-experts`) and run on native AVX-VNNI.
  2. **INT8 Tensor Core prompt kernels + 2 GiB borrowed staging (`--prompt-batch 2048`)**: Delivers **1,135.6 tok/s `pp2k`** and **1,301.6 tok/s `pp8k`** (7.2x over `-ub 256` `llama.cpp`, 1.79x over elastic `llama.cpp-inkling2-lean`) with **zero** post-8k decode dip (**75.7 tok/s** immediately after an 8k prompt).
  3. **Merged 4 bit-exact PolyStrata PRs onto `vendor/polystrata` branch `polystrata-4070`**: `#284` (warp-per-row MMVQ slots), `#288` (6 fewer launches/layer in doorbell decode), `#291` (eager prompt kernel load at startup), and `#271` (`STRATA_INK_PRESTAGE`, tested and left `0`: `1` regressed `pp8k` `1,302 -> 1,218` and `post_tg` `75.7 -> 67.2`). Together `#284 + #288 + #291` lifted steady 128k decode from `72.1 -> 74.65 tok/s` (+2.55 tok/s).
  4. **Why no draft model on 12 GB**: Tested `dspark-Inkling-2-Mini-Q8_0_Old.gguf` (matches PolyStrata's `dspark_fuse.weight` schema; newer `dspark-Inkling-2-Mini-Q8_0.gguf` is `markov2`): DSpark's ~1.0-1.3 GiB weights + KV evicted 272 resident GPU experts (`1,688 -> 1,416`), dropping decode to `68.5 tok/s` and `pp8k` to `1,176 tok/s`. On 12 GB VRAM, keeping 272 more hot experts on GPU beats DSpark.
  5. **Anti-bloat policy preserved**: `vendor/strata-lean` stays strictly focused on Qwen3.8-Flash-Next; PolyStrata lives in its own isolated `vendor/polystrata` checkout with zero-VRAM CPU `strata-vision` (`vendor/polystrata/build-vision/bin/strata-vision`).

### Breaking the Prefill vs Decode Trade-off (`vendor/llama.cpp-inkling2-lean`)
Sweep 9 proved that `-cmoe` prefill is **not** stuck on CPU: `ggml_backend_cuda_device_offload_op()` already offloads `MUL_MAT_ID` to GPU whenever `n_tokens >= 32` by streaming host experts over PCIe, so `pp2k` scaled linearly with `-ub` (`184 -> 331 -> 550 -> 892 tok/s`). The only reason `-ub 2048` couldn't coexist with a 7.5 GiB MoE cache on a 12 GB RTX 4070 was static VRAM reservation:
1. **DSpark draft context bloat**: By default, `dspark-Inkling-2-Mini-Q8_0.gguf` (481 MiB) inherited `-ub 2048` and tried to reserve a **3,151.45 MiB** compute buffer. Passing `--spec-draft-n-ubatch 256` (128 at 128k) cut that waste immediately.
2. **Elastic Borrowed-Cache Prefill (`LLAMA_MOE_CACHE_ELASTIC=1`)**: `llama_moe_cache` is already inactive for `ubatch.n_tokens > 32` (`max_batch = 32`). In `vendor/llama.cpp-inkling2-lean` (branch `inkling2-lean`, built on `b12651` + 16 Inkling-2-only Unsloth pins + PR `#264` CUDA MoE prompt kernels), `llama_context::sched_reserve()` reserves only a 32-token decode compute buffer at startup. Whenever `process_ubatch()` enters a prefill ubatch (`n_tokens > 32`), it releases the MoE cache device buffers (`release_device_buffers()`, preserving CRF frequency history) so the `-ub 2048` prefill compute buffer fits cleanly in VRAM; when `n_tokens <= 32` returns, it resets the prefill scheduler and re-acquires the full 7,800 MiB MoE cache (`acquire_device_buffers()`).
3. **Visit-Weighted Group Budget (`LLAMA_MOE_CACHE_VISIT_BUDGET=1`)**: Vanilla `llama_moe_cache` split VRAM across tensor-layout groups by `host_bytes`, handing `blk.20-23` (4 visits / 48 passes = 8.3%) 27.4% of the cache while starving `blk.0-19` (44 visits / 48 passes = 91.7%). Weighting group budget by `visit_bytes` shifted ~950 MiB of cache slots into the looped core group, lifting decode alongside the extra `7500 -> 7800 MiB` budget reclaimed from the smaller startup reserve to **68.66 tok/s steady** (three consecutive probes at 69.8 to 69.9 tok/s).
4. **PR `#264` CUDA MoE prompt kernels** (`unslothai/llama.cpp#264`, 6 CUDA files: compact MMQ tiling, device expert counting sort, vectorized `GET_ROWS`): lifted `pp8k` from **578 to 726.5 tok/s** (+26%) on top of the elastic cache, beating `strata-inkling` on both decode (**2.67x** at 32k, **2.17x** at 128k) and prefill (**820 / 727 tok/s** vs `666 / 686 tok/s`).

### Head-to-Head Verdict: `PolyStrata` (`polystrata-4070`) vs `llama.cpp-inkling2-lean` vs `strata-inkling`
- **`PolyStrata` (`vendor/polystrata`, branch `polystrata-4070`) is the clear winner on both decode AND prefill at full 128K**: **74.65 tok/s steady decode** (75.7 tok/s post-`pp8k`), **1,135.6 tok/s `pp2k`**, and **1,301.6 tok/s `pp8k`** at **128K elastic FP8 KV** in **11,620 MiB VRAM** with zero draft model required.
- **`llama.cpp-inkling2-lean` stays as the GGUF/llama.cpp fallback (`inkling-2-mini-llama`)**: **68.66 tok/s decode** and **820 / 727 tok/s prefill** at 32k (**55.96 tok/s** at 128k), still beating upstream `strata-inkling` (`v0.2.20`) by **2.67x** on decode.
- **Anti-bloat decision stands**: Keep `vendor/strata-lean` strictly untouched and focused on Qwen3.8-Flash-Next; isolate `PolyStrata` cleanly in `vendor/polystrata`.

### Why `--moe-cache-cpu-share 0.70` Won (and Why `1.00` Collapsed)
- At `0.00` (default): 100% of MoE cache misses serialize over PCIe Gen4 -> **58.22 tok/s**.
- At `0.35` -> `0.50` -> **`0.70`**: 70% of misses run concurrently on the i5-12600K's 12 P-core threads while 30% stream over PCIe Gen4 and admit into the GPU LRU cache -> **65.80 tok/s** (+7.6 tok/s over default; **68.66 tok/s** with `m7800` + `VISIT_BUDGET=1`).
- At `1.00`: 100% of misses run on CPU with zero GPU PCIe overlap -> **40.98 tok/s** (-24.8 tok/s cliff).

---

## 4. Promoted Serving Configs (`llama-swap.yaml` & `polystrata-inkling2-256k.json`)

Primary tier (`inkling-2-mini` + unlisted alias `inkling-2-mini-128k`, backed by `vendor/polystrata/polystrata-inkling2-256k.json`):

```yaml
  "inkling-2-mini":
    name: "Inkling 2 Mini 26.4B-A5.4B (PolyStrata, UD-Q4_K_XL + CPU Vision, 256k)"
    ttl: 3600
    cmd: |
      taskset -c ${cpu_range} ${strata_python}
      ${polystrata_server}
      --engine strata
      --config ${polystrata_inkling2_config}
      --fit-max-tokens
      --host 0.0.0.0
      --port ${PORT}
```

Engine flags inside `vendor/polystrata/polystrata-inkling2-256k.json`:
- Binary: `vendor/polystrata/build/strata-poly` (branch `polystrata-4070`, `v0.2.20-poly.4-hotfix` + PRs `#271`, `#284`, `#288`, `#291`, `#296`, `#297`, `#301`, `#313`, Ada `sm_89` + AVX-VNNI + cuBLASLt MXFP8)
- Args: `--max-context 262144 --kv fp8 --vram-pool --loop-bf16 --prompt-batch 2048 --lock-experts --threads 12 --window 6 --lookup 3` + 4-slot / 3 GiB conversation cache + persistent `--expert-profile`
- Env: `STRATA_INK_PRESTAGE=0` (benchmarked: `1` regressed `pp8k` `1,302 -> 1,218` and `post_tg` `75.7 -> 67.2`), `STRATA_PREFILL_NVFP4=w4a8` (routes all 44 NVFP4 visits to Ada INT8 Tensor Cores, lifting prefill `+74-97%` to `1,483 / 2,036 / 2,153 tok/s`), `STRATA_INK_FAST_CTRL=1` (bit-exact faster loop-controller down kernel)
- Vision: zero-VRAM pure-CPU `vendor/polystrata/build-vision/bin/strata-vision` (`gpu: false`, `threads: 6`, `mmproj-F16.gguf`)
- Fallback tier `inkling-2-mini-llama` preserves `vendor/llama.cpp-inkling2-lean` (32k elastic `m7800` + `VISIT_BUDGET=1` + DSpark `n=3`) in `llama-swap.yaml`.

---

## 5. Long-Context Scaling Roadmap (256K -> 512K -> 1M on RTX 4070 12 GB)

Inkling 2 Mini is natively trained for **1M context** (`inkling2_loop.context_length = 1048576`, `rope.dimension_count = 0`) and only has **8 full-attention KV slots** across all 48 layer passes (`blk.5` x1, `blk.11` x3, `blk.17` x3, `blk.23` x1; the other 40 passes use fixed 1,024-token sliding windows). At `n_kv = 8, head_dim = 128`, FP8 KV (`--kv fp8`) costs **~16.5 KiB/token** total (~2.06 GiB per 128K tokens).

Against PolyStrata's **8.27 GiB** elastic `--vram-pool` on the RTX 4070 12 GB:

| Context | Full FP8 KV | Static Score Buf | Remaining Pool for GPU Experts | Status / Requirement |
| :--- | ---: | ---: | ---: | :--- |
| **128K** | ~2.06 GiB | ~13 MiB | ~6.2 GiB (~1,290 experts) | Proven baseline |
| **256K** *(promoted)* | ~4.12 GiB | ~26 MiB | ~4.1 GiB (~860 experts) | **Active today**; `--vram-pool` grows on demand, so short chats still keep ~1,685 GPU experts (~75 t/s) and pay only ~13 MiB extra at boot |
| **384K** | ~6.19 GiB | ~39 MiB | ~2.0 GiB (~420 experts) | Fits in VRAM, slower decode once full |
| **512K** | ~8.25 GiB | ~52 MiB | ~0 GiB (pool exhausted) | Needs **`--kv q4_0`** (~4.3 GiB KV) or **`--kv-resident`** (host-RAM KV streaming) |
| **1M** | ~16.50 GiB | ~105 MiB | Exceeds 12 GB VRAM | Needs **`--kv-resident`** streaming cold KV rows from host RAM (~50+ GB free on this 64 GB host) |

**Future unlock note:** PolyStrata already implements both `--kv q4_0` and `--kv-resident` in `vendor/polystrata/engine/src/poly/engine.cpp` (lines 2016-2020), currently gated to `qwen35moe` in v1. Once PolyStrata extends `--kv-resident` / `--kv q4_0` to `inkling2` (or we port them), 512K and native 1M become viable on this 12 GB + 64 GB RAM box without leaving `UD-Q4_K_XL`.

---

## 6. Follow-Up 3-Way Shootout (2026-11-24): PolyStrata vs Project Maya PR `#1208` vs Strata `v0.2.21`

Apples-to-apples HTTP server shootout on RTX 4070 12 GB + i5-12600K (`performance` governor, baseline VRAM 30 MiB) running identical `Inkling-2-Mini-UD-Q4_K_XL.gguf`, 3 coding warmups + 3 measured 256-token steady coding probes + 5,087-token prefill + 20,258-token prefill + 256-token post-20K decode:

| Engine & Config | Context | Boot | Steady Decode | 5K Prefill | 20K Prefill | Post-20K Decode | Peak VRAM | Notes |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **PolyStrata (`polystrata-4070` + `#296/#297/#301/#313`)** | **256k** | **10.5s** | **71.93 tok/s** *(pk 72.2)* | **1,193.6 tok/s** | **1,316.6 tok/s** | **59.20 tok/s** | **11,612 MiB** | **Winner**; exact dense FP8 KV, streaming tool headers (`#313`), OOM graph retry (`#301`) |
| Project Maya PR `#1208` (`IK2_KV=q8` stream + Heat-LFU + Quest, no draft) | 256k | 10.5s | 63.77 tok/s *(pk 67.3)* | 874.4 tok/s | 824.0 tok/s | 60.00 tok/s | 10,942 MiB | Required local `inkling2_loop` loader patch; Quest sparse >4K |
| Project Maya PR `#1208` (`c2048` + `mtp.safetensors` DSpark) | 128k | 10.5s | 60.87 tok/s *(pk 64.7)* | 873.3 tok/s | 822.4 tok/s | **64.90 tok/s** | 10,978 MiB | Best post-20K decode (Quest sparse + DSpark), 38% slower prefill |
| Project Maya PR `#1208` (`c512` + `IK2_ROOM_MB=64` + DSpark) | 128k | 11.0s | *(71.8 warmup)* | - | - | - | 11,874 MiB | Hung 90s on warmup 3 (engine stall / tight VRAM) |
| Upstream Strata `v0.2.21` (`strata-inkling 0.1.67`, no draft) | 128k | 29.0s | 46.83 tok/s *(pk 47.9)* | 833.1 tok/s | 817.7 tok/s | 43.40 tok/s | 11,702 MiB | Big jump over `v0.2.20` (25.7), still 1.54x slower than PolyStrata |
| Upstream Strata `v0.2.21` (`strata-inkling 0.1.67` + `mtp.safetensors`) | 128k | 25.5s | 42.93 tok/s *(pk 43.1)* | 796.4 tok/s | 757.3 tok/s | 28.50 tok/s | 11,710 MiB | DSpark hurts decode and post-20K on 12 GB |

---

## 7. Prefill Hill-Climb (2026-11-24): Unlocking Ada INT8 Tensor Cores (`STRATA_PREFILL_NVFP4=w4a8`)

Auditing `vendor/polystrata/engine/src/poly/mmq_own.cu` revealed that on Ada (`sm_89`), all 44 NVFP4 visits (`blk.0-19`) defaulted to the FP16 halves kernel (`60.5 TFLOPS`) unless `STRATA_PREFILL_NVFP4=w4a8` was set (which sends exact NVFP4 weights x Q8_1 activations to Ada's INT8 Tensor Cores, matching `llama.cpp` MMQ). Paired-control A/B with isolated per-arm profiles (`ctl_1 -> w4a8 + fast_ctrl -> ctl_2`, 256k elastic):

| Arm | Steady Decode (`tg`) | `pp2k` (`2,205` tok) | `pp8k` (`8,640` tok) | `pp20k` (`21,510` tok) | Post-20K `tg` | Peak VRAM | Verdict |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `ctl_1` (default FP16 NVFP4) | 59.07 tok/s | 631.7 tok/s | 1,089.9 tok/s | 1,192.7 tok/s | 39.7 tok/s | 11,622 MiB | Control 1 |
| **`w4a8 + STRATA_INK_FAST_CTRL=1`** | **71.63 tok/s** | **1,482.9 tok/s** | **2,036.2 tok/s** | **2,152.5 tok/s** | **55.8 tok/s** | **11,622 MiB** | **PROMOTED (+74-97% prefill, beats Qwen3.8-Flash-Next)** |
| `ctl_2` (default FP16 NVFP4) | 69.03 tok/s | 751.0 tok/s | 1,134.9 tok/s | 1,234.1 tok/s | 49.5 tok/s | 11,622 MiB | Control 2 |

Rejected from the 7-arm isolation sweep: `STRATA_FAST_ATTN=1` (`pp20k` `1,250.5` vs `1,263.9`, dropped `post20k_tg` `51.2 -> 45.3`). All 3 paired arms retrieved the exact same top latency node (`edge-69 at 430 ms`) out of 21,510 tokens of synthetic telemetry.

---

## 8. Speculative Decoding Sweet-Spot Sweep (2026-11-24): DSpark PR `#311` vs Zero-VRAM `--lookup`

To verify we hadn't dismissed DSpark prematurely on a single untuned `--window 8` run, merged PolyStrata PR `#311` (51 MB INT8 Markov fold + fused argmax) and added a `STRATA_DS_CTX=16384` ring cap (cutting DSpark eviction nearly in half from 272 to 154 GPU experts), then swept `--window 3 / 4 / 5 / 6 / 8` and zero-VRAM `--lookup 3` with isolated per-arm profiles:

| Arm | GPU Experts | Fresh Coding (`tg`) | Code Edit (`tg`) | `pp8k` | Post-8K `tg` | Peak VRAM | Verdict |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `ctl_plain_1` (`--window 1`) | 1,688 | 72.57 (pk 75.2) | 72.7 | 1,637.0 | 51.4 | 11,604 MiB | Control 1 |
| `dspark_w3` (`PR #311` + 16K KV) | 1,534 | 69.27 | 65.4 | 1,794.2 | 46.1 | 11,656 MiB | Rejected |
| `dspark_w4` (`PR #311` + 16K KV) | 1,534 | 66.53 | 74.4 | 1,761.0 | 50.0 | 11,682 MiB | Rejected |
| `dspark_w5` (`PR #311` + 16K KV) | 1,533 | 65.63 | 86.1 | 1,772.1 | 46.3 | 11,706 MiB | Rejected |
| `dspark_w6` (`PR #311` + 16K KV) | 1,532 | 67.40 | 77.1 | 1,781.9 | 40.6 | 11,724 MiB | Rejected |
| `dspark_w8` (`PR #311` + 16K KV) | 1,530 | 67.90 | 78.1 | 1,770.1 | 44.0 | 11,772 MiB | Rejected |
| `lookup_w4` (`--window 4 --lookup 3`) | 1,684 | 73.17 (pk 75.5) | 89.1 | 1,692.7 | 51.7 | 11,678 MiB | Beat DSpark & plain |
| `ctl_plain_2` (`--window 1`) | 1,688 | 73.17 (pk 73.6) | 73.0 | 1,850.3 | 53.8 | 11,604 MiB | Control 2 |

Follow-up clean `polystrata-4070` lookup window sweep (`ctl_1 -> w4 -> w6 -> w8 -> ctl_2`):
- `ctl_1`: `74.13` fresh coding, `70.5` edit (`1,688` exps, `11,596 MiB`)
- `lookup_w4`: `76.20` fresh coding, `90.5` edit (`1,684` exps, `11,660 MiB`)
- **`lookup_w6` (`--window 6 --lookup 3` — PROMOTED):** **`75.57` fresh coding**, **`99.7 tok/s` code edits (+37%)** (`1,681` exps, `11,704 MiB`)
- `lookup_w8`: `75.27` fresh coding, `91.4` edit (`1,680` exps, `11,748 MiB`)
- `ctl_2`: `75.83` fresh coding, `75.3` edit (`1,688` exps, `11,596 MiB`)

---

## 12 GB VRAM Budget, `--vram-pool`, and Long-Context Behavior (190K-223K)

Live serving observations at 190K-223K context (`polystrata-inkling2-256k.json` on RTX 4070 `12.28 GiB`):

| VRAM Bucket | Allocation | Contents |
| :--- | ---: | :--- |
| Always-resident model weights | ~`2.7 GiB` | Embeddings, LM head, all attention (Q/K/V/O) layers, shared experts, loop controllers, norms |
| CUDA runtime & scratch | ~`0.8 GiB` | CUDA context, graph captures, staging buffers, `--prompt-batch` workspace |
| Driver safety headroom | ~`0.55 GiB` | Free buffer preventing `cudaGraphInstantiate` OOM spikes |
| Shared `--vram-pool` | **~`8.2 GiB`** | Dynamically split between **GPU MoE experts** (~`1,681-1,809` of `3,072` at short context) and **live FP8 KV cache** (`~16.5 KiB/tok`) |

### Key Operational Clarifications (Common Misconceptions)

1. **`--conversation-cache-mib 3072` / `--conversation-cache-slots 4` uses `0 MiB` of VRAM:**
   - Verified in `generate.cpp:871`: conversation parking lives **100% in host DDR5 RAM** (capped at 3 GiB out of 64 GiB DDR5, where Inkling 2 Mini uses only ~7 GiB total RAM).
   - Reducing `--conversation-cache-mib` or `--conversation-cache-slots` frees **zero VRAM** and adds **zero GPU experts**; it only causes extra re-prefills when juggling multiple chats.
   - Qwen3.8-Flash-Next (`strata-ud-iq2_xs-ple_q8.json`) uses the exact same host-RAM budget (`3072 MiB` / `4` slots).
2. **Lowering `--max-context 262144` -> `131072` does not expand the `8.2 GiB` pool:**
   - With `--vram-pool`, KV cache is allocated **elastically** based on *live* sequence length, not `--max-context` (only ~`6-13 MiB` static score-buffer overhead separates 128K from 256K at idle).
   - At ~190K-223K *live* tokens, the 8 full-attention layers' FP8 KV cache alone grows to **`~3.2-3.7 GiB`**, evicting ~`3.4 GiB` of GPU experts out to host RAM (`3,389 -> 2,486 -> 491 MiB` churn observed in live logs). Decode drops from `~75 tok/s` (short context) to `26-33 tok/s`, and a `186,561`-token cold prompt reads at `1,172 tok/s` (`159 s`) due to quadratic attention growth (with 99%+ prefix cache reuse on subsequent turns).
   - **Operational takeaway:** Keep `--max-context 262144` for occasional large repo dumps, and compact or start a fresh conversation around `64K-100K` tokens to retain `55-75+ tok/s` interactive decode. For native 200K+ speed on 12 GB, track PolyStrata PR `#336` (`STRATA_INK_ZHA`, offloading 4-bit full-attention KV leaves to pinned DDR5 RAM so 200K uses ~`115 MiB` VRAM instead of ~`3.3 GiB`).

---

## PolyStrata `v0.2.20-poly.5` & PR `#320` (`--prompt-batch auto`) Bracketed A/B (2026-11-25)

Measured on RTX 4070 12 GB (`sm_89`, i5-12600K `performance` governor, `/mnt/dev/models` `ext4`, `Inkling-2-Mini-UD-Q4_K_XL.gguf`, `STRATA_PREFILL_NVFP4=w4a8`, `STRATA_INK_FAST_CTRL=1`, `STRATA_INK_PRESTAGE=0`, `--kv fp8 --vram-pool --loop-bf16 --window 6 --lookup 3`, isolated per-arm `--expert-profile`, logged in `logs/results/inkling-refresh-ab.jsonl`):

| Arm | Binary / Flags | Load | GPU Experts | Fresh `tg` (last 4) | Edit `tg` | `pp2k` (`2,455`t) | `pp8k` (`9,751`t) | `pp20k` (`24,343`t) | Post-20K `tg` | Peak VRAM | Verdict |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `base_start` | `polystrata-4070` (`poly.4-hotfix` + PRs `#271..#313`, `--prompt-batch 2048`) | 13.7 s | 1,681 | 77.30 | 64.30 | 1,556.7 | 2,337.9 | 2,313.7 | 82.16 | 11,746 MiB | Control 1 |
| `v5_pb2048` | `v0.2.20-poly.5` (`d7970a11` + `#343`, `--prompt-batch 2048`) | **7.8 s** | 1,678 | **77.47** | 69.80 | **1,833.3** (+14.5%) | 2,296.1 | 2,269.5 | 71.41 | 11,722 MiB | Candidate (+14.5% pp2k, prefix-cache fix `#343`) |
| `v5_pr320_auto` | `v0.2.20-poly.5` + PR `#320` (`--prompt-batch auto`) | 8.3 s | 1,801 -> ~1,400 | 56.55 (-27%) | 57.84 | 1,203.7 (-25%) | 1,729.9 (-25%) | 1,871.1 (-17%) | 66.84 | 11,720 MiB | **Rejected** |
| `base_end` | `polystrata-4070` (`--prompt-batch 2048`) | 9.8 s | 1,681 | 77.46 | 87.29 | 1,645.5 | 2,260.3 | 2,207.1 | 77.46 | 11,746 MiB | Control 2 |

### Key Takeaways
- **PR `#320` (`--prompt-batch auto`) is rejected on RTX 4070 12 GB + `w4a8`:** Despite starting with `+120` more GPU experts (`1,801` vs `1,681`), widening to `4096`-token batches evicted `1,994 MiB` of experts per long prompt (triggering `4.6 GiB` of `--lock-experts` RAM churn and VMM remap overhead) and slowed short prompts (`1,390 ms` vs `867 ms` on a 31-token prompt), regressing fresh decode `-27%` (`56.55` vs `77.4 tok/s`) and prefill `-17%` to `-25%`. Fixed `--prompt-batch 2048` remains optimal on 12 GB Ada with `w4a8`.
- **`v0.2.20-poly.5` (`--prompt-batch 2048`, PROMOTED 2026-11-25):** Rebased `polystrata-4070` onto `v0.2.20-poly.5` (`d62dfacb`) + cherry-picked PR `#313` (`f2578edf`, tool-call header streaming; old `poly.4` tip preserved on `polystrata-4070-poly4-backup`). Matches `poly.4-hotfix` on fresh decode (`77.47` vs `77.38`), `pp8k` (`2,296` vs `2,299`), and `pp20k` (`2,270` vs `2,260`), gains **`+14.5%` on `pp2k`** (`1,833` vs `1,601`) and faster cold load (`7.8 s`), and brings critical fixes (`#343` Inkling 2 system-prompt prefix cache fix + `strata_pin`, `vram_ahead()` staging race fix, temp-0 tool-call JSON fix). Also deleted unused `mtp.safetensors` (`607 MB` raw HF safetensors drafter).
