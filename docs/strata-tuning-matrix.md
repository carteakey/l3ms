# Strata Ada sm_89 Tuning Matrix

Empirical reference of all runtime switches, environment variables, and flags evaluated on the RTX 4070 (12 GB) Platinum Tier (`qwen38-flash-next-plat` / `carteakey/strata-lean`).

---

## 1. Active Production Serving Flags (Promoted)

These switches are declared in `vendor/strata-lean/strata-ud-iq2_xs-ple_q8.json` and `llama-swap.yaml`.

| Switch / Env Var | Configured Value | Date Promoted | Empirical Impact & Rationale |
| :--- | :--- | :--- | :--- |
| **`STRATA_HC_Q8`** | `"1"` | 2026-11-09 | **Quantizes hyper-connection up/down weights on GPU to Q8_0** (`native_dense.cpp`). Frees **0.39 GiB of VRAM**, which `--expert-cache auto` + `--kv-grow` turns directly into **+238 GPU expert slots (`2,711 -> 2,949 slots`, `4.33 -> 4.72 GiB`)**. Steady decode increases from 67.87 $\to$ **73.00 tok/s (+5.13 tok/s / +7.6%)**, decode cache hit rate rises from 68.2% $\to$ **72.4%**, and pp2048 increases from 1,114 $\to$ **1,136 tok/s**. Numerically well-behaved: KL 0.0003–0.0010, passes 4k needle, and greedy output matches `base` across 4/7 fixed prompts (identical to the engine's intra-boot noise floor). |
| **`STRATA_PREFILL_LEND_PCT`** | `"40"` | 2026-11-09 | **Caps prefill cache loan to 40% (812 slots / 1.31 GiB)**. Lifts `lend_first` from 283 to 1,899, expanding `--kv-grow` reclaimable headroom from 155 slots (248 MiB) to **1,771 slots (2.87 GiB)**. Completely eliminates K/V growth OOM and `verify: instantiate: out of memory` crashes up to 128k context without sacrificing short-context decode speed. |
| **`STRATA_PREFILL_CPU_SHARE_MAX`** | `"3072"` | 2026-11-07 | **Arms CPU expert prefill for prompt chunks $\le 3,072$ tokens**. 2k prefill throughput increases from 939 $\to$ **1,116 tok/s (+18.8%)** with zero degradation to 8k prefill or decode speed. First-token KL divergence remains minimal (~0.011 nats at 1–3k). |
| **`STRATA_ADAPT_FETCH`** | `"1"` | 2026-11-07 | **Admits PCIe-streamed misses into the GPU cache** (cherry-pick `20c19b89`). Sustained decode throughput jumps from 55.6 $\to$ **70.4 tok/s (+26.6%)**, and decode cache hit rate rises from 46.6% $\to$ **67.2%**. Prefill is unaffected. |
| **`--kv-grow`** | Enabled (`args`) | 2026-10-28 | **Elastic K/V allocation**. Maps initial 16,384 cells at boot (0.21 GiB) instead of reserving 1.32 GiB upfront. Gives an extra ~270–500 expert slots to the cache at short/medium context. Decode increases from 70.7 $\to$ **73.5+ tok/s** with hit rates up to 68.6%. Neutral at deep context (64.95 tok/s at 98k context). |
| **`--kv`** | `"k8v4"` | 2026-10-15 | **Hybrid FP8 Key / Int4 Value cache**. Compresses full 131,072-token context history into just **1.32 GiB** VRAM (versus 3.35 GiB at int8 or 5.3+ GiB at fp16), enabling full 128k context on 12 GB. |
| **`--spec` / `--spec-min-p`** | `"4"`, `"0.5"` | 2026-10-15 | **Native MTP Speculative Decoding** with 106,299-token vocabulary draft head. Produces 2.76–2.93 tokens/step with 77%–90%+ acceptance. |
| **`STRATA_ROUTE_TAIL_SKIP`** | `0` (off) | Built-in | **Strict mathematical routing**. Rejects upstream Strata's rank $\ge 7$ expert dropping to eliminate KL 0.003–0.006 drift. Default in `strata-lean`. |

---

## 2. Evaluated & Rejected Switches

Switches evaluated during tuning sweeps (`bench-models/bench-strata-tuning-ab.sh`) that failed quality, task, or performance gates on Ada sm_89.

| Switch / Env Var | Tested Value | Empirical Result | Rejection Reason |
| :--- | :--- | :--- | :--- |
| **`STRATA_ROUTE_PRIOR`** | `1.5` | 102.23 tok/s (from 71.13 pair mean), 93.0% hit rate | **Fails Agent Quality Gate**. Re-biases expert routing toward resident slots. Upstream notes flag agent-task completion drops from 5.6 to 4.5/10. Not served. |
| **`STRATA_ROUTE_RESIDENT`** | `4` | 78.33 tok/s (vs 67.87 base), 81.4% hit rate | **Fails Identity Gate (7/7 intra-boot diffs)**. Margin-4 swap in tail ranks 6–9 changes expert selection; `json_strict` runs to 96 tokens instead of stopping at 84. |
| **`STRATA_GR_NORM_FAST`** | `1` | 64.85 tok/s (vs 71.13 pair mean) | **Regression on Ada sm_89 (-6.3 tok/s slower)**. Designed and measured by upstream for sm_70 only; breaks kernel fusion efficiency on sm_89. |
| **`STRATA_GR_UP75`** | `1` | 58.17 tok/s (vs 67.87 base) | **Regression on Ada sm_89 (-9.7 tok/s slower)**. 8-row-per-warp Turing kernel regresses decode on Ada SMs. |
| **`--aux-cpus`** | `auto` (`taskset 0-15`) | 57.70 tok/s (vs 67.53 base) | **CPU Thread Pool Degradation (-9.8 tok/s)**. Mask `0-15` recruits Alder Lake E-cores 12–15 into `pool_workers` (9 workers instead of 5 on P-cores 2,4,6,8,10), introducing barrier latency. |
| **`--ple-io`** | `mmap` | 66.90 tok/s (vs 67.53 base) | **Page Cache Contamination Trap**. Speed matches `direct`, but mmapping 30.5 GB PLE tables displaces the 39.4 GB expert arena from OS RAM, causing severe SSD refaulting (`0.1756 MB/tok`) in subsequent runs. |
| **`--prefill-head-alias`** | Flag | Engine exits at load | **Incompatible with default configuration**. Engine log explicitly reports: `--prefill-head-alias is unavailable: it needs --short-read 0`. |
| **`STRATA_HC_UPMIX`** | `1` | 51.97 tok/s (vs 67.53 base) | **Decode Regression (-15.6 tok/s)**. Fused hyper-connection up projection + `gr_mix_r` kernel regresses decode throughput on Ada sm_89. |
| **`--lookup-chain`** | `5` | 56.43 tok/s (vs 67.87 base) | **Graph Overhead Penalty (-11.4 tok/s)**. Captures 8 CUDA graph windows instead of 6, reducing available VRAM and slowing down verification passes. |
| **`--suffix-draft`** | `0` | 65.80 tok/s (vs 67.87 base) | **Slower Decode (-2.1 tok/s)**. Disabling suffix drafting reduces draft acceptance from 80.5% down to 78.6% on coding benchmarks. |
| **`STRATA_PROMPT_ATTN_IMMA`** | `1` | 67.10 tok/s (vs 67.53 base) | **No-op on `--kv k8v4`**. INT8 MMA prompt attention path only triggers on `k8v8` (`pools.v_q4 == nullptr`); `k8v4` branches into `launch<3>`. |
| **`STRATA_FUSE_WAIT`** | `2` | 67.70 tok/s (vs 67.87 base) | **Neutral**. Folding GPU plan host-flag wait into copy kernel yields no measurable speedup on sm_89. |
| **`STRATA_DF_PLE`** | `1` | 67.57 tok/s (vs 67.87 base) | **Neutral**. Window-batched PLE post-projection matches per-token verify loop within measurement noise. |
| **`--spec`** | `5` | 61.00 tok/s | **Over-speculation penalty (-10.1 tok/s slower)**. 5th draft candidate acceptance drops below break-even threshold on complex code. |
| **`--mtp-draft-vocab`** | `data/draft_vocab_en.bin` | 65.78 tok/s | **-5.3 tok/s slower** than the full 106k native runtime draft head. |
| **`STRATA_ATTN_DAILY`** | `1` | 72.03 tok/s (vs 77.07 control) | Undecided/negative decode delta (-5.0 tok/s). |
| **`STRATA_QSA_ROWS`** | `1` | 70.43 tok/s (vs 77.07 control) | Slower decode (-6.6 tok/s), increased I/O churn. |
| **`STRATA_B6_MMVQ_ROWS`** | `1` | 72.70 tok/s (vs 77.07 control) | Negative delta on Ada tensor cores. |

---

## 3. Monitored & Candidate Levers

Levers with potential benefits currently under evaluation or serving as operational reserves:

| Switch / Env Var | Candidate Value | Measured Status | Notes |
| :--- | :--- | :--- | :--- |
| **`--vram-reserve-mib`** | `832` / `960` | 68.75 tok/s (+134 MiB headroom) | Frees additional card headroom for CUDA graph instantiations. Engine warning `#2065` recommends 832 on tight setups. Current baseline is 700. |
| **`STRATA_DEQUANT_WIDE`** | `1` | pp8k: 1,978 tok/s; pp32k: 1,992 tok/s | Repeat ladder on performance governor confirmed repeatable prefill gain (+45 to +52 tok/s) at 8k and 32k context. |
| **`STRATA_KV_GROW_HOLD`** | `1` | Reduces re-trim churn | Keeps K/V sized to longest parked conversation; prevents expert shuffling between turns in multi-turn sessions. |
| **`--mtp vendor/Strata-data/mtp/rt-lamina`** | Dir | Measured 68.9 tok/s | Lamina draft head with 40,525 vocab. Frees ~132 MiB of VRAM which can be converted to reserve headroom via `--vram-reserve-mib 960`. |

---

## 4. Parameter Discovery & Audit Workflow (Docs-First)

When auditing new runtime parameters or candidate levers for Strata:

1. **Check Upstream Documentation First**:
   Upstream documentation in `vendor/strata-lean/docs/` contains comprehensive descriptions of defaults, opt-in behavior, intended hardware targets (e.g. sm_70 vs sm_89 vs ROCm vs CPU), and known constraints:
   - Primary reference: [`vendor/strata-lean/docs/DETAILS.md`](file:///home/kchauhan/repos/l3ms/vendor/strata-lean/docs/DETAILS.md) (covers switches like `STRATA_PREFILL_CPU_SHARE`, `STRATA_ROUTE_TAIL_SKIP`, `STRATA_ADAPT_FETCH`, `STRATA_HC_Q8`, `--aux-cpus`, `--ple-io`, etc.).
   - Specialized guides: [`vendor/strata-lean/docs/VRAM_ELASTIC.md`](file:///home/kchauhan/repos/l3ms/vendor/strata-lean/docs/VRAM_ELASTIC.md), [`vendor/strata-lean/docs/PREFILL_HEAD_ALIAS.md`](file:///home/kchauhan/repos/l3ms/vendor/strata-lean/docs/PREFILL_HEAD_ALIAS.md), [`vendor/strata-lean/docs/QSAROWS.md`](file:///home/kchauhan/repos/l3ms/vendor/strata-lean/docs/QSAROWS.md), [`vendor/strata-lean/docs/BATCHING.md`](file:///home/kchauhan/repos/l3ms/vendor/strata-lean/docs/BATCHING.md), [`vendor/strata-lean/docs/TROUBLESHOOTING.md`](file:///home/kchauhan/repos/l3ms/vendor/strata-lean/docs/TROUBLESHOOTING.md).

2. **Fallback to Source Code Grepping**:
   For unreleased developer flags, newly merged PR levers, or undocumented environment variables, scan the codebase:
   ```bash
   git -C vendor/strata-lean grep -ho "STRATA_[A-Z0-9_]*" src/ | sort -u
   ```
   Cross-reference candidate switches against the hardware target (filter out ROCm/HIP, non-Ada CUDA compute architectures like sm_70/sm_75, or incompatible KV formats such as `k8v8`-only routines).
