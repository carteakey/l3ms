# Strata Ada sm_89 Tuning Matrix

Empirical reference of all runtime switches, environment variables, and flags evaluated on the RTX 4070 (12 GB) Platinum Tier (`qwen38-flash-next-plat` / `carteakey/strata-lean`).

---

## 1. Active Production Serving Flags (Promoted)

These switches are declared in `vendor/strata-lean/strata-ud-iq2_xs-ple_q8.json` and `llama-swap.yaml`.

| Switch / Env Var | Configured Value | Date Promoted | Empirical Impact & Rationale |
| :--- | :--- | :--- | :--- |
| **`STRATA_PREFILL_LEND_PCT`** | `"40"` | 2026-11-09 | **Caps prefill cache loan to 40% (812 slots / 1.31 GiB)**. Lifts `lend_first` from 283 to 1,899, expanding `--kv-grow` reclaimable headroom from 155 slots (248 MiB) to **1,771 slots (2.87 GiB)**. Completely eliminates K/V growth OOM and `verify: instantiate: out of memory` crashes up to 128k context without sacrificing short-context decode speed. |
| **`STRATA_PREFILL_CPU_SHARE_MAX`** | `"3072"` | 2026-11-07 | **Arms CPU expert prefill for prompt chunks $\le 3,072$ tokens**. 2k prefill throughput increases from 939 $\to$ **1,116 tok/s (+18.8%)** with zero degradation to 8k prefill or decode speed. First-token KL divergence remains minimal (~0.011 nats at 1–3k). |
| **`STRATA_ADAPT_FETCH`** | `"1"` | 2026-11-07 | **Admits PCIe-streamed misses into the GPU cache** (cherry-pick `20c19b89`). Sustained decode throughput jumps from 55.6 $\to$ **70.4 tok/s (+26.6%)**, and decode cache hit rate rises from 46.6% $\to$ **67.2%**. Prefill is unaffected. |
| **`--kv-grow`** | Enabled (`args`) | 2026-10-28 | **Elastic K/V allocation**. Maps initial 16,384 cells at boot (0.21 GiB) instead of reserving 1.32 GiB upfront. Gives an extra ~270–500 expert slots to the cache at short/medium context. Decode increases from 70.7 $\to$ **73.5+ tok/s** with hit rates up to 68.6%. |
| **`--kv`** | `"k8v4"` | 2026-10-15 | **Hybrid FP8 Key / Int4 Value cache**. Compresses full 131,072-token context history into just **1.32 GiB** VRAM (versus 3.35 GiB at int8 or 5.3+ GiB at fp16), enabling full 128k context on 12 GB. |
| **`--spec` / `--spec-min-p`** | `"4"`, `"0.5"` | 2026-10-15 | **Native MTP Speculative Decoding** with 106,299-token vocabulary draft head. Produces 2.76–2.93 tokens/step with 77%–90%+ acceptance. |
| **`STRATA_ROUTE_TAIL_SKIP`** | `0` (off) | Built-in | **Strict mathematical routing**. Rejects upstream Strata's rank $\ge 7$ expert dropping to eliminate KL 0.003–0.006 drift. Default in `strata-lean`. |

---

## 2. Evaluated & Rejected Switches

Switches evaluated during tuning sweeps (`bench-models/bench-strata-tuning-ab.sh`) that failed quality, task, or performance gates on Ada sm_89.

| Switch / Env Var | Tested Value | Empirical Result | Rejection Reason |
| :--- | :--- | :--- | :--- |
| **`STRATA_ROUTE_PRIOR`** | `1.5` | 102.23 tok/s (from 71.13 pair mean), 93.0% hit rate | **Fails Agent Quality Gate**. Re-biases expert routing toward resident slots. Upstream notes flag agent-task completion drops from 5.6 to 4.5/10. Not served. |
| **`STRATA_GR_NORM_FAST`** | `1` | 64.85 tok/s (vs 71.13 pair mean) | **Regression on Ada sm_89 (-6.3 tok/s slower)**. Designed and measured by upstream for sm_70 only; breaks kernel fusion efficiency on sm_89. |
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
| **`--vram-reserve-mib`** | `832` | 68.75 tok/s (+134 MiB headroom) | Frees additional card headroom for CUDA graph instantiations. Engine warning `#2065` recommends 832 on tight setups. Current baseline is 700. |
| **`STRATA_DEQUANT_WIDE`** | `1` | pp8k: 1,978 tok/s; pp32k: 1,992 tok/s | Only switch in Sweep 2 showing consistent prefill speedup (+40 to +60 tok/s) against control. Requires identity verification before promotion. |
| **`STRATA_KV_GROW_HOLD`** | `1` | Reduces re-trim churn | Keeps K/V sized to longest parked conversation; prevents expert shuffling between turns in multi-turn sessions. |
| **`--prefill-head-alias`** | Flag | Unrun on serving tier | Unloads native output head during prompt ingestion to expand prompt workspace, then reloads for decode. |
