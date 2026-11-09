# Incident Postmortem: Strata Elastic K/V (`--kv-grow`) OOM at Extended Context

- **Date**: 2026-11-09
- **Impacted Tier**: `qwen38-flash-next-plat` (`strata-ud-iq2_xs-ple_q8`) on RTX 4070 (12 GB)
- **Symptoms**:
  - `Error: verify: instantiate: out of memory` during token generation (CUDA graph instantiate failure).
  - `Error: the K/V cannot grow to this prompt: no VRAM is left` upon incoming multi-turn requests at long context (>50k–70k tokens).
  - Engine crash (exit code 1) and cold reboot cycle.

---

## 1. Root Cause Summary

The elastic K/V feature (`--kv-grow`) dynamically allocates K/V cache blocks in steps (default 8,192 cells) up to `--max-context 131072`, reclaiming physical VRAM memory from the dynamic expert cache arena (`xcache`) as context grows.

However, on a 12 GB VRAM card:
1. **Prefill Over-Subscription**: At boot, `--prefill auto` sizes the prompt processing loan up to `kAutoLendPct = 90%` of all expert cache slots (`generate.cpp:7211`). Out of 2,711 total slots (4.33 GiB), it reserved **2,428 slots (3.88 GiB)** for 6,912-token prefill chunks.
2. **Artificial Growth Ceiling**: At server initialization (`generate.cpp:8368`), `kvg_start` sets `kvg.top = lend_first = 283`. Because `kvg.floor` is 128, `--kv-grow` was strictly clamped to:
   $$\text{Available Slots} = 283 - 128 = 155 \text{ slots } (\approx 248 \text{ MiB})$$
3. **Exhaustion & Unreserved Allocation**: The full 128k K/V cache requires **1.11 GiB ($\approx 693$ slots)**. When context crossed ~50k–70k tokens, all 155 slots were exhausted.
4. **False Floor & CUDA Driver Exhaustion**: Rather than reclaiming more slots from the **2,556 expert slots still sitting in VRAM**, `kvg_ensure` treated the cache as "at its floor" and fell back to `vmm_chunk_new()` (`cuMemCreate`). This consumed the remaining 412 MiB of free card memory until only **8 MiB** remained.
5. **Graph Instantiation Failure**: When decoding attempted to capture/instantiate verify graphs (`cudaGraphInstantiate`), which require 20–30 MiB of VRAM, the driver failed with `cudaErrorMemoryAllocation`, triggering `verify: instantiate: out of memory`. When a subsequent request required more K/V, `vmm_chunk_new()` failed, producing `the K/V cannot grow to this prompt: no VRAM is left`.

---

## 2. Quantitative Verification

| Metric | Pre-Fix (Default `--prefill auto`) | Post-Fix (`STRATA_PREFILL_LEND_PCT=40`) |
| :--- | :--- | :--- |
| **Total Expert Slots** | 2,711 slots (4.33 GiB) | 2,711 slots (4.33 GiB) |
| **Prefill Auto Chunk** | 6,912 tokens | 2,560 tokens |
| **Prefill Borrowed Slots** | 2,428 slots (3.88 GiB) | 812 slots (1.31 GiB) |
| **`lend_first` (`kvg.top`)** | 283 | 1,899 |
| **`kvg.floor`** | 128 | 128 |
| **Reclaimable K/V Slots** | **155 slots (248 MiB)** | **1,771 slots (2.87 GiB)** |
| **Slots Needed for 128k K/V** | 693 slots (1.11 GiB) | 693 slots (1.11 GiB) |
| **Margin to Floor at 128k** | **$-538$ slots (Deficit $\to$ OOM)** | **$+1,078$ slots (Headroom)** |
| **Startup / Short Context Decode Speed** | 63.5–70+ tok/s (100% cache active) | 63.5–70+ tok/s (100% cache active) |

Notice that `STRATA_PREFILL_LEND_PCT=40` does not sacrifice a single slot of expert cache during normal operation. All 2,711 slots are fully populated in VRAM for maximum decode throughput. It only bounds how much cache prefill scratch space can borrow during prompt ingestion.

---

## 3. Implemented Fix

In `vendor/strata-lean/strata-ud-iq2_xs-ple_q8.json`:
```json
  "env": {
   "STRATA_PREFILL_CPU_SHARE_MAX": "3072",
   "STRATA_ADAPT_FETCH": "1",
   "STRATA_PREFILL_LEND_PCT": "40"
  }
```

Verified at boot:
```text
strata serve: prompt chunk auto: 2560 tokens, a 8-slot ring
strata serve: the prompt path borrows 812 CUDA0 cache slots (1.31 GiB)
strata generate: elastic K/V: 16384 of 131072 cells in VRAM (0.21 of 1.32 GiB);
the expert cache (2711 slots) gives the K/V room below slot 1899 as the context grows
```
Room below slot 1899 allows `--kv-grow` to scale smoothly to the full 128k context without ever requesting unreserved VRAM from the CUDA driver.
