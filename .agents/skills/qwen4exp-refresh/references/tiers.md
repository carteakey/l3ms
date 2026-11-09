# Qwen3.8-Flash-Next (qwen4exp) Tiers Reference

### Platinum Tier (`strata`)
* **Engine:** Strata Lean (`vendor/strata-lean/build/strata`, branch `strata-lean`).
* **Config:** `vendor/strata-lean/strata-ud-iq2_xs-ple_q8.json` (text), `strata-vision-ple_q8.json` (vision), `strata-sc117-vision-ple_q8.json` (abliterated vision). Tracked backups in [`maintenance/configs/strata/`](file:///home/kchauhan/repos/l3ms/maintenance/configs/strata/).
* **Hardware Profile:** 2,711 to 3,277 hot expert pairs cached in GPU VRAM (11.5-11.7 GB) with elastic K/V (`--kv-grow`) and `STRATA_PREFILL_LEND_PCT=40` (812-slot prompt loan cap, leaving 1,771 reclaimable slots / 2.87 GiB up to 128k context). Native MTP draft head (Q2_0 over 106,299 tokens), K8V4 KV cache @ 128k context (`131072`), `STRATA_ADAPT_FETCH=1`, `STRATA_PREFILL_CPU_SHARE_MAX=3072`.
* **Memory Management:** `cudaHostRegister` on 4 KB pages (`vm.nr_hugepages=0`) over ~39.4 GB host RAM expert arena; unquantized Q8_0 PLE table (54.4 GB) streamed from NVMe.
* **Observed Metrics:** 70-76 t/s steady decode on code, 1,116 t/s pp2048, 1,876-1,891 t/s pp8192.

### Gold Tier (`master`)
* **Engine:** `vendor/llama.cpp-master/build/bin/llama-server`.
* **Flags:**
  ```bash
  -m <model.gguf> --fit on --fit-target 512 -c 98304 --parallel 1 -b 4096 -ub 2048 \
  -fa on --jinja -ctk q8_0 -ctv q8_0 -t 10 --threads-batch 12 --prio 2 --lazy-mode on \
  --spec-type ngram-mod --spec-ngram-mod-n-match 60 --spec-ngram-mod-n-min 12 --spec-ngram-mod-n-max 24
  ```
* **Observed Metrics (b11475):** 17.3 t/s steady decode, 460–590 t/s prefill, 11,128 MiB VRAM.

### Experimental MoE Stack Tier (`exp_cache`)
* **Engine:** `vendor/llama.cpp-exp-latest/build/bin/llama-server`.
* **Key Enhancements:**
  * PR #29887 (`--moe-cache-mib 2048`): GPU-resident LRU cache for host-offloaded MoE experts.
  * PR #28671: Radix-select TOP_K for wide CUB fallback rows.
* **Flags:** Same as gold, with `-c 65536 -ub 1024 --moe-cache-mib 2048`.
* **Observed Metrics:** 23.9 t/s steady decode (+40% over plain offload), 11,454 MiB VRAM.

### Quantization Compatibility
* **Primary Homelab Fast NVMe:** Unsloth `UD-IQ2_XS` (`Qwen3.8-Flash-Next-UD-IQ2_XS`, 39.4 GB fast weights) and `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (`IQ3_XXS`), both paired with the shared 54.4 GB `Q8_0-PLE` shard on NVMe.
* **Archived to NAS (`/mnt/storage`):** `AD-4.27bpw-Q4_K_M-M64` (33 shards, 89 GB).
