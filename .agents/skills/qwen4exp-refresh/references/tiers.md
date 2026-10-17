# Qwen3.8-Flash-Next (qwen4exp) Tiers Reference

### Platinum Tier (`strata`)
* **Engine:** Strata (`vendor/strata` or `vendor/strata-latest`, build at `build/strata` or `engine/strata`).
* **Config:** `strata-iq3_xxs.json` or `strata-vision.json`.
* **Hardware Profile:** 3,086 hot experts cached in GPU VRAM (11.5–11.7 GB). Native MTP draft head (Q2_0), K8V4 KV cache @ 128k context (`131072`).
* **Memory Management:** Linux Transparent Huge Pages (`MADV_HUGEPAGE`) on 43 GB host RAM expert arena (native since Strata v0.1.40+).
* **Observed Metrics:** 53–60 t/s steady decode, 950–2,000 t/s prefill.

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
* **Primary Homelab Fast NVMe:** `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (`IQ3_XXS`, 2 shards: 47 GB fast weights + 28.8 GB PLE shard).
* **Archived to NAS (`/mnt/storage`):** `AD-4.27bpw-Q4_K_M-M64` (33 shards, 89 GB).
