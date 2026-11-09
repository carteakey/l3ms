# Strata Production Serving Configs

Tracked backup and golden reference for all runtime JSON serving profiles used by `llama-swap` on the RTX 4070 (12 GB) Platinum tiers (`carteakey/strata-lean`).

Because `vendor/` is gitignored, these copies guarantee that serving flags, environment variables, and memory limits are version-controlled in the repository.

---

## Active Profiles

| Config File | Served Tier / Role | Key Features |
| :--- | :--- | :--- |
| `strata-ud-iq2_xs-ple_q8.json` | **`qwen38-flash-next-plat` (daily driver)** | Unsloth UD-IQ2_XS + Q8_0 PLE table, `--kv-grow`, `STRATA_PREFILL_LEND_PCT=40`, `STRATA_PREFILL_CPU_SHARE_MAX=3072`, `STRATA_ADAPT_FETCH=1`, 128k ctx |
| `strata-vision-ple_q8.json` | `qwen38-flash-next-plat-vision` | MTMD vision CPU encoder (`mmproj-F16.gguf`), zero VRAM overhead, 512px warmup patch |
| `strata-sc117-vision-ple_q8.json` | `qwen38-flash-next-plat-abliterated-vision` | Uncensored SC117 weights + vision CPU encoder |
| `strata-sc117-iq3_xxs-ple_q8.json` | `qwen38-flash-next-plat-abliterated` | Uncensored SC117 text-only weights |
| `strata-iq3_xxs-ple_q8.json` | Baseline control / benchmark peer | ISTA-DASLab GSQ-RCO IQ3_XXS |

---

## Deployment & Sync

To restore or deploy these tracked configs into the local vendor checkout:

```bash
cp maintenance/configs/strata/*.json vendor/strata-lean/
```

To update these tracked configs after promoting new tuning switches:

```bash
cp vendor/strata-lean/*.json maintenance/configs/strata/
```
