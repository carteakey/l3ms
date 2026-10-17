---
name: model-inventory-manager
description: Audit and cleanup model files and configurations. Use when identifying old models, cleaning up disk space, or reconciling serving configs with physical files.
---

# Model Inventory Manager

This skill provides workflows for inspecting homelab model footprints across all mounted NVMe drives (`/home`, `/mnt/dev`, `/run/media/kchauhan/Games`), cross-referencing against active serving tiers (`llama-swap.yaml`, Strata JSON configs) and media pipelines (`media-runtimes.json`, ComfyUI symlinks), and safely pruning unreferenced models.

## Workflow

### 1. Generate Model Inventory & Status Report
Run the audit script:

```bash
python3 .agents/skills/model-inventory-manager/scripts/audit_models.py
```

To output structured JSON instead of markdown:
```bash
python3 .agents/skills/model-inventory-manager/scripts/audit_models.py --json
```

The report categorizes every tracked model into:
- **`ACTIVE (llama-swap)`**: Currently wired in `llama-swap.yaml` (including Strata config packs and shard splits). Safe to keep; deletion is blocked.
- **`MEDIA (ComfyUI / L2M2)`**: Referenced in `toolkits/l2m2/media-runtimes.json` or active ComfyUI model directories (diffusers, text encoders, VAEs, LoRAs).
- **`UNREFERENCED`**: Found on disk but not wired in active inference routes. Candidate for user review and cleanup.

### 2. Safely Prune Unreferenced Models
Once an unreferenced candidate is selected by the user:

```bash
python3 .agents/skills/model-inventory-manager/scripts/audit_models.py --delete /path/to/model.gguf
```

The script verifies that the target model is NOT currently served before deleting and reclaiming disk space.

### 3. Preserved Infrastructure
Never remove active primary tiers unless explicitly instructed by the user:
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` (`qwen38-flash-next-plat`)
- `Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS` (`qwen38-flash-next-plat-abliterated` / `abliterated-vision`)
- Active media generators (FLUX.3 Klein, NoctQ, Krea, Breeze-TTS)
