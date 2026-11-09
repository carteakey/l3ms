---
name: strata-tokens
description: Audit, monitor, and report token throughput and cumulative token accounting for Strata under L3MS across live sessions, current active deployments, and host history. Use when checking total tokens processed, inspecting prompt/generation ratios, or analyzing KV cache reuse.
---

# Strata Token Accounting

This skill provides utilities and procedures for inspecting cumulative token throughput, live session metrics, prefix/KV-cache reuse, and historical processing statistics for **Strata** (Platinum tier) running under the **L3MS** model server.

## Overview & Architecture

Under L3MS, Strata serves Qwen3.8-Flash-Next (and multimodal vision variants) with speculative decoding (MTP), dynamic expert VRAM caching, and persistent conversation / KV caching. Token telemetry is available at three layers:

1. **Live Runtime Metrics**: Strata's built-in HTTP server (`serve/server.py`) exposes real-time generation and cumulative session counters via `GET http://127.0.0.1:<PORT>/metrics`.
2. **Current Deployment Logs**: The active worktree (currently `vendor/strata-lean`) maintains engine logs (`strata-ud-iq2_xs-ple_q8.log`, `strata-sc117-vision-ple_q8.log`, `strata-iq3_xxs-ple_q8.log`) logging exact per-request prompt prefill, reuse, and decode tokens.
3. **Historical Host Logs**: Archived engine logs across version checkouts (`strata`, `strata-v0.1.44.1`, `strata-latest`, `strata-lean`) provide all-time host accounting.

### Token Classification

Every logged request records:
- **Prompt Tokens**: The total context size of the prompt.
- **Reused Tokens**: Tokens matched in the persistent conversation cache / KV cache prefix (0 ms compute cost).
- **Read Tokens**: Net new tokens ingested during prefill phase (`Prompt Tokens - Reused Tokens`).
- **Generated Tokens**: Completion tokens produced during the speculative autoregressive decode phase.

Engine log format:
```
strata serve: prompt <PROMPT> tokens = <REUSED> reused + <READ> read in <PREFILL_MS> ms, <GENERATED> generated in <DECODE_MS> ms ...
```

---

## Workflow

### 1. Run Complete Token Accounting Audit

Execute the bundled token counter script:

```bash
python3 .agents/skills/strata-tokens/scripts/count_tokens.py
```

For programmatic pipelines and JSON output:
```bash
python3 .agents/skills/strata-tokens/scripts/count_tokens.py --json
```

The script automatically:
- Detects the active Strata server port from running processes (defaults to `10022`).
- Pulls live completed requests, prompt reuse, and in-flight request status from `/metrics`.
- Scans and sums active deployment logs (`vendor/strata-lean/*.log`).
- Aggregates all non-benchmark historical host logs in `vendor/*/*.log` with inode deduplication to eliminate hardlinked worktrees.

---

### 2. Live Runtime Inspection (Ad-hoc)

To inspect only the currently running engine instance directly:

```bash
# Total session counters
curl -s http://127.0.0.1:10022/metrics | jq '.totals'

# In-flight generation progress (phase, prompt, generated count, decode rate)
curl -s http://127.0.0.1:10022/metrics | jq '.live'

# Recent completed requests with timing breakdown
curl -s http://127.0.0.1:10022/metrics | jq '.requests[0:3]'
```

---

### 3. Monitoring via `llama-swap` Journal

When inspecting active requests streaming through the `llama-swap` user service:

```bash
# Monitor live generation logs
journalctl --user -u llama-swap.service -f | grep -E "\[strata\] (reading the prompt|thinking|writing|done)"

# Count requests completed during the current llama-swap daemon lifetime
journalctl --user -u llama-swap.service --no-pager | grep -c "\[strata\] done:"
```

---

## Log Locations Reference

- Active deployment:
  - `vendor/strata-lean/strata-ud-iq2_xs-ple_q8.log` (Text Platinum UD-IQ2_XS + Q8 PLE daily driver)
  - `vendor/strata-lean/strata-sc117-vision-ple_q8.log` (Multimodal vision + SC117 abliterated Q8 PLE)
  - `vendor/strata-lean/strata-iq3_xxs-ple_q8.log` (Text Platinum IQ3_XXS + Q8 PLE)
- Configuration files:
  - `llama-swap.yaml` (`strata_server`, `strata_python`, model cmd entries)
  - `vendor/strata-lean/*.json` (tracked copies in [`maintenance/configs/strata/`](file:///home/kchauhan/repos/l3ms/maintenance/configs/strata/))
