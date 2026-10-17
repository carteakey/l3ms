---
name: refresh-llama
description: Inspect upstream llama.cpp changes, track open pull requests, update/fast-forward master (Gold tier), maintain the Unified QSA Sparsity + MTP branch, compile server/bench binaries, and verify model serving through llama-swap. Use this skill when the user asks to refresh llama, update llama.cpp, check upstream commits or PRs, rebuild llama-server/llama-bench, or verify gold/experimental serving parity.
---

# LLaMA Upstream Refresh & Build Management

Provides automated auditing, fetching, compiling, and verification for `llama.cpp` builds serving `Qwen3.8-Flash-Next` and other models within `L3MS`.

Detailed tier layouts and VRAM limits are documented in [Tiers Reference](./references/tiers.md).

## Quick Execution

Run the refresh helper script directly:
```bash
# Check status, commits, and upstream PRs without modifying files
~/.gemini/config/skills/refresh-llama/scripts/refresh.sh --dry-run

# Refresh Gold tier only (vendor/llama.cpp-master) and run smoke test
~/.gemini/config/skills/refresh-llama/scripts/refresh.sh --gold --smoke

# Refresh both Gold and MTP tiers, followed by router smoke tests
~/.gemini/config/skills/refresh-llama/scripts/refresh.sh --all --smoke

# Check GitHub for active Qwen/MTP/QSA pull requests
~/.gemini/config/skills/refresh-llama/scripts/refresh.sh --check-prs
```

Or run via the repository maintenance script:
```bash
/home/kchauhan/repos/l3ms/maintenance/refresh-llama.sh [OPTIONS]
```

---

## Standard Workflow for the Agent

When requested to refresh `llama.cpp` or check upstream sync:

### 1. Preflight Audit
1. Query active pull requests on GitHub using `gh pr list --repo ggerganov/llama.cpp --limit 100`.
2. Inspect `vendor/llama.cpp-master`:
   ```bash
   git -C vendor/llama.cpp-master fetch origin
   git -C vendor/llama.cpp-master log HEAD..origin/master --oneline
   ```
3. Check for relevant kernel fusions, architecture changes (`qwen4exp`, `mtp`, `moe`, `cuda graph`), or breaking CLI flag updates.

### 2. Refresh Gold Tier (`vendor/llama.cpp-master`)
1. Fast-forward the master working copy:
   ```bash
   git -C vendor/llama.cpp-master merge --ff-only origin/master
   ```
2. Build the binaries:
   ```bash
   cmake --build vendor/llama.cpp-master/build --config Release --target llama-server llama-bench -j$(nproc)
   ```
   > [!IMPORTANT]
   > Always pass `--target llama-server llama-bench`. Do not invoke bare `cmake --build build`, as GCC 16 LTO (`lto1`) crashes with an internal compiler error on the `test-fusion` test target.

### 3. Maintain Experimental & MTP Stack (`vendor/llama.cpp-pr-test-28770-28699-28213`)
This branch carries the Unified QSA Sparsity + MTP stack:
- **#28770** (Sparse Flash Attention for qwen4 - merged upstream in master `b11062`)
- **#28699** (Incremental pooled-key cache for QSA indexer)
- **#28213** (Gather-based sparse attention for QSA decode)
- **#28243** (Daniel Han shared-module MTP head)
- **#29166** (Multi-sequence block bias indexing bugfix)

When updating, rebase or merge upstream master commits and recompile targets:
```bash
cmake --build vendor/llama.cpp-pr-test-28770-28699-28213/build --config Release --target llama-server llama-bench -j$(nproc)
```

### 4. Hot-Reload & Smoke Test
1. If `llama-swap.yaml` paths were changed, hot-reload the router:
   ```bash
   kill -HUP $(pgrep '[l]lama-swap')
   ```
2. Run a smoke test chat completion with `max_tokens >= 64` (reasoning models need `>= 64` tokens to ensure output moves beyond `<think>`):
   ```bash
   API_KEY=$(grep LLAMA_SWAP_API_KEY ~/.config/systemd/user/llama-swap.service.d/api-key.conf | cut -d'=' -f2)
   curl -s http://127.0.0.1:8080/v1/chat/completions \
     -H "Authorization: Bearer ${API_KEY}" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "qwen38-flash-next",
       "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
       "max_tokens": 128,
       "temperature": 0.0
     }'
   ```
3. Verify the `system_fingerprint` in the response matches the new git commit/tag.
4. Verify peak VRAM via `nvidia-smi` (must stay under 12,000 MiB).

### 5. Document & Synchronize
Record updated commit hashes, tags, and benchmarks in:
- [AGENTS.md](file:///home/kchauhan/repos/l3ms/AGENTS.md)
- [docs/qwen38-flash-next-internal.md](file:///home/kchauhan/repos/l3ms/docs/qwen38-flash-next-internal.md)
- [CHANGELOG.md](file:///home/kchauhan/repos/l3ms/CHANGELOG.md)
