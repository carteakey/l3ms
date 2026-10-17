---
name: qwen4exp-refresh
description: Audit, refresh, build, and benchmark both Strata (Platinum tier) and llama.cpp (Gold & Experimental tiers) for Qwen3.8-Flash-Next (qwen4exp). Tracks merged and open PRs (native MTP, MoE LRU cache, fast FA, indexer optimizations), maintains worktrees, compiles server binaries, and executes standardized 7-way parity benchmarks against identical model quants.
---

# Qwen3.8-Flash-Next (qwen4exp) Refresh & A/B Evaluation Workflow

Provides end-to-end orchestration to inspect upstream changes, track pull requests, compile test branches, and perform parity benchmarking across **llama.cpp** and **Strata** for `Qwen3.8-Flash-Next` (`qwen4exp`).

## Overview & Tiers Evaluated

| Tier | Engine / Stack | Description & Target |
| :--- | :--- | :--- |
| **Platinum (`strata`)** | Strata (`vendor/strata` or `vendor/strata-latest`) | High-throughput tier (50–60+ t/s decode, ~2,000 t/s 8k prefill) with dynamic VRAM expert caching, native MTP speculative decoding, K8V4 KV streaming, and native 2MB Transparent Huge Pages (`MADV_HUGEPAGE`). |
| **Gold (`master`)** | llama.cpp master (`vendor/llama.cpp-master`) | Plain upstream master. Native CUDA sparse Flash Attention (#28770), RMS_NORM+SCALE CUDA fusion (#29393), native MTP (#29761), and NVMe row prefetching (#29599). |
| **Experimental (`exp`)** | llama.cpp PR stack (`vendor/llama.cpp-exp-latest`) | Master plus cutting-edge MoE optimizations: PR #29887 (GPU LRU cache for host MoE experts, `--moe-cache-mib`), PR #28671 (Radix-select `TOP_K`), and speculative optimizations. |

---

## 1. Quick Execution

```bash
# 1. Audit active PRs for llama.cpp and check Strata upstream delta
~/.gemini/config/skills/qwen4exp-refresh/scripts/audit-prs.sh

# 2. Rebuild Gold master, Experimental stack, and Strata latest
~/.gemini/config/skills/qwen4exp-refresh/scripts/build-all.sh

# 3. Execute standardized 7-way A/B benchmark suite
~/.gemini/config/skills/qwen4exp-refresh/scripts/run-ab-bench.sh
```

---

## 2. Standard Workflow for the Agent

When asked to check qwen4exp PRs, test Strata current vs latest, or test llama.cpp gold vs experimental vs latest refresh:

### Step 1: Preflight Audit & PR Identification

1. **llama.cpp GitHub PR Audit:**
   ```bash
   # List merged qwen4exp PRs
   gh pr list --repo ggml-org/llama.cpp --state merged \
     --search "qwen4exp OR qwen4 OR Qwen3.8 OR nextn in:title" --limit 25

   # List open qwen4exp PRs
   gh pr list --repo ggml-org/llama.cpp --state open \
     --search "qwen4exp OR qwen4 OR Qwen3.8 OR Flash-Next OR nextn in:title" --limit 25
   ```
   *Look specifically for:*
   - Native MTP status (PR #29761 merged into master).
   - Indexer optimizations (PR #29824, #29825, #29901).
   - Host MoE LRU cache (PR #29887: `--moe-cache-mib N`).
   - Sparse attention & radix select (PR #28770, #28671).

2. **Strata Upstream Inspection:**
   ```bash
   git -C vendor/strata fetch origin --tags
   git -C vendor/strata log HEAD..origin/main --oneline -20
   git -C vendor/strata tag --sort=-v:refname | head -5
   ```
   *Verify if Strata has updated kernel batches (e.g. #783 series) and whether Transparent Huge Page support (`MADV_HUGEPAGE`) is native in `src/core/pinned.cu`.*

---

### Step 2: Build & Worktree Management

1. **Gold Tier (`vendor/llama.cpp-master`):**
   ```bash
   cd vendor/llama.cpp-master
   git fetch origin
   git merge --ff-only origin/master
   cmake --build build --config Release --target llama-server llama-bench -j$(nproc)
   # Archive binary snapshot by tag/commit:
   cp build/bin/llama-server "build/bin/llama-server-$(git describe --tags --always)"
   ```

2. **Experimental Tier (`vendor/llama.cpp-exp-latest`):**
   Create or update a clean worktree from latest master and merge active PRs:
   ```bash
   cd vendor/llama.cpp-master
   git worktree add ../llama.cpp-exp-latest -b exp-latest origin/master
   cd ../llama.cpp-exp-latest
   # Fetch and merge active PR heads (e.g. #28671 and #29887)
   git fetch origin pull/28671/head:pr-28671
   git fetch origin pull/29887/head:pr-29887
   git merge --no-edit pr-28671 pr-29887

   # Configure and build
   cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
     -DGGML_NATIVE=ON -DGGML_LTO=ON -DGGML_OPENMP=ON -DGGML_CUDA_GRAPHS=ON \
     -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF -DLLAMA_BUILD_TESTS=OFF
   cmake --build build --config Release --target llama-server llama-bench -j10
   ```

3. **Strata Latest Tier (`vendor/strata-latest`):**
   ```bash
   cd vendor/strata
   git worktree add ../strata-latest <latest-tag-or-branch>
   cd ../strata-latest
   export PATH="/home/kchauhan/repos/l3ms/vendor/strata/.venv/bin:$PATH"
   cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
     -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc
   cmake --build build --target strata -j8
   ```

---

### Step 3: Standardized Benchmarking Harness

1. **Verify and Lock CPU Governor to `performance`:**
   ```bash
   sudo cpupower frequency-set -g performance
   # Verify:
   cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor  # must output 'performance'
   ```
   *(Running under `powersave` degrades decode by ~15–25% due to CPU clock fluctuation during memory-bound MoE stalls).*

2. **Full-VRAM Experiments require stopping the router:**
   ```bash
   systemctl --user stop llama-swap.service
   ```

3. **Execute the standardized A/B orchestrator (`bench-models/bench-qwen38-refresh-ab.sh`):**
   ```bash
   ARMS="strata_old strata_new gold_old gold_new" \
     bash bench-models/bench-qwen38-refresh-ab.sh
   ```

The harness runs:
1. System frontmatter emission (`bench-models/bench-env.sh`) recording CPU governor, RAM MT/s, and baseline VRAM.
2. Identical model quant (`ISTA-DASLab IQ3_XXS`) for direct cross-engine comparison.
3. Warmup generations (discarded to eliminate cold KV and caching transients).
4. Four code generation probes (256 tokens) logging steady-state decode throughput ($t/s$).
5. Two unique-prefix prefill probes at 2,048 and 8,192 tokens.
6. A 4,096-token needle-in-a-haystack verification check.
7. Peak VRAM and available system RAM measurement.

Results append directly to `logs/results/qwen38-refresh-ab.jsonl`.

---

### Step 4: Post-Run Restoration & Router Sync

1. Restart the user router service immediately:
   ```bash
   systemctl --user start llama-swap.service
   ```
2. Synchronize production serving configuration via `scripts/sync-serving.sh`:
   ```bash
   ~/.gemini/config/skills/qwen4exp-refresh/scripts/sync-serving.sh qwen38-flash-next-plat
   ```
   *This automated step:*
   - Snapshots `llama-swap.yaml` (`llama-swap.yaml.bak-YYYYmmdd-HHMMSS`).
   - Verifies all binaries exist.
   - Hot-reloads the router via `kill -HUP $(pgrep '[l]lama-swap')`.
   - Runs a smoke test with `max_tokens >= 64` (checking `reasoning_content` for reasoning models).

---

### Step 5: Live Site, Post, Leaderboard & Tooling Updates

1. **Update Local Blog Post (`docs/qwen3-8-flash-next-local-post.md`):**
   - Update frontmatter `updated: YYYY-MM-DD`.
   - Update or add an evolution Callout (`{% callout "tip", "..." %}`) documenting the latest throughput figures, expert cache hit rates, and context support.
   - Append a dated changelog entry to the markdown table at the bottom of the post.

2. **Update Live Homelab Leaderboard (`l3ms.carteakey.dev`):**
   - Update benchmark figures, model descriptions, and progression timeline milestones in `docs/dashboard-meta.json`.
   - Recompile dashboard payload:
     ```bash
     python3 docs/generate_dashboard_data.py
     python3 -m unittest docs/test_generate_dashboard_data.py
     ```
   - Update cache-busting timestamp in `docs/index.html`:
     `<script src="generated-models.js?v=YYYYMMDD"></script>`

3. **Synchronize Local Agent Configs (`pi`):**
   - Update `~/.pi/agent/models.json` with updated engine versions, context windows, and any newly added tiers/aliases (e.g. `qwen38-flash-next-plat-abliterated`, `abliterated`).
   - Validate using `pi --list-models` and run a smoke test (`pi -p --model <model> "..."`).

4. **Git Commit & Deployment:**
   - Commit changes to `docs/` and skills:
     ```bash
     git add docs/ .agents/skills/qwen4exp-refresh/
     git commit -m "feat(dashboard): refresh qwen4exp Strata benchmarks, leaderboard, and post"
     git push github main
     ```
   - *(If deploying to personal blog, sync `docs/qwen3-8-flash-next-local-post.md` to `carteakey.dev`).*
