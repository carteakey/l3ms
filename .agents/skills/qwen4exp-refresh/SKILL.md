---
name: qwen4exp-refresh
description: Audit, refresh, build, and benchmark both Strata (Platinum tier) and llama.cpp (Gold & Experimental tiers) for Qwen3.8-Flash-Next (qwen4exp). Tracks merged and open PRs (native MTP, MoE LRU cache, fast FA, indexer optimizations), maintains worktrees, compiles server binaries, and executes standardized 7-way parity benchmarks against identical model quants.
---

# Qwen3.8-Flash-Next (qwen4exp) Refresh & A/B Evaluation Workflow

Provides end-to-end orchestration to inspect upstream changes, track pull requests, compile test branches, and perform parity benchmarking across **llama.cpp** and **Strata** for `Qwen3.8-Flash-Next` (`qwen4exp`).

## Overview & Tiers Evaluated

| Tier | Engine / Stack | Description & Target |
| :--- | :--- | :--- |
| **Platinum (`strata`)** | `vendor/strata-lean` (fork branch `strata-lean`, cut from upstream `v0.1.45`) | Served tier (`strata-ud-iq2_xs-ple_q8.json`, tracked backup in [`maintenance/configs/strata/`](file:///home/kchauhan/repos/l3ms/maintenance/configs/strata/)). Serves Unsloth Dynamic V3.1 `UD-IQ2_XS` (Q5_0 down-experts) paired with unquantized Q8_0 PLE table on NVMe (75.8 GB iso-size, +11.5 pt DeepSWE over GSQ-RCO). **73.0 t/s steady decode on code** (`STRATA_HC_Q8=1` Q8_0 hyper-connections yielding +238 GPU expert slots to 2,949 slots / 4.72 GiB; `STRATA_ADAPT_FETCH=1`, hit rate 72.4%), **1,136 t/s pp2048** (`STRATA_PREFILL_CPU_SHARE_MAX=3072`, +19%), **1,440–1,885 t/s pp8192**, 80.8% draft acceptance, elastic K/V (`--kv-grow`) with `STRATA_PREFILL_LEND_PCT=40` (caps prompt loan to 812 slots / 2,560 tokens, giving `--kv-grow` 1,771 reclaimable slots / 2.87 GiB up to 128k context; see [`docs/incidents/2026-11-09-strata-kv-grow-oom.md`](file:///home/kchauhan/repos/l3ms/docs/incidents/2026-11-09-strata-kv-grow-oom.md) and [`docs/strata-tuning-matrix.md`](file:///home/kchauhan/repos/l3ms/docs/strata-tuning-matrix.md)). Upstream is at `v0.2.12`; the 2026-11-07 re-slice (`vendor/strata-v0.2.12`) is kept as a reference arm. |
| **Gold (`master`)** | llama.cpp master (`vendor/llama.cpp-master`, b11913 `7aca0d2ea`) | Plain upstream master. Native CUDA sparse Flash Attention (#28770), RMS_NORM+SCALE CUDA fusion (#29393), native MTP (#29761), NVMe row prefetching (#29599). Measured 22.77 t/s decode, 595 t/s pp8192 on the same quant. |
| **Experimental (`exp`)** | llama.cpp PR stack (`vendor/llama.cpp-exp-latest`, b11924 `86d69ec57`) | Master plus cutting-edge MoE and speculative work. PR #29887 (GPU LRU cache for host MoE experts, `--moe-cache-mib`) is **merged upstream** now; open follow-ups are #27861 (GPU-resident LRU), #27210 (adaptive MTP draft depth), and #31469 (--moe-direct-io). Measured `exp_cache` 24.53 t/s decode. |
| **MTP (served tier)** | `vendor/llama.cpp-exp-stack-v2` (`qwen38_mtp_server`) | The tree the served `qwen38-flash-next-mtp` tier runs from, and the one that has fallen behind: b11268 from 2026-09-28, against master's b11913. Rebuild this tree, not `exp-latest`, if the MTP tier should keep up. |

A small `llama-server` (16 KB) in a vendor `build/bin/` is normal for recent llama.cpp: it links
`libllama-server-impl.so` and `libggml-cuda.so.0.*` beside it. Check with `llama-server
--version` before declaring a binary broken.

---

## 1. Quick Execution

```bash
# 1. Audit active PRs for llama.cpp and check Strata upstream delta
.agents/skills/qwen4exp-refresh/scripts/audit-prs.sh
.agents/skills/strata-cherry-pick/scripts/audit_upstream.sh   # Strata: tag distance, what is already upstream

# 2. Rebuild Gold master, Experimental stack, and Strata latest
.agents/skills/qwen4exp-refresh/scripts/build-all.sh

# 3. Execute standardized 7-way A/B benchmark suite
.agents/skills/qwen4exp-refresh/scripts/run-ab-bench.sh

# 4. Measure a flag or env switch on the served Strata tier (no rebuild)
ARMS="base kv_grow routeprior15 base" bash bench-models/bench-strata-tuning-ab.sh
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

2. **Strata Upstream Inspection** (the served tier is the fork, so audit the fork):
   ```bash
   .agents/skills/strata-cherry-pick/scripts/audit_upstream.sh
   # manual equivalent:
   git -C vendor/strata-lean fetch --tags -q origin
   NEWEST=$(git -C vendor/strata-lean tag --sort=-v:refname | head -n1)
   git -C vendor/strata-lean log "$NEWEST..HEAD" --oneline          # what is still ours
   git -C vendor/strata-lean grep -n STRATA_ROUTE_PRIOR "$NEWEST" -- src   # already upstream?
   ```
   Do **not** audit `vendor/strata` (a stale v0.1.30 checkout that still carries an
   uncommitted local patch to `src/core/pinned.cu`). Check huge pages by what the engine
   reports at load, not by hoping for `MADV_HUGEPAGE`: upstream stopped requesting it under
   THP `always` (#771, #2126), the lever on this box is `vm.nr_hugepages` (currently 0).
   See `docs/strata-perf-next-steps-2026-10.md` P1-9.

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
   Create or update a clean worktree from latest master and merge active PRs. #29887 is
   merged, so the current open stack to consider is #27861, #27210, #29143 and #31037.
   The **served MTP tier** is a different tree (`vendor/llama.cpp-exp-stack-v2`, b11268 from
   2026-09-28): refresh that one when the goal is to keep a served tier current.
   ```bash
   cd vendor/llama.cpp-master
   git worktree add ../llama.cpp-exp-latest -b exp-latest origin/master
   cd ../llama.cpp-exp-latest
   # Fetch and merge active PR heads (e.g. #28671 and #27861)
   git fetch origin pull/28671/head:pr-28671
   git fetch origin pull/27861/head:pr-27861
   git merge --no-edit pr-28671 pr-27861

   # Configure and build
   cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
     -DGGML_NATIVE=ON -DGGML_LTO=ON -DGGML_OPENMP=ON -DGGML_CUDA_GRAPHS=ON \
     -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF -DLLAMA_BUILD_TESTS=OFF
   cmake --build build --config Release --target llama-server llama-bench -j10
   ```

3. **Strata Latest Tier (`vendor/strata-latest`):**
   For a plain upstream comparison arm (`strata_v026` style), slice a tag. For anything that
   should be served, work in `vendor/strata-lean` and follow the re-slice recipe in the
   `strata-cherry-pick` skill.
   ```bash
   cd vendor/strata-lean
   git worktree add ../strata-v0.2.7 v0.2.7      # plain upstream comparison arm
   cd ../strata-v0.2.7
   export PATH="/home/kchauhan/repos/l3ms/vendor/strata/.venv/bin:$PATH"   # the only strata venv on this box
   cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
     -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc
   cmake --build build --target strata -j8
   ```
   There is no `vendor/strata-lean/.venv` on this host; the Strata python environment lives at
   `vendor/strata/.venv` (serve/server.py runs from it). `build_lean.sh` prefers a lean venv
   when one exists and falls back to that path otherwise.

---

### Step 3: Standardized Benchmarking Harness

1. **Verify and Lock CPU Governor to `performance`:**
   ```bash
   ./maintenance/cpu-power-state.sh check
   # If failing or unapplied:
   sudo ./maintenance/cpu-power-state.sh apply-cpupower --yes
   sudo systemctl restart cpupower.service
   ```
   *(Running under `powersave` degrades decode from 59.7 tok/s to 54.6 tok/s due to CPU clock idling at 800 MHz during MoE stalls).*

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

4. **Switch-only tuning sweep on the served Strata tier** (no rebuild, arms overlaid on the
   production run config, pp probed at 2,048 / 8,192 / 32,768):
   ```bash
   systemctl --user stop llama-swap.service
   bash bench-models/bench-strata-tuning-ab.sh              # default arm ladder
   ARMS="base hcq8 aux_ecores ple_mmap base" bash bench-models/bench-strata-tuning-ab.sh
   systemctl --user start llama-swap.service
   ```
   Rows: `logs/results/strata-tuning-ab.jsonl`. Engine self-report (expert slots, decode hit
   rate, PCIe share, VRAM free at request start, `#2065` warning, arena page backing, draft
   head token count, `read_bytes` per generated token): `logs/results/strata-tuning-diag.jsonl`.
   Read the diag row before believing a speed number: tens of MB/token means the run got
   faster by spilling to SSD.

### Step 3b: Strata Param Sweep & Switch Discovery Workflow

Use this workflow to discover, audit, benchmark, and promote candidate flags or `STRATA_*` environment switches:

1. **Consult Strata Documentation First (Fallback to Source Code Grepping):**
   Always check the Strata documentation first before exploring or testing candidate parameters. Upstream documentation explains default values, opt-in flags, intended target architectures, and architectural interactions:
   - Primary reference: [`vendor/strata-lean/docs/DETAILS.md`](file:///home/kchauhan/repos/l3ms/vendor/strata-lean/docs/DETAILS.md) (especially § "Short prompts: the CPU shares the experts", § "Missed experts in a verify window: the tail is skipped", § "More opt-in switches measured for 0.1.41", § "Thread placement: --aux-cpus").
   - Topic guides: [`vendor/strata-lean/docs/VRAM_ELASTIC.md`](file:///home/kchauhan/repos/l3ms/vendor/strata-lean/docs/VRAM_ELASTIC.md), [`vendor/strata-lean/docs/PREFILL_HEAD_ALIAS.md`](file:///home/kchauhan/repos/l3ms/vendor/strata-lean/docs/PREFILL_HEAD_ALIAS.md), [`vendor/strata-lean/docs/QSAROWS.md`](file:///home/kchauhan/repos/l3ms/vendor/strata-lean/docs/QSAROWS.md), [`vendor/strata-lean/docs/BATCHING.md`](file:///home/kchauhan/repos/l3ms/vendor/strata-lean/docs/BATCHING.md), [`vendor/strata-lean/docs/TROUBLESHOOTING.md`](file:///home/kchauhan/repos/l3ms/vendor/strata-lean/docs/TROUBLESHOOTING.md).
   - **Fallback to source code scanning** only for undocumented switches, newly merged PRs, or unpublished developer flags:
     ```bash
     git -C vendor/strata-lean grep -ho "STRATA_[A-Z0-9_]*" src/ | sort -u
     ```
   Filter out non-Ada (`HIPCC`, `sm_70`, `sm_75`, `gfx12`), multi-GPU (`split_at`), or conflicting branches (e.g. `STRATA_PROMPT_ATTN_IMMA` only acts on `k8v8`, not our `k8v4`).

2. **Define Candidate Arms in `bench-models/bench-strata-tuning-ab.sh`:**
   Add new candidate specs to `arm_spec()` and document them in `list_arms()`. Verify syntax without touching the GPU:
   ```bash
   DRY_RUN=1 ARMS="base hcq8 gr_up75 fuse_wait2 base" ./bench-models/bench-strata-tuning-ab.sh
   ```

3. **Execute Paired A/B Bracketed Ladders:**
   Always bracket arms with identical `base` controls at both ends (`base <candidates...> base`) to detect memory-spill contamination or thermal drift:
   - **Ladder A (Prompt / I/O / CPU topology)**: `aux_ecores`, `prefill_alias`, `imma`, `kg_upmix`, `ple_mmap`.
   - **Ladder B (Decode / Speculation / Residency)**: `hcq8`, `no_suffix`, `lookup`, `route_resident`, `gr_up75`, `fuse_wait2`, `df_ple`.
   - For long runs where dark detached execution is preferred:
     ```bash
     bench-models/run-strata-tuning-dark.sh
     ```

4. **Judge Results via Diagnostic Separation (`strata-tuning-diag.jsonl`):**
   - **VRAM Reallocation Win**: `STRATA_HC_Q8=1` (+5.13 t/s steady decode, +238 expert slots to 2,949, 72.4% hit rate, zero quality loss).
   - **Page Cache Contamination Trap**: `--ple-io mmap` matches `--ple-io direct` on speed, but pollutes the Linux page cache with 30.5 GB of PLE tables, causing subsequent runs to spill expert arena pages to SSD (`io_mb_per_token: 0.1756` vs `0.0290`). **Always keep `--ple-io direct`**.
   - **CPU Pool Straggler Trap**: `--aux-cpus auto` with `taskset -c 0-15` recruits Alder Lake E-cores `12-15` into `pool_workers` (9 workers instead of 5 on P-cores `2,4,6,8,10`), dropping decode by -9.8 t/s.
   - **Residency Routing Drift**: `STRATA_ROUTE_RESIDENT=4` gains +10.5 t/s decode (81.4% hit rate) but changes expert picks, causing 7/7 intra-boot prompt diffs. Reject for agent tasks.

5. **Promote Winner to Production Config:**
   Snapshot `vendor/strata-lean/strata-ud-iq2_xs-ple_q8.json`, add the switch under `"env"` or `"args"`, copy to `maintenance/configs/strata/`, and restart `llama-swap.service`. Run a chat completion smoke test with `max_tokens >= 64`.

---

### Step 4: Post-Run Restoration & Router Sync

1. Restart the user router service immediately:
   ```bash
   systemctl --user start llama-swap.service
   ```
2. Synchronize production serving configuration via `scripts/sync-serving.sh`:
   ```bash
   .agents/skills/qwen4exp-refresh/scripts/sync-serving.sh qwen38-flash-next-plat
   ```
   *This automated step:*
   - Snapshots `llama-swap.yaml` (`llama-swap.yaml.bak-YYYYmmdd-HHMMSS`).
   - Verifies all binaries exist.
   - Hot-reloads the router via `kill -HUP $(pgrep '[l]lama-swap')`.
   - Runs a smoke test with `max_tokens >= 64` (checking `reasoning_content` for reasoning models).

---

### Step 5: Live Site, Post, Leaderboard & Tooling Updates

1. **Update Evaluation Docs, Matrix & Blog Post:**
   - Update [`docs/strata-tuning-matrix.md`](file:///home/kchauhan/repos/l3ms/docs/strata-tuning-matrix.md) and [`docs/strata-perf-next-steps-2026-10.md`](file:///home/kchauhan/repos/l3ms/docs/strata-perf-next-steps-2026-10.md) with newly promoted or rejected switches.
   - Sync updated Strata JSON configs into [`maintenance/configs/strata/`](file:///home/kchauhan/repos/l3ms/maintenance/configs/strata/) (`cp vendor/strata-lean/*.json maintenance/configs/strata/`).
   - Update [`docs/strata-evaluation-internal.md`](file:///home/kchauhan/repos/l3ms/docs/strata-evaluation-internal.md) (§18) and [`docs/qwen38-flash-next-variant-ranking.md`](file:///home/kchauhan/repos/l3ms/docs/qwen38-flash-next-variant-ranking.md) (indexed in [`docs/README.md`](file:///home/kchauhan/repos/l3ms/docs/README.md)).
   - Sync or draft companion posts in `carteakey.dev/src/posts/local-inference/` following Human Writer skill rules (zero em dashes, first-person voice).

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
