---
name: strata-cherry-pick
description: Audit, evaluate, cherry-pick, and port inference optimizations from upstream Strata releases, llama.cpp pull requests, and community CUDA kernels into the custom minimal-bloat Strata fork (carteakey/Strata on branch 4070-main-lite). Use when reviewing new Strata releases or pull requests, porting algorithmic improvements from llama.cpp or vLLM, compiling lean Ada sm_89 binaries, and executing before/after A/B benchmarks.
---

# Strata Cherry-Pick & Inference Porting Workflow

This skill provides procedures for selectively porting high-value inference improvements into my custom Strata fork (`carteakey/Strata` on rolling branch `4070-main-lite` at `vendor/strata-lean`) while completely rejecting upstream bloat (multi-vendor backends, unreleased architectures, redundant tests).

## Engine Strategy & Context

- **Strata is the primary serving engine**: Strata delivers ~70 t/s steady decode and ~1,890 t/s 8k prefill on an RTX 4070 12 GB, outperforming llama.cpp (~20 to 25 t/s decode) by roughly 3x.
- **Selective intake policy**: Upstream Strata continuously pulls in code for Metal (Apple Silicon), SYCL (Intel Arc), ROCm tuning, web UI wrappers, and unreleased engines. My fork strips all non-Ada backends, targets `sm_89` strictly, and only pulls in verified inference kernels, speculative decoding logic, and memory routing.
- **Porting from external engines**: Algorithmic ideas and CUDA kernel fusions from `llama.cpp` (e.g. MTP draft vocab trimming, direct quantized Q8_0 KV in FlashAttention) or `vLLM` can be ported directly into Strata rather than running them in slower runtimes.

---

## Hardware & Quant Baseline

All evaluations are conducted against this fixed homelab target:
- **GPU**: NVIDIA GeForce RTX 4070 (12 GB VRAM, Ada Lovelace `sm_89`, 46 SMs, PCIe 4.0 x16).
- **CPU & RAM**: Intel Core i5-12600K (10 cores / 16 threads), 64 GB DDR5-5200 (4x16 GB).
- **OS & Kernel**: Linux 7.3.0 CachyOS (`powersave` governor, `performance` EPP).
- **Model**: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/` (`IQ3_XXS`, unquantized Q8_0 PLE table on NVMe).
- **Baseline numbers**:
  - Upstream Strata v0.1.45: 63.83 t/s decode, 1,926 t/s pp.
  - `4070-main-lite` with Ada MMVQ (PR #3453): **69.60 t/s decode (peak 76.1 t/s)**, 1,883 t/s pp, 100% 4k needle retrieval accuracy.

---

## 1. Quick Execution

```bash
# 1. Audit upstream Strata commits and tags
.agents/skills/strata-cherry-pick/scripts/audit_upstream.sh

# 2. Recompile lean Strata binary (sm_89)
.agents/skills/strata-cherry-pick/scripts/build_lean.sh

# 3. Benchmark before/after with safe GPU handoff
.agents/skills/strata-cherry-pick/scripts/bench_strata.sh
```

---

## 2. Standard Workflow for the Agent

### Step 1: Upstream Strata Audit & Filtering

Inspect what upstream has changed since the base tag (`v0.1.45`) or between releases:

```bash
cd vendor/strata-lean
git fetch origin --tags -q

# List commits touching only core inference, CUDA kernels, and speculative decoding
git log v0.1.45..origin/main --oneline -- src/core src/kernels/cuda src/spec
```

**Filter Rules**:
- **Accept**: Changes in `src/kernels/cuda/` (GEMV, MMVQ, FlashAttention, norms, hyper-connections), `src/core/` (expert cache, MTP draft verification, pinned memory, THP), and `src/spec/` (draft policies, adaptive windows).
- **Reject**: Anything in `src/kernels/metal/`, `src/kernels/sycl/`, `src/kernels/rocm/`, `src/qwen35/`, `web/`, `installer/`, or multi-GPU wrappers.

---

### Step 2: Audit llama.cpp & External PRs for Ports

Search open and merged pull requests in `ggml-org/llama.cpp` for techniques applicable to `qwen4exp` or Ada Lovelace:

```bash
# Scan merged PRs touching qwen or speculative decoding
gh pr list --repo ggml-org/llama.cpp --state merged \
  --search "qwen4exp OR qwen4 OR MTP OR draft OR vocab in:title" --limit 25

# Inspect specific PR diff
gh pr diff <PR_NUMBER> --repo ggml-org/llama.cpp | head -n 100
```

**High-value port targets**:
1. **MTP Draft Vocab Trimming (PR #31037 / FR-Spec)**:
   - Evaluates only the active top-K token logits during speculative draft proposals instead of all 152k tokens.
   - Target location in Strata: `src/core/mtp.cpp`.
2. **Quantized Q8_0 KV in FlashAttention (PR #30792 / #30790)**:
   - Direct int8 math in the tile attention kernel, eliminating intermediate FP16 scratch buffer expansion.
   - Target location in Strata: `src/kernels/cuda/flash_attn.cu` or `src/kernels/cuda/kv_q8.cu`.

---

### Step 3: Cherry-Picking & Implementing the Change

When cherry-picking a clean Strata commit:
```bash
cd vendor/strata-lean
git cherry-pick -x <COMMIT_HASH>
```

If there are conflicts due to stripped backends or directory layouts:
- Resolve conflicts manually, ensuring only CUDA `sm_89` paths are updated.
- Verify that `CMakeLists.txt` does not re-enable stripped directories (`metal`, `sycl`, `qwen35`).

When porting a feature from llama.cpp or writing a custom kernel:
- Implement the change in `src/kernels/cuda/` or `src/core/`.
- Maintain register bounds and 128-byte cache line alignment required for Ada Lovelace (`sm_89`).

---

### Step 4: Compiling with Lean Configuration

Build with Ninja and verify binary size:

```bash
cd vendor/strata-lean
export PATH="$PWD/.venv/bin:/opt/cuda/bin:$PATH"

cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DSTRATA_ENABLE_QWEN35=OFF \
  -DSTRATA_BUILD_TESTS=OFF

cmake --build build --config Release -j$(nproc)
```

Verify binary size:
```bash
ls -lh build/strata
# Expected: ~73 MB. If binary balloons to >77 MB, check for stray enabled targets.
```

---

### Step 5: A/B Parity Benchmarking

Always benchmark using `bench-models/bench-qwen38-refresh-ab.sh` with the standard system frontmatter:

1. Stop the router to free VRAM:
   ```bash
   systemctl --user stop llama-swap.service
   ```
2. Run the probe suite:
   ```bash
   ARMS="strata_lean" bash bench-models/bench-qwen38-refresh-ab.sh
   ```
3. Restart the router immediately:
   ```bash
   systemctl --user start llama-swap.service
   ```
4. Parse results from `logs/results/qwen38-refresh-ab.jsonl`:
   - Steady-state decode (`tg_mean`, `tg_min`, `tg_max`).
   - Prefill throughput (`pp["8192"]`).
   - Needle retrieval test (`needle_4k: true`).
   - Memory footprint (`vram_mb`).

**Decision Rule**:
- If decode improves and needle passes: Keep the change.
- If decode regresses (as happened with `--mtp-q4 head` dropping to 67.8 t/s): Revert immediately.

---

### Step 6: Documentation & Remote Tracking

1. Document the change in `vendor/strata-lean/README.md`:
   - Add the commit hash, PR link, description of the change, and before/after numbers.
2. Commit and push to `carteakey/Strata`:
   ```bash
   cd vendor/strata-lean
   git commit -am "feat(cuda): <description of change>"
   git push carteakey 4070-main-lite
   ```
3. Hot-reload `llama-swap`:
   ```bash
   kill -HUP $(pgrep '[l]lama-swap')
   ```
