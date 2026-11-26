---
name: inkling-refresh
description: Audit, discover, build, and A/B benchmark Inkling 2 Mini inference engines on RTX 4070 12 GB (PolyStrata primary, llama.cpp fallback, upstream Strata, Maya, and new community engines). Use when checking for Inkling 2 Mini engine/PR updates, hunting new prefill or decode wins, scouting new inference engines, or promoting a faster PolyStrata build/config.
---

# Inkling 2 Mini Engine Refresh & Hill-Climb Workflow

End-to-end loop for tracking upstream Inkling 2 Mini engines, scouting new community engines, compiling lean Ada `sm_89` test builds, and promoting verified wins on the RTX 4070 12 GB (`yeti-cachy`).

Reference context (read on demand, not repeated here):
- Sweep ledger, 5-engine shootout, and rejected arms: [`docs/inkling-2-mini-optimization.md`](file:///home/kchauhan/repos/l3ms/docs/inkling-2-mini-optimization.md)
- Measured RTX 4070 silicon ceilings (`455.8 GB/s` VRAM, `25.0 GB/s` PCIe H2D, `60.0 GB/s` DDR5, `60.5 TFLOPS` FP16, `3.92 GB/tok` Inkling traffic): [`docs/rtx4070-inference-ceilings.md`](file:///home/kchauhan/repos/l3ms/docs/rtx4070-inference-ceilings.md)
- Served config & binary: [`vendor/polystrata/polystrata-inkling2-256k.json`](file:///home/kchauhan/repos/l3ms/vendor/polystrata/polystrata-inkling2-256k.json) (`vendor/polystrata`, branch `polystrata-4070`; keep `vendor/strata-lean` strictly untouched for Qwen3.8-Flash-Next).

---

## Step 1: Audit Upstream Engines & Scout New Engines

Run the radar script first:

```bash
.agents/skills/inkling-refresh/scripts/audit-upstream.sh
```

It inspects:
1. **`simplexai-labs/PolyStrata`** — new releases, commits since local `polystrata-4070`, and open PRs (CUDA prefill/decode/VMM/KV/lookup/prefix-pinning).
2. **`Niko1221/Strata`** — commits/releases touching Inkling or shared CUDA kernels.
3. **`ggml-org/llama.cpp`** — open/merged PRs matching `inkling`, `moe-cache`, `nvfp4`, `dspark`.
4. **New engine discovery** — GitHub repo search for newly published Inkling 2 / `inkling2_loop` CUDA engines (e.g. Maya forks, custom kernels, vLLM/SGLang/ExLlama support).

**Fast-reject filter for RTX 4070 12 GB (`sm_89` + i5-12600K):**
- Skip Blackwell-only (`sm_120`), AMD HIP/ROCm-only, Metal-only, and SYCL-only PRs.
- Skip AVX-512-only CPU expert paths (Alder Lake i5-12600K has AVX-VNNI, no AVX-512).
- Skip neural drafters that evict `>150` GPU experts unless they cut VRAM footprint far below `500 MiB` (DSpark already lost at 154 experts evicted; zero-VRAM `--window 6 --lookup 3` wins on 12 GB).
- Check `UD-Q4_K_XL` relevance (e.g. `Q2_0`-only kernels do not fire on `UD-Q4_K_XL`).

**Done when:** You have a concrete shortlist of 1–3 candidate releases, PRs, or engines worth compiling, with expected mechanism (prefill, decode, long-context KV, or prefix reuse).

---

## Step 2: Build Isolated Test Worktree(s)

Never overwrite `vendor/polystrata/build/strata-poly` before benchmarking. Build candidates in a detached worktree under `/tmp/poly-refresh-wt`:

```bash
git -C vendor/polystrata fetch origin --tags
git -C vendor/polystrata worktree add --detach /tmp/poly-refresh-wt <tag-or-commit>
cmake -G Ninja -S /tmp/poly-refresh-wt/engine -B /tmp/poly-refresh-wt/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_POLY=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_BUILD_HY4=OFF \
  -DSTRATA_BUILD_SPLIT=OFF \
  -DSTRATA_BUILD_CONVERSATION_TESTS=OFF
cmake --build /tmp/poly-refresh-wt/build --target strata-poly -j10
```

For an open PR test arm, snapshot the base binary (`cp /tmp/poly-refresh-wt/build/strata-poly /tmp/strata-poly-base`), cherry-pick the PR commit (`git -C /tmp/poly-refresh-wt cherry-pick --no-edit <sha>`), and run an incremental `cmake --build` (~15 s).

If testing a **brand-new engine**, build in `/tmp/bench-<engine>` and verify it loads `/mnt/dev/models/unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf` cleanly before benchmarking.

**Done when:** Candidate binaries exist in `/tmp/` and `--help` succeeds.

---

## Step 3: Run Noise-Controlled Bracketed A/B Benchmark

1. **Gate CPU governor:**
   ```bash
   ./maintenance/cpu-power-state.sh check
   ```
   Do not run or quote numbers under `powersave`.
2. **Free the RTX 4070:**
   ```bash
   systemctl --user stop llama-swap.service
   ```
3. **Run bracketed A/B (`base` at start AND end to detect thermal/RAM drift):**
   ```bash
   python3 .agents/skills/inkling-refresh/scripts/bench-inkling-ab.py \
     --arm base_start:/home/kchauhan/repos/l3ms/vendor/polystrata/build/strata-poly \
     --arm cand_v5:/tmp/strata-poly-v5 \
     --arm cand_pr320:/tmp/strata-poly-v5-pr320:--prompt-batch=auto \
     --arm base_end:/home/kchauhan/repos/l3ms/vendor/polystrata/build/strata-poly
   ```

The harness enforces:
- Isolated `--expert-profile` per arm (prevents shared-profile steady-decode drift).
- Identical model (`Inkling-2-Mini-UD-Q4_K_XL.gguf` on `/mnt/dev/models` `ext4`) and production env (`STRATA_PREFILL_NVFP4=w4a8`, `STRATA_INK_FAST_CTRL=1`, `STRATA_INK_PRESTAGE=0`).
- Metrics: `tg_steady` (6 fresh coding prompts x 256 tok, mean of last 4), `edit_tg` (code-edit prompt measuring `--window 6 --lookup 3`), `pp2k` / `pp8k` / `pp20k` (unique prefixes), `post20k_tg` (128 tok after 20K prompt), `vram_peak_mib`, `experts_gpu`, and `load_s`.
- Appends structured JSONL rows with `bench-env.sh` system frontmatter to `logs/results/inkling-refresh-ab.jsonl`.

**Done when:** Bracketed controls (`base_start` vs `base_end`) agree within ~2% and every candidate arm has complete decode, prefill, post-20K, and VRAM telemetry.

---

## Step 4: Promote Winner & Restore Live Serving

Promote **only** arms that win clearly on the RTX 4070 without regressing decode, post-20K decode, or 256K stability:

1. Update `vendor/polystrata` branch `polystrata-4070` (preserve still-unmerged wins such as `#313` tool-call header streaming if not upstreamed), rebuild `vendor/polystrata/build/strata-poly` for `sm_89`, and remove `/tmp/poly-refresh-wt`.
2. Update `vendor/polystrata/polystrata-inkling2-256k.json` (keep `allowed_hosts` intact) and run:
   ```bash
   python3 maintenance/check-serving-configs.py --local-paths
   ```
3. Restart and verify live serving + Tailscale Web UI:
   ```bash
   systemctl --user start llama-swap.service
   maintenance/l3ms-ctl.sh wake inkling-2-mini
   maintenance/l3ms-ctl.sh smoke inkling-2-mini
   maintenance/l3ms-ctl.sh ui-check
   ```
4. Record the measured A/B table (promoted AND rejected arms) in [`docs/inkling-2-mini-optimization.md`](file:///home/kchauhan/repos/l3ms/docs/inkling-2-mini-optimization.md), update summary numbers in [`AGENTS.md`](file:///home/kchauhan/repos/l3ms/AGENTS.md) and [`CHANGELOG.md`](file:///home/kchauhan/repos/l3ms/CHANGELOG.md), and sync `~/repos/homelabbr/wiki/wiki/services/l3ms.md` if endpoints or headline speeds changed.
