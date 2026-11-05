---
name: strata-cherry-pick
description: Audit, evaluate, cherry-pick, and port inference optimizations from upstream Strata releases, llama.cpp pull requests, and community CUDA kernels into the custom minimal-bloat Strata fork (carteakey/strata-lean on branch strata-lean). Use when reviewing new Strata releases or pull requests, porting algorithmic improvements from llama.cpp or vLLM, compiling lean Ada sm_89 binaries, and executing before/after A/B benchmarks.
---

# Strata Cherry-Pick & Inference Porting Workflow

This skill provides procedures for selectively porting high-value inference improvements into my custom Strata fork (`carteakey/strata-lean` on rolling branch `strata-lean` at `vendor/strata-lean`) while completely rejecting upstream bloat (multi-vendor backends, unreleased architectures, redundant tests).

## Engine Strategy & Context

- **Strata is the primary serving engine**: Strata delivers 65 to 70 t/s steady decode and ~1,890 t/s 8k prefill on an RTX 4070 12 GB, outperforming llama.cpp (~20 to 25 t/s decode) by roughly 3x. Run-to-run spread on this tier is about 2 t/s, so a claimed win under ~3 t/s needs a paired control run in the same boot.
- **Anti-Bloat Intake Policy (Never Blindly Re-slice)**: Upstream Strata continuously pulls in code for Metal (Apple Silicon), SYCL (Intel Arc), ROCm tuning, web UI wrappers, test suites, and unreleased engines (e.g. 2,200+ commits between 0.1.45 and 0.2.7). Do NOT suggest or execute a whole-tree rebase or "re-slice" onto upstream. My fork strips all non-Ada backends, targets `sm_89` strictly, and only pulls in verified inference kernels, speculative decoding logic, and memory routing.
- **Optimization Priority**: 
  1. *Switch first*: Most of the immediate headroom on this card is already compiled in as a runtime flag or a `STRATA_*` env switch (e.g. `--kv-grow` yielding +2.8 t/s with zero code changes). Measure with `bench-models/bench-strata-tuning-ab.sh` before touching C++/CUDA code.
  2. *Cherry-pick second*: Selectively port verified CUDA kernels and algorithmic wins (from upstream Strata or llama.cpp PRs) into `vendor/strata-lean`.
  3. *Re-slice NEVER*: Do not pull multi-platform churn into the lean Ada tree.
- **Porting from external engines**: Algorithmic ideas and CUDA kernel fusions from `llama.cpp` (e.g. MTP draft vocab trimming, direct quantized Q8_0 KV in FlashAttention) or `vLLM` can be ported directly into Strata rather than running them in slower runtimes. llama.cpp is a source of ideas, not a speed tier. The Qwen3.8-specific fork family (Lamina, PolyStrata, basalt) and what to take from each is Step 2b; for this model they are a data file and a handful of switches, not code.

---

## Hardware & Quant Baseline

All evaluations are conducted against this fixed homelab target:
- **GPU**: NVIDIA GeForce RTX 4070 (12 GB VRAM, Ada Lovelace `sm_89`, 46 SMs, PCIe 4.0 x16).
- **CPU & RAM**: Intel Core i5-12600K (10 cores / 16 threads), 64 GB DDR5-5200 (4x16 GB).
- **OS & Kernel**: Linux 7.3.0 CachyOS (`powersave` governor, `performance` EPP).
- **Model**: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/` (`IQ3_XXS`, unquantized Q8_0 PLE table on NVMe).
- **Baseline numbers** (`logs/results/qwen38-refresh-ab.jsonl`, same quant, same probe):

  | arm | engine | decode mean | min / max | pp8192 |
  | --- | --- | --- | --- | --- |
  | `strata_old` | v0.1.44.1 | 60.58 | 56.4 / 64.2 | 1911 |
  | `strata_new` | v0.1.45 | 63.83 | 62.0 / 67.7 | 1919 |
  | `strata_v026` | v0.2.6 | 65.53 | 62.2 / 70.5 | 1876 |
  | `strata_lean` (before the sm_89 table) | fork | 65.08 | 59.3 / 72.9 | 1891 |
  | `strata_lean` (after PR #3453 table) | fork | **69.60** | 65.4 / 76.1 | 1876 |
  | `strata_lean` repeat, same day | fork | 67.88 | 63.0 / 71.6 | 1890 |

  All these runs passed `needle_4k`. Prefill has been flat (1,876 to 1,996) across every
  Strata arm since 2026-10-07 while decode moved 52 to 69.6: the prompt lane is the part
  of the fork that has not kept pace, and the 4070 is behind an unpruned v0.1.44.1 there.

- **Fork size and upstream distance**: upstream went from `v0.1.45` to `v0.2.11`.
  `git log v0.2.11..HEAD` retains our core lean commits: the hyper-connection
  inject/gate-mean fold (`8645328d`), build gates (`2b829e1e`), non-Ada backend pruning
  (`3c3312d4`), serve repeat-wrap fix (`9aafcbf0`), QSA fused RMSNorm + RoPE (`STRATA_QSA_FUSE_NR`),
  strict mathematical routing default (`STRATA_ROUTE_TAIL_SKIP=0`), and now the **Unsloth Dynamic V3.1
  intake**: cherry-picked `Q5_0` down-expert activation offset fix (`aa678f10`) and `tools/iq_pack.py`
  v4 split-shard layout handling. This enables native execution of `UD-IQ2_XS` (75.8 GB iso-size
  contender) with zero non-Ada bloat.

---

## 1. Quick Execution

```bash
# 1. Audit upstream Strata: how far ahead it is, what is already upstream, what is still ours
.agents/skills/strata-cherry-pick/scripts/audit_upstream.sh

# 2. Measure a switch that is already compiled in (flags and env only, no rebuild)
MODE=tune .agents/skills/strata-cherry-pick/scripts/bench_strata.sh
TUNE_ARMS="base kv_grow routeprior15 draft_en grnormfast base" MODE=tune \
  .agents/skills/strata-cherry-pick/scripts/bench_strata.sh

# 3. Recompile lean Strata binary (sm_89)
.agents/skills/strata-cherry-pick/scripts/build_lean.sh

# 4. Benchmark the rebuilt fork with safe GPU handoff
.agents/skills/strata-cherry-pick/scripts/bench_strata.sh          # MODE=code, the default
```

`bench_strata.sh` stops `llama-swap.service`, runs, and restarts it on exit (trap on EXIT).
Calling `bench-models/bench-strata-tuning-ab.sh` directly refuses to run while the router is
active; through `bench_strata.sh` it passes `FORCE=1` because the handoff already happened.

---

## 2. Standard Workflow for the Agent

### Step 1: Upstream Strata Audit & Filtering

Inspect what upstream has changed since the base tag (`v0.1.45`) or between releases.
Compare against the newest tag, not against the tag the fork was cut from:

```bash
cd vendor/strata-lean
git fetch origin --tags -q
NEWEST=$(git tag --sort=-v:refname | head -n1)     # v0.2.7 on 2026-10-28

# Core inference / CUDA / speculative commits between the fork's merge base and the newest tag
git log "$(git merge-base HEAD "$NEWEST")..$NEWEST" --oneline -- src/core src/kernels/cuda src/spec

# What is still ours, i.e. what a re-slice would have to re-apply
git log "$NEWEST..HEAD" --oneline

# Is a switch already upstream? Ask the tag, not the release notes
git grep -n "STRATA_ROUTE_PRIOR" "$NEWEST" -- src
git show "$NEWEST:src/kernels/cuda/route_prior.cu" | head -20

# New env switches since the fork's base (what the re-slice would unlock)
git grep -ho "STRATA_[A-Z0-9_]*" "$(git merge-base HEAD "$NEWEST")" -- src | sort -u > /tmp/old_env.txt
git grep -ho "STRATA_[A-Z0-9_]*" "$NEWEST" -- src | sort -u > /tmp/new_env.txt
comm -13 /tmp/old_env.txt /tmp/new_env.txt
```

Before porting a llama.cpp or Strata PR, check whether 0.2.x already implemented the same
idea as a switch. On 2026-10-28 that was true for: our own sm_89 MMVQ table (PR #3453, on
by default), `STRATA_GR_NORM_FAST`, cache-conditioned routing (`STRATA_ROUTE_PRIOR`,
default for NVIDIA cards at or under 13 GiB), int8 prompt attention
(`STRATA_PROMPT_ATTN_IMMA`), and the trimmed MTP draft head (`--mtp-draft-vocab`).

**Filter Rules**:
- **Accept**: Changes in `src/kernels/cuda/` (GEMV, MMVQ, FlashAttention, norms, hyper-connections), `src/core/` (expert cache, MTP draft verification, pinned memory, huge pages), and `src/spec/` (draft policies, adaptive windows).
- **Reject**: Anything in `src/kernels/metal/`, `src/kernels/sycl/`, `src/kernels/rocm/`, `src/qwen35/`, `web/`, `installer/`, or multi-GPU wrappers.

### Step 1b: The re-slice recipe (preferred over a long cherry-pick chain)

```bash
cd vendor/strata-lean
git worktree add -b 4070-main-lite-027 ../strata-v0.2.7 v0.2.7
cd ../strata-v0.2.7
git cherry-pick -x 8645328d 2b829e1e 3c3312d4 9aafcbf0    # ours; skip what is already upstream
../.venv/... build_lean.sh   # the fork's build gate, then the standard A/B
```

Expect the conflict to land in `src/core/gr.cpp` / `src/kernels/cuda/fused_gr.cu`, because
0.2.x reworked the hyper-connection path that `8645328d` folds. Two of our seven commits
(`0cef00fa` the sm_89 table, `7ce60ad3` `GR_NORM_FAST`) do not need re-applying at all.
Keep the old branch until the re-sliced one benches at or above it; the comparison is
`ARMS="strata_lean strata_v027"` once the arm exists in `bench-qwen38-refresh-ab.sh`.

---

### Step 1c: Merge patterns I actually use (five shapes, cheapest first)

1. **Switch-only promote (no merge at all).** The change is a flag, a `STRATA_*` env var, a
   taskset mask, or a data file (`--expert-profile`, `--mtp-draft-vocab`). It is measured by
   `strata-tuning`, and a winner lands as one line in `strata-iq3_xxs-ple_q8.json` plus a
   `name`/`description` refresh in `llama-swap.yaml`. No git history changes. This is the most
   common shape and it should be tried before any merge.
2. **Single-commit cherry-pick.** Only when the commit is one kernel or one policy file that my
   fork has not touched, and only after `git show "$TAG:path/to/file"` proves the same file exists
   in the new base with the same shape. `git cherry-pick -x <sha>` keeps the upstream SHA in the
   message so the next audit can see it is not ours to maintain.
3. **Re-slice.** When upstream distance is the problem (2,252 commits, 8 of them ours), retarget
   the branch instead of stacking picks: new branch off the tag, re-apply the substantive commits,
   keep the old branch until the new one benches at or above it. Apply them in this order:
   engine code first (`8645328d`), then the build gate (`2b829e1e`), then the backend prune
   (`3c3312d4`) so the prune runs against the new tree and never conflicts with a file that no
   longer exists, then tooling or serve fixes (`9aafcbf0`).
4. **Re-implement on top of upstream's rewrite.** When upstream reworked the same file my commit
   edits (the hyper-connection path in `src/core/gr.cpp` and `src/kernels/cuda/fused_gr.cu` is the
   live example), do not merge through the conflict. Read upstream's version, re-apply the intent
   by hand, then prove the fold survived: the launch count I fold is 6 dispatches down to 4 per
   verification step, and if the engine log or a launch-count probe does not show 4, the merge
   silently dropped the optimization.
5. **Reject with evidence.** A rejected experiment gets a written row (what changed, the measured
   before and after, the reason) so nobody re-runs it: `--mtp-q4 head` went 69.60 to 67.88 t/s and
   was reverted because on-the-fly dequant cost more than the bandwidth it saved.

Hard rule across all five: upstream backends never come in. Metal, SYCL, ROCm tuning headers,
`src/qwen35/`, `web/` and installer code are pruned in every re-slice, and `CMAKE_CUDA_ARCHITECTURES=89`
stays pinned in the build gate.

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

**Port targets and watchlist** (checked against GitHub on 2026-10-28; verify state before acting, these move):

| PR | state | read |
| --- | --- | --- |
| llama.cpp #31037 "MTP draft with a trimmed output head" | open | Same idea Strata already ships as `--mtp-draft-vocab` plus `tools/draft_vocab.py`. Use it as a second opinion, not as a port. |
| llama.cpp #29143 d2t draft-vocabulary trim (qwen35) | open | Same lane. |
| llama.cpp #27210 adaptive MTP draft depth | open | The llama.cpp shape of upstream Strata's cost-aware window (`STRATA_SPEC_COST`, 0.2.x). |
| llama.cpp #30792 native q8_0 KV in the flash-attn tile kernel | open | Real. Direct int8 tile math for `-ctk q8_0` parity work. |
| llama.cpp **#30790** | does not exist | Earlier versions of this skill named it; drop it. |
| llama.cpp #29887 GPU cache for MoE experts kept in host memory | merged | Already in `vendor/llama.cpp-exp-latest`; Strata's expert cache does this natively. |
| llama.cpp #27861 GPU-resident LRU for host-offloaded MoE experts | open | The same lane as #29887. |
| llama.cpp #30815 qwen4exp QSA with chunking | open | Prompt-side idea worth reading before tuning `--prefill auto:...` further. |
| llama.cpp #30592, #30316 qwen4exp nextn/MTP fixes, padding, multimodal MTP | open | Blocks any llama.cpp MTP tier re-bench. |
| Strata #3453 Ada sm_89 MMVQ table | merged into `v0.2.7` (PR still shows open) | Our `0cef00fa`. Do not re-apply on a re-slice. |
| Strata #4121 `STRATA_PIPED_MTP` batch-groups | open | Only matters if `--batch` ever gets used here. |
| Strata #3060 `STRATA_RAM_ADAPT` | closed | Expert refill path, carried by 0.2.x. |

---

### Step 2b: The Qwen3.8-Flash-Next fork family (audited 2026-11-05)

Three forks of Strata now matter, and they matter only for this one model. None is a re-slice
candidate. Each is a source of a switch, a data file, or an idea, in that order of preference.

| source | what it is | runs on Ada `sm_89`? | what to take |
| --- | --- | --- | --- |
| `eddoursul/Lamina` | Strata fork, diverged 2026-10-29; NVIDIA only, extra GPUs as expert tiers | yes, but do not build it | the retrained MTP draft head, as a data file |
| `simplexai-labs/PolyStrata` | Strata fork plus a second engine for 11 other models | for Qwen3.8 it runs "the Strata project's engine, unchanged" | documented opt-in switches, and its poly.7 RAM/VRAM work |
| `jesdga95/basalt` | Strata fork, Blackwell `sm_120` only | **no** | ideas and measurement method, never code |

**Lamina draft head (switch-only promote, the cheapest shape).**
`eddoursul/Lamina-Qwen3.8-Flash-Next-MTP` is not a GGUF and cannot go in a llama.cpp tier. It
ships `dense.txt`, `dense.bin`, `experts.bin`, `draft_vocab.bin`, which is exactly the runtime
directory `--mtp` already loads (the `tools/mtp_rt.py` layout). Verified 2026-11-05: the same 29
tensor names, dtypes and dims, the same `dense.bin` (116,099,072 B) and `experts.bin`
(707,788,800 B) sizes, and SHA-256 matching the model card. The only structural difference is the
draft vocab: 40,525 tokens against the served 106,299.

- Local copy: `vendor/Strata-data/mtp/rt-lamina`. Arm `mtp_lamina` in
  `bench-models/bench-strata-tuning-ab.sh` appends `--mtp <that dir>`; the engine's parser takes
  the last `--mtp` (`src/program/generate.cpp:2478`), so it replaces the config's head. The dir's
  own `draft_vocab.bin` is picked up automatically, so this is a head swap, not a vocab swap.
- Its own claim: 2.489 against 2.402 tokens a verify pass on IQ3_S (+3.6%), 2.433 against 2.344 on
  UD-Q4_K_XL (+3.8%). The baseline there is *basalt's* dense part, not the stock unsloth head, so
  it is not a like-for-like claim against what we serve.
- It was trained on IQ3_S and UD-Q4_K_XL output. Re-measure on whatever quant is actually served;
  acceptance does not transfer across quants.
- The 40,525-token vocab is the same size as the rejected `draft_en` arm, which cost about 7%
  (65.78 against ~71 t/s). The retrained dense part has to earn that back.
- Verdict: pending (paired A/B `base mtp_lamina base` launched 2026-11-05 on UD-IQ2_XS).

**PolyStrata: nothing for the model, three switches for this box.**
For Qwen3.8-Flash-Next it runs Strata's engine unchanged, so its Qwen numbers are Strata's
numbers. Its own engine covers the other eleven models and is out of scope here.

- `--adapt-async 1` - already in `strata-lean` (`src/program/generate.cpp:2484`) but in **no
  served config**. It needs the resident RAM mode or the arena on one GPU, which is what the plat
  tier is. PolyStrata measures +4-5% decode, +7-10% on code and long documents, on an RTX 5070
  (Windows, Q2_0). This is a switch, so measure it with `strata-tuning` before touching code.
- `STRATA_PIN_BY_NEED=1` - already present. Their +4.4% prompt-reading win is a Windows WDDM
  pinning cap (`RAM/2 - 8 GiB`); on Linux expect little.
- `--ram-experts-mib` - **absent** from `strata-lean` (a poly.7 addition). Holds less RAM for
  experts at 3-14% slower writes. Only worth carrying when RAM headroom is the binding
  constraint, which it is not here (about 14 GB free).
- poly.7 also claims 350-400 MiB less VRAM for reading prompts, taken from the expert cache. That
  is our `#2065` line (410 MiB free at request start), so read their `docs/BACKENDS.md` section
  before the next prefill tuning pass.

**basalt: not portable, still the best-documented source.**
`sm_120` only (`docs/platforms.md`), NVFP4 packs, so no engine code from it runs on Ada. Do not
attempt a port. Read it for three things:

- `docs/precision/isolated-costs.md` - a per-switch paired KLD gate: 100 chunks of wikitext-2, a
  t-based 95% interval, and each switch ruled lossless or measurable. That is the quality gate our
  identity probe cannot give us (greedy output is not reproducible here), so copy the *method*.
- The two-stage Q8_0 head: a six-bit screen then an exact pass, lossless by proof
  (`q80_vhead.cu`). A head-GEMV idea for `sm_89`.
- Its drafter (Q8_0/Q2_0 weights, fp16 draft attention, 2-bit head screen) is the ancestor of
  Lamina's dense part; `convert/mtp/rt/dense-original.bin` is the pre-Lamina baseline, useful as
  the control when judging any head swap.

**Reject on sight:** basalt's NVFP4 and `sm_120` kernels, PolyStrata's second engine (other
architectures), and Lamina's engine tree (a whole-tree divergence, 4,000+ commits from the same
Strata base). Take the data file, not the tree.

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

### Step 5a: Measuring a flag or env switch (no rebuild)

Use `bench-models/bench-strata-tuning-ab.sh` when the lever is a switch that already exists
in the binary. It overlays arms on the **production** run config
(`vendor/strata-lean/strata-iq3_xxs-ple_q8.json`), so a winner is one config line from
being served, and it writes a second file of engine self-reporting:

```bash
systemctl --user stop llama-swap.service
bash bench-models/bench-strata-tuning-ab.sh              # default 19-arm ladder
ARMS="base kv_grow routeprior15 draft_en base" bash bench-models/bench-strata-tuning-ab.sh
systemctl --user start llama-swap.service
# or let the skill script do the handoff:
MODE=tune .agents/skills/strata-cherry-pick/scripts/bench_strata.sh
./bench-models/bench-strata-tuning-ab.sh --arms   # list the arms without touching the GPU
```

Rows go to `logs/results/strata-tuning-ab.jsonl` (probe numbers, comparable to the refresh
A/B rows) and `logs/results/strata-tuning-diag.jsonl` (expert slots, decode hit rate, PCIe
share, VRAM free at request start, the `#2065` warning, pool worker placement, the arena
page-backing note, draft head token count, window-capture floor, and `read_bytes` per
generated token). The diag file is what separates a real win from a run that got faster by
spilling to SSD: MB/token should stay in this arch's PLE design traffic band (about 5
KB/token of n-gram rows); tens of MB/token means a spill.

The tier's own current diag numbers, for reference when judging an arm: 2,583 expert slots
(4.24 GiB) of 24,576 pairs, decode cache hit rate 58.6%, 3.8% of routed experts re-read
over PCIe, 410 MiB free at request start with the `#2065` warning firing, window captures
down to 282 MiB, arena on 4 KB pages (`vm.nr_hugepages=0`), draft head over 106,299 tokens.

### Step 5b: A/B parity benchmarking for a code change

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

**Decision Rules**:
- Pair the arm: run the same control arm before and after it in one boot, same CPU governor
  (`powersave` costs 15 to 25% of decode against `performance`). A delta under ~3 t/s is
  inside this tier's spread.
- If decode improves and needle passes: keep the change.
- If decode regresses (as happened with `--mtp-q4 head`, 69.60 to 67.88 t/s): revert
  immediately, and say so in the fork README so nobody re-runs it.
- If the lever changes routing or the draft vocabulary (`STRATA_ROUTE_PRIOR`,
  `--mtp-draft-vocab`, a rebuilt `--expert-profile`), the needle probe is necessary but not
  sufficient: read one real Pi session's output before it goes into the served config.
  Upstream measured an agent-task pass-rate dip at `ROUTE_PRIOR=1.5` and copy/code defects
  at 2.0; 1.5 is the ceiling here.
- Check the diag row for a spill before believing a speed number.

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
