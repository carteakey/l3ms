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
  1. *Switch first*: Most of the immediate headroom on this card is already compiled in as a runtime flag or a `STRATA_*` env switch (e.g. `--kv-grow` yielding +2.8 t/s with zero code changes, paired with `STRATA_PREFILL_LEND_PCT=40` to preserve 1,771 reclaimable slots up to 128k context; see [`docs/strata-tuning-matrix.md`](file:///home/kchauhan/repos/l3ms/docs/strata-tuning-matrix.md) and [`docs/incidents/2026-11-09-strata-kv-grow-oom.md`](file:///home/kchauhan/repos/l3ms/docs/incidents/2026-11-09-strata-kv-grow-oom.md)). Measure with `bench-models/bench-strata-tuning-ab.sh` before touching C++/CUDA code.
  2. *Cherry-pick second*: Selectively port verified CUDA kernels and algorithmic wins (from upstream Strata or llama.cpp PRs) into `vendor/strata-lean`.
  3. *Re-slice NEVER*: Do not pull multi-platform churn into the lean Ada tree.
- **Porting from external engines**: Algorithmic ideas and CUDA kernel fusions from `llama.cpp` (e.g. MTP draft vocab trimming, direct quantized Q8_0 KV in FlashAttention) or `vLLM` can be ported directly into Strata rather than running them in slower runtimes. llama.cpp is a source of ideas, not a speed tier. The Qwen3.8-specific fork family (Lamina, PolyStrata, basalt) and what to take from each is Step 2b; for this model they are a data file and a handful of switches, not code.

---

## Hardware & Quant Baseline

All evaluations are conducted against this fixed homelab target:
- **GPU**: NVIDIA GeForce RTX 4070 (12 GB VRAM, Ada Lovelace `sm_89`, 46 SMs, PCIe 4.0 x16).
- **CPU & RAM**: Intel Core i5-12600K (10 cores / 16 threads), 64 GB DDR5-5200 (4x16 GB).
- **OS & Kernel**: Linux 7.3.1 CachyOS (`performance` governor and `performance` EPP via `cpupower.service` since 2026-10-29).
- **Model**: Unsloth Dynamic V3.1 `UD-IQ2_XS` (`strata-ud-iq2_xs-ple_q8.json`, primary served Plat tier since 2026-11-05) and `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/` (`IQ3_XXS`, baseline peer), both paired with the unquantized Q8_0 PLE table on NVMe. Tracked config backups live in [`maintenance/configs/strata/`](file:///home/kchauhan/repos/l3ms/maintenance/configs/strata/).
- **Baseline numbers** (`logs/results/qwen38-refresh-ab.jsonl`, same quant, same probe):

  | arm | engine | decode mean | min / max | pp8192 |
  | --- | --- | --- | --- | --- |
  | `strata_old` | v0.1.44.1 | 60.58 | 56.4 / 64.2 | 1911 |
  | `strata_new` | v0.1.45 | 63.83 | 62.0 / 67.7 | 1919 |
  | `strata_v026` | v0.2.6 | 65.53 | 62.2 / 70.5 | 1876 |
  | `strata_lean` (before the sm_89 table) | fork | 65.08 | 59.3 / 72.9 | 1891 |
  | `strata_lean` (after PR #3453 table) | fork | 69.60 | 65.4 / 76.1 | 1876 |
  | `strata_lean` + `STRATA_ADAPT_FETCH=1` | fork | 70.43 | 68.2 / 72.8 | 1885 |
  | `strata_lean` + `STRATA_HC_Q8=1` (2,949 slots) | fork | **73.00** | **69.0 / 75.0** | **1888** |

  All these runs passed `needle_4k`. Prefill has been flat (1,876 to 1,996) across every
  Strata arm since 2026-10-07 while decode moved 52 to 69.6: the prompt lane is the part
  of the fork that has not kept pace, and the 4070 is behind an unpruned v0.1.44.1 there.

- **Fork size and upstream distance**: upstream went from `v0.1.45` to `v0.2.12`
  (3,502 commits from the fork's base `477ac13c`, 2026-10-16). Most of what the fork
  carried is now upstream, so the 2026-11-07 audit found only two picks still needed:
  the external PLE shard (`9842df41`) and the serve thinking wrap (`9aafcbf0`). Already
  upstream and not re-applied: the Ada `sm_89` MMVQ tables (`kIlRows_89`, default on),
  `STRATA_QSA_FUSE_NR` (default on), `STRATA_GR_NORM_FAST`, the hyper-connection
  inject/gate-mean fold (`gr.cu` is byte-identical to ours), the Q5_0 expert-dot offset
  fix (`aa678f10`) and `tools/iq_pack.py` v4, which is what the Unsloth Dynamic V3.1
  intake needed to run `UD-IQ2_XS` natively (75.8 GB iso-size contender, zero non-Ada
  bloat). Still ours and not upstream: the build
  gate (`2b829e1e`), the backend prune (`3c3312d4`), `STRATA_ROUTE_TAIL_SKIP=0`
  (`3733343d`) and the PLE shard. The re-slice measured equal decode and 12% worse 8k
  prefill, so it is a reference arm, not served; the one promotable change it exposed was
  a default, not a commit (Step 1d).

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

**Read the release notes first.** They are the fastest filter there is: a release note names the
card class, the quant, the switch and the measured delta, which is exactly the tuple needed to
decide whether a change can touch this box. The commit log cannot be read that way, and 3,502
commits is a lot of noise for four candidates. Use the notes to build a shortlist, then use the
tag to prove each item's state.

The filter is: **RTX 4070, 12 GB, `sm_89`, Linux, single GPU, UD-IQ2_XS, Q8_0 PLE on NVMe, MTP
draft head, 128k context.** Anything naming Mac, Metal, Strix Halo, gfx12/gfx1201, DGX Spark,
Intel Arc, a peer/second GPU, a layer split, or a `UD-Q4`/`UD-IQ4`/`IQ3_S` quant is not ours.
For the 0.2.12 notes that leaves:

| note item | applies? | why |
| --- | --- | --- |
| 12 GB cards: file-tier re-check no longer counts the engine's own memory | **check, probably no action** | the fallback line `expert reads now go through the file cache (X GiB of RAM free for them, Y GiB limit)` never appears in our log; the `expert arena read through the file cache` line is the normal arena load, not the fallback |
| `STRATA_ADAPT_FETCH=1\|2`, fetched misses into the card's cache, decode +10.2% | **taken** | our decode ran 10-12% PCIe misses, exactly the regime; ported as `20c19b89` and served as `=1`. Isolated +9.42 t/s on the reference tree, 55.6 -> 70.4 on the fork, hit rate 46.6% -> 67.2% |
| `STRATA_SELECT_TWO_PASS=1`, 128K +1.65% | yes, small | we serve 128k; prefill-only, low value |
| tool-call fixes (#5312, #5409, #5276, #5281, #4828, #4832, #4622) | yes, quality not speed | they live in `serve/server.py`; worth reading before any agent-facing change |
| row-table over-read that faults long CUDA prompts (#5108) | yes, correctness | 128k context is the exposure |
| tuned DFlash2 drafter | no | 12 GB gains are prompt-side on UD-IQ4_XS-v31; we run IQ2_XS and MTP |
| `STRATA_IQ4_W4A8` | no | IQ4_XS / IQ4_NL experts only |
| `STRATA_PF_FUSED_Q50` | no | Radeon gfx12 and Q5_0 layers |
| `STRATA_VNNI_TIER` | no | CPUs with AVX-512 but no VNNI; Alder Lake has neither |
| Mac, Strix Halo, Radeon, DGX Spark, peer/layer split, #3559, #5158 | no | wrong backend or multi-GPU |

A release note's headline can also be a bug fix rather than a kernel. The 3060's "2K 87 to 264
tokens per second" is mostly the file-tier fix, not a faster prompt path. On this box the fix
does not fire, so the applicable slice of that headline was the CPU share limit (+19% at 2k),
which is what the measurement found. Read the note, then check which half of it your box is in.

**The fork is cut from 0.1.45, so every release since then is unclaimed, not just the newest.**
Read the notes from the fork's base forward. What 0.2.7 to 0.2.11 still hold for this box, and
whether it has already been tried:

| item | release | status here |
| --- | --- | --- |
| `STRATA_ROUTE_PRIOR=1.5`, residency-biased routing on 12 GB NVIDIA | 0.2.7 | **measured +43.7% (102.23 t/s) on 2026-10-28 and deliberately not served**: it re-picks experts, and upstream's own agent-task check read 4.5 of 10 against 5.6. Upstream reverted it as a default in 0.2.8 for the same reason. Do not re-litigate without an agent-quality gate. |
| `STRATA_MMVQ_IL_89` RTX 40 tables | 0.2.7 | taken (`0cef00fa`), now upstream and on by default |
| `--slots 2`, +10-12% aggregate | 0.2.7 | not applicable to a single-user Pi; it pays only for concurrent requests |
| `--ple-io ram`, prompts +16-30% | 0.2.7 | impossible here: the Q8_0 PLE table is 54.4 GB and the non-PLE weights are 43.8 GB against 64 GB RAM |
| `--mtp-draft-vocab draft_vocab_en.bin`, +2.7-3.1% for English | 0.2.7 | **measured slower here** (65.78 against ~71 t/s); rejected |
| `--kv q4_0` on a 12 GB card | 0.2.7 | upstream could not reproduce a gain; we serve K8V4 |
| `STRATA_KVA=1`, long prompts +11-18% | 0.2.7 | opt-in and approximate (last 12 layers); it can drop details from long lists, so it needs a task-suite gate before it is worth a look |
| DFlash2 drafter, one card +5.8% to +12% | 0.2.10 | sm_89 is a supported architecture, but no 12 GB Ada number exists; it is a new drafter plus a large intake, and MTP already accepts 77-88% here |
| CPU/PCIe miss share follows the CPU, ceiling 0.95 | 0.2.11 | the `strata_v0212` arm already ran it and decode was equal (62.59 against 62.60), so the resident fraction here is not the regime upstream's +40% single-12-GB rows were measured in |
| `STRATA_ADAPT_FETCH=1\|2`, decode +10.2% | 0.2.12 | **ported (`20c19b89`) and served as `=1`**: +9.42 t/s isolated on the reference tree, 55.6 -> 70.4 on the fork, hit rate 46.6% -> 67.2%. `docs/strata-perf-next-steps-2026-10.md` §10 |
| tool-call and server fixes (#5312, #5409, #5276, #5281, #4828, #4832, #4622, #5040, #4701, #2518, #2892) | 0.2.11 | **absent from the fork**: its `serve/server.py` is 7,164 lines against 10,540 in 0.2.12. These are agent-quality and correctness gaps, not tps, and they are the largest unclaimed block for daily Pi use |

The filter that does the work: a note is only a candidate if it names **this** card class, **this**
quant and **one** card. Everything naming a second card, a layer split, a peer device, Mac,
Metal, Strix Halo, gfx12/gfx1201, DGX Spark, Intel Arc, `UD-Q4`/`UD-IQ4`/`IQ3_S`, or a K-quant
expert is not ours. `STRATA_MMQ_KQ` is K-quant only; `STRATA_IQ4_W4A8` is IQ4 only;
`STRATA_PF_FUSED_Q50` is Radeon gfx12 and Q5_0; `STRATA_VNNI_TIER` needs AVX-512 without VNNI,
which Alder Lake does not have.

Inspect what upstream has changed since the base tag (`v0.1.45`) or between releases.
Compare against the newest tag, not against the tag the fork was cut from:

```bash
cd vendor/strata-lean
git fetch origin --tags -q
NEWEST=$(git tag --sort=-v:refname | head -n1)     # v0.2.12 on 2026-11-07

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
git worktree add -b 4070-main-lite-0.2.12 ../strata-v0.2.12 v0.2.12
cd ../strata-v0.2.12

# 1. Prove which of our commits are already upstream BEFORE picking anything.
#    An empty diff means upstream already has it; do not re-apply.
git diff --stat v0.2.12 HEAD -- src/kernels/cuda/gr.cu src/kernels/cuda/native_mmvq.cu

# 2. Pick only the still-ours commits, smallest first.
git cherry-pick -x 9842df41 9aafcbf0

# 3. Build with the lean flags. The gate and prune are optional for a reference arm.
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc
cmake --build build --target strata -j6
```

`-j6` is not a preference. `-j14` crashed `nvcc` on `mmq-instance-tqk6.cu` (rc 139) while
the served model held 14 GB; the retry at `-j6` finished clean. A worktree build is 1.0 to
1.5 GB, so it fits in the 28 GB free on `/home`.

Expect a conflict in `CMakeLists.txt` if the build gate is picked. For a reference arm, skip
the gate and the prune: neither changes performance, and the prune conflicts with upstream's
new backend directories. Keep the old branch until the re-sliced one benches at or above it;
the comparison is `ARMS="strata_lean_ud strata_v0212"` once the arms exist in
`bench-qwen38-refresh-ab.sh`.

The 2026-11-07 run is the worked example: two picks, a clean build, and a result that was
equal on decode and worse on 8k prefill. That is a valid outcome, not a failure. Write it down
and move to Step 1d, because the tree being worse does not mean the upstream work has nothing
in it.

---

### Step 1c: Merge patterns I actually use (five shapes, cheapest first)

1. **Switch-only promote (no merge at all).** The change is a flag, a `STRATA_*` env var, a
   taskset mask, or a data file (`--expert-profile`, `--mtp-draft-vocab`). It is measured by
   `strata-tuning`, and a winner lands as one line in the run config's `args` or `env` block,
   synced to `maintenance/configs/strata/`, recorded in [`docs/strata-tuning-matrix.md`](file:///home/kchauhan/repos/l3ms/docs/strata-tuning-matrix.md),
   plus a `name`/`description` refresh in `llama-swap.yaml`. No engine git history changes. This is
   the most common shape and it should be tried before any merge. It is also how an upstream *default*
   change gets taken when the tree itself is not worth carrying (Step 1d: the CPU share limit).
2. **Single-commit cherry-pick.** Only when the commit is one kernel or one policy file that my
   fork has not touched, and only after `git show "$TAG:path/to/file"` proves the same file exists
   in the new base with the same shape. `git cherry-pick -x <sha>` keeps the upstream SHA in the
   message so the next audit can see it is not ours to maintain. Worked example (2026-11-07):
   `24c6c2f6` (`STRATA_ADAPT_FETCH`) applied cleanly to the headers, `verify.cpp` and
   `verify_kernels.cu`, and conflicted in three places. Resolve by intent, not by taking a side:
   the pruned `metal/` stub is dropped, `expert_source.cpp` keeps the fork's simpler lookahead
   (the upstream hunk there was an unrelated confidence-scoring refactor), and `generate.cpp`
   keeps the fork's verify call and gains only the new diagnostic block. Then prove the port is
   inert with the switch off before measuring it on: the ported binary read 55.63 t/s against a
   closing control of 56.47, and only then 70.43 with the switch on.
3. **Re-slice.** When upstream distance is the problem (3,502 commits from the 2026-10-16 base,
   16 of them ours and only 2 still needed on 2026-11-07), retarget the branch instead of stacking
   picks: new branch off the tag, re-apply the substantive commits, keep the old branch until the
   new one benches at or above it. Apply them in this order: engine code first (`8645328d`), then
   the build gate (`2b829e1e`), then the backend prune (`3c3312d4`) so the prune runs against the
   new tree and never conflicts with a file that no longer exists, then tooling or serve fixes
   (`9aafcbf0`).
4. **Re-implement on top of upstream's rewrite.** When upstream reworked the same file my commit
   edits (the hyper-connection path in `src/core/gr.cpp` and `src/kernels/cuda/fused_gr.cu` is the
   live example), do not merge through the conflict. Read upstream's version, re-apply the intent
   by hand, then prove the fold survived: the launch count I fold is 6 dispatches down to 4 per
   verification step, and if the engine log or a launch-count probe does not show 4, the merge
   silently dropped the optimization.
5. **Reject with evidence.** A rejected experiment gets a written row (what changed, the measured
   before and after, the reason) so nobody re-runs it: `--mtp-q4 head` went 69.60 to 67.88 t/s and
   was reverted because on-the-fly dequant cost more than the bandwidth it saved. The Lamina draft head
   (2026-11-05) stayed out for the same reason at the other end of the scale: its acceptance gain did not
   reproduce (pooled +0.6 points at SE 0.56 over three ladders) and its decode effect flipped sign between
   runs, even though it does free a reproducible 132 MiB. Row: `docs/strata-perf-next-steps-2026-10.md` §9.
   The v0.2.12 re-slice (2026-11-07) is the same shape at tree scale: equal decode, 12% worse 8k
   prefill, kept as a reference arm at `vendor/strata-v0.2.12` rather than served. Row: §10.

Hard rule across all five: upstream backends never come in. Metal, SYCL, ROCm tuning headers,
`src/qwen35/`, `web/` and installer code are pruned in every re-slice, and `CMAKE_CUDA_ARCHITECTURES=89`
stays pinned in the build gate.

---

### Step 1d: Audit the defaults, not only the commits

A re-slice can be a wash and still contain the win, because upstream often changes a *default*
the fork pinned. On 2026-11-07 the tree move measured equal decode (62.59 against 62.60 paired)
and 12% worse 8k prefill, yet the same afternoon produced a +19% 2k prefill win with no 8k or
decode cost: upstream had raised the CPU prefill share limit from 1,024 to 3,072 tokens.

Do this after the tree A/B, before writing the tree off:

```bash
# 1. Diff the default constants the fork pins against the tag.
git diff v0.2.12 HEAD -- src/prefill/prefill.cpp include/strata/prefill/share_rules.hpp

# 2. Read the new constant's comment. It usually names the measurement and the card class.
git show v0.2.12:include/strata/prefill/share_rules.hpp | head -30

# 3. Test it as an env arm on the CURRENT binary. No rebuild.
ARMS="strata_lean_ud strata_lean_share3072 strata_lean_ud" \
  bash bench-models/bench-qwen38-refresh-ab.sh
```

The fork's `cpu_share_max()` returns `kStreamAllMinDefault` (1,024) when the default is armed.
Setting `STRATA_PREFILL_CPU_SHARE_MAX=3072` explicitly bypasses that guard, so the switch is
measurable without a build. The winner lands as one `env` line in the run config, not a commit:

```json
"env": { "STRATA_PREFILL_CPU_SHARE_MAX": "3072" }
```

The same shape applies to every default the fork overrides: `STRATA_ROUTE_TAIL_SKIP`,
`STRATA_PCIE_AUTO_SPLIT`, `STRATA_MMQ_KQ`, the bare `--prefill auto` chunk ceiling. Diff the
constant, read its comment, measure it as a switch, and only then decide whether the tree is
worth carrying.

Two traps in this step:

- **A hardcoded log line is not the value.** The fork prints "the CPU share is ON by default for
  prompt chunks below 1024 tokens" from a string literal in `0.1.45`, even with the env set to
  3072. Confirm the switch reached the engine from `/proc/<pid>/environ`, not from the log.
- **The tree's own reservation can be the regression.** v0.2.12 reserves an extra 256 MiB "for
  what is allocated after the cache", which costs 184 expert slots (2,527 against 2,711) and
  shrinks the prompt chunk from 6,912 to 6,144. That, not the prompt path, is the 8k loss:
  `STRATA_PF_SMALL=0` recovered only 4 of the 13 points. When a re-slice loses prefill, read
  the slot count and chunk size in the engine log before blaming a kernel.

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
- Verdict, measured 2026-11-05 on UD-IQ2_XS (two paired ladders, `base` bookending):
  **acceptance does not reproduce.** Run 1 read 79.9% to 82.1% (+1.4 points); run 2 read 80.1/81.5%
  to 80.8% (-0.1). Pooled over both, 80.8% (11,837 of 14,647) to 81.4% (5,906 of 7,254), which is
  +0.6 points at a standard error of 0.56 - not significant. Decode contradicts itself the same way:
  +3.9 t/s in run 1, -2.9 in run 2. The README's +3.6% is against basalt's dense part on IQ3_S, not
  against the stock head on the quant we serve.
- What *is* reproducible is the VRAM. The head is Q6_K over the vocabulary, so 106,299 tokens is
  212.9 MiB and 40,525 is 81.2 MiB: **131.7 MiB freed**, confirmed in both runs. With
  `--expert-cache auto` the engine spends it on expert slots immediately, 2,711 to 2,790 (+79, +2.9%),
  at unchanged headroom (412 to 414 MiB free with everything loaded). It never shows up as free VRAM.
- The headroom option, measured 2026-11-05 in a third ladder (`base base_r960 mtp_lamina_r960 base`,
  no spill, io/tok 0.023-0.032): `--vram-reserve-mib 960` clears `#2065` in both variants (674 MiB free
  with everything loaded, 538-540 MiB at the worst prompt capture) but costs very differently.
  Stock head at 960: 2,552 slots, hit rate 39.6%, **52.30 t/s against a 62.30 base pair (-16%)**.
  Lamina head at 960: 2,633 slots, hit rate 47.8%, **60.77 t/s (-1.5, inside the spread)**.
- So the head's real value is not the drafts, it is that 132 MiB is worth 81 expert slots at a fixed
  reserve, which is worth about 8.5 t/s once the reserve has already been paid. That reverses the
  earlier "do not promote" call: carry the head if the `#2065` margin is wanted, because it is what
  makes the reserve increase affordable. Do not carry it for acceptance, which is flat at 80.5-80.8%
  in every arm measured.
- Still true: with no reserve increase, `base` is the fastest configuration measured (62.30 t/s,
  2,711 slots) and the head buys nothing there.
- **Decision 2026-11-05: not served.** The plat tier stays on the stock head and the stock reserve.
  The measured write-up is `docs/strata-perf-next-steps-2026-10.md` §9. The head stays on disk at
  `vendor/Strata-data/mtp/rt-lamina` and the `mtp_lamina` / `mtp_lamina_r960` / `base_r960` arms stay
  in the harness, so re-testing it is one ladder, not a re-download.

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
(`vendor/strata-lean/strata-ud-iq2_xs-ple_q8.json`, tracked backup in `maintenance/configs/strata/`),
so a winner is one config line from being served, and it writes a second file of engine self-reporting:

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

The tier's own current diag numbers, for reference when judging an arm: 2,949 expert
slots (`STRATA_HC_Q8=1` saving 0.39 GiB of hyper-connection VRAM for +238 slots, paired with
`--kv-grow` elastic K/V and `STRATA_PREFILL_LEND_PCT=40` capping prompt loan to 881
slots / 1.41 GiB, leaving 1,771+ reclaimable slots up to 128k
context), decode cache hit rate ~72.4% (`STRATA_ADAPT_FETCH=1`), 412 MiB free at request
start with the `#2065` warning firing, arena on 4 KB pages (`vm.nr_hugepages=0`), draft head
over 106,299 tokens.

### Step 5b: A/B parity benchmarking for a code change

Always benchmark using `bench-models/bench-qwen38-refresh-ab.sh` with the standard system frontmatter:

1. Stop the router to free VRAM:
   ```bash
   systemctl --user stop llama-swap.service
   ```
2. Run the probe suite. The served quant is UD-IQ2_XS, so use the `_ud` arms, not the
   IQ3_XXS one:
   ```bash
   ARMS="strata_lean_ud strata_v0212 strata_lean_ud" bash bench-models/bench-qwen38-refresh-ab.sh
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
  inside this tier's spread. Read decode and prefill separately, because they behave
  differently: on 2026-11-07 prefill held to about 1% across three ladders while identical
  control arms read 51.4 and 57.2 t/s in one ladder and 61.8 and 63.4 in the next, with the
  desktop stack (plasmashell, firefox, zed, the agent itself) as the only variable. Only paired
  arms inside one ladder are comparable, and a control pair that disagrees by more than 3 t/s
  invalidates that ladder's decode rows, not its prefill rows. When a decode delta matters,
  run the same arm twice beside the control (the 2026-11-07 decisive ladder was
  `control, v0212, control, v0212` and settled 62.60 against 62.59).
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
2. Commit and push to `carteakey/strata-lean`:
   ```bash
   cd vendor/strata-lean
   git commit -am "feat(cuda): <description of change>"
   git push origin strata-lean
   ```
3. Hot-reload `llama-swap`:
   ```bash
   kill -HUP $(pgrep '[l]lama-swap')
   ```
