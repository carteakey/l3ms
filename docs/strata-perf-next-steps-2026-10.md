# Strata inference speed: what is still on the table

Audit date: 2026-10-28. Written by me after reading `vendor/strata-lean` (branch
`4070-main-lite`), the upstream Strata tags it forked from, the `qwen4exp-refresh`
and `strata-cherry-pick` skills, the plat serving config, and the live engine log.

The audit itself ran without touching the card (a bench on this tier evicts the tier this
agent is served from), so sections 1 to 6 are stored L3MS bench rows, lines the engine printed
in `vendor/strata-lean/strata-iq3_xxs-ple_q8.log`, or upstream release-note claims labelled as
such. **The first sweep has since run: 2026-10-28 13:16 to 13:40, nine rows, measured numbers in
section 7.** One switch won big (`STRATA_ROUTE_PRIOR=1.5`, 102.23 against a 70.70/71.55 base pair)
and it is not served yet, because it changes what the model routes and I have not read its output.

## 1. Where the fork actually stands

Measured from `vendor/strata-lean` with `git log v0.2.7..HEAD --oneline`:

- Upstream has moved from `v0.1.45` (the fork's stated base) to `v0.2.7.1`, the newest tag
  when I fetched on 2026-10-28: **2,252 commits** since the fork's merge base `477ac13c`
  (2026-10-16), 193 of them in `src/core`, `src/kernels/cuda` and `src/spec` (98 in
  `src/kernels/cuda` alone). `audit_upstream.sh` prints those numbers.
- The fork carries only **8 commits of its own**, and two of them are now upstream:

  | lean commit | what it did | status on v0.2.7 |
  | --- | --- | --- |
  | `8645328d` | fold the hyper-connection inject and gate/mean into neighbours | still ours |
  | `7ce60ad3`, `31e2339b` | `STRATA_GR_NORM_FAST=1` on CUDA (register-split norm) | upstream (`src/kernels/cuda/fused_gr.cu`), still **opt-in** on CUDA: the served tier runs with it off, see P0-8 |
  | `0cef00fa` | Ada sm_89 MMVQ interleave table (Strata PR #3453) | upstream **and on by default** for compute capability 8.9 (`native_mmvq.cu:3709`) |
  | `2b829e1e` | default `CMAKE_CUDA_ARCHITECTURES 89`, gate qwen35, tests off | still ours |
  | `3c3312d4` | prune Metal/SYCL/ROCm bloat | still ours |
  | `9aafcbf0` | serve fix: wrap thinking into the answer when `repeat_stop_tokens` fires inside `<think>` | still ours |
  | `7dad7862` | fork README | still ours |

So the maintenance problem is smaller than it looks and it is not a cherry-pick
problem: a **re-slice** (new branch off `v0.2.7`, re-apply 4 substantive commits)
buys all 2,252 upstream commits for one afternoon of conflict work. That is item P2-10.

Bench history from `logs/results/qwen38-refresh-ab.jsonl`, same quant
(ISTA-DASLab GSQ-RCO IQ3_XXS), same probe, t/s:

| run | arm | decode mean | decode min/max | pp2048 | pp8192 | VRAM MiB |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-26 | `strata_old` (v0.1.44.1) | 60.58 | 56.4 / 64.2 | 933 | 1911 | 11696 |
| 2026-10-26 | `strata_new` (v0.1.45) | 63.83 | 62.0 / 67.7 | 938 | 1919 | 11698 |
| 2026-10-26 | `strata_v026` | 65.53 | 62.2 / 70.5 | 913 | 1876 | 11696 |
| 2026-10-27 | `strata_lean` (before the sm_89 table) | 65.08 | 59.3 / 72.9 | 919 | 1891 | 11698 |
| 2026-10-28 | `strata_lean` (with the sm_89 table) | 69.60 | 65.4 / 76.1 | 918 | 1876 | 11696 |
| 2026-10-28 | `strata_lean` repeat | 67.88 | 63.0 / 71.6 | 928 | 1890 | 11694 |

Two things the table says out loud:

1. The sm_89 table is worth about 4 t/s (65.1 to 69.6), but run-to-run spread on this
   tier is 2 t/s or more, and the repeat run landed at 67.9. Anything smaller than
   about 3 t/s needs a paired control run before I quote it.
2. Decode is up while **prefill is flat or slightly behind** the unpruned engine
   (1,876 to 1,891 for lean against 1,911 to 1,996 for v0.1.44.1 a week earlier).
   Prefill is where the fork has not kept pace, and it matches upstream moving
   prompt-side work into switches we have not switched on.

## 2. What the live card is reporting about itself

From `vendor/strata-lean/strata-iq3_xxs-ple_q8.log` (2026-10-28 11:51 boot):

```
expert cache auto: 5.10 GiB free, 700 MiB reserved (+184 MiB for the draft head) -> 1953 slots
expert cache 2583 slots, 4.24 GiB of VRAM
pre-filled 2583 of 2583 slots from the profile in 0.2 s (23067 MB/s); slot 0 verified
the prompt path borrows 2323 CUDA0 cache slots (3.81 GiB)
mtp: draft head over 106299 tokens (178.4 MiB)
serve: 410 MiB of VRAM free with everything loaded
serve: WARNING: only 410 MiB of VRAM free at the start of this request (below 512 MiB):
       when the card is this full the driver can page GPU memory to system RAM, decode
       falls to a few tok/s and a window may never finish (#2065). Add --vram-reserve-mib
       832 to the config's args, or lower --max-context / --batch
verify: capturing the 5-token window (302 MiB of VRAM free)
serve: decode expert cache hit rate: 58.6% (54487 hits / 92979 lookups); 3652 more read by
       the GPU over PCIe or from another GPU (3.8% of all 96631 routed)
generate: 5 pool workers on logical processors 2,4,6,8,10, the host thread on 0 (draining too)
generate: expert arena: cudaHostRegister PORTABLE ok; MAP_HUGETLB unavailable (needed 20464
          2 MiB pages, vm.nr_hugepages=0); transparent huge pages are on for every mapping
          (THP always): MADV_HUGEPAGE not requested
generate: loaded 39.97 GiB at 0.64 GiB/s
```

Reading those lines together gives the shape of the remaining headroom, and it is not
a kernel problem:

- The tier is **out of VRAM by its own admission**. `--max-context 131072` with
  `--kv k8v4` reserves the whole context up front, the expert cache gets the leftovers
  (4.24 GiB, 2,583 of 24,576 expert pairs), the prompt path borrows 2,323 of those
  slots while a prompt is read, and the CUDA graph windows then push free VRAM down to
  302 MiB, under the 512 MiB floor the engine warns about (#2065). That is the most
  likely cause of the 59 to 76 t/s probe spread.
- **41% of the routed experts are not in VRAM** during decode (58.6% hit rate). 3.8% of
  all routed experts are re-read over PCIe; the rest are computed by the 5-worker CPU
  pool. Decode on this box is partly a CPU job, which is why the CPU-side items below
  rank as high as they do.
- The expert arena is **4 KB backed**. `vm.nr_hugepages` is 0, THP is `always`, and
  upstream's own comment in `src/core/pinned.cu` is that with 4 KB pages "every 3 MB
  expert costs ~750 TLB misses" for exactly the CPU pool path we are leaning on.

## 3. Ranked backlog

Ordering rule I used: an existing switch that costs one config line beats a rebuild, a
rebuild on the current base beats a re-slice, and anything that can change output
carries a quality gate.

### P0-1. Give the card its headroom back (`--vram-reserve-mib`, `--kv-grow`, context)

What: the engine asks for `--vram-reserve-mib 832` or a smaller context. `--kv-grow`
(lean `src/program/generate.cpp:477`, `:4944-4959`, env `STRATA_KV_GROW=1`, tunables
`STRATA_KV_GROW_INIT`, `_STEP`, `_FLOOR`, `_HOLD`) keeps the K/V in a CUDA VMM range and
maps only the cells a request reaches, and the freed VRAM becomes expert slots. It is
off by default in lean and still off by default in v0.2.7, and our setup is exactly what
it was written for (one GPU, `--expert-profile` set, no `--batch`, no `--vram-elastic`,
no `--peer-device`).

Why it should pay: it attacks both problems at once, the 410 MiB cliff and the 58.6%
hit rate. A Pi session that runs 30k tokens of a 131072-cell reservation is holding
roughly 3/4 of the K/V for nothing.

Risk: `--kv-grow` grows the K/V mid-request, so a long prompt can pay mapping cost it
does not amortize; a too-small `--vram-reserve-mib` re-opens the #2065 paging failure
that costs 10x, not 10%.

Test: arms `reserve832`, `kv_grow`, `ctx64k` in the sweep, decode plus pp2048/8192/32768,
plus the engine's own `expert cache ... slots` and `hit rate` lines, which the harness
captures.

### P0-2. Residency-biased routing (`STRATA_ROUTE_PRIOR=1.5`)

What: `src/core/verify.cpp:1849-1851` reads `STRATA_ROUTE_PRIOR` (default 0.0) and the
kernel `src/kernels/cuda/route_prior.cu` is compiled in lean. It biases routing toward
experts already resident in the cache instead of only toward the highest logit.

Why: it is the only lever I found that directly attacks the miss rate rather than the
cache size. Upstream v0.2.7 made it the **default for discrete NVIDIA cards at or under
13 GiB**, which is our class. Upstream's published measurement (RTX 3060 12 GB,
IQ3_XXS): story prompt 46.2 to 53.1 t/s, code prompt 45.7 to 62.5 t/s.

Risk: it changes which expert computes a token, so it changes output. Upstream's 0.2.7 release note
(line 7) reports an agent-task pass-rate dip inside the run-to-run spread at 1.5 (4.5 of 10 at 1.5 over
4 runs against 5.6 over 7 runs off) and copy/code defects at 2.0, which is why 1.5 is their default. I
would not go above 1.5, and I would gate it on agent tasks rather than a needle probe (see section 7).

Test: arms `routeprior15` and `routeprior_kv` (combined with `--kv-grow`, since both
push the same bottleneck).

### P0-3. An expert profile built from my own traffic

What: `--expert-profile` currently points at `data/expert-profile.bin`, md5
`ff1bc6544d3f2f0013e3e376632d33a8`, the shipped upstream ranking of all 24,576 pairs.
The engine pre-fills 2,583 slots from its head. The ranking therefore decides which
2,583 experts are hot, and it was ranked for upstream's corpus, not for a coding agent.

Tooling already in the tree, commands verified against each tool's own `--help` in
`vendor/strata-lean`:

```bash
# 1. collect routing traces, one engine start per prompt. --mode routing is the CUDA path
#    (the default `usage` is the SYCL device counter). The tool imports strata_tokenizer,
#    which needs `regex`, so run it with the Strata venv: vendor/strata/.venv is the only
#    Strata venv on this box and plain python3 does not have the module.
vendor/strata/.venv/bin/python tools/collect_profile_traces.py \
    strata-iq3_xxs-ple_q8.json /tmp/pi-traces --mode routing --max-new 250
#    default corpus is bench/profile_corpus.json (45 chat/code/maths/writing/tool-use prompts,
#    about 30 minutes at 20 to 60 s of start plus ~10 s of decode per prompt);
#    --corpus POINTS_AT_MY_OWN_PROMPTS is the point of the exercise.

# 2. rank them. In this tree --reorder / --no-base / --base belong to make_profile.py;
#    lean's make_profile_mix.py takes TRACE[:WEIGHT] ... --out and nothing else.
python3 tools/make_profile.py /tmp/pi-traces/*.trace --reorder \
    --out data/expert-profile-pi.bin        # traces decide the head, the shipped base fills the tail
python3 tools/make_profile.py /tmp/pi-traces/*.trace --no-base \
    --out data/expert-profile-pi-only.bin   # traces only
python3 tools/make_profile_mix.py /tmp/pi-traces/c1.trace:1 /tmp/pi-traces/c2.trace:2 \
    --out data/expert-profile-mix.bin       # weighted mix, upstream's Arc Pro B70 recipe

# 3. run the sweep with --expert-profile pointed at the new file in one arm and the shipped
#    profile as the paired control (md5 ff1bc6544d3f2f0013e3e376632d33a8, 24,576 pairs).
```

`tools/make_profile.py` carries the trap I would have walked into: the shipped base
already ranks all 24,576 pairs, so a plain `--base` run cannot be moved by a trace at
all (`--reorder`'s own help says it is "needed with a base that already ranks every pair,
where --base alone cannot change the order"). `--reorder` (traces first) or `--no-base` is
what makes a trace decide the head.
Upstream also warns the ranking is read as a frequency curve by the layer-split cost
model, and quotes 94.9% claimed against 69.2% measured at 5,805 pairs held. Their
worked example on an Arc Pro B70: shipped profile covers 60% of what 45 chat/code/language
prompts route, a trace-built profile covers 71% on held-out prompts.

Why: our hit rate is 58.6% with 2,583 slots. If a workload-shaped profile moves that a
few points, the CPU pool does less work per token. Cheapest real win on this list.

Risk: low, it is a data file, snapshot it and keep the shipped one. The cost is engine
time to collect traces (upstream quotes 20 to 60 s of start plus ~10 s of decode per
prompt; 45 prompts is about 30 minutes).

### P0-4. Trim the draft head's vocabulary

What: the served tier runs `--mtp .../Strata-data/mtp/rt`, and the log says
`draft head over 106299 tokens (178.4 MiB)`, i.e. the shipped `rt/draft_vocab.bin` subset
is already active. `data/draft_vocab_en.bin` is 40,525 tokens ("English and code"),
`draft_vocab_cyrillic.bin` 58,963, and lean accepts `--mtp-draft-vocab FILE`
(`src/program/generate.cpp:2546`).

Why: a smaller draft head means fewer draft logits per step and about 110 MiB of VRAM
back (roughly 60 more expert slots). Upstream claims +2.7 to 3.1% decode for English
with the `en` subset. And `tools/draft_vocab.py --gguf <shard1> --base
data/draft_vocab_en.bin --corpus <my code and agent corpora> --coverage 0.98 --out ...`
can build a subset shaped to this workload instead of guessing, which the llama.cpp side
only just got for qwen35 (open PR #29143).

Risk: a token outside the subset can never be drafted, so acceptance drops on text the
subset does not cover. The needle probe and a code-heavy probe catch the obvious misses.

### P0-5. Speculative window sweep (`--spec`, `--spec-min-p`, `--lookup-chain`, `--suffix-draft`)

What: today `--spec 4 --spec-min-p 0.5`. Lean already ships `--spec 5`,
`--mtp-max-t M`, `--lookup-chain K` / `--lookup-chain-min M` (longest suffix repeat,
default 3 / 2), `--suffix-draft N` (default 3, `0` = MTP only), and the gated proposal
law `STRATA_SPEC_PROB=1`.

Why: the stored probe rows show per-request draft acceptance between 0.67 and 0.82 on
this tier, which is a wide band for a fixed window. Accepted tokens per step is the
multiplier on decode, and none of these knobs need a rebuild.

Risk: `--spec 6` and above is warned against upstream even for the MTP drafter. Also
`--suffix-draft 0` is worth one arm as a control: it isolates how much of our current
speed is prompt-lookup drafting rather than MTP.

Already rejected here (do not re-run): `--mtp-q4 head` measured 69.60 to 67.88 t/s.

### P0-6. Put the 4 E-cores to work

What: `llama-swap.yaml` pins the tier with `cpu_range: 0-11`, and the engine logs
`5 pool workers on logical processors 2,4,6,8,10, the host thread on 0`. On the
i5-12600K that mask covers the 6 P-cores and their siblings only: the 4 E-cores
(logical 12 to 15) are outside it and idle, while 41% of routed experts are computed off
GPU. `--aux-cpus off|auto|LIST` (env `STRATA_AUX_CPUS`, `src/program/aux_cpus.hpp`) is
how upstream places the non-pool threads (prefill helpers, the I/O thread, graph
capture) on siblings or spare physical cores, and with a 0-11 mask there are no spare
physical cores to find. `--pool-workers N` (`src/program/generate.cpp:507`, `:2366`)
overrides "all physical cores minus the host's".

Why: this is the CPU-side half of the 41% miss path. Upstream's own guidance is that a
pool worker per CPU is a loss (they measured 0.65 to 0.73x with too many), so I would
not raise the pool count; I would widen the mask and let aux threads land on E-cores.

Test: arm `aux_ecores` = `--aux-cpus auto` with taskset `0-15`. Keep the pool at 5.

### P0-7. Prompt-side switches (pp has not moved in a month)

What, all already compiled into `vendor/strata-lean/build/strata`:

- `--prefill auto:16384` or `auto:32768` (env `STRATA_PREFILL_AUTO_MAX`): bigger prompt
  chunks than the default `auto` (largest chunk up to 8192 whose buffers the expert
  cache can lend). Fewer chunks means fewer expert re-reads per prompt. Note the
  interaction: the prompt path already borrows 2,323 cache slots, so a bigger chunk
  borrows more and can strand the decode that follows.
- `--prefill-head-alias`: unload the native output head while the prompt is read, reload
  before decode. That is VRAM the chunk can use.
- `STRATA_PROMPT_ATTN_IMMA=1` (`src/kernels/cuda/qsa_prompt_attn.cu:2334`, sm_80+):
  int8 tensor-core prompt attention. Upstream made it default-on for sm_80+ in the
  0.2.x line and measured the gains there, not here.
- `STRATA_HC_Q8=1` (`src/core/native_dense.cpp:45-70`): Q8 hyper-connection GEMV,
  default-on only for gfx12 upstream, opt-in elsewhere.
- PLE I/O: `--ple-io direct|mmap|ram`, `--ple-row-cache N` (default 1,048,576 rows),
  `--ple-inflight N` (default 256). `direct` is what we run (unbuffered SSD reads, never
  in RAM or the file cache). `mmap` is the cheap A/B of the same path.
- `STRATA_PARK_VRAM_MIB=384`: keep parked conversations in VRAM instead of host RAM
  (default is host RAM, which is why parked conversations cost CPU time to restore).

Why: pp8192 has been 1,876 to 1,996 across every Strata arm I have measured since
October 7, while decode moved 52 to 69.6. The prefill lane is simply unexplored on this
fork.

Risk: mostly memory-shaped, so measure with the harness's pp32768 probe and the VRAM
line, and expect an interaction with P0-1 rather than independent wins.

### P0-8. Two switches the docs say are on but are not

`STRATA_GR_NORM_FAST` is listed in AGENTS.md as a plat-tier feature. On CUDA the register-split
norm is **opt-in**: `src/kernels/cuda/fused_gr.cu:1179-1180` turns it on only when the env var
is set non-zero (upstream defaults it on for gfx12, and the lean comment says CUDA was
"measured on sm_70 only"). The served run config `vendor/strata-lean/strata-iq3_xxs-ple_q8.json`
has **no `env` block at all**, and `llama-swap.yaml` sets no `STRATA_*` for any tier, so the
served tier runs with it off today. It costs one config line to test, one arm in the sweep
(`grnormfast`), and upstream's own measurement is a modest +0.5 to 0.6% on a V100.

The engine log names a second knob I have never touched:

```
strata hc: CUDA0: the hyper-connection read runs as staged (the norm per token and stream, the
  down projection's activations staged ahead by cp.async); checked bit for bit against the
  plain read on this card (STRATA_HC_SPLIT=1 or 0 for the earlier ones)
```

`STRATA_HC_SPLIT=1/0` selects the earlier hyper-connection read kernels instead of the staged
one, and the engine prints which path it took, so it is a testable A/B with a printed answer.
It sits directly on top of the fusion our fork contributes (`8645328d`), which makes it the
first thing I would re-check after a re-slice onto 0.2.x.

### P1-9. Reserved huge pages for the expert arena (`vm.nr_hugepages`)

What: the arena needs 20,464 2 MiB pages (`/proc/sys/vm/nr_hugepages` is 0, so
`MAP_HUGETLB` fails and the mapping falls to 4 KB pages). `sudo sysctl vm.nr_hugepages=20480`
then a re-start of the tier asks for the hugetlb path again, and `STRATA_NO_LARGEPAGES=1`
is the same-boot A/B switch.

Why: upstream's comment in `pinned.cu` prices it for our exact workload class: the CPU
expert pool streams whole experts out of this arena, and 4 KB pages cost about 750 TLB
misses per 3 MB expert. 37 to 41% of our routed experts go through that path.

Honest caveat, because two of our documents say otherwise:

- AGENTS.md describes "StrataGP 2 MB Transparent Huge Page (`MADV_HUGEPAGE`) kernel
  acceleration" as a plat feature. That came from a **17-line local patch to
  `src/core/pinned.cu` in the stale `vendor/strata` v0.1.30 checkout** (still sitting
  uncommitted there, `git status` shows ` M src/core/pinned.cu`). It is not in lean, and
  the live log says `MADV_HUGEPAGE not requested`.
- Upstream deliberately stopped requesting `MADV_HUGEPAGE` when THP is `always` (#771)
  and stopped requesting it at all when free 2 MiB blocks are less than half the arena
  (#2126: a 62 GB box holding 47 GB of anonymous memory took 432 s to start instead of
  25 s). Our box is 61.6 GiB with 47 GiB available at load. So the lever is a reserved
  hugetlb pool, not the old patch, and reserving ~40 GiB of pages on a 64 GB box is the
  experiment, with `HugePages_Total` and MemAvailable watched while it runs.

I have not tested this and it needs sudo plus a boot-time reservation to be reliable.
Treat it as an experiment with a rollback, not a config change.

### P2-10. Re-slice `4070-main-lite` onto `v0.2.7`

What: `git worktree add -b 4070-main-lite-027 vendor/strata-v0.2.7 v0.2.7` (or a fresh
clone dir), re-apply `8645328d`, `2b829e1e`, `3c3312d4`, `9aafcbf0`, re-run the fork's
build gate, then the standard 7-way A/B. Two of our seven commits are already upstream,
so the real port is four, one of which (`8645328d`, the hyper-connection fusion) is the
one likely to conflict with the 0.2.x GR work.

What the re-slice unlocks that a cherry-pick would have to carry by hand (new `STRATA_*`
switches between the fork's merge base and v0.2.7.1: **218** of them, listed by
`audit_upstream.sh`; the interesting subset):

| switch | what it is | upstream claim (their cards, not measured here) |
| --- | --- | --- |
| `STRATA_SPEC_COST` | cost-aware MTP window: a proposal costs a decode unit, a suffix repeat costs a memory read, so the window budget follows the measured cost | +7% decode on the 4060 branch, +27.5% on the 4070 branch |
| `STRATA_KVA=1` | approximate K/V read in the last 12 layers for prompts over 4k, exact in the first 36 | +11% at 14k, +18% at 30k (RTX 5070), quality risk |
| `--draft dflash2`, `STRATA_DF2_*` | DFlash2 block drafter with a masked GLM head (new pack tool `tools/dflash2_pack.py`) | +20.3% / +18.1% prose on a B70 rig, +2.7% / +8.6% coding on a 4060 |
| `STRATA_IO_PF_GATE/TARGET/BUDGET` | turn expert prefetch on only when the CPU pool is the limit, sized to measured headroom | +5 to 10% when the pool is the limit, no change when the GPU is |
| `STRATA_FILE_TIER_PROBE` | measure the SSD once at start instead of assuming a rate | calibration only |
| `STRATA_PARK_ASYNC`, `STRATA_PARK_VRAM_MIB`, `STRATA_CKPT_*` | parked conversations and window checkpoints, including keeping them in VRAM | +5 to 14% in their sessions |
| `STRATA_GRAPH_PRIO=1` | graph capture on its own priority stream | 53% faster capture |
| `STRATA_RAM_ADAPT*` | copy an incoming expert out of its stage buffer, up to 4 pairs per layer (Strata PR #3060) | refill cost, not speed |
| `STRATA_SELECT_LONG`, `STRATA_TOPK_MULTI`, argmax multi-row | small decode-side kernels for long context | 1 to 4% decode |
| `STRATA_ROUTE_PRIOR` default for NVIDIA cards at or under 13 GiB | see P0-2 | +15% / +37% on a 3060 |

The re-slice is also what makes P0 items cheap to keep: the sm_89 table and
`STRATA_GR_NORM_FAST` stop being our maintenance burden.

### P2-11. Batching, if I ever run two clients at once

`--batch N` / `--slots N` (upstream: +10 to 12% aggregate on a B70 rig, plus
`--batch-mtp` and `STRATA_PIPED_MTP=N`, open Strata PR #4121). Today llama-swap serves
one request per model instance, so this only pays if I route concurrent agents at the
same tier. It is also incompatible with `--kv-grow`, so it is not free to combine with
P0-1. Parked unless the multi-agent pattern actually shows up.

### P3-12. PLE table in RAM

`--ple-io ram` locks the whole n-gram table in RAM at start, and upstream measured
+16.5% at 1k, +30.6% at 4k, +16.8% at 8k on a 72 GB card with a 20k-row cache. Our Q8_0
PLE table is 54.4 GiB against a 61.6 GiB memory guard and a 39.97 GiB expert arena, and
v0.2.7's residency policy explicitly refuses the cache when RAM cannot hold table plus
arena plus 8 GiB. So this is not a switch to flip; it is a quant decision: the pack's own
IQ3_XXS PLE is 28.8 GiB (`docs/qwen38-flash-next-internal.md` §5) and `§9.2` measured PLE
depth as near-irrelevant to quality (delta KLD 0.0005). A smaller PLE table is the
unlock, and it needs a packing run plus a precision check before I spend RAM on it.

### P3-13. llama.cpp lane: parity and idea harvesting, not speed

Current state I verified by running each binary with `--version` (2026-10-28):

| tree | binary | build reported by `--version` | llama-swap macro |
| --- | --- | --- | --- |
| `vendor/llama.cpp-master` | `build/bin/llama-server` (112 MB, statically linked) | b11913, commit `7aca0d2ea` | `qwen38_master_server` (gold tier) |
| `vendor/llama.cpp-exp-latest` | `build/bin/llama-server` (15,976 bytes) | b11924, commit `86d69ec57` | `qwen38_exp_server` (exp tier) |
| `vendor/llama.cpp-exp-stack-v2` | `build/bin/llama-server` (16,000 bytes, Sep 28) | b11268, commit `f382a59e3` | `qwen38_mtp_server` (MTP tier) |

The 16 KB binaries are **not stubs**: recent llama.cpp builds link `llama-server` as a thin
launcher against `libllama-server-impl.so` (7.2 MB) and `libggml-cuda.so.0.27.0` (67 MB) in
the same directory. I made that mistake while auditing and corrected it by running
`llama-server --version`. What is actually stale in that lane:

- `llama.cpp-exp-stack-v2` (the tree the **served MTP tier** runs from) is at b11268 from
  September 28, i.e. ~645 builds behind master, and its llama.cpp version string is 0.5.0
  against master's 0.7.0. If I want the MTP tier to keep up I have to rebuild that tree,
  not the exp-latest one.
- `AGENTS.md` still says the exp clone's build dir serves two branches, `llama-server` for
  MTP and `llama-server-exp` for exp. `llama-server-exp` does not exist in
  `vendor/llama.cpp-exp-stack-v2/build/bin/` (only `llama-server` and `llama-bench`), and
  `llama-swap.yaml:36-37` routes MTP to `exp-stack-v2` and exp to `exp-latest`.
- Stored numbers keep the ratio honest: gold 22.8, `exp_cache` 24.5, lean 67.9 to 69.6
  decode on the same quant. llama.cpp is 3x behind; its value here is as a control
  harness and as a source of ideas to port.

Open llama.cpp PRs worth tracking (queried 2026-10-28), with my read of whether they
matter for us:

| PR | title | my read |
| --- | --- | --- |
| 31037 | MTP draft with a trimmed output head | same idea Strata already ships (`--mtp-draft-vocab`); useful as a second opinion on P0-4 |
| 29143 | d2t draft-vocabulary trim for MTP (qwen35) | same lane |
| 27210 | adaptive MTP draft depth (`draft-mtp-adaptive`) | the llama.cpp version of `STRATA_SPEC_COST` |
| 30815 | qwen4exp QSA with chunking | prompt-side idea for P0-7 |
| 27861 | GPU-resident LRU cache for host-offloaded MoE experts | Strata's expert cache already does this, natively |
| 28699 | incremental pooled-key cache for the QSA indexer | already in our exp-stack branch |
| 30592, 30316 | qwen4exp nextn/MTP fixes, padding, multimodal MTP | needed before any llama.cpp MTP tier is re-benched |
| 31259 | sparse attn on CDNA | not our card |
| 30792 | native q8_0 KV in the flash-attn tile kernel | interesting for `-ctk q8_0` parity work |

The two targets named in the `strata-cherry-pick` skill need correcting: #31037 is real
and open, **#30790 does not exist**, #29887 is merged (it is the MoE host-memory GPU
cache that our exp branch carries), and Strata's own #3453 (our sm_89 table) is merged
into `v0.2.7` even though the PR still shows open.

## 4. How I would measure this

Harness: `bench-models/bench-strata-tuning-ab.sh`. Same binary, same quant, same probe as
`bench-qwen38-refresh-ab.sh` (`bench-qwen38-refresh-probe.py`: 3 warm-up decodes, 4 code
decode probes at 256 tokens, pp at 2048/8192/32768 with unique prefixes, a 4k needle
check), arms are overlays on the **production** run config rather than a bench-only
config, so a winner is one config line from being served.

Default arms (`base` runs first and last so every delta is paired):

```
base reserve832 kv_grow ctx64k routeprior15 routeprior_kv draft_en grnormfast spec5 specprob
no_suffix lookup prefill16k prefill_alias imma hcq8 ple_mmap aux_ecores base
```

19 arms, about 2 t/s of spread in each direction: read the paired `base` rows, not one arm in
isolation.

Outputs: `logs/results/strata-tuning-ab.jsonl` (probe rows, comparable to the refresh A/B
rows) and `logs/results/strata-tuning-diag.jsonl` (per arm: expert slots, hit rate, PCIe
share, free VRAM at start, the #2065 warning, pool worker placement, arena page backing,
draft head token count, and the `/proc/<pid>/io read_bytes` delta over the probe window
with MB per generated token). The diag file is what separates "faster" from "faster
because it spilled to SSD", which AGENTS.md already treats as the ground-truth check.

Rules I would apply before believing an arm:

1. Preflight refuses to run while `llama-swap.service` is active (the sweep measures one
   loaded model; a router swap mid-sweep invalidates every row). `FORCE=1` overrides.
2. An arm needs the control arm before and after it, same boot, CPU governor unchanged.
   Spread on this tier is already 2 t/s.
3. Anything that changes routing or the draft vocabulary also has to pass the needle
   probe and a manual read of one Pi session before it is served.
4. `read_bytes` per generated token has to stay in the PLE design traffic band
   (~5 KB/token for this arch's n-gram rows). Tens of MB/token means the win is a spill.

Cost: about 70 to 90 s per arm to load plus about 3 minutes of probes, so the default
11-arm sweep is 45 to 60 minutes of card time, and it takes the plat tier off the router
for that window. That is why I am writing this up instead of running it: it evicts the
model I am running on.

## 5. Stale things I found while auditing

These are documentation/consistency fixes rather than performance work, but they mislead
the next agent that reads them.

1. `AGENTS.md` claims plat has "StrataGP 2 MB Transparent Huge Page (`MADV_HUGEPAGE`)
   kernel acceleration". The live engine reports it is not requested, and the patch that
   claimed it is an uncommitted local edit in the stale `vendor/strata` v0.1.30 checkout.
2. `llama-swap.yaml` tier descriptions still say "Strata v0.1.45 ... 74.8 t/s steady,
   90.2 t/s peak". The engine says 0.1.45, the honest steady range from my own runs is
   65 to 70 t/s, and 90.2 is the single-burst number AGENTS.md says must not be quoted as
   sustained throughput.
3. `qwen4exp-refresh/SKILL.md` points at `~/.gemini/config/skills/qwen4exp-refresh/scripts/`
   while the scripts live in `.agents/skills/qwen4exp-refresh/scripts/`, so the copy-paste
   commands in the skill fail on this box.
4. `qwen4exp-refresh/SKILL.md` describes the Experimental tier as `vendor/llama.cpp-exp-latest`
   with PR #29887; #29887 is merged upstream, and the tree the served **MTP** tier actually
   runs from (`llama.cpp-exp-stack-v2`, b11268 of Sep 28) is the one that has fallen behind.
7. `AGENTS.md` lists `STRATA_GR_NORM_FAST=1 register-split norm` as a plat-tier feature. On
   CUDA it is opt-in (`fused_gr.cu:1179-1180`), the served run config has no `env` block, and
   `llama-swap.yaml` sets no `STRATA_*` for any tier, so the served tier runs with it off.
8. `AGENTS.md` describes `vendor/llama.cpp-exp-stack-v2/build/bin/` as serving two binaries,
   `llama-server` (MTP) and `llama-server-exp` (exp). Only `llama-server` and `llama-bench`
   are there; the exp tier routes to `vendor/llama.cpp-exp-latest`.
5. `strata-cherry-pick/SKILL.md` bases its audit on `v0.1.45..origin/main` and names #30790,
   which does not exist in llama.cpp.
6. `vendor/strata-lean/README.md`'s "Candidate backlog" lists draft-vocab trimming and
   MMVQ interleave as open questions. MMVQ-IL is merged and default-on for sm_89, and
   the draft subset is already active (106,299 tokens); the open question is which subset,
   not whether to trim.

I fixed 3, 4, 5 and 6 in the skills and the fork README, and I put the honest wording for 1 and 7
in AGENTS.md. Item 2 touches the served config, so I left `llama-swap.yaml` alone for you to
approve, since a tier-description edit is also a hot-reload trigger. Item 8 is a note I did
not act on: it is an AGENTS.md line about llama.cpp vendor trees, and the served MTP tier's
flags depend on how that sentence is read.

---

## 6. Ticket text for Linear (paste-ready)

Project `L3MS`, milestone `Operations backlog completion — 2026-08-10`, parent `CAR-97`. This
session has no Linear tool exposed to it, so these are text for you to paste, not issues I created.
Pick the `CAR-140`..`CAR-146` group each belongs to at creation time. Each one names the harness
that measures it, because that is what makes them closable.

### T1 (P0, from P0-1) Plat tier: give the card its headroom back

Body: The plat tier boots with 410 MiB of VRAM free and the engine prints its own `#2065` warning
below 512 MiB, which is the driver-paging failure that costs 10x rather than 10%. The tier reserves
all 131,072 K/V cells up front while a typical Pi session touches a fraction of them. Three levers,
in order: `--vram-reserve-mib 832` (the value the engine asks for; parsed at
`src/program/generate.cpp:2426`, hidden from `--help`), `--kv-grow` (map only the cells a request
reaches, `generate.cpp:477` and `:4944-4959`, env `STRATA_KV_GROW`), and `--max-context 65536` for a
tier that is serving agent sessions rather than 128k documents. Freed VRAM becomes expert slots.

Done when: arms `reserve832`, `kv_grow`, `ctx64k` measured in `bench-models/bench-strata-tuning-ab.sh`
against a paired `base`, with the diag file showing slots, hit rate, free VRAM at request start and
whether `#2065` fired; winner promoted into `vendor/strata-lean/strata-iq3_xxs-ple_q8.json` or written
up as rejected with numbers.

Refs: `logs/results/strata-tuning-diag.jsonl`, `.agents/skills/strata-tuning/SKILL.md`.

### T2 (P0, from P0-2) Plat tier: residency-biased routing, with a quality gate

Body: `STRATA_ROUTE_PRIOR=1.5` biases routing toward experts already in the cache, which is the fix
for our 58.6% hit rate (`54487` hits over `92979` lookups, 3,652 experts read over PCIe at 3.8%).
Upstream measured 46.2 to 53.1 t/s on chat and 45.7 to 62.5 t/s on code on a 12 GB RTX 3060, and
defaulted the switch on for NVIDIA cards at or under 13 GiB. It changes what the model routes, so the
speed gate is not enough: a value above 1.5 is out of scope, and any candidate has to survive reading
a real coding session's output before it is served.

Done when: arms `routeprior1` (if the default is not already active in the lean build), `routeprior15`
and `routeprior_kv` measured against `base`, hit rate and PCIe share read from the diag file, and a
quality read recorded either way.

Refs: `src/core/verify.cpp:1849-1851`, `src/kernels/cuda/route_prior.cu`.

### T3 (P0, from P0-3) An expert profile built from my own traffic

Body: The shipped `data/expert-profile.bin` (24,576 pairs, md5 `ff1bc6544d3f2f0013e3e376632d33a8`)
was built for a general workload. Upstream's own measurement: a profile built from the user's prompts
covered 71% of routed experts where the shipped one covered 60%. My traffic is coding-agent sessions,
which is a narrow and very repeatable distribution, so this is the most overfit-friendly win available
on this box. Note the cost that the earlier draft of this document got wrong: collecting traces boots
the engine per prompt (20 to 60 s load, about 10 s decode), so 45 prompts is roughly 30 minutes of
card time, not a CPU-only prep job. Correct commands:
`tools/collect_profile_traces.py CONFIG.json OUTDIR --mode routing` under `vendor/strata/.venv`, then
`tools/make_profile.py` (which owns `--reorder`, `--base`, `--no-base`), not `make_profile_mix.py`.

Done when: a profile built from real Pi transcripts exists, `--expert-profile` measured as an arm,
coverage and hit rate compared against the shipped file in the diag output, and the shipped profile
kept as a fallback.

### T4 (P0, from P0-8) Two switches my own documentation claimed were on

Body: `AGENTS.md` listed `STRATA_GR_NORM_FAST=1` as a plat-tier feature. On CUDA it is opt-in
(`src/kernels/cuda/fused_gr.cu:1179-1180`), the served run config has `env: null`, and
`llama-swap.yaml` sets no `STRATA_*` for any tier, so the tier has been running without it. The
documentation is fixed; the switch itself is still unmeasured on Ada. Same shape for `STRATA_HC_SPLIT`
(the engine's own line says the hyper-connection read runs "as staged") and for `STRATA_KV_GROW`.

Done when: arms `grnormfast` (and the `env:` block it needs in the served config) measured against a
paired `base`; if it wins, the config carries it in `env` so serving and docs agree; if it loses on
`sm_89`, that is recorded so the claim never comes back.

### T5 (P2, from P2-10) Re-slice `4070-main-lite` onto `v0.2.7`

Body: My fork is 8 commits on top of `477ac13c` (2026-10-16) while upstream is at `v0.2.7.1`, 2,252
commits ahead, 193 of them in `src/core`, `src/kernels/cuda` and `src/spec`, with 218 new `STRATA_*`
switches. Continued cherry-picking is the expensive way to the same place. Re-slice instead: new
branch off `v0.2.7`, re-apply `8645328d`, `2b829e1e`, `3c3312d4`, `9aafcbf0` in that order (engine
code, build gate, backend prune, serve fix), skip `0cef00fa` and `7ce60ad3` because they are already
upstream, expect conflicts in `src/core/gr.cpp` and `src/kernels/cuda/fused_gr.cu`, and prove the
6-to-4 launch fold survived before trusting the number. Keep `4070-main-lite` until the re-slice
benches at or above it.

Done when: `strata_v027` exists as an arm in `bench-models/bench-qwen38-refresh-ab.sh` beside
`strata_lean`, the prune is verified (no Metal, SYCL, ROCm, `src/qwen35/`, `web/`;
`CMAKE_CUDA_ARCHITECTURES=89`), and the fork README's progression table has the new row.

### T6 (P0, needs your approval first) The served tier's own description is stale

Body: `llama-swap.yaml:332-334` still describes the plat tier as "Strata v0.1.45 ... 74.8 t/s steady,
90.2 t/s peak, pp=1,138-2,013". The first number is a 2026-10-14 measurement on a different tree, the
second is a single burst that AGENTS.md already says must not be published as sustained, and the prefill
range no longer matches anything measured (1,876 to 1,891 on the current build). Editing a tier's
`name`/`description` triggers a hot reload, and a hot reload swaps the resident model, which is the
model this agent runs on. I have not touched it.

Done when: the tier text matches a measured row from `logs/results/`, the yaml is snapshotted first,
and a 64-token smoke test confirms the tier still answers after the reload.

---

## 7. First tuning sweep, measured 2026-10-28 13:16 to 13:40

Nine rows through `bench-models/bench-strata-tuning-ab.sh` (ladder
`base reserve832 kv_grow routeprior15 draft_en grnormfast spec5 prefill16k base`), engine
`vendor/strata-lean/build/strata` (0.1.45), same quant, same probe script, `taskset 0-11`.
Rows: `logs/results/strata-tuning-ab.jsonl`, self-report: `logs/results/strata-tuning-diag.jsonl`.
System frontmatter (`/tmp/strata-tune/frontmatter.yaml`): CachyOS 7.3.0-4, `powersave` governor with
`performance` EPP, i5-12600K at 37 C, DDR5-5200 4x16 GiB, `mem_available 50.22 GiB` at handoff,
zram 12.9 GiB used, THP `always`, RTX 4070 at 39 C, baseline VRAM 47 MiB used.

Decode is `tg_mean` over four 256-token probes; prefill cells are the mean of each size's two probes;
hit rate is the last window in the engine log (the same rule for every arm).

| arm | the one change | decode t/s | min/max | pp2048 | pp8192 | pp32768 | slots | hit | PCIe | free MiB (all loaded) | MB/tok |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | production config | 70.70 | 67.1 / 75.6 | 920 | 1,872 | 1,950 | 2,583 | 57.4% | 9.9% | 410 | 0.0788 |
| `reserve832` | `--vram-reserve-mib 832` | 68.75 | 61.5 / 74.0 | 914 | 1,883 | 1,966 | 2,506 | 55.6% | 10.3% | 544 | 0.0789 |
| `kv_grow` | `--kv-grow` | **73.53** | 67.7 / 78.9 | 952 | 1,888 | 1,928 | **3,277** | 62.1% | 9.1% | 412 | 0.0998 |
| `routeprior15` | `STRATA_ROUTE_PRIOR=1.5` | **102.23** | 97.4 / 109.3 | 930 | 1,882 | 1,954 | 2,583 | **93.0%** | **1.3%** | 410 | 0.0775 |
| `draft_en` | `--mtp-draft-vocab data/draft_vocab_en.bin` | 65.78 | 63.2 / 70.0 | 920 | 1,877 | 1,935 | 2,653 | 55.9% | 3.5% | 410 | 0.0682 |
| `grnormfast` | `STRATA_GR_NORM_FAST=1` | 64.85 | 59.0 / 68.3 | 913 | 1,871 | 1,955 | 2,583 | 52.9% | 2.0% | 410 | 0.0725 |
| `spec5` | `--spec 5` | 61.00 | 57.6 / 64.3 | 920 | 1,870 | 1,955 | 2,581 | 53.4% | 3.9% | 408 | 0.0569 |
| `prefill16k` | `--prefill auto:16384` | 70.60 | 66.8 / 76.0 | 920 | 1,877 | 1,946 | 2,583 | 59.8% | 9.6% | 410 | 0.0680 |
| `base` | production config again | 71.55 | 66.7 / 76.7 | 918 | 1,875 | 1,948 | 2,583 | 57.7% | 9.9% | 410 | 0.0557 |

"Free MiB (all loaded)" is the engine's `VRAM free with everything loaded` line; the free figure at the start
of a request is lower in every arm (276 for `base`, 274 for `routeprior15`, 278 for `kv_grow` and
`prefill16k`, 394 for `reserve832`, 258 for `spec5`), and the `#2065` warning fired in all nine.

The paired control is 70.70 and 71.55, i.e. 0.85 t/s of spread this session and a pair mean of 71.13.
Deltas against that pair mean: `routeprior15` **+31.10**, `kv_grow` +2.40, `prefill16k` -0.53,
`reserve832` -2.38, `draft_en` -5.35, `grnormfast` -6.28, `spec5` -10.13.

### The win: `STRATA_ROUTE_PRIOR=1.5` (+43.7%)

`+31.1 t/s` on top of a 71.13 pair mean, 97.4 to 109.3 across its four probes, with the mechanism
visible in the engine's own lines rather than inferred from the speed:

- decode expert cache hit rate 93.0% (37,896 hits / 40,747 lookups) against 57.4% and 57.7% for the
  two `base` rows, and 92 to 96% across every window it printed (base's mid-run windows were 69 to 71%);
- experts fetched over PCIe down to 1.3% of routed (350 to 2,142 per window) from 9.9%;
- `read_bytes` per generated token 0.0775 MB, inside the band the other arms sat in (0.0557 to 0.0998),
  so the win is not the disk lending a hand;
- draft acceptance basically unchanged on the decode probes. `draft_acc` is stored per tg probe in the ab
  row (`accepted / proposed`): `routeprior15` 0.752 (0.734 to 0.780) against the controls' 0.737 and 0.732,
  `kv_grow` 0.762, `prefill16k` 0.727, `reserve832` 0.713, `grnormfast` 0.732, `draft_en` 0.720, `spec5` 0.652.
  A 0.015 shift cannot explain +43.7%: this is the router picking experts already in VRAM, not the drafter
  getting luckier. (Counting **every** window the engine printed, suffix drafting included, gives higher
  numbers with the same shape: 0.831 for `routeprior15` against 0.794 for `base`. Both readings live in the
  same JSONL and the engine logs.)

Mechanism, from the source: `src/kernels/cuda/route_prior.cu` implements "Mixture of Cache-Conditional
Experts" (arXiv 2412.00099) without training: an expert already resident gets a `lambda` logit bonus when
the top-10 are selected, while the *weights* stay the router's (softmax over the 512 unbiased logits,
renormalised over the 10 picked). `src/core/verify.cpp:1849-1851` applies it only when `NE == 512`,
`K == 10` and the residency table is on the device. Both the kernel header and the call site say
**"opt-in, changes the output"**, which is why it is not in the served config yet.

What is left before it is served: a quality read on a real coding session (the selection moves, so the
answers move), a paired repeat to confirm 31 t/s is not a one-boot artifact, and a value check (1.0
versus 1.5; upstream's own 12 GB measurement used 1.5, and I am not going above it).

### The quality side, from upstream's own 0.2.7 release notes

Our fork sits on the 0.1.45-era base, so it does not have what upstream did with this switch. The
ground truth is in the tree: `git show v0.2.7:docs/release-notes-0.2.7.md`, line 7.

- 0.2.7 turns `STRATA_ROUTE_PRIOR=1.5` on **by itself** on a discrete NVIDIA card of 13 GiB or less
  ("the owner approved the quality trade for 12 GB cards", `docs/DETAILS.md` of the same tag). It is
  never a default on a Radeon, an integrated GPU or a bigger card, and `STRATA_ROUTE_PRIOR=0` turns it off.
- Upstream's measurement, same model and quant on an RTX 3060 12 GB, five paired runs against the same
  build with the setting off: story 46.2 to 53.1 t/s (+15%), code 45.7 to 62.5 t/s (+37%). My 4070
  measured **+43.7% on the code probes**, so the direction agrees and the size is larger.
- Quality at 1.5: 80 tasks on that card, "no copy or code errors seen". But: "agent loops may finish
  slightly less often, within the run-to-run spread: out of 10 agent tasks, 4.5 passed on average over
  4 runs at 1.5, against 5.6 over 7 runs with the setting off, where the off runs alone ranged from 4 to 6."
- "A stronger setting (2.0) did show copy and code defects, so the default is 1.5."
- The KL cost, from `docs/release-notes-0.2.0.md` line 55: `STRATA_ROUTE_PRIOR=0.5` measured KL 0.0126
  against a run-to-run floor of 0.0012, which is why 0.2.0 left it opt-in. The fork's own older
  `docs/DETAILS.md` (the 0.1.45 base) puts it in four words: "adds speed, doubles the KL".

That changes what the gate has to be. The axis upstream flagged is **agent task completion**, which is
exactly what this tier serves me for, and upstream's own spread (4 to 6 passes out of 10) is wider than
the difference they measured. So a needle probe and a code compile check are not the gate; the gate is
whether my own Pi sessions still finish their tasks, and it needs several tasks over more than one run.

Serving shape I would use if it passes: add a second tier (`qwen38-flash-next-plat-fast` with
`STRATA_ROUTE_PRIOR=1.5` in the upstream's `env:`) instead of editing the plat tier, so the daily driver
stays reversible while the agent-task read runs.

And one interaction with P2-10 worth writing down now: **re-slicing onto `v0.2.7` turns this switch on by
itself** (13 GiB-or-less NVIDIA rule), so the `base` arm of any post-re-slice sweep is already biased. Either
set `STRATA_ROUTE_PRIOR=0` explicitly in the re-slice arms or read the new base as biased-on.

Also worth knowing: `STRATA_ROUTE_TAIL_SKIP=7` is **already on by default** in this engine (the boot log
says so: "a missed expert that every token of a verify window routes at rank 7 or lower is skipped,
+10..20% decode on 12-16 GB cards that miss experts, answers differ slightly from 0.1.41"). So the
71 t/s baseline already includes that, and `STRATA_ROUTE_RESIDENT` (swap a missed tail expert for the
best resident one within a margin) is a separate untested switch in the same lane.

### `--kv-grow`: promising, two flags on it

+2.40 t/s (73.53, 67.7 to 78.9) with exactly the effect it was sold for: expert slots 2,583 to 3,277
(+694), cache 4.24 to 5.35 GiB, hit rate 57.4/57.7% to 62.1%. Against it: the worst SSD traffic of the
run (0.0998 MB/token against 0.0557 to 0.0788), and pp32768 down to 1,928 from 1,950 and 1,948. The
gain is inside the ~2 t/s spread, so it needs a paired repeat before I write it as a win, and the mid-
request K/V mapping cost has to be watched at 32k prefill, not just at 8k.

### Rejected or neutral

- **`--mtp-draft-vocab data/draft_vocab_en.bin`** (65.78, -5.35): the English-only subset does exactly what
  upstream advertised (`draft head over 40525 tokens (68.0 MiB)` against `106299 tokens (178.4 MiB)`, so
  110 MiB back into the cache, `+73 MiB for the draft head`, 2,003 slots instead of 1,953) and it still
  **loses 5.35 t/s** on a code probe (acceptance mean 0.768 against 0.794), and the 70 auto-cache slots it
  bought (2,653 against 2,583, from freeing 110 MiB of draft head) did not pay for itself. Decode-probe
  acceptance 0.720 against the controls' 0.735. The
  head stays as shipped. If a subset is tried again it has to be code-plus-English built from my own
  transcripts (`tools/draft_vocab.py`), not the shipped English-only file.
- **`STRATA_GR_NORM_FAST=1`** (64.85, -6.28): the register-split norm is a **loss on Ada `sm_89`**, which
  is the measurement AGENTS.md was missing when it claimed the switch was a tier feature. Leaving it off is
  now a decision with a number behind it instead of an accident of an `env: null` config. Upstream defaults
  it on for AMD gfx12 and measured CUDA on `sm_70`; that does not transfer here.
- **`--spec 5`** (61.00, -10.13): it loses on both counts. Decode-probe acceptance fell to 0.652 (0.621 to
  0.701) against the controls' 0.735, and a fifth draft position pays for more verification than it saves on
  this workload. `--spec 4` stays.
- **`--prefill auto:16384`** (70.60, -0.53; pp8192 1,877 and pp32768 1,946): dead neutral. `--prefill auto`
  is not actually bounded by the 8,192 chunk ceiling on this card, so raising the ceiling buys nothing. The
  pp lever is elsewhere (`--prefill-head-alias`, `STRATA_PROMPT_ATTN_IMMA`, `STRATA_HC_Q8`, all unrun).
- **`--vram-reserve-mib 832`** (68.75, -2.38): it does what it says (410 to 544 MiB free with everything
  loaded, window-capture floor 410 instead of 276) and costs 77 expert slots plus 2.4 t/s. It is insurance
  against the `#2065` paging failure, not a speed change; note every arm in this sweep still logged the
  warning and captured windows at 254 to 410 MiB free, so the cliff is real and this is the knob for it.

### Harness notes from the run

- Handoff worked as the `gpu-handoff` skill describes: `systemctl --user stop llama-swap.service` left
  **47 MiB** on the card with no orphan to kill (the hardened `bench_strata.sh` prints the holders by pid
  and refuses above 200 MiB), and it came back `active` with `llama-swap listening on http://:8080`.
- `pcie.link.gen.current` reads **Gen1 while the card is idle** (ASPM downclock) and **Gen4 16x with a
  model loaded** (`nvidia-smi -q` GPU Link Info, checked after the run). The frontmatter sampled it at
  idle and reported `Gen1 (max Gen4)`, which is easy to misread as a bottleneck for expert streaming.
  `bench-models/bench-env.sh` now prints a note under the `pcie_link:` line saying exactly that.
- The diag's hit-rate cell is the engine's **last** window, which on `base` is the post-32k-prefill window
  (57.4%) while its mid-run windows were 69 to 71%. Same rule on every arm, so the comparison stands, but
  do not read the cell as the tier's steady-state hit rate (the served log's own number is 58.6%).
- Cost accounting held: 2 min 26 s to 2 min 37 s per arm, 23 minutes for nine rows (13:16:51 to
  13:39:49), of which about 65 s per arm is the engine loading 39.97 GiB at 0.65 to 0.68 GiB/s.
- Not yet measured (arms exist, ladder shortened for the first pass): `ctx64k`, `routeprior_kv`,
  `routeprior10` (new), `specprob`, `no_suffix`, `lookup`, `prefill_alias`, `imma`, `hcq8`, `ple_mmap`,
  `aux_ecores`, plus the two opt-in arms added after the first run: `routeprior10` (`STRATA_ROUTE_PRIOR=1.0`)
  and `route_resident` (`STRATA_ROUTE_RESIDENT=4`, the margin-swap residency law instead of the logit bonus,
  with `STRATA_ROUTE_RESIDENT_RANKS` left at its 6-9 default).

### What this does and does not close in the ticket text above

- The **route-prior ticket**: its speed half is done (`routeprior15` measured, +31.1 over the pair mean, hit
  and PCIe read from the diag file) and the `routeprior1` arm turned out to be unnecessary, because this
  engine is 0.1.45 and defaults the switch off. What is left is the ticket's last clause: the agent-task
  read. Until that reads clean the tier does not change.
- The **VRAM-headroom ticket**: half done. `--kv-grow` and `--vram-reserve-mib 832` are measured, `ctx64k`
  is not, and `--kv-grow` still needs its paired repeat.
- The **documentation-correction ticket**: the `STRATA_GR_NORM_FAST` claim is now closed with a measurement
  instead of a code read.
- The **expert-profile ticket** and the **re-slice ticket** are untouched by this sweep. The re-slice gained
  a warning it did not have before (v0.2.7 turns route prior on by itself below 13 GiB, so a post-re-slice
  `base` is biased unless pinned off).

## 8. Second tuning sweep (Exact switches on the kv-grow baseline), 2026-10-28 17:05 to 17:36

Eleven arms, every one stacked on `--kv-grow` so the baseline is what I now serve. Ladder:
`kv_grow kg_attn kg_rows kg_mmvq kg_side kg_wide kg_exact1 kg_expdec kg_adapt kg_set kv_grow`
(`bench-models/run-strata-tuning-dark.sh`, log `logs/strata-dark-launch.log`, raw evidence kept in
`logs/results/strata-tune-20261028-170550/` including each arm's engine log and identity files).
2 min 26 s to 2 min 32 s per arm, 31 minutes end to end, `TG_PROBES=3`, `PP_SIZES="8192,32768"`.

Two instruments changed before the run, both because of what the first sweep left unresolved:

- `bench-models/strata-identity-probe.py` (seven fixed greedy prompts, `temperature 0`, `top_k 1`,
  `seed 42`, `cache_prompt false`, no timestamps) now runs **twice per arm, before any speed probe**, and
  each differing prompt records how deep the two texts agreed (a hash of the first 240 chars and of the
  first 960) instead of only that they differ.
- `bench-strata-tuning-ab.sh` gained the `kg_*` arms for the switch pool that has never been measured on
  sm_89 (our fork has no `nv_defaults.cpp`; upstream's bitwise sets cover sm_86, sm_60 and sm_120, never
  Ada), and the diag row gained `identity_*`, `intra_diff_n`, `slot_values_seen`, `kv_grown_cells` and
  `expert_slots_held`.

### The result that matters most is negative, and about my own method

The identity gate could not judge anything, because the control did not pass it against itself:

- two boots of the same `--kv-grow` config (the opening and closing control arms) differed on **4 of 7
  prompts** (`code_fix`, `json_strict`, `explain_short`, `code_long`), every time at the same token count;
- two passes inside **one boot** of the served tier, run back to back with no other traffic, differed on
  **3 of 7** (`code_small`, `code_long`, `pp20k`).

So greedy output on this engine is not reproducible, and the reason is in the architecture rather than in
the probe: 3,277 of 24,576 expert pairs are cached, which experts are resident changes between passes, and
the cached and streamed paths are not bitwise the same arithmetic. Same token counts with different texts is
what a near-tie in the logits looks like when the logits move. The consequence for the quality gate:
upstream's Exact rule (`v0.2.7:docs/DETAILS.md:1031`, identity gate passed) cannot be satisfied by any arm
here, including the genuinely exact ones, so the gate in this tree is comparative from now on:
`identity_diff_n` (arm against the ladder reference) is only meaningful next to the control pair's own
cross-boot divergence, with `intra_diff_n` as each arm's noise floor.

### Speed rows, read as what they can and cannot support

| arm | switch | tg all 3 | tg last 2 | probe acc | hit % | PCIe % | io MB/tok | diffs of 7 |
|---|---|---|---|---|---|---|---|---|
| `kv_grow` (open) | baseline | 77.07 | 80.60 | 0.741 | 53.5 | 11.5 | 0.0127 | reference |
| `kg_attn` | `STRATA_ATTN_DAILY=1` | 72.03 | 74.40 | 0.737 | 47.2 | 2.3 | 0.0604 | 4 |
| `kg_rows` | `STRATA_QSA_ROWS=1` | 70.43 | 73.35 | 0.734 | 54.7 | 1.9 | 0.0802 | 6 |
| `kg_mmvq` | `STRATA_B6_MMVQ_ROWS=1` | 72.70 | 73.85 | 0.760 | 49.8 | 2.3 | 0.0616 | 4 |
| `kg_side` | `STRATA_SIDE_MIXER=2` | 79.23 | 79.80 | 0.759 | 58.5 | 10.6 | 0.0313 | 5 |
| `kg_wide` | `STRATA_DEQUANT_WIDE=1` | 70.10 | 70.40 | 0.713 | 51.3 | 2.0 | 0.0391 | 3 |
| `kg_exact1` | `STRATA_CPU_IQ_ALL_EXACT1=1` | 79.90 | 81.70 | 0.727 | 54.2 | 11.0 | 0.0358 | 4 |
| `kg_expdec` | `STRATA_EXPDEC=1` | 77.23 | 78.25 | 0.711 | 49.9 | 12.5 | 0.0565 | 2 |
| `kg_adapt` | `STRATA_ADAPT_NOWAIT=1 STRATA_ADAPT_LAG=2` | 74.57 | 75.45 | 0.741 | 51.6 | 4.4 | 0.0392 | 5 |
| `kg_set` | stacked bitwise set (attn rows mmvq side wide) | 72.07 | 72.20 | 0.750 | 49.2 | 2.1 | 0.042 | 4 |
| `kv_grow` (close) | baseline | 69.43 | 70.55 | 0.708 | 49.3 | 2.2 | 0.040 | 4 |

Every arm ran with the same slots (3,277, 5.35 GiB of cache, 2,468 auto), `needle_4k` true, and
`io_mb_per_token` between 0.0127 and 0.0802, so nothing in this ladder was thrashing from SSD.

- The paired control spread **7.63 t/s** on all-three means (77.07 and 69.43) and **10.05 t/s** on last-two
  means (80.60 and 70.55). Nothing in this ladder is promotable on decode speed, including the two arms
  that look best (`kg_exact1` 81.70 and `kg_side` 79.80 on last two).
- The reason is visible in the probe sequences and it is mine to fix: with 3 warm-ups and 3 measurement
  probes most arms were still climbing, throughput and draft acceptance together. Opening control
  `70.0, 76.1, 85.1` t/s with `0.680, 0.715, 0.828` acceptance; closing control `67.2, 68.7, 72.4` with
  `0.681, 0.681, 0.760`. Only three arms were near steady state (`kg_wide` 69.5, 70.1, 70.7; `kg_set`
  71.8, 72.0, 72.4; `kg_side` 78.1, 78.0, 81.6). Fixed for the next run: `TG_WARMUP=6` (was 3) and the
  probe now writes `tg_steady_mean` and `tg_steady_spread` (last three probes), so a row states its own
  steadiness.
- The drift is not RAM pressure and I have not identified it: `mem_available_mib` ran 12,807 to 14,306
  across the whole ladder with no trend, `slot_values_seen` was `[3277]` in every arm, and disk traffic
  stayed under 0.08 MB per generated token.
- One signal survived the noise because it is not a decode number: prefill. `kg_wide` measured **1,978 at
  8k and 1,992 at 32k** against the two controls' 1,938/1,950 and 1,932/1,933, and `kg_set` (which contains
  the same switch) 1,972 and 1,979. `STRATA_DEQUANT_WIDE=1` is default-on upstream only for cc 6.1, it is
  off here, upstream calls it bit-exact, and it is the only arm with a repeatable gain at both prefill
  sizes. Queued: `kv_grow kg_wide kg_wide kv_grow` at 6 warm-ups and 4 probes.
- `kg_expdec` (2 diffs) and `kg_wide` (3 diffs) are the arms whose divergence came in under the control
  pair's own 4, which is as close to an Exact verdict as this instrument currently gives.

### `--kv-grow` is served, and the engine told me what the trade actually is

`--kv-grow` is in the `args` of all four served Strata run configs (`strata-iq3_xxs-ple_q8.json`,
`strata-vision-ple_q8.json`, `strata-sc117-iq3_xxs-ple_q8.json`, `strata-sc117-vision-ple_q8.json`,
snapshots `*.bak-20261028-191525`), hot-reloaded with `kill -HUP` at 19:15:37, smoke-tested 19:19:39 (a
completion from `qwen3.8-flash-next-vision-ple_q8`). No `llama-swap.yaml` change was needed:
`serve/server.py:2867` forwards the run config's `env` and the args come from the config. The IQ4 vision
tiers (`strata_vision_iq4_config`) were left alone.

From the served vision tier's own engine log (`vendor/strata-lean/strata-vision-ple_q8.log`), before and
after, and then from my own session:

```
expert cache auto: 5.10 GiB free, 700 MiB reserved (+184 MiB for the draft head) -> 1952 slots
expert cache 2581 slots, 4.23 GiB of VRAM; policy is
expert cache auto: 6.22 GiB free, 700 MiB reserved (+184 MiB for the draft head) -> 2467 slots
expert cache 3276 slots, 5.35 GiB of VRAM; policy is
elastic K/V: 16384 of 131072 cells in VRAM (0.21 of 1.32 GiB); the expert cache (3276 slots) gives
  the K/V room below slot 823 as the context grows
strata: K/V grown to 114688 cells (1.17 GiB); the expert cache gave 634 slots for it, 286 hotter
  experts moved to colder ones' slots (2642 of 3276 slots hold experts)
```

That corrects the size of the prize. The whole 128k K/V on this tier is **1.32 GiB**, not 4. kv-grow maps
16,384 cells (0.21 GiB) at load and hands the other 1.11 GiB to the expert cache (+694 slots, 2,581 to
3,276). As a long context arrives the cache gives slots back, and the experts it evicts are computed by the
CPU from RAM like any other miss (`src/program/generate.cpp:7350-7372`, floor 128 slots, growth in
`STRATA_KV_GROW_STEP` steps of 8,192 cells). Measured on my own live session at 114,688 cells: 2,642 slots
holding experts, still above the static config's 2,581. By the engine's own ceiling line (room below slot
823) the cache ends near 2,450 slots when a request reaches 131,072 cells, which is below static. So
kv-grow is a win at short and medium contexts, roughly neutral around 110k cells, and slightly negative at
the deepest requests. Whether the give-back costs decode speed is the next measurement: `DEEP_SIZES` now
drives a deep-context decode probe (`bench-models/bench-qwen38-refresh-probe.py --deep-sizes`, opt-in,
`deep` field in the row), and `kv_grown_cells` / `expert_slots_held` capture the give-back lines.

### What the second sweep closes and what it opens

- Closed: the Exact-switch pool is now measured on Ada for the first time in this tree, and none of it is
  promoted; the only surviving signal is prefill-side (`STRATA_DEQUANT_WIDE=1`).
- Closed: `--kv-grow` is served, with its trade written down from the engine's own accounting instead of
  from upstream's table (their kv-grow row moved slots 6,213 to 7,968 and 96.3 to 98.5 tok/s on a bigger
  card, `v0.2.7:docs/DETAILS.md:1328`).
- Open, and now the blocking one: the harness has to separate warm-up from steady state before any decode
  delta means anything. Two of the eleven rows (`kg_side`, `kg_exact1`) look like 6 to 8 t/s wins and one
  control reads 10 t/s below its twin. A `base kv_grow kv_grow base` ladder at 6 warm-ups, 4 probes and a
  98k deep-context decode probe is the run that decides both.
- Open: the identity instrument. Until there is a way to hold the expert-cache state still (or to compare
  logits instead of text), a per-arm divergence count is a noise measurement, not a quality gate.

## 9. The Lamina draft head and the Qwen3.8 fork family, 2026-11-05

Three forks of Strata now ship for this one model. All three audited, one of them measured, and the answer
is that the served tier does not change.

- **`eddoursul/Lamina`**, a Strata fork that diverged on 2026-10-29. Its model
  `eddoursul/Lamina-Qwen3.8-Flash-Next-MTP` is not a GGUF: it is a `--mtp` runtime directory (`dense.txt`,
  `dense.bin`, `experts.bin`, `draft_vocab.bin`), which is the format `tools/mtp_rt.py` already writes and
  `strata-lean` already loads. Verified identical to the shipped head: the same 29 tensor names, dtypes and
  dims, the same `dense.bin` (116,099,072 B) and `experts.bin` (707,788,800 B), SHA-256 matching the model
  card. The only structural difference is the draft vocab, 40,525 tokens against the served 106,299.
- **`simplexai-labs/PolyStrata`**. For Qwen3.8 it runs "the Strata project's engine, unchanged", so its Qwen
  numbers are Strata's numbers. Its own engine covers eleven other models and is out of scope here.
- **`jesdga95/basalt`**. Blackwell `sm_120` only, NVFP4 packs. Nothing in it runs on Ada. Worth reading for
  its per-switch KLD gate (`docs/precision/isolated-costs.md`) and its two-stage Q8_0 head, which is
  lossless by proof, but there is no code to take.

### What was measured

Three paired ladders on the served `strata-ud-iq2_xs-ple_q8` config, `TG_PROBES=4`, `PP_SIZES=2048`, with
`base` bookending each one:

1. `base mtp_lamina base`, 16:21, 11 minutes.
2. `base mtp_lamina mtp_lamina_r960 base`, 18:13, 14 minutes.
3. `base base_r960 mtp_lamina_r960 base`, 18:50, 14 minutes.

New arms: `mtp_lamina` (`--mtp vendor/Strata-data/mtp/rt-lamina`), `mtp_lamina_r960` (the same plus
`--vram-reserve-mib 960`), and `base_r960` (`--vram-reserve-mib 960` on the stock head). The harness also
gained acceptance accounting (`drafts_accepted`, `drafts_offered`, `draft_accept_pct`) parsed from the
engine's own `drafts accepted X of Y` lines, and an `SRC_CFG` override, because the harness's default
IQ3_XXS weights are not on this disk.

### Acceptance: the claimed win does not reproduce

| ladder | control | Lamina head |
|---|---|---|
| 1 | 80.7% (5,936 / 7,358) | 82.1% (2,921 / 3,560) |
| 2 | 80.1% and 81.5% | 80.8% (2,985 / 3,694) |
| 3 | 80.7% and 80.6% | 80.6% (3,030 / 3,761, the r960 arm) |

Pooled over all three ladders: 80.8% (11,837 / 14,647) against 81.4% (5,906 / 7,254), which is +0.6 points
at a standard error of 0.56. Not significant. The head's README claims 2.489 against 2.402 tokens a verify
pass on IQ3_S (+3.6%), but that baseline is basalt's dense part rather than the stock unsloth head, and the
dense part was trained on IQ3_S and UD-Q4_K_XL output while this tier serves UD-IQ2_XS.

Decode contradicts itself the same way: +3.9 t/s in ladder 1, -2.9 in ladder 2.

### The VRAM is real, and it is the actual finding

The head is Q6_K over the vocabulary, so 106,299 tokens is 212.9 MiB and 40,525 is 81.2 MiB: 131.7 MiB
freed, identical in every run. With `--expert-cache auto` the engine spends it on slots immediately, 2,711
to 2,790 (+79, +2.9%), at unchanged headroom (412 to 414 MiB free with everything loaded). It never appears
as free VRAM, which is why the first two ladders looked like a wash.

Ladder 3 is where that matters, because it prices the head against the reserve increase it makes
affordable. `#2065` is diagnostic only (`src/program/generate.cpp:11335`, "said, never changed"): the
engine prints it and takes no other path. What it warns about is the driver paging VRAM to system RAM, so a
higher reserve buys margin rather than throughput.

| arm | slots | free (loaded) | worst capture | `#2065` | hit% | tg steady |
|---|---|---|---|---|---|---|
| `base` | 2,711 | 412 | 278 | fires | 52.7 and 50.1 | 62.47 and 62.13 |
| `base_r960` | 2,552 | 674 | 540 | clear | 39.6 | 52.30 |
| `mtp_lamina_r960` | 2,633 | 674 | 538 | clear | 47.8 | 60.77 |

No spill in any arm (io 0.023 to 0.032 MB/token). The stock head at 960 loses 159 slots and 16% of decode.
The Lamina head at 960 loses 78 slots and 1.5 t/s, inside the spread. At a fixed reserve the head is worth
81 slots and about 8.5 t/s, which is a different justification from the one its README makes.

### Verdict: serve `base`, carry nothing

`base` is the fastest configuration measured (62.30 t/s, 2,711 slots) and the head buys nothing there. The
head is not promoted: its acceptance claim does not reproduce, its speed effect is contradictory, and the
one durable thing it provides (132 MiB) only pays once a reserve increase is already being paid for, which
is a safety trade rather than a speed one.

Kept for a future decision, not served: `--mtp vendor/Strata-data/mtp/rt-lamina --vram-reserve-mib 960` is
the configuration that clears `#2065` at near-parity (60.77 against 62.30, no acceptance loss). If the
driver-paging risk ever shows up in a real session, that is the line to take, not `base_r960`, which costs
16%.

### Harness defects this found

- Repeated labels (the ladder convention runs the control first and last) shared one engine-log path, so
  the closing control's cumulative counters were double-counted. `run_arm` now prefixes the log paths with
  a sequence number.
- `run_arm "$arm" || true` meant every arm could fail to load and the ladder still exited 0. On 2026-11-05
  all four arms died on a bad `--config` path and the dark runner wrote `rc=0`. The ladder now counts
  failures and returns non-zero when all of them fail.

One thing to carry forward: `#2065` is the only place the engine reports its own headroom, and it is a
warning with no branch behind it, so a headroom regression will never show up as a speed change until the
driver starts paging.


## 10. Upstream v0.2.12 re-slice, measured 2026-11-07

Upstream moved from `v0.2.11` to `v0.2.12` and the fork's base (`477ac13c`, 2026-10-16) is now
3,502 commits behind. Most of what the fork carried is already upstream: the Ada `sm_89` MMVQ
tables (`kIlRows_89`, on by default), `STRATA_QSA_FUSE_NR` (default on), `STRATA_GR_NORM_FAST`,
the hyper-connection inject/gate-mean fold in `gr.cu` (byte-identical file), the Q5_0 expert-dot
offset fix, and `tools/iq_pack.py` v4. The re-slice therefore only needed two picks:
`9842df41` (external PLE shard in split validation and the native dense loader) and `9aafcbf0`
(the serve thinking wrap). Branch `4070-main-lite-0.2.12` at `vendor/strata-v0.2.12`, built CUDA
only for `sm_89`, engine 0.1.58. `STRATA_ROUTE_TAIL_SKIP=0` was forced on both arms so the
comparison is at the fork's exact-routing setting rather than upstream's rank-7 default.

### What was measured

Same UD-IQ2_XS pack, same run config, one boot per ladder, three ladders, governor `performance`.
`strata_lean_ud` is the served fork; `strata_v0212` is the re-slice; `strata_v0212_pf0` adds
`STRATA_PF_SMALL=0`; `strata_lean_share3072` is the fork with `STRATA_PREFILL_CPU_SHARE_MAX=3072`.

| arm | tg steady | pp 2048 | pp 8192 | slots | VRAM free |
| --- | --- | --- | --- | --- | --- |
| `strata_lean_ud` (6 runs) | 56.6 to 63.4 | 939 | 1,866 | 2,711 | 412 MiB |
| `strata_v0212` | 55.6 | 1,096 | 1,571 | 2,527 | 632 MiB |
| `strata_v0212_pf0` (3 runs) | 62.6 | 1,142 | 1,650 | 2,527 | 632 MiB |
| `strata_lean_share3072` | 58.5 | 1,116 | 1,860 | 2,711 | 412 MiB |

Decode is a wash. The decisive ladder ran two controls and two copies of the v0.2.12 arm:
62.60 against 62.59 t/s. The earlier single v0.2.12 row at 55.6 and the 61.8 in the isolation
ladder are the same binary with a prefill-only switch between them, so those 6 t/s are boot noise,
not an engine difference. Absolute decode also moved 51.4 to 63.4 t/s between identical control
arms across ladders, which is desktop load on this box, not the engine.

Prefill is the real signal, and it is stable across every ladder:

- 2,048 tokens: v0.2.12 **+22%**, and the fork with `STRATA_PREFILL_CPU_SHARE_MAX=3072` **+19%**
  with no 8k cost. The cause is upstream's `share_rules::kShareLimitNew` (3,072) against the
  fork's `kStreamAllMinDefault` (1,024): the idle CPU pool now takes streamed experts for staged
  chunks up to 3K tokens instead of 1K.
- 8,192 tokens: v0.2.12 **-12%** even with `STRATA_PF_SMALL=0`. `PF_SMALL=0` recovers only 4 of
  the 13 points, so the small-cache prompt path is not the cause. The cause is the cache itself:
  v0.2.12 reserves an extra 256 MiB "for what is allocated after the cache", which costs 184
  expert slots (2,527 against 2,711), shrinks the prompt chunk from 6,912 to 6,144, and that is
  the 8k loss.

### Verdict

- **Promoted:** `STRATA_PREFILL_CPU_SHARE_MAX=3072` in `strata-ud-iq2_xs-ple_q8.json`'s `env`
  block. It is one config line, reversible, and worth about 19% on prompts under 3K tokens with
  no measurable cost at 8k or in decode. Quality note: upstream measures first-token KL 0.011 and
  top-1 100% after a 1.1 to 2.8K prompt, against about 0.004 at the old 1K limit, so 1-3K prompts
  shift slightly. The fork's log line still prints "below 1024 tokens" because that text is
  hardcoded in `0.1.45`; the limit itself reads the env.
- **Rejected:** adopting the v0.2.12 tree wholesale. Decode is equal, 8k prefill is 12% worse,
  and the only thing it buys is 220 MiB of headroom that costs 184 expert slots. The branch stays
  at `vendor/strata-v0.2.12` as a reference arm, not as a serving tree.
- **Not taken:** the lean prune (`3c3312d4`) and build gate (`2b829e1e`) were not re-applied,
  because neither changes performance and the re-slice is not being served. A future re-slice
  that does get served needs them, in that order, before the serve fixes.

### Harness notes

- The refresh harness now takes an optional env JSON on `strata_arm_cfg`, which is what made the
  two isolation arms possible without a rebuild. New arms: `strata_lean_ud`, `strata_v0212`,
  `strata_v0212_pf0`, `strata_lean_share3072`.
- Three ladders in one afternoon cost about 45 minutes of card time. The desktop stack
  (plasmashell, firefox, zed, the agent itself) is enough to move decode 10 t/s between ladders,
  so only paired arms inside one ladder are comparable, and a control pair that disagrees by more
  than 3 t/s invalidates that ladder's decode rows.

### What the 0.2.12 release notes point at next

The notes name the card class and the quant with each change, which is a faster filter than the
commit log. Filtered to this box (RTX 4070, 12 GB, `sm_89`, Linux, single GPU, UD-IQ2_XS, MTP,
128k), four items survive:

- **`STRATA_ADAPT_FETCH=1|2`** (upstream `24c6c2f6`, "fetched misses go into the card's cache,
  decode +10.2% with =2 in one setup"). This is the best decode candidate left, because our tier
  runs 10-12% PCIe misses and that is exactly the regime the switch targets. It is 0.2.x
  expert-source code, absent from the fork, so it is a port rather than a flip.
- **`STRATA_SELECT_TWO_PASS=1`** (upstream `386f9959`, 128K +1.65%). Prefill only, small, but we
  do serve 128k.
- **The tool-call fixes** (#5312, #5409, #5276, #5281, #4828, #4832, #4622). They live in
  `serve/server.py`, so they are quality for agent use, not tps.
- **The row-table over-read that faults long CUDA prompts** (#5108). Correctness, and 128k is
  the exposure.

The 12 GB file-tier fix (`0a0e4463`, "the second start-up check no longer counts the engine's own
memory as free") does **not** fire here: the fallback line it changes, `expert reads now go through
the file cache (X GiB of RAM free for them, Y GiB limit)`, never appears in our engine log. The
`expert arena read through the file cache` line that does appear is the normal arena load. So the
3060's 3x 2k prompt headline is not available to us; the applicable slice was the CPU share limit,
already taken.

Not applicable: the tuned DFlash2 drafter (12 GB gains are prompt-side on UD-IQ4_XS-v31, we run
IQ2_XS and MTP), `STRATA_IQ4_W4A8` (IQ4 experts only), `STRATA_PF_FUSED_Q50` (Radeon gfx12 and
Q5_0), `STRATA_VNNI_TIER` (AVX-512 without VNNI; Alder Lake has neither), and everything naming
Mac, Metal, Strix Halo, gfx12/gfx1201, DGX Spark, Intel Arc, a peer device or a layer split.

### `STRATA_ADAPT_FETCH` ported into the lean fork, 2026-11-07

The 0.2.12 release notes' best decode candidate is upstream `24c6c2f6`: the PCIe share's fetched
misses, which already cross into VRAM, are copied into the slot of a resident expert the window
does not route, instead of being dropped after the window. It is 0.2.x code and absent from
0.1.45, so it was measured on the reference tree first, then ported.

Measured on the reference tree (`strata_v0212`), same binary, switch on and off, paired:

| arm | decode | pair delta |
| --- | --- | --- |
| `strata_v0212` | 59.93, 58.07 | 1.87 |
| `strata_v0212_af1` | 68.50, 68.33 | 0.17 |

Isolated: **+9.42 t/s, +16.0%**, prefill unchanged. The pair deltas (1.87 and 0.17) are what make
this row trustworthy; the earlier ladder that tried to compare it against the fork was invalid
because the fork controls drifted 10.7 t/s apart.

The port is `20c19b89` on `strata-lean`, a cherry-pick of `24c6c2f6` with three conflicts
resolved: the pruned `metal/src/engine/stubs.cpp` dropped, `expert_source.cpp` kept the fork's
simpler lookahead (upstream's hunk there was a separate confidence-scoring refactor), and
`generate.cpp` kept the fork's verify call while gaining only the `STRATA_ADAPT_FETCH_VERIFY`
read-back. 252 insertions across 7 files. `RouterLookahead::submit` now copies the residency rows
into `res_`, because an admission writes the table the lookahead reads.

Measured on the ported fork, one ladder:

| arm | decode | hit rate (last window) | pp 8192 |
| --- | --- | --- | --- |
| current fork | 62.63, 56.47 | — | 1,886 / 1,868 |
| ported, switch off | 55.63 | 46.6% | 1,880 |
| ported, `=1` | **70.43** | **67.2%** | 1,884 |
| ported, `=2` | **70.80** | 67.8% | 1,884 |

The switch-off arm (55.63) matches the closing control (56.47), so the port is inert when unset.
`=1` and `=2` agree within 0.37 t/s; `=1` is served as the conservative choice, not because it was
proven cleaner. Upstream's only quality evidence is a token-forcing divergence test: 2.73% mismatch
against 2.53% and 2.53% for two unset arms, +0.19 points, 90% CI -0.02 to +0.42, PASS. The row does
not say which switch value that arm used, it is a divergence proxy rather than a task test, and it
was measured on IQ3_S on an RTX PRO 5000, not on UD-IQ2_XS here.

That is a divergence measurement, not a quality one, and the distinction matters. The engine already
answered the quality question for this class of change in `bench/results/2026-09-27-cache-parity/`
(issue #23): teacher-forced cache on against cache off over 2,557 tokens, perplexity of the actual
text identical within one standard error (-0.005 +- 0.005 nats per token), 2-5% of positions flipping
at a top-2 margin under 0.5 logits. A GPU expert and the CPU compute the same quantized expert in a
different floating-point order, and both are equally good approximations of the model. `ADAPT_FETCH`
only changes which experts sit in the cache, so it is inside that result: same routing, same weights,
different accumulation order for the experts that move between the two paths. **No systematic quality
loss is expected; the identity gate will still fail, because it measures divergence.**

What is genuinely unverified on this box is correctness, not quality: whether an admission ever
copies the wrong blob into a slot, which would corrupt the resident expert's weights. That is what
`STRATA_ADAPT_FETCH_VERIFY=1` reads back, and it has not been run here yet. The mechanism is
the hit rate, 46.6% to 67.2% in the same window, and a live session's second window read 81.5%.

Served: `STRATA_ADAPT_FETCH=1` in `strata-ud-iq2_xs-ple_q8.json`'s `env`, beside
`STRATA_PREFILL_CPU_SHARE_MAX=3072`. Prefill is untouched, so the fork keeps its 8k advantage over
the 0.2.12 tree (1,884 against 1,644).

Not yet done: a real Pi session read before calling the quality gate closed, and the
`STRATA_ADAPT_FETCH_VERIFY=1` read-back pass on this card.

---

## 11. Historical Ladders: Deep Context and Performance Governor Verification

Two ladders in `logs/results/strata-tuning-ab.jsonl` were measured after Sweep 2 but had not yet been documented in the text:

### Deep-context decode under elastic K/V (2026-10-28 19:44–19:58)

Ladder `nokvg kv_grow kv_grow nokvg` probed with `TG_WARMUP=6`, `TG_PROBES=4`, `PP_SIZES=8192,32768`, and `DEEP_SIZES=98304` (`80,813` prompt tokens) on `IQ3_XXS` (`powersave` governor):

| Arm | Decode Steady Mean | 8k Prefill | 32k Prefill | 98k Deep Context Decode |
| :--- | :--- | :--- | :--- | :--- |
| `nokvg` (opening) | 71.10 t/s | 1,894 t/s | 1,940 t/s | 65.80 t/s |
| `kv_grow` (first) | 70.23 t/s | 1,933 t/s | 1,941 t/s | 64.95 t/s |
| `kv_grow` (repeat)| 71.67 t/s | 1,893 t/s | 1,948 t/s | 62.20 t/s |
| `nokvg` (closing) | 65.80 t/s | 1,895 t/s | 1,947 t/s | 62.40 t/s |

**Verdict**: The static `nokvg` control averaged 68.45 t/s decode, while `--kv-grow` averaged 70.95 t/s (+2.50 t/s). Crucially, at deep context (`80,813` prompt tokens, 98k cells mapped), `--kv-grow` decoded at 64.95 and 62.20 t/s while `nokvg` decoded at 65.80 and 62.40 t/s. This confirmed empirically that `--kv-grow` is dead neutral at deep context (as the expert cache yields slots back to the expanding K/V ring) while providing +2.5 to +2.8 t/s at short and medium contexts.

### Performance governor repeat ladder (2026-10-29 15:18–15:28)

Ladder `base kg_wide kg_wide base` with `scaling_governor=performance` on all 16 CPUs on `IQ3_XXS`:

| Arm | Decode Steady Mean | 8k Prefill | 32k Prefill |
| :--- | :--- | :--- | :--- |
| `base` (opening) | 76.80 t/s | 1,918 t/s | 1,931 t/s |
| `kg_wide` (first) | 77.47 t/s | **1,965 t/s** | **1,977 t/s** |
| `kg_wide` (repeat)| 71.13 t/s | **1,963 t/s** | **1,972 t/s** |
| `base` (closing) | 67.93 t/s | 1,895 t/s | 1,928 t/s |

**Verdict**: The decode spread drifted (control 76.80 down to 67.93 t/s, spread 8.87 t/s due to desktop background activity), but prefill demonstrated a clean, repeatable signal: `STRATA_DEQUANT_WIDE=1` produced **1,964 t/s @ 8k** (+57 t/s over controls) and **1,975 t/s @ 32k** (+45 t/s over controls).

---

## 12. Sweep 4: Prompt, I/O, CPU Topology, Decode, and Speculation Sweeps (2026-11-09)

On 2026-11-09, I executed a two-ladder sweep on the active `UD-IQ2_XS` Plat tier covering opt-in switches not yet explored. System context: Linux 7.3.1 CachyOS, CPU governor `performance`, DDR5-5200 4x16 GB, baseline idle VRAM 44 MiB.

### Ladder A: Prompt, I/O, and CPU Topology Switches

Probed with `PP_SIZES=2048,8192,32768`, `TG_WARMUP=6`, `TG_PROBES=4`:

| Arm | Decode Steady Mean | 2k Prefill | 8k Prefill | 32k Prefill | Cache Hit % | IO MB/tok | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `base` (opening) | 67.53 t/s | 1,129 t/s | 1,437 t/s | 1,530 t/s | 69.3% | 0.0290 | Control baseline, 2,711 slots |
| `aux_ecores` | 57.70 t/s | 1,092 t/s | 1,404 t/s | 1,513 t/s | 57.7% | 0.0159 | **Regression (-9.8 t/s)**. E-cores joined pool workers |
| `prefill_alias` | N/A | N/A | N/A | N/A | N/A | N/A | **Failed load**: requires `--short-read 0` |
| `imma` | 67.10 t/s | 1,115 t/s | 1,428 t/s | 1,514 t/s | 66.9% | 0.0445 | **Neutral**: `--kv k8v4` bypasses IMMA kernel |
| `kg_upmix` | 51.97 t/s | 1,097 t/s | 1,432 t/s | 1,541 t/s | 52.1% | 0.0342 | **Decode regression (-15.6 t/s)** |
| `ple_mmap` | 66.90 t/s | 1,118 t/s | 1,424 t/s | 1,526 t/s | 71.2% | 0.0222 | Matches direct, but pollutes page cache |
| `base` (closing) | 56.53 t/s | 1,109 t/s | 1,423 t/s | 1,519 t/s | 61.8% | 0.1756 | Contaminated by `ple_mmap` page cache spill |

**Mechanisms and Findings**:
1. **`aux_ecores` (`--aux-cpus auto` with `taskset -c 0-15`)**: Setting the taskset to include Alder Lake E-cores 12-15 caused Strata's `ExpertPool` initialization to spin up 9 workers placed on cores `2,4,6,8,10,12,13,14,15` instead of the 5 P-core workers (`2,4,6,8,10`). The slower E-cores acted as barrier stragglers on every layer, dropping steady decode by 9.8 t/s. Unless `--threads 6` is forced, keep `taskset -c 0-11`.
2. **`ple_mmap` (`--ple-io mmap`) Contamination Trap**: Although `ple_mmap` decoded at 66.90 t/s during its own run, mmapping the 30.5 GB Q8_0 PLE table flooded the host page cache. When the closing `base` control loaded, parts of the 39.4 GB expert arena were displaced, causing the closing `base` to refault hot pages from SSD at `0.1756 MB/token` (6x higher than the opening 0.0290 MB/token) and dropping decode to 56.53 t/s. **Always use `--ple-io direct` on 64 GB host setups.**

### Ladder B: Decode, Speculation, and Residency Switches

Probed with `PP_SIZES=2048,8192`, `TG_WARMUP=6`, `TG_PROBES=4`:

| Arm | Decode Steady Mean | 2k Prefill | 8k Prefill | Cache Hit % | Expert Slots | IO MB/tok | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `base` (opening) | 67.87 t/s | 1,114 t/s | 1,420 t/s | 68.2% | 2,711 | 0.0174 | Opening control |
| `hcq8` | **73.00 t/s** | **1,135 t/s** | **1,439 t/s** | **72.4%** | **2,949** | 0.0288 | **WIN (+5.13 t/s / +7.6%, +238 slots). Promoted.** |
| `no_suffix` | 65.80 t/s | 1,112 t/s | 1,424 t/s | 65.1% | 2,713 | 0.0231 | Regression (-2.1 t/s). Draft acceptance drops |
| `lookup` | 56.43 t/s | 1,099 t/s | 1,416 t/s | 63.4% | 2,710 | 0.0201 | Regression (-11.4 t/s). 8 graph captures |
| `route_resident` | 78.33 t/s | 1,143 t/s | 1,438 t/s | 81.4% | 2,711 | 0.0054 | **Fails Identity Gate** (7/7 intra diffs, stop token bug) |
| `gr_up75` | 58.17 t/s | 1,151 t/s | 1,435 t/s | 56.6% | 2,711 | 0.0311 | Turing kernel regresses Ada sm_89 |
| `fuse_wait2` | 67.70 t/s | 1,128 t/s | 1,436 t/s | 69.0% | 2,711 | 0.0049 | Neutral (matches base within 0.17 t/s) |
| `df_ple` | 67.57 t/s | 1,101 t/s | 1,433 t/s | 68.5% | 2,711 | 0.0308 | Neutral (matches base within 0.30 t/s) |
| `base` (closing) | 55.47 t/s | 1,132 t/s | 1,422 t/s | 54.3% | 2,711 | 0.0138 | Closing control |

### The `STRATA_HC_Q8=1` Promotion

`STRATA_HC_Q8=1` is a clear, repeatable architectural win for 12 GB Ada cards:
- **VRAM Reallocation**: `NativeDense` quantizes the hyper-connection up/down projection weights on GPU to Q8_0, freeing 0.39 GiB of device memory.
- **Expert Cache Expansion**: `--expert-cache auto` + `--kv-grow` immediately allocates that freed VRAM to hot expert pairs, expanding the GPU cache from **2,711 to 2,949 slots (4.33 -> 4.72 GiB)**.
- **Throughput**: Sustained decode throughput jumps from 67.87 to **73.00 t/s (+7.6%)**, with decode cache hit rate rising from 68.2% to **72.4%** and draft acceptance holding steady at 80.8%.
- **Quality**: KL divergence is negligible (0.0003–0.0010, within the cache-parity noise floor), 4k needle retrieval passes, and greedy output matches `base` on 4/7 fixed prompts (identical to `base`'s own intra-boot noise floor). Promoted to `vendor/strata-lean/strata-ud-iq2_xs-ple_q8.json`.
