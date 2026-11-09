# L3MS Documentation Index

Central knowledge base and operational index for `L3MS` (Language, Media, Decisions).

---

## 1. Strata Engine & Kernels (Platinum Tier)

Deep-dive documentation for `carteakey/strata-lean` (branch `strata-lean`, optimized Ada `sm_89` on RTX 4070 12 GB):

- **[Strata Tuning Matrix](strata-tuning-matrix.md)** - **Single source of truth** for all active, rejected, and candidate runtime switches and env flags.
- **[Strata Lean Fork Overview](strata-lean.md)** - Architecture of the lean fork: stripping bloat, preserving Ada sm_89, cherry-pick policy, and precision trade-offs.
- **[Strata Performance Next Steps (2026-10)](strata-perf-next-steps-2026-10.md)** - Comprehensive audit of upstream releases, tuning sweep records (Sweeps 1 & 2), and optimization backlogs.
- **[Strata Evaluation & Intake](strata-evaluation-internal.md)** - Baseline benchmarks, VMM dynamic expert caching architecture, and comparison against llama.cpp.

---

## 2. Model Architectures, Quants & Speculation

Architecture, quant formats, and speculative decoding details for Qwen3.8-Flash-Next (qwen4exp):

- **[Qwen3.8-Flash-Next Internal Guide](qwen38-flash-next-internal.md)** - Canonical reference for Gold, MTP, Exp, and Platinum serving configurations.
- **[PLE Table Precision & I/O](qwen3-8-flash-next-ple-precision.md)** - Unquantized Q8_0 PLE n-gram table offloading to NVMe, direct I/O vs mmap.
- **[Variant Ranking & Quant Matrix](qwen38-flash-next-variant-ranking.md)** - Comparative analysis of Unsloth UD-IQ2_XS vs ISTA-DASLab IQ3_XXS vs IQ3_S.
- **[Decision Models Guide](decision-models.md)** - TypeSafe System One non-causal decision model runtimes (vLLM-SR d3-flash, Clef).

---

## 3. Operational Runbooks & Tooling

Day-to-day operations, service supervision, and benchmarking guides:

- **[llama-swap Runbook](llama-swap-runbook.md)** - Router operations, config reloading (`kill -HUP`), process supervision, and zero-downtime swaps.
- **[Bench Runbook](bench-runbook.md)** - Standardized benchmarking protocols, CPU power management (`cpupower`), frontmatter logging, and isolation rules.
- **[Bench Troubleshooting](bench-troubleshooting.md)** - Diagnosing SSD read thrashing, memory spills, and thermal throttling during sweeps.
- **[Media Models Runbook](media-generation-runbook.md)** - ComfyUI integration (`comfyui_auto`), MiniMax-H3 video, and audio.cpp workflows.
- **[Model Onboarding Playbook](model-onboarding-playbook.md)** - Step-by-step checklist for bringing a new model into `llama-swap.yaml`.

---

## 4. Incidents & Postmortems

Postmortems, root-cause analyses, and regression investigations:

- **[2026-11-09: Strata Elastic K/V (`--kv-grow`) OOM at 128k](incidents/2026-11-09-strata-kv-grow-oom.md)** - Full postmortem on prefill loan over-subscription, 155-slot clamp, and `STRATA_PREFILL_LEND_PCT=40` resolution.
- **[Slow Generation (tg) Investigation](slow-tg-investigation.md)** - Diagnosing SSD major page faults and cold page-in effects on SSD-offloaded models.

---

## 5. Web Dashboard & Telemetry

Visualizations, benchmark graphs, and static site assets:

- **[Dashboard Metadata](dashboard-meta.json)** - Serving and benchmark metadata powering the web dashboard.
- **[Generated Model Catalog](generated-models.js)** - Auto-generated model profiles and performance numbers.
- **[Community Runs Schema](community-runs.schema.json)** - JSON schema for community benchmark submissions.

---

## 6. Publications & Blog Drafts

Technical posts mirrored from and to `carteakey.dev`:

- **[The Rise of Overfit Inference Engines](qwen3-8-flash-next-local-post.md)** - Local inference deep dive on Strata, MTP, and Qwen3.8-Flash-Next.
- **[Strata Self-Tuning Post](strata-self-tuning-local-post.md)** - Draft blog post detailing automated tuning sweeps on consumer hardware.
- **[Qwen 27B vs Flash-Next](qwen3-8-27b-vs-flash-next-local-post.md)** - Comparative study between dense 27B hybrid and MoE Flash-Next.
