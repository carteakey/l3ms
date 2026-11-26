# L3MS Documentation Index

Central knowledge base and operational index for `L3MS` (Language, Media, Decisions).

---

## 1. Architecture, Roofline & Audits

System design, silicon bounds, remote burst compute, and stack audits:

- **[Stack Audit, Website & Dashboard Redesign (2026-11-26)](stack-audit-and-redesign-2026-11-26.md)** - Empirical audit of the live `L3MS` stack on `yeti-cachy`, public site redesign, local dashboard plan, and monorepo cleanup.
- **[RTX 4070 12 GB Inference Roofline](rtx4070-inference-ceilings.md)** - Measured VRAM/PCIe/DDR5/compute ceilings and theoretical decode/prefill bounds on `yeti-cachy`.
- **[Colab Burst Lane Plan](colab-burst-lane.md)** - On-demand Google Colab compute (`pro`/`ultra`, L4/A100/G4/H100) for fine-tuning, oversized models, and non-evicting remote endpoints.
- **[Local Live Control Dashboard](local-dashboard.md)** - Tailscale live control plane (`:8085`) for hardware telemetry, `llama-swap` model switching, active engine config, and smoke runs.
- **[Workbench Design](workbench-design.md)** - Sidebar, toolbar, tabs, and sections proposal for the L3MS TUI/GUI shell.
- **[Platform Portability](platform-portability.md)** - Notes on Linux/Windows path and runtime portability.

---

## 2. Strata & PolyStrata Engines

Deep-dive documentation for `carteakey/strata-lean` (Ada `sm_89` for Qwen3.8-Flash-Next) and `polystrata-4070` (Inkling 2 Mini):

- **[Strata Tuning Matrix](strata-tuning-matrix.md)** - **Single source of truth** for all active, rejected, and candidate runtime switches and env flags.
- **[Strata Lean Fork Overview](strata-lean.md)** - Architecture of the lean fork: stripping bloat, preserving Ada `sm_89`, cherry-pick policy, and precision trade-offs.
- **[Strata Performance Next Steps (2026-10)](strata-perf-next-steps-2026-10.md)** - Comprehensive audit of upstream releases, tuning sweep records, and optimization backlogs.
- **[Strata Evaluation & Intake](strata-evaluation-internal.md)** - Baseline benchmarks, VMM dynamic expert caching architecture, and comparison against `llama.cpp`.
- **[Inkling 2 Mini Optimization Log](inkling-2-mini-optimization.md)** - PolyStrata sweeps, 5-engine shootout, DSpark speculative decoding, and RTX 4070 12 GB tuning.

---

## 3. Language, Vision, Media & Decisions

Architecture, quant formats, and per-lane setup guides:

- **[Qwen3.8-Flash-Next Internal Guide](qwen38-flash-next-internal.md)** - Canonical reference for Gold, MTP, Exp, and Platinum serving configurations.
- **[PLE Table Precision & I/O](qwen3-8-flash-next-ple-precision.md)** - Unquantized `Q8_0` PLE n-gram table offloading to NVMe, direct I/O vs mmap.
- **[Variant Ranking & Quant Matrix](qwen38-flash-next-variant-ranking.md)** - Comparative analysis of Unsloth `UD-IQ2_XS` vs ISTA-DASLab `IQ3_XXS` vs `IQ3_S`.
- **[Nemotron 3.5 Transcribe](nemotron-transcribe.md)** - Local speech transcription (`speech.cpp`) notes.
- **[Decision Models Guide](decision-models.md)**, **[Decision Toolkit](decision-toolkit.md)** & **[Decision Models CLI](decision-models-cli.md)** - TypeSafe System One non-causal decision model runtimes (`vllm-d3-flash`, `cloudflare-clef-flash`).
- **[Classifier Triage (Jev / pi codemode)](pi-jev-classifier-workflow.md)** - Automated benchmark and log triage via `/jev-bench` and `/jev-log`.
- **[Media Models Overview](media-models.md)** & **[Media Generation Runbook](media-generation-runbook.md)** - ComfyUI (`comfyui_auto`) and `audio.cpp` (`audiocpp_auto`) routing through `llama-swap`.
- **[ComfyUI Setup](comfyui-setup.md)**, **[LTX-2.5 / YuE2 ComfyUI Notes](comfyui-ltx-yue2.md)**, **[Qwen Image 2.1](qwen-image-2.1-setup.md)**, **[Qwen Image 2.1 Turbo](qwen-image-2.1-turbo-setup.md)**, **[Krea 2](krea-2-setup.md)**, **[FLUX.2 Klein 9B](flux-2-klein-9b-setup.md)**, **[Z-Image Turbo](z-image-turbo-setup.md)** - Image and video workflow runbooks.
- **[audio.cpp Runtimes](audio-cpp-runtimes.md)**, **[YuE2 Setup](yue2-setup.md)**, **[MiniMax-H3 Quality Notes](minimax-h3-quality-notes.md)** - Audio, TTS, and music generation notes.
- **[Blender MCP Setup](blender-mcp-setup.md)** - Headless and interactive Blender 5.2 LTS MCP workflows.
- **[CPU-Native Workflows](cpu-native-workflows.md)** & **[Native GPU Workflows](native-gpu-workflows.md)** - Direct runtime profiles and GPU handoff constraints.
- **[ComfyUI MCP & Tailscale Notes](2026-09-25-comfyui-mcp-tailscale.md)** & **[Qwen Code CLI Setup](qwen-code-cli.md)** - Client integration guides.

---

## 4. Operational Runbooks & Benchmarking

Day-to-day operations, service supervision, and benchmarking guides:

- **[llama-swap Runbook](llama-swap-runbook.md)** - Router operations, config reloading, process supervision, and exclusive VRAM swaps.
- **[Bench Runbook](bench-runbook.md)** - Standardized benchmarking protocols, CPU power management (`cpupower`), frontmatter logging, and isolation rules.
- **[Bench Results](bench-results.md)** - Measured per-model reference results and quickstarts.
- **[Bench Troubleshooting](bench-troubleshooting.md)** - Diagnosing SSD read thrashing, memory spills, and thermal throttling during sweeps.
- **[Model Onboarding Playbook](model-onboarding-playbook.md)** - Step-by-step checklist for bringing a new model into `llama-swap.yaml`.
- **[GPU Inference Switching](gpu-inference-switching.md)** & **[GPU Unlock Evaluation](gpu-unlock-evaluation.md)** - GPU handoff and exclusive access notes.
- **[Agent Environment Retrospective (2026-11-25)](retro-2026-11-25.md)** - Session audit covering `AGENTS.md` trimming, serving guardrails, `l3ms-ctl.sh`, and wiki deduplication.
- **[Benchmark Harnesses Index](../bench-models/README.md)** - Active benchmark scripts vs `bench-models/archive/`.

---

## 5. Incidents, Experiments & Migration Evidence

- **[2026-11-09: Strata Elastic K/V (`--kv-grow`) OOM at 128k](incidents/2026-11-09-strata-kv-grow-oom.md)** - Full postmortem on prefill loan over-subscription and `STRATA_PREFILL_LEND_PCT=40`.
- **[Slow Generation (tg) Investigation](slow-tg-investigation.md)** - Diagnosing SSD major page faults and cold page-in effects on SSD-offloaded models.
- **[Experiments (`experiments/`)](experiments/)** - Empirical comparisons for Qwen Image 2.1 Turbo vs Base, Qwen vs Krea 2, Breeze-TTS-2, and MiniMax-H3.
- **[Migration & Runtime Policy (`migration/`)](migration/)** - Shared-engine runtime policy (`migration/runtime-policy.md`), `llama-swap` media integration (`migration/media-through-llama-swap.md`), and acceptance records.

---

## 6. Web Dashboard & Publications

- **[Dashboard Metadata](dashboard-meta.json)** & **[Stack Metadata](stack-meta.json)** - Serving and benchmark metadata powering `l3ms.carteakey.dev`.
- **[Generated Model Catalog](generated-models.js)** - Auto-generated model profiles and performance numbers.
- **[Community Runs](community-runs.md)** & **[Schema](community-runs.schema.json)** - Community benchmark submission format.
- **[Model Post Template](model-post-template.md)** & **[Sarvam Local Post](sarvam-local-post.md)** - Publication template and tracked post (`docs/*-post.md` drafts for `carteakey.dev` are gitignored locally).

---

## 7. Archive (`.archive/`)

Unreferenced historical plans, code reviews, point-in-time checkpoints, superseded scripts, and legacy scratch transcripts live in [`.archive/`](.archive/):

- Historical plans & proposals: `implementation_plan_v1.md`, `implementation_plan_v2.md`, `monorepo-consolidation-plan.md`, `interface-direction.md`, `workbench-plan.md`, `extraction.md`
- Point-in-time reviews & checkpoints: `code-review-2026-09-03.md`, `code-review-2026-09-28.md`, `checkpoint-2026-10-17.md`, `state-2026-11-12.md`, `local-inference-experiment-handoff.md`
- Superseded artifacts & scratch: `tui-color-theme-draft.patch`, `vibe_configuration.md`, `srv_start.sh`, `thread.md`
