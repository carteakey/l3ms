# Benchmark Harnesses (`bench-models/`)

Active local LLM benchmark scripts, shared runners, and structured JSONL logs for `L3MS` (RTX 4070 12 GB `sm_89` + i5-12600K + 64 GB DDR5). See [docs/bench-runbook.md](../docs/bench-runbook.md) for preflight hygiene, CPU governor checks (`performance`), and steady-state measurement rules.

## Shared Infrastructure

| File / Directory | Purpose |
| :--- | :--- |
| [`bench-env.sh`](bench-env.sh) | Host preflight warnings and YAML/JSON system frontmatter capture. |
| [`log-result.sh`](log-result.sh) | Append structured benchmark rows to `logs/results/<MODEL_KEY>.jsonl`. |
| [`run-llama-bench.sh`](run-llama-bench.sh) | Shared `llama-bench` runner invoked by per-model wrapper scripts. |
| [`run-llama-fit-bench.sh`](run-llama-fit-bench.sh) | Shared `--fit` solver benchmark runner. |
| [`run-llama-fit-params.sh`](run-llama-fit-params.sh) | Inspect `llama-fit-params` layer placement without running generation. |
| [`run-ik-llama-bench.sh`](run-ik-llama-bench.sh) | Shared `ik_llama.cpp` benchmark runner. |
| [`run-strata-tuning-dark.sh`](run-strata-tuning-dark.sh) | Unattended overnight Strata tuning wrapper (stops/restarts `llama-swap`). |
| [`strata-identity-probe.py`](strata-identity-probe.py) | Greedy identity probe used by `bench-strata-tuning-ab.sh` quality gates. |
| [`presets/`](presets/) | Reusable environment presets (`fast.env`, etc.). |
| [`templates/`](templates/) | Template scripts for onboarding new models. |
| [`prompts/`](prompts/) | Fixed prompt sets for reproducible A/B sweeps. |
| [`logs/`](logs/) | Tracked structured results (`logs/results/*.jsonl`); raw `.log` files are gitignored. |

## Active Model Harnesses

### Qwen3.8-Flash-Next (`Strata` Platinum & `llama.cpp` Gold/Exp/MTP)

- [`bench-strata-tuning-ab.sh`](bench-strata-tuning-ab.sh) — Canonical Strata `strata-lean` A/B switch harness (with warmup/steady split and identity gate).
- [`bench-strata-lookahead-ab.sh`](bench-strata-lookahead-ab.sh) — Strata lookahead / speculation A/B sweep.
- [`bench_strata_long_horizon.py`](bench_strata_long_horizon.py) — Long-context Strata scaling probe.
- [`bench-qwen38-refresh-ab.sh`](bench-qwen38-refresh-ab.sh) & [`bench-qwen38-refresh-probe.py`](bench-qwen38-refresh-probe.py) — Multi-tier parity refresh benchmark (Strata vs `llama.cpp` master/exp).
- [`bench-llama-qwen38-flash-next-build-ab.sh`](bench-llama-qwen38-flash-next-build-ab.sh), [`bench-llama-qwen38-flash-next-mtp.sh`](bench-llama-qwen38-flash-next-mtp.sh), [`bench-llama-qwen38-flash-next-quant-ab.sh`](bench-llama-qwen38-flash-next-quant-ab.sh), [`bench-llama-qwen38-flash-next-pp.sh`](bench-llama-qwen38-flash-next-pp.sh), [`bench-llama-qwen38-flash-next-graphopt.sh`](bench-llama-qwen38-flash-next-graphopt.sh) — `llama.cpp` fallback tier sweeps.
- [`bench-llama-qwen38-28243-q4km.sh`](bench-llama-qwen38-28243-q4km.sh), [`bench-llama-qwen38-28243-unique.sh`](bench-llama-qwen38-28243-unique.sh), [`bench-llama-qwen38-flash-next-ik-campaign.sh`](bench-llama-qwen38-flash-next-ik-campaign.sh), [`bench-llama-qwen38-flash-next-ik-mtp.sh`](bench-llama-qwen38-flash-next-ik-mtp.sh), [`bench-llama-qwen38-flash-next-ik-unique.sh`](bench-llama-qwen38-flash-next-ik-unique.sh), [`bench-llama-qwen38-flash-next-mtp144-sweep.sh`](bench-llama-qwen38-flash-next-mtp144-sweep.sh), [`bench-llama-qwen38-plain-compare.sh`](bench-llama-qwen38-plain-compare.sh) — PR/MTP/ik comparative harnesses.
- [`bench-llama-qwen38-direct-reads.py`](bench-llama-qwen38-direct-reads.py), [`bench-llama-qwen38-mtp-qsa.py`](bench-llama-qwen38-mtp-qsa.py), [`bench-llama-qwen38-qsa-sparsity.py`](bench-llama-qwen38-qsa-sparsity.py), [`bench_pp_sweep.py`](bench_pp_sweep.py), [`test-gold-layer-fitting.sh`](test-gold-layer-fitting.sh), [`test_ub2048_feasibility.py`](test_ub2048_feasibility.py), [`test_vision_model.py`](test_vision_model.py), [`challenge_3d_train.py`](challenge_3d_train.py) — Targeted PLE I/O, QSA sparsity, prefill, vision, and quality probes.

### Inkling 2 Mini, Nemotron 3.5 Bolt, Qwen3.8-27B, Gemma 4 26B & Humanizer 12B

- [`bench-llama-cpp-inkling-2-mini.sh`](bench-llama-cpp-inkling-2-mini.sh) — Inkling 2 Mini `llama.cpp` baseline harness.
- [`bench-llama-cpp-nemotron-3.5-bolt.sh`](bench-llama-cpp-nemotron-3.5-bolt.sh) — Nemotron 3.5 Bolt (74B-A9B) harness.
- [`bench-llama-qwen3-8-27b-q2-ctx.sh`](bench-llama-qwen3-8-27b-q2-ctx.sh), [`bench-llama-qwen3-8-ridge-baseline.sh`](bench-llama-qwen3-8-ridge-baseline.sh), [`bench-llama-qwen3-8-ridge-mtp.sh`](bench-llama-qwen3-8-ridge-mtp.sh) — Qwen3.8-27B dense pure-VRAM / long-context harnesses.
- [`bench-llama-cpp-gemma-4-26b-a4b.sh`](bench-llama-cpp-gemma-4-26b-a4b.sh) (plus `-fit`, `-strategies`, `-q6-x-l*`, and [`bench-llama-gemma-26b-qat-mtp.sh`](bench-llama-gemma-26b-qat-mtp.sh)) — Gemma 4 26B QAT / MTP harnesses.
- [`bench-llama-humanizer.sh`](bench-llama-humanizer.sh) — Humanizer 12B Q6_K full-GPU harness.

## Archive (`archive/`)

[`archive/`](archive/) holds 32 historical benchmark scripts for retired models whose weights have been deleted or archived off the primary node:

- `gpt-oss-*` (120B, 20B, Puzzle 88B)
- `mistral-small-4-119b-*`
- `qwen3-coder-next-*`
- `qwen3-5-122b-a10b-*`
- `qwen3-6-*` (35B-A3B Reddit baselines)
- `nemotron-super-120b-*`
- `sarvam-30b-*`
- `qwen3-30b-a3b` / `qwen3-cpu-only`

Archived scripts are kept for historical reproducibility reference only and are not run in CI syntax checks.
