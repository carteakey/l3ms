# Local Inference Experiment Handoff

Status: ready for the homelab node  
Created: 2026-06-22  
Target hardware: RTX 4070 12GB, i5-12600K, 32GB DDR5-6000

This document holds changes that sound useful but should not become guide advice or production `llama-swap.yaml` defaults until they win on the actual L3MS machine.

## Rules

1. Do not tune production profiles in place. Add a bench script or named temporary profile.
2. Record the llama.cpp commit, model checksum/path, context, cache types, batch sizes, environment variables, free VRAM, and CPU power state with every run.
3. Run each configuration at least three times. Discard the first run when it includes model loading or kernel compilation.
4. Use the same prompts, seed, output length, and context for every comparison.
5. Record PP, TG, TTFT, acceptance rate, peak VRAM, and whether output/tool calling remained correct.
6. A throughput win that introduces OOMs, malformed tool calls, thinking loops, or lower draft acceptance is not a win.

## Baseline capture

Before each experiment:

```bash
git -C vendor/llama.cpp rev-parse HEAD
sudo bash preflight-check.sh
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv
```

Use these reference profiles:

- `qwen3-6-mtp`: MTP with q8 target KV.
- `gemma-4-12b-qat-mtp`: fast MTP profile with the strongest existing result.
- `gemma-4-26b-qat-mtp`: Gemma configuration that previously showed KV sensitivity.
- `qwen3-coder-next`: non-MTP coding baseline for PP/cache/ubatch tests.

Use the existing task set from `mtp_bench.py` for speculative tests. Add one repeated-code editing prompt and one agent/tool-call transcript for cache and production-workflow tests.

## Experiment A: n-gram speculation

Goal: determine where n-gram speculation helps alone and whether combining it with MTP improves end-to-end throughput.

Test each applicable model with:

1. No speculation.
2. Current MTP configuration.
3. `--spec-type ngram-mod` with upstream defaults.
4. `--spec-type draft-mtp,ngram-mod` with the current MTP draft length.
5. If useful, sweep `--spec-ngram-mod-n-match`, `--spec-ngram-mod-n-min`, and `--spec-ngram-mod-n-max` around upstream defaults.

Prioritize repeated code editing, summarization, and reasoning-to-final-answer prompts. N-gram techniques depend on repetition and may do little for unrelated short chat.

Record accepted draft tokens separately for MTP and n-gram modes when the logs expose them. Current llama.cpp gives draftless speculative methods precedence when they are combined with a draft model, so a “stacked” result must be interpreted from logs rather than assumed additive.

Promotion gate: improve median end-to-end generation time by at least 10% on two relevant tasks without slowing the rest of the suite by more than 5%.

## Experiment B: target KV versus draft KV

Goal: replace the current Gemma-specific KV rule with model-specific evidence.

Target cache sweep:

```text
-ctk f16  -ctv f16
-ctk q8_0 -ctv q8_0
-ctk q5_1 -ctv q5_1   # only if supported and VRAM pressure justifies it
```

Draft cache sweep, holding the target cache fixed:

```text
-ctkd f16  -ctvd f16
-ctkd q8_0 -ctvd q8_0
```

Run this on Gemma 4 12B MTP, Gemma 4 26B MTP, and Qwen3.6 MTP. Record acceptance rate, TG, PP, and VRAM. Do not infer draft-cache behavior from a target-cache test; these are separate buffers and flags.

Promotion gate: publish per-model recommendations. Do not create one MTP-wide default unless all tested families agree.

## Experiment C: CUDA graph optimization

Goal: decide whether `GGML_CUDA_GRAPH_OPT=1` helps any current profile.

For each reference profile, compare unset/`0` against `1` at short context and after a long agent-style prompt. Record startup, PP, TG, peak VRAM, graph recapture messages, and OOM behavior.

Promotion gate: keep `1` only on profiles where the repeated-run median improves and long-context stability is unchanged. Otherwise remove the environment variable.

## Experiment D: ubatch and prompt processing

Goal: improve PP without presenting another machine-specific value as universal.

Sweep:

```text
--batch-size 2048 --ubatch-size 512
--batch-size 2048 --ubatch-size 1024
--batch-size 2048 --ubatch-size 2048
```

Use 4k, 16k, and 32k prompts. Record PP, TTFT, peak VRAM, and prefill OOMs. Repeat separately for a dense profile and a hybrid MoE profile.

Promotion gate: choose a per-profile value that improves PP without reducing available VRAM below the profile’s stability margin.

## Experiment E: prompt caching and cache reuse

Goal: measure the workflow users actually feel during repeated coding-agent turns.

Compare:

1. Prompt caching enabled with `--cache-reuse 0`.
2. `--cache-reuse 256`.
3. `--cache-reuse 512`.

Send a shared long system/repository prefix followed by small changing suffixes. Record cached tokens, newly processed tokens, TTFT, and total response time. Also test a substantially changed prompt to catch cache-shift regressions.

Promotion gate: document the measured reuse threshold and workload. Keep the upstream default when the benefit is inconsistent.

## Experiment F: imatrix/IQ dense-model fit

Goal: answer the “27B on 12GB” question with measurements instead of a blanket yes/no.

Select one current 27B-class dense model with reputable imatrix quants. Compare at least two quant levels that fit or nearly fit. Record model size, context, KV type, GPU layers, PP, TG, and a small quality suite covering reasoning, tool calls, and thinking-loop behavior.

Promotion gate: describe a profile as usable only if it fits at the documented context and passes the quality checks. “Loads successfully” is not enough.

## Experiment G: multi-GPU

Blocked until a second GPU is available. When hardware exists, follow upstream `docs/multi-gpu.md` and test layer/row modes first. Tensor mode is experimental, requires non-quantized KV, and is unavailable for many MoE architectures.

Record both GPUs, interconnect, PCIe generation, split mode, `--tensor-split`, `--main-gpu`, NCCL/RCCL availability, PP, TG, and power draw.

## Result format

Save each run through the existing logging path and summarize a completed experiment with:

```markdown
## Experiment: <name>

- llama.cpp commit:
- model / quant:
- context:
- unchanged flags:
- variable under test:
- runs per configuration:

| Config | PP | TG | TTFT | Accept | Peak VRAM | Correctness | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |

Decision:
Production change:
Guide wording:
Dashboard update:
```

## Handoff sequence

1. Run the baseline capture.
2. Complete n-gram and KV-cache experiments first; they directly affect the speculative-decoding section.
3. Run CUDA graph and ubatch sweeps next.
4. Run cache reuse with a realistic coding transcript.
5. Treat imatrix/27B and multi-GPU as separate hardware/model projects.
6. Update `llama-swap.yaml` only after a promotion gate passes.
7. Regenerate `docs/generated-models.js`, update the dashboard metadata, and then revise the public guide with the measured result.

