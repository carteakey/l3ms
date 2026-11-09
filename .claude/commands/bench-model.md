# /bench-model

Run benchmarks for a model following the experiment sequence in `docs/bench-runbook.md`.
Always run the system preflight first. Stop when results are good enough.

## Usage

```
/bench-model <model-key> [--stage <stage>] [--ik] [--fast]
```

**Arguments:**
- `<model-key>` — llama-swap.yaml key / bench script suffix (e.g. `qwen3-coder-next`, `qwen3-6-35b-a3b`)
- `--stage <stage>` — run only one stage: `baseline | strategies | fit | kv | ik | ik-strategies`
- `--ik` — prefer ik_llama.cpp scripts if available
- `--fast` — single repetition (`REPETITIONS=1`); use for quick validation, not for recorded results

## Prerequisites

1. Check that the bench scripts exist:
   ```bash
   ls bench-models/bench-llama-cpp-<model-key>*.sh
   ```
2. Check that the model file is present (look for the path in the bench script `MODEL=` line)
3. Run `/preflight` or manually verify CPU governor, EPP, VRAM state

**Result logging is automatic** — every run writes a JSONL record to
`bench-models/results/<model-key>.jsonl` via `log-result.sh`.
Set `STRATEGY=<label>` before running to tag the record (e.g. `STRATEGY=partial-cpu`).
Set `NOTES=<text>` to add a free-text annotation.

## Experiment sequence (from bench-runbook.md §4)

Run in order. Stop when tg is satisfactory.

### Stage 4a — Baseline (N_CPU_MOE)
```bash
./bench-models/bench-llama-cpp-<model-key>.sh
```
- Sets a floor for pp and tg
- Try adjusting `N_CPU_MOE` up/down to find VRAM sweet spot:
  ```bash
  N_CPU_MOE=30 ./bench-models/bench-llama-cpp-<model-key>.sh  # more on GPU
  N_CPU_MOE=48 ./bench-models/bench-llama-cpp-<model-key>.sh  # all on CPU
  ```

### Stage 4b — Manual -ot strategy sweep
```bash
STRATEGY=all-cpu-moe  ./bench-models/bench-llama-cpp-<model-key>-strategies.sh
STRATEGY=partial-cpu  ./bench-models/bench-llama-cpp-<model-key>-strategies.sh
STRATEGY=up-down-cpu  ./bench-models/bench-llama-cpp-<model-key>-strategies.sh
```
- Skip `up-cpu` if on 12 GB VRAM — will OOM (512 experts × projections)
- `up-down-cpu` also OOMs on 12 GB for dense-expert models; try `partial-cpu` first

### Stage 4c — llama-fit-params auto-placement
```bash
./bench-models/bench-llama-cpp-<model-key>-fit.sh
```
Useful knobs:
```bash
FIT_TARGET=2048 ./bench-models/bench-llama-cpp-<model-key>-fit.sh   # more headroom
FIT_CTX=65536   ./bench-models/bench-llama-cpp-<model-key>-fit.sh   # 64k context floor
FIT_CTX=131072  ./bench-models/bench-llama-cpp-<model-key>-fit.sh   # 128k context floor
```

Dry-run only (print fit params without running bench):
```bash
MODEL=/path/to/model.gguf ./bench-models/run-llama-fit-params.sh
```

### Stage 4d — Poll level (optional, usually flat)
```bash
POLL=0   ./bench-models/bench-llama-cpp-<model-key>.sh
POLL=50  ./bench-models/bench-llama-cpp-<model-key>.sh
POLL=100 ./bench-models/bench-llama-cpp-<model-key>.sh
```
- On hybrid CPU+GPU inference, effect is typically within noise. Run once to confirm, then skip.

### Stage 4e — KV cache quantization
```bash
CACHE_TYPE_K=q8_0 CACHE_TYPE_V=q8_0 ./bench-models/bench-llama-cpp-<model-key>.sh
CACHE_TYPE_K=q4_0 CACHE_TYPE_V=q4_0 ./bench-models/bench-llama-cpp-<model-key>.sh
```
- At 512-token bench context, impact on t/s is near zero
- Matters at 8k+ context (KV cache grows large, VRAM pressure increases)
- `q8_0` is recommended: lossless for most purposes, saves ~50% KV VRAM vs f16

### Stage 4f — ik_llama.cpp (fused-moe)
```bash
./bench-models/bench-ik-llama-cpp-<model-key>.sh
```
- `FUSED_MOE=1` by default — major tg win vs stock llama.cpp (+10-30%)
- Compare directly against stage 4a llama.cpp baseline
- To test without fused-moe: `FUSED_MOE=0 ./bench-models/bench-ik-llama-cpp-<model-key>.sh`

### Stage 4g — ik_llama merge-qkv (optional)
```bash
STRATEGY=fused-mqkv ./bench-models/bench-ik-llama-cpp-<model-key>-strategies.sh
```

### Stage 4h — ik_llama merge-up-gate (only if tg-critical + ≥75 GB RAM free)
```bash
STRATEGY=fused-muge ./bench-models/bench-ik-llama-cpp-<model-key>-strategies.sh
```
- ⚠️ Adds ~27 GB RAM (repacks up+gate weights)
- ⚠️ pp regresses significantly (-200-300 t/s)
- Only use when tg is the sole metric and RAM headroom exists

### Stage 4i - Strata runtime switch tuning sweep (RTX 4070 Platinum tier)
```bash
systemctl --user stop llama-swap.service
ARMS="base hcq8 aux_ecores ple_mmap base" ./bench-models/bench-strata-tuning-ab.sh
systemctl --user start llama-swap.service
```
- Consult `vendor/strata-lean/docs/DETAILS.md` and related topic guides first for switch definitions and defaults; fallback to `git -C vendor/strata-lean grep "STRATA_"` for undocumented switches
- Overlays candidate flags and `STRATA_*` environment variables directly on top of the served production config (`strata-ud-iq2_xs-ple_q8.json`)
- Evaluates steady-state decode (`tg_steady_mean`, `TG_WARMUP=6`, `TG_PROBES=4`), multi-length prefill (2k, 8k, 32k), and 4k needle retrieval
- Emits engine diagnostics (`expert_slots`, cache hit rate, PCIe fraction, `io_mb_per_token`) to `logs/results/strata-tuning-diag.jsonl`
- Use detached dark execution (`bench-models/run-strata-tuning-dark.sh`) when benchmarking the very engine serving the agent

## After benchmarking

1. Results are **auto-logged** to `bench-models/results/<model-key>.jsonl`
   ```bash
   # Quick summary of all recorded runs
   cat bench-models/results/<model-key>.jsonl | \
     python3 -c "import sys,json; [print(f'{r[\"ts\"][:10]}  {r.get(\"strategy\",\"?\"):20}  pp={r.get(\"pp_ts\",\"?\")}  tg={r.get(\"tg_ts\",\"?\")}') for r in map(json.loads, sys.stdin)]"
   ```
2. Run `/add-bench-result <model-key>` to update the human-readable results in `docs/bench-results.md`
3. Run `/optimize-model <model-key>` to derive the production llama-swap.yaml config
