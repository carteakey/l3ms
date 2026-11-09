# /optimize-model

Convert raw bench data into a production llama-swap.yaml serving config.
Replaces `--fit` with a deterministic static `-ngl` + `--override-tensor` derived from `llama-fit-params`.
Reads `docs/bench-results.md` for existing results before proposing changes.

## Usage

```
/optimize-model <model-key> [--ctx <context-size>] [--fit-target <mib>]
```

**Arguments:**
- `<model-key>` — llama-swap.yaml key (e.g. `qwen3-coder-next`, `gpt-oss-120b`)
- `--ctx <context-size>` — serving context window (default: 65536; use 131072 for long-context models)
- `--fit-target <mib>` — VRAM headroom for VMM pool (default: 512; use 2048 for vision)

## Workflow

### Step 1 — Read existing bench results
Read `docs/bench-results.md` for the model's recorded results.
Read the current `llama-swap.yaml` entry for the model to understand what is already deployed.
Check `bench-models/logs/` for the most recent `.log` files for the model.

### Step 2 — Run llama-fit-params dry run
Get the fit-derived placement for the target serving context without starting the server:

```bash
# Standard context
MODEL=/path/to/model.gguf \
FIT_CTX=<context-size> \
FIT_TARGET=<fit-target> \
CACHE_TYPE_K=q8_0 CACHE_TYPE_V=q8_0 \
./bench-models/run-llama-fit-params.sh

# Or call llama-fit-params directly:
vendor/llama.cpp/build/bin/llama-fit-params \
  -m /path/to/model.gguf \
  -fitt <fit-target> \
  -fitc <context-size> \
  -ctk q8_0 -ctv q8_0
```

The output will be something like: `-c 65536 -ngl 49 -ot "blk\.7\.ffn_down.*=CPU,..."`

### Step 3 — Run bench at fit-derived placement
Verify the fit placement actually improves on the baseline:

```bash
OVERRIDE_TENSOR="<ot-from-fit>" \
./bench-models/bench-llama-cpp-<model-key>.sh
```

Compare pp and tg against the baseline from `docs/bench-results.md`.

### Step 4 — Confirm VRAM safety at serving context
Check that the fit placement holds at the full serving context (not just 512-token bench):

```bash
# Check what VRAM will be consumed by KV at serving context:
# Formula: ctx_size × n_kv_heads × head_dim × 2 × bytes_per_element
# q8_0: ~0.5 GB per 32k ctx on most MoE models; f16: ~1 GB per 32k ctx
```

If using `FIT_TARGET=512` and serving at 64k+ context, verify the VMM pool won't OOM:
- `GGML_CUDA_GRAPH_OPT=0` is required when context depth varies (graph re-capture → VMM pool growth)
- Increase to `FIT_TARGET=2048` for vision or aggressive long-context configs

### Step 5 — Update llama-swap.yaml

For llama.cpp models, replace the `--fit` based entry with the static placement. Follow this pattern:

```yaml
"<model-key>":
  name: "<Display name> (<quant>, optimized, bench-derived static placement)"
  description: "pp=<pp> tg=<tg> t/s @ <ctx> ctx. See docs/bench-results.md."
  env:
    - "LLAMA_SET_ROWS=1"
    - "GGML_CUDA_GRAPH_OPT=0"   # 0 if context varies; 1 only for fixed-context servers
  cmd: |
    taskset -c ${cpu_range} ${llama_server}
    -m <model-path>
    --alias "<alias>"
    # bench-derived: blk 0-<N> on GPU, blk <N+1>+ experts on CPU
    -ngl <ngl>
    --override-tensor "<ot-pattern>"
    --no-mmap --mlock
    -ctk q8_0 -ctv q8_0
    --ctx-size <context-size> --parallel 1
    --threads 10 --threads-batch 12
    --flash-attn on
    --batch-size 2048 --ubatch-size 512
    --temp <temp> --top-p <top_p> --top-k <top_k> --min-p <min_p> --repeat-penalty 1.0
    --host 0.0.0.0 --port ${PORT}
    --jinja --prio 2 --no-warmup
```

Also add a `<model-key>-legacy` variant (set `unlisted: true`) that keeps the old `--fit` entry for regression comparison.

### Step 6 — Key rules from bench experience

**MoE shared expert gotcha:**
Some models (Qwen3.5-122B, some gpt-oss variants) have both routed experts (`_exps`) and shared experts (`_shexp`). Use `(ch|)exps` to match both:
```
.ffn_(up|down|gate)_(ch|)exps=CPU    ✓ matches both _exps and _shexp
.ffn_.*_exps.=CPU                     ✗ only matches routed experts (OOM risk)
```

**GGML_CUDA_GRAPH_OPT:**
- Set to `0` for models with variable context depth (coding sessions, long-context models)
- CUDA graph re-capture at new context depths triggers VMM pool growth → OOM with tight fit margins
- Set to `1` only for fixed-context servers where context doesn't vary mid-session

**Bench tg vs server tg:**
- Bench tg uses 512-token context (tiny KV cache)
- Server tg at 64k context with q8_0 KV ≈ 2 GB consumed from VRAM
- Always use `FIT_CTX=<serving-ctx>` (not bench context) to get a placement valid under real load

**q8_0 KV is always recommended:**
- Halves KV VRAM vs f16 — frees room for extra GPU layers
- Effectively lossless for most purposes
- At 64k context: ~2 GB (q8_0) vs ~4 GB (f16) — significant difference on 12 GB cards

### Step 7 — Restart and verify

```bash
systemctl --user restart llama-swap.service
sleep 5
curl -s http://localhost:8080/v1/models | jq '.data[].id'

# Quick functional test
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model-key>","messages":[{"role":"user","content":"say hi"}]}'
```

### Step 8 — Update docs and CHANGELOG

In `docs/bench-results.md`:
- Add or update the "Recommended serving configuration" table
- Add the static `-ot` placement and context to the results table

In `CHANGELOG.md` under `## [Unreleased]` → `### Changed`:
```markdown
- `<model-key>`: replaced --fit with bench-derived static -ot (pp=<pp>, tg=<tg> t/s)
```
