# L3DS — Local Low-latency Decision System

A keyboard-first, script-first terminal toolkit and benchmark harness for local **Decision Models** ("System 1" models). 
Companion to **[`l3ms`](https://github.com/carteakey/l3ms)** (Language Models) and **[`l2m2`](https://github.com/carteakey/l2m2)** (Latent Media Models).

Unlike autoregressive LLMs (System 2) that generate token-by-token JSON, decision models map a state and a closed option set into typed outputs (`choice`, `score`, `noul`). The latency figures below are manifest expectations, not verified local benchmark results.

---

## Supported Decision Engines

| Profile | Engine | Architecture | Tier | Deployment | Expected Latency |
|---|---|---|---|---|---|
| `nagi-smol` | Nagi | ModernBERT-large / M2' | Smol (~480M) | Local (CPU/CUDA) | ~17ms |
| `nagi-big` | Nagi | Qwen3.5-4B + Adapter | Big (4B) | Local (CUDA 8.5GB) | ~70ms |
| `laya-en` | Laya | ModernBERT-large | Smol (~350M) | Local (CPU/CUDA) | ~13ms |
| `semif-openjev` | SemIf | Causal Logit Extraction | Mid (~3B) | Local (CUDA 7.0GB) | ~51ms |
| `jev-api` | Jev | Proprietary Hosted | Cloud API | Remote API | ~250–500ms (WAN) |

---

## Quickstart

From the L3MS monorepo root, enter the L3DS child before running these
commands:

```bash
cd toolkits/l3ds
```

Cargo sets the CLI's default `--root` to this package directory
(`toolkits/l3ds`) through `CARGO_MANIFEST_DIR`. The manifest, Python runner,
benchmark suites, and `.venv` are resolved beneath that directory even when the
binary is launched from another working directory. To select another
compatible checkout, run:

```bash
./target/release/l3ds --root /path/to/l3ds --list
```

### Build the CLI
```bash
cargo build --release
```

### Prepare the Nagi-Smol Python environment

The new pinned environment installer is deliberately a no-op by default. From this
directory, inspect its plan first:

```bash
./maintenance/prepare-nagi-smol-env.sh --dry-run
```

The existing `maintenance/setup-env.sh` retains its original install behavior
for standalone compatibility; the pinned Nagi-only path is the separate script
above and does not run as part of the migration checks.

Only after separately approving the Python/CUDA package download, run
`./maintenance/prepare-nagi-smol-env.sh --install`. It creates a fresh child `.venv` from
[`requirements/nagi-smol.lock.txt`](requirements/nagi-smol.lock.txt), pinned to
the Nagi source commit and the dependency versions observed in the original
L3DS environment. The lock is Nagi-only: it excludes SemIf, whose declared
exact Torch and Hugging Face pins conflict with the observed Nagi stack. An
existing `.venv` is never overwritten, and staging is discarded if installation
fails.

This setup may download multi-gigabyte CUDA wheels. It installs Python packages
only; it does not fetch model weights, run inference, or contact a hosted model
API. The pins are reproducible environment inputs, not evidence that Nagi
inference or the manifest's expected latency has been validated. They target
CPython 3.12.13 on Linux x86_64 with CUDA 12.4 wheels; other platforms need a
separately validated lock. The shared Hugging Face cache had no model snapshots
at the latest migration audit, so a real smoke run still needs a separately
approved Nagi-Smol model acquisition. No model weights have been acquired by
this setup work, and the model artifact revision/checksum remain unrecorded;
capture them during that authorized acquisition before calling inference
reproducible.

After the weights are available, use the Hub's offline mode for a cache-only
smoke test; this prevents the test from silently initiating model acquisition:

```bash
HF_HUB_OFFLINE=1 ./target/release/l3ds --profile nagi-smol \
  --state '{"alert":"test"}' \
  --question '{"route":{"type":"choice","criteria":{"infra":"Infrastructure","billing":"Billing"}}}'
```

For Phase 3 evidence, require a successful native result labeled
`"engine": "nagi-native"` (never `--simulate`), then run the bounded benchmark
and retain its P50/P95 and option-flip results:

```bash
./target/release/l3ds --bench --profile nagi-smol \
  --suite routing-gate --iterations 10
```

Neither the environment setup nor a successful CLI build/list substitutes for
those live checks.

### Inspect Profiles
```bash
./target/release/l3ds --list
```

### Run a Single Decision
```bash
# Default routing check
./target/release/l3ds --profile nagi-smol

# Custom state and question
./target/release/l3ds --profile nagi-big \
  --state '{"alert": "OOMKilled on worker node gpu-04"}' \
  --question '{"squad": {"type": "choice", "instructions": "Assign team", "criteria": {"infra": "Infrastructure", "billing": "Billing"}}}'
```

Normal profile and benchmark commands fail if the native model is unavailable;
they never substitute fabricated decisions. Use `--simulate` only for an
explicit synthetic demo, not performance or migration acceptance. The SemIf
bridge currently has no native forward pass and therefore requires that demo
flag until its adapter is implemented. Hosted profiles require a configured API
credential for real requests.

The native Python runner also accepts `--input-stdin` for coordinator or script
callers. Feed a UTF-8 JSON object with exactly `state` and `questions` (at most
1 MiB) from a private file or pipe; these payloads then do not appear in the
runner's process arguments. Invalid input is rejected before model loading.

```bash
.venv/bin/python decision-models/runner.py \
  --engine nagi --model nagisanzeninz/nagi-smol-v0 --device cuda \
  --input-stdin < request.json
```

This option cannot be combined with `--state`, `--question`, or `--bench-suite`.
It does not install dependencies/models or enable coordinated GPU scheduling.

### Run Comparative Latency & Option-Flip Benchmark
```bash
./target/release/l3ds --bench --suite routing-gate --iterations 10
```

### Run the Real-Time Game Arena
Test competitors under real-time clock deadlines (e.g. 100ms ticks):
```bash
./target/release/l3ds --arena --tick-ms 100
```

---

## Real-Time Game Target

Under a real-time game clock (such as Snake, Drone Swarm, or Bomber Arena):
1. **Clock Deadlines:** The simulation advances every 100ms.
2. **Local vs Remote WAN:** The manifest targets **~17ms (Smol)** and **~70ms (Big)** on the intended host, versus a longer expected remote roundtrip. These are design targets until a native, reproducible run is recorded.
3. **Accuracy vs Pure Speed:** The smaller Laya/Nagi-Smol and larger Nagi-Big profiles explore this tradeoff. Actual quality and deadline performance require real model measurements; `--simulate` output is not evidence.

---

## Repository Structure

```text
l3ds/
├── Cargo.toml                  # Fast Rust CLI manifest
├── decision-runtimes.json      # Declarative model & engine registry
├── src/
│   ├── main.rs                 # CLI dispatch
│   ├── manifest.rs             # Profile definitions and validation
│   ├── bench.rs                # Latency & option-flip benchmark harness
│   └── arena.rs                # Real-time game simulation harness
├── decision-models/
│   ├── runner.py               # Unified Python decision engine bridge
│   └── arena_sim.py            # Real-time Bomber Arena game loop
├── suites/
│   ├── routing-gate.json       # Agentic triage & classification suite
│   └── latency-micro.json      # Micro-benchmark for forward pass speed
├── AGENTS.md                   # Operational guidelines for AI agents
└── README.md
```

## License

MIT
