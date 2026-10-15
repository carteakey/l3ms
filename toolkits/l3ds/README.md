# L3DS — Local Low-latency Decision System

A keyboard-first, script-first terminal toolkit and benchmark harness for local **Decision Models** ("System 1" models). 
Companion to **[`l3ms`](https://github.com/carteakey/l3ms)** (Language Models) and **[`l2m2`](https://github.com/carteakey/l2m2)** (Latent Media Models).

Unlike autoregressive LLMs (System 2) that generate token-by-token JSON, decision models map a state and a closed option set into typed outputs (`choice`, `score`, `noul`). The table lists declared profiles and their current readiness; manifest latency values are targets, not general performance guarantees. An offline Nagi CPU execution has been recorded, but CUDA performance and model quality remain unverified.

---

## Declared Decision Profiles and Readiness

For the current llama.cpp-backed local candidates, prefer the separately named
`vllm-d3-flash` and `cloudflare-clef-flash` profiles. They send native typed
questions through llama-swap v262 or newer using only Python's standard
library; they do not require a Nagi/Laya environment or silently fall back to
those models. See [native decision routing](../../docs/decision-models-cli.md) for
recipes, environment variables, live-use restrictions and unfinished gates.

| Profile | Engine | Architecture / tier | Declared deployment | Manifest expected latency | Readiness |
|---|---|---|---|---:|---|
| `vllm-d3-flash` | llama.cpp / llama-swap | Qwen3.5-9B, Q8_0 | Local router-owned CUDA | ~110ms historical warm text | Real isolated CPU text/image sanity checks pass; GPU/quality acceptance pending. |
| `cloudflare-clef-flash` | llama.cpp / llama-swap | Qwen3.5-9B, Q4_K_M | Local router-owned CUDA | ~65ms historical warm routed text | Real isolated CPU text sanity check passes; image/GPU/quality acceptance pending. |
| `nagi-smol` | Nagi | ModernBERT-large / M2', Smol (~480M) | Local (CUDA; CPU fallback) | ~17ms | CUDA target is unverified; CPU evidence is for the separate explicit CPU profile. |
| `nagi-smol-cpu` | Nagi | ModernBERT-large / M2', Smol (~480M) | Local (CPU only) | ~1,600ms | Offline native execution used the installed environment and cached weights; this is not a quality or clean-install claim. |
| `nagi-big` | Nagi | Qwen3.5-4B + Adapter, Big (4B) | Local (CUDA, 8.5GB) | ~70ms | Declared profile; native execution, latency and quality are unverified. |
| `laya-en` | Laya | ModernBERT-large, Smol (~350M) | Local (CUDA; CPU fallback) | ~13ms | Declared profile; native execution, latency and quality are unverified. |
| `semif-openjev` | SemIf | Causal Logit Extraction, Mid (~3B) | Local (CUDA, 7.0GB) | ~51ms | No native forward pass: standalone execution refuses unless `--simulate` is explicit; simulated output is demo-only, not native acceptance. |
| `jev-api` | Jev | Proprietary hosted | Remote API | ~250–500ms (WAN) | No paid live-provider acceptance is recorded. |

The Nagi CPU profile is explicitly named `nagi-smol-cpu`; current direct-CPU
acceptance details are in [Direct CPU decision acceptance](../../docs/migration/direct-decision-acceptance.md).
The earlier [Nagi preparation and acceptance record](../../docs/migration/nagi-acceptance.md)
documents pinned environment inputs and hash-verified model acquisition. The
latest direct CPU run reused the installed environment and offline cache; it
did not repeat a clean-machine installation. Neither record verifies CUDA
latency, CUDA execution quality, or decision accuracy.

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
API. The pins target CPython 3.12.13 on Linux x86_64 with CUDA 12.4 wheels;
other platforms need a separately validated lock. The Nagi artifact revision
and hashes were recorded during the separate acquisition documented in the
acceptance record above. The CPU evidence used that already-installed
environment and cached weights in offline mode; it does not show a clean
machine can install and run without separate environment/model preparation.
The declared CUDA target and model quality remain unverified.

To repeat a cache-only smoke test, use the Hub's offline mode; this prevents
the test from silently initiating model acquisition:

```bash
HF_HUB_OFFLINE=1 ./target/release/l3ds --profile nagi-smol-cpu \
  --state '{"alert":"test"}' \
  --question '{"route":{"type":"choice","criteria":{"infra":"Infrastructure","billing":"Billing"}}}'
```

For Phase 3 evidence, require a successful native result labeled
`"engine": "nagi-native"` (never `--simulate`), then run the bounded benchmark
and retain its P50/P95 and option-flip results:

```bash
./target/release/l3ds --bench --profile nagi-smol-cpu \
  --suite routing-gate --iterations 10
```

Neither the environment setup nor a successful CLI build/list substitutes for
those live checks.

### Prepare the hosted TypeSafe runtime

Hosted profiles use the official TypeSafe System One API, not an OpenRouter
protocol adapter. Both profile IDs are preserved (`jev-api`, `systemone`), with
the provider's canonical `jev-latest` model alias. This alias can move; choose a
versioned model in your manifest if your application needs a fixed model.

The monorepo foreground path uses a separate `.venv-hosted/bin/python`; it never
changes the Nagi `.venv`. With Python 3.12 and uv installed, create it only if
neither a directory nor symlink already exists:

```sh
test ! -e .venv-hosted && test ! -L .venv-hosted &&
  uv venv --python 3.12 .venv-hosted &&
  uv pip sync --strict --python .venv-hosted/bin/python requirements/hosted.lock.txt
```

The lock records the observed SDK 0.7.1 runtime dependency closure; a fresh
installation still needs verification. These commands may download packages,
not models; nothing installs automatically. `L3MS_HOSTED_DECISIONS_PYTHON` can
instead select an explicitly prepared absolute Python executable for the shared
L3MS path. Provider credentials stay in `TYPESAFE_API_KEY`; `TYPESAFE_BASE_URL`
optionally configures the endpoint. The legacy `OPENROUTER_API_KEY` fallback is
not an OpenRouter adapter.

Root L3MS sends `--single-attempt`: SDK retries are disabled and SDK payload
logging is suppressed. Child cancellation or timeout cannot prove the remote
request stopped; unresolved calls are never replayed. Standalone runner defaults
remain unchanged; use `--single-attempt` explicitly for a one-shot hosted call.
No paid provider acceptance has been run. Official contracts:
[retry policy](https://docs.typesafe.ai/sdk/python/api/retries),
[models](https://docs.typesafe.ai/models).

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
bridge currently has no native forward pass: standalone SemIf requests refuse
without `--simulate`, and a simulated answer is not native acceptance. Hosted
profiles require a configured API credential for real requests.

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
2. **Local vs Remote WAN:** The manifest's **~17ms (Smol)** and **~70ms (Big)** figures are CUDA design targets, not accepted measurements. The explicit Nagi CPU profile has an offline native run around **1.6 seconds per forward pass**; it is not a 100ms-tick CPU target. See the acceptance records above.
3. **Accuracy vs Pure Speed:** The smaller Laya/Nagi-Smol and larger Nagi-Big profiles explore this tradeoff. The recorded CPU run verifies execution and typed output only, not semantic accuracy; CUDA quality and deadline performance still need real measurements. `--simulate` output is not evidence.

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
