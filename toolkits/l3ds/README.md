# L3DS — Local Low-latency Decision System

A keyboard-first, script-first terminal toolkit and benchmark harness for local **Decision Models** ("System 1" models). 
Companion to **[`l3ms`](https://github.com/carteakey/l3ms)** (Language Models) and **[`l2m2`](https://github.com/carteakey/l2m2)** (Latent Media Models).

Unlike autoregressive LLMs (System 2) that generate token-by-token JSON, decision models map a state and a closed option set into typed outputs (`choice`, `score`, `noul`) with calibrated probabilities in **a single forward pass** (13ms – 70ms).

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

### Build the CLI
```bash
cargo build --release
```

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

## Why Local Decision Models Win in Real-Time Games

Under a real-time game clock (such as Snake, Drone Swarm, or Bomber Arena):
1. **Clock Deadlines:** The simulation advances every 100ms.
2. **Local vs Remote WAN:** Local inference on an RTX 4070 takes **~17ms (Smol)** or **~68ms (Big)**, consistently beating the deadline. Remote APIs (Jev) experience WAN roundtrips (150ms – 400ms+), resulting in missed ticks, frozen units, and eliminations.
3. **Accuracy vs Pure Speed:** While smaller encoders like Laya are ultra-fast (~13ms), Nagi-Big (4B) provides the spatial reasoning capacity needed to evade dynamic traps while still easily clearing the 100ms threshold.

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
