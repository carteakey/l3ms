# Local LLM Inference Optimization Guide — Revised Structure Plan

Target file: `docs/local-llm-optimization.md`
Audience: technically literate homelab users; generalized, not tied to any single machine.
Style: deep dive reference — comprehensive, opinionated, living document.

---

## Revised Document Structure

---

### 0. Preamble — What This Guide Is

Short framing paragraph:
- This is an opinionated, continuously-updated reference for squeezing maximum performance
  from locally-hosted LLMs on consumer hardware.
- We start from first principles (what is inference?) and drill all the way down to
  specific flags, tradeoffs, and failure modes.
- Everything except GPU/VRAM sizing applies regardless of whether you have 8 GB or 80 GB.
- This doc assumes CUDA. Notes for Vulkan and CPU-only paths are called out where they differ.

---

### 1. Glossary

Before the deep dive — a reference to come back to.

| Term | Definition |
| --- | --- |
| **GGUF** | File format for quantized LLM weights, used by llama.cpp. Successor to GGML. |
| **Quantization** | Reducing weight precision (e.g. FP16 → INT4) to shrink model size and speed up inference. Q4, Q5, Q8, etc. Higher = more accurate, larger. |
| **PP (Prompt Processing)** | Tokens/second during the prefill phase — how fast the model reads your input. |
| **TG (Token Generation)** | Tokens/second during autoregressive decode — how fast you see output stream. **This is what you feel.** |
| **KV Cache** | Stores attention key/value tensors for prior tokens. Grows with context length. Lives in VRAM. |
| **Context Window** | Maximum tokens (input + output) the model can "remember" in one session. |
| **Dense Model** | Standard transformer: all parameters active for every token. Must fit entirely in VRAM for best performance. |
| **MoE (Mixture of Experts)** | Architecture where only a subset of parameter "experts" is active per token. Allows very large models (80B–120B+) to run with active compute equivalent to a much smaller model. |
| **VRAM** | Video RAM — GPU memory. Primary inference resource; lowest latency. |
| **Perplexity** | Measure of how "surprised" a model is by text; proxy for quality. Lower = better. Used to compare quant levels. |
| **llama-bench** | CLI tool to measure pp and tg synthetic throughput. |
| **llama-fit-params** | CLI tool that probes VRAM and computes optimal layer placement without running a server. |
| **`-ngl` / n-gpu-layers** | Number of transformer blocks loaded onto GPU. |
| **`-ot` / override-tensor** | Regex-based per-tensor placement override (CPU vs GPU). |
| **Fit** | `--fit on` flag: auto-placement using llama-fit-params at startup. |

---

### 2. The Inference Landscape — From Cloud to Local

Start from the universe and zoom in.

#### 2.1 Why run locally at all?
- Privacy: no data leaves your machine
- Cost: amortized over time; no per-token billing
- Control: no rate limits, no model deprecation, no "we're changing pricing"
- Availability: works offline; unaffected by API outages
- Experimentation: run any model, any quant, any parameters

#### 2.2 Cloud vs Local — Honest tradeoffs
Table comparing:
- Hosted API (OpenAI, Anthropic, Gemini): zero setup, best models, per-token cost, privacy concerns
- Self-hosted cloud (RunPod, Lambda, Vast.ai): your model, their GPU, still pays per hour
- Local: zero marginal cost, full control, constrained by hardware

Guidance on when each makes sense. Not black/white.

#### 2.3 Local Inference Backends — Choosing Your Stack

| Tool | Best for | Tradeoffs |
| --- | --- | --- |
| **llama.cpp** | Performance tuning, full parameter control, any hardware | Complex flags; build from source recommended |
| **Ollama** | Zero-config, quick deployment, model management | Fixed defaults; limited tuning surface |
| **LM Studio** | Desktop GUI, Windows/Mac users | Less control than llama.cpp; good for evaluation |
| **vLLM** | Multi-user server, batched inference, production | Needs full VRAM; not suited for consumer hybrid setups |
| **exllamav2** | High-speed CUDA inference for dense models | CUDA-only; less community support for MoE |
| **mlx** | Apple Silicon / Metal | macOS only; excellent for unified memory setups |

**This guide focuses on llama.cpp.** Most concepts transfer.

#### 2.4 Backend / Hardware Backends within llama.cpp

llama.cpp supports multiple compute backends. Build flag determines which is active.

| Backend | Build flag | Best for |
| --- | --- | --- |
| **CUDA** | `GGML_CUDA=ON` | NVIDIA GPUs; best performance |
| **Vulkan** | `GGML_VULKAN=ON` | AMD/Intel GPUs; NVIDIA fallback; cross-platform |
| **Metal** | (macOS only, auto) | Apple Silicon; uses unified memory |
| **CPU-only** | (default, no GPU flag) | Baseline; usable for small models or testing |
| **RPC** | `GGML_RPC=ON` | Distribute across machines (experimental) |

> **This document assumes CUDA.** Notes for Vulkan or CPU-only are marked where behavior differs.

---

### 3. Hardware

Not a buying guide — but understanding why each component matters is essential for tuning decisions.

#### 3.1 The Memory Hierarchy (Most Important Mental Model)

Token generation speed is limited by how fast you can read model weights.
Hierarchy from fastest to slowest:

```
VRAM (GPU memory)
  >> Unified memory (Apple Silicon, AMD Strix Halo)
    >> System RAM (DDR5 > DDR4; bandwidth matters more than capacity)
      >>>> SSD (NVMe > SATA; avoid — generates 1–5 t/s at best)
        >>>>>>> HDD (effectively unusable)
```

- **Dense models**: must fit entirely in VRAM for full speed. Any spill to RAM = major slowdown.
- **MoE models**: designed for partial offload. Active experts on GPU; inactive expert weights in RAM.
  Token generation speed ∝ how fast you can stream those weights from RAM.
  This means memory _bandwidth_ (GB/s) matters more than capacity for MoE t/s.
  A fast RAM kit at rated speed with XMP/EXPO enabled can give 3× the bandwidth of the same kit running at BIOS default speed.

#### 3.2 GPU / VRAM
- VRAM is the primary bottleneck
- More VRAM = more layers on GPU = faster inference
- Practical capability tiers (rough, quant-dependent):
  | VRAM | Dense models | MoE models |
  | --- | --- | --- |
  | 8 GB | 7B–13B Q4 | 30B–70B with heavy RAM offload |
  | 12 GB | 13B–20B Q4; 7B FP16 | 70B–120B with RAM offload |
  | 24 GB | 34B Q4; 13B FP16 | 120B+ with moderate RAM offload |
  | 48 GB+ | 70B FP16 | Most models fully on GPU |
  | 80 GB+ | 120B+ FP16 | No offload needed |

- iGPU tip (desktop, NVIDIA): route display output through motherboard HDMI (iGPU) instead of dedicated GPU → frees 500–1000 MB VRAM for inference

#### 3.3 System Memory (RAM)
- For dense models: RAM is not used for inference; capacity requirement is just for OS + model loading
- For MoE models: expert weights live in RAM during inference
  - Bandwidth matters more than capacity for t/s
  - Enable XMP/EXPO in BIOS — on many boards "Auto" runs at JEDEC base (far below rated speed)
  - Verify with: `sudo dmidecode -t memory | grep "Configured Memory Speed"`
  - Minimum ~64 GB for 60–120B models with CPU offloading

#### 3.4 CPU
- For dense models (all on GPU): CPU is mostly idle during inference; minimal impact
- For MoE models: CPU executes expert computations; P-core count and clock speed matter
- **Hybrid CPU architectures** (Intel 12th gen+): P-cores (fast) + E-cores (slow)
  - E-cores drag down expert compute significantly
  - Always pin inference threads to P-cores: `taskset -c <p-core-range>`
  - More threads ≠ better; optimal is P-core count − 1 or 2

#### 3.5 What is "Good Enough" Throughput?

| TG speed | User experience |
| --- | --- |
| < 5 t/s | Painful; barely usable for short responses |
| 5–10 t/s | Functional; near human reading speed (~7 t/s) |
| 10–20 t/s | Comfortable for interactive chat |
| 20–40 t/s | Fast; agentic loops and long outputs feel snappy |
| 40+ t/s | Near-instant for most tasks |

**For coding agents**: t/s is the dominant metric. Long outputs (500–2000 tokens) at 10 t/s is 50–200 seconds. Every t/s gained compounds across an entire session.

---

### 4. OS Choice

Not a forced recommendation — just honest tradeoffs.

#### 4.1 Linux
- **~20% higher throughput** than Windows in practice (CUDA scheduler + lower driver overhead)
- Best support for system-level tuning (CPU governor, NUMA, huge pages, cgroups)
- Required for some CUDA toolkit versions
- CachyOS/Arch: real-time kernel available; best latency tuning surface
- Ubuntu/Debian: best CUDA package support; easier setup

#### 4.2 Windows (Native)
- Reasonable performance; CUDA works fine
- Slightly higher overhead from Windows scheduler and CUDA runtime
- Power profile management: use "High Performance" or "Ultimate Performance" power plan
- WSL2: gets close to native for CUDA workloads; some VRAM overhead; generally acceptable

#### 4.3 macOS
- Metal backend; excellent for Apple Silicon unified memory models
- Unified memory = both CPU and GPU share the same pool; model can exceed GPU "VRAM" limit
- M-series chips are strong for models up to their unified memory limit
- No CUDA; CUDA-specific guidance in this doc does not apply

#### 4.4 OS-Level Tuning (Linux)

**CPU Governor & Power Profile** — most impactful tuning after hardware:

```bash
# Check governor (must be "performance")
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Check EPP — must be "performance", not just governor
cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference

# Check actual CPU frequency
grep "cpu MHz" /proc/cpuinfo | sort -rn | head -4
```

The `power-profiles-daemon` trap (KDE/GNOME default): can silently set a non-performance HWP state on some boots even when sysfs shows `governor=performance` and `EPP=performance`. All standard diagnostics look clean, but tg drops 20–30%.

Fix — replace with `tuned-ppd`:
```bash
# Arch/CachyOS
sudo pacman -S tuned-ppd
sudo systemctl enable --now tuned
sudo tuned-adm profile throughput-performance
```

**Transparent Huge Pages:**
```bash
cat /sys/kernel/mm/transparent_hugepage/enabled
# Recommended: [always]
```

**Headless mode** (maximum VRAM recovery):
```bash
# Enter headless (stops compositor + display manager; frees 200–400 MB RAM + compositor VRAM)
sudo systemctl isolate multi-user.target
sudo sync && sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
free -h

# Restore GUI
sudo systemctl isolate graphical.target
```

Use `zellij` in a TTY for split panes without a display server.

---

### 5. Why llama.cpp / Building It

#### 5.1 Why not Ollama?

Ollama uses llama.cpp under the hood but wraps it with fixed defaults.
The flags in §6–§11 below are all unavailable or unexposed in Ollama.
If you care about squeezing performance, you need the raw tool.

#### 5.2 Building from Source (CUDA)

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=ON \
  -DGGML_NATIVE=ON \
  -DGGML_LTO=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_F16=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89   # 89=RTX40, 86=RTX30, 75=RTX20, 61=GTX10
cmake --build . --config Release \
  --target llama-server llama-bench llama-fit-params --parallel
```

> **Vulkan (AMD/Intel)**: replace `-DGGML_CUDA=ON` with `-DGGML_VULKAN=ON`. Most flags below marked [CUDA] do not apply.

Build notes:
- `GGML_NATIVE=ON`: CPU-specific tuning; do not use for distributed binaries
- `GGML_LTO=ON`: link-time optimization; slower build, faster binary
- `GGML_CUDA_GRAPHS=ON`: CUDA graph capture; see env var note in §10
- **Rebuild regularly** — MoE kernel performance has improved significantly across versions

#### 5.3 Key Binaries

| Binary | Purpose |
| --- | --- |
| `llama-server` | OpenAI-compatible inference server |
| `llama-bench` | Synthetic pp/tg benchmarking |
| `llama-fit-params` | Dry-run VRAM probe → outputs optimal `-ngl` and `-ot` flags |
| `llama-cli` | Interactive shell for quick tests |
| `llama-sweep-bench` | Sweep over params for batch optimization |

---

### 6. Model Selection and Quantization

#### 6.1 Dense vs MoE — Choosing a Model Type

| Type | Example | Tuning approach |
| --- | --- | --- |
| Dense | Llama 3 70B, Gemma 4 | Must fit in VRAM; `-ngl 99`; RAM not a factor |
| MoE | Qwen3-Coder-Next, gpt-oss-120b | Partial offload; RAM bandwidth = tg bottleneck |

#### 6.2 Quantization Levels

| Quant | Size vs FP16 | Quality | Use case |
| --- | --- | --- | --- |
| Q2_K | ~25% | Significant degradation | Absolute smallest footprint |
| Q4_K_M | ~35% | Good; small loss | Best size/quality balance for consumer |
| Q5_K_M / Q5_K_XL | ~40% | Very close to FP16 | Recommended when VRAM allows |
| Q6_K | ~50% | Near-lossless | High VRAM or RAM serving |
| Q8_0 | ~65% | Effectively lossless | If you have the space |
| F16 | 100% | Reference | Max quality; largest size |
| MXFP4 (native) | ~35% | Model-native; not post-quantized | gpt-oss series; better quality than Q4 at same size |

**UD (Unsloth Dynamic)** quants: layer-importance-aware quantization. Higher bits on sensitive layers, lower on robust layers. Better perplexity than uniform quants at same size.

---

### 7. Layer Placement — The Core Optimization for MoE

For dense models fully on GPU, skip to §8.
For MoE/hybrid setups, this is the most impactful tuning surface.

#### 7.1 `--n-gpu-layers` (`-ngl`)
#### 7.2 `--n-cpu-moe` — Integer coarse control
  - ⚠️ RAM ceiling warning for 60 GB+ models
#### 7.3 `--override-tensor` (`-ot`) — Fine-grained regex control
  - Pattern syntax
  - ⚠️ Shared expert gotcha (`_shexp` vs `_exps`; `(ch|)exps` pattern)
  - Partial-CPU pattern examples
#### 7.4 `--fit on` — Recommended "just works" auto-placement
  - `--fit-ctx`: minimum context to guarantee (KV accounted for)
  - `--fit-target`: VRAM headroom; use ≥512 MiB to survive VMM pool growth
  - `llama-fit-params` dry run
#### 7.5 Static vs Dynamic placement tradeoffs
  - `--fit` = startup delay, best for experimentation
  - Hardcoded `-ngl` + `-ot` = zero delay, deterministic, best for production

---

### 8. Context and KV Cache

#### 8.1 `--ctx-size` — choosing your context
  - Direct VRAM cost table: at various ctx sizes and quant levels
#### 8.2 KV Cache quantization (`-ctk`, `-ctv`)
  - `q8_0`: halves KV VRAM vs f16; effectively lossless; recommended default
  - `q4_0`: saves more; quality tradeoff at long context
  - Effect at bench context vs real serving context
  - How q8_0 unlocks extra GPU layers at large context
#### 8.3 `--parallel` — concurrent inference slots
  - Each slot × n_parallel multiplies KV VRAM
  - Single user homelab: `--parallel 1` to reclaim KV VRAM for weights
#### 8.4 `--flash-attn on` (`-fa`)
  - Required for large context; no downside on CUDA; always enable
  - [CUDA / Metal note: Vulkan flash-attn support varies]

---

### 9. Batch Sizes

#### 9.1 `--batch-size` (`-b`) — pp throughput
#### 9.2 `--ubatch-size` (`-ub`) — physical micro-batch
  - Vision models: ubatch must be ≥ image token count
#### 9.3 Tradeoff table across values

---

### 10. Sampling Parameters

Guidance on where defaults come from (model cards) and what each knob does.

#### 10.1 `--temp` — creativity / determinism
#### 10.2 `--top-k` — vocabulary truncation; speed vs diversity
  - `top-k 100` as safe speed optimization (from gpt-oss-120b findings)
  - `top-k 0` = full vocabulary (slowest, most diverse)
#### 10.3 `--top-p` — nucleus sampling
#### 10.4 `--min-p` — minimum probability floor
#### 10.5 `--repeat-penalty` — repetition control; keep at 1.0 for code

Model-specific recommended defaults table: Gemma 4, Sarvam, Qwen3, gpt-oss, generic.

---

### 11. Threading and CPU Control

#### 11.1 `--threads` (`-t`)
  - Set to P-core count; more ≠ better
#### 11.2 `--threads-batch`
  - For pp phase; can be set to full thread count
#### 11.3 `taskset -c` — P-core pinning [Linux]
  - Intel hybrid arch: E-cores drag down expert GEMM significantly
  - Example: `taskset -c 0-11` on 6P+4E processor
#### 11.4 `--poll`
  - **Flat on hybrid CPU+GPU inference** — do not tune
  - 0 = yield, 100 = spin; no measurable difference in GPU-bound workloads
#### 11.5 `--numa`
  - Only relevant on multi-socket / multi-die systems
  - Single socket: use `taskset` instead; NUMA modes are neutral or worse

---

### 12. Memory Control

#### 12.1 `--no-mmap`
  - Loads entire model into RAM before inference
  - Eliminates page-fault jitter during expert access (non-sequential access pattern)
  - **Recommended for all hybrid setups**
#### 12.2 `--mlock`
  - Pins RAM pages; prevents swapping under memory pressure
  - Especially important if `vm.swappiness` is high (many Linux distros default to 60–150 with zram)
  - Requires sufficient RAM headroom

---

### 13. Priority and Process Settings

#### 13.1 `--prio`
  - 0–3; `2` recommended for homelab
#### 13.2 `--no-warmup`
  - Skips initial kernel warmup; reduces startup time; safe

---

### 14. CUDA-Specific Settings [CUDA]

#### 14.1 Environment variables
  - `LLAMA_SET_ROWS=1` — CPU cache locality for MoE row access
  - `GGML_CUDA_GRAPH_OPT=1` — CUDA graph batching; **disable if context depth varies** (graph re-capture triggers VMM pool growth → OOM)

#### 14.2 Build-time flags (recap of §5.2 additions)
  - `GGML_CUDA_FA_ALL_QUANTS=ON` — flash-attn across all quant types
  - `GGML_CUDA_F16=ON` — f16 CUDA ops
  - `GGML_CUDA_GRAPHS=ON` — enable CUDA graph capture
  - `GGML_CUDA_FORCE_CUBLAS=OFF` — keep OFF; GGML MMQ kernels outperform cuBLAS for mxfp4 / MoE at consumer batch sizes; tested and confirmed

---

### 15. Vision / Multimodal [Dense-only / specific models]

- `--mmproj` flag and where to get projector files
- OOM failure modes:
  1. Aggressive fit headroom → mmproj allocation fails
  2. Image token batch > n_ubatch → assertion failure
- Safe vision profile on 12 GB: values + explanation
- Text and vision server separation when VRAM-constrained

---

### 16. The ik_llama.cpp Fork [Advanced / Optional]

#### 16.1 When upstream tops out
#### 16.2 Added flags and what they do
  | Flag | Effect | Tradeoff |
  | --- | --- | --- |
  | `-fmoe` | Fused MoE kernel; +10–30% tg | pp halves |
  | `-muge` | Repack up+gate; +tg | +27 GB RAM |
  | `-mqkv` | Merge QKV projections | marginally better tg |
  | `-ger` | Grouped expert routing | variable |
#### 16.3 Decision guidance: when to prefer ik_llama vs upstream

---

### 17. Diagnostic Checklist

Pre-bench and pre-run checks — bash snippet for each.
Covers: RAM speed, CPU governor, EPP, CPU MHz, free VRAM, temperature, background load.
Includes the full variability checklist from `bench-runbook.md` with status notes.

---

### 18. Optimization Priority Checklist — Summary

Numbered, ordered by impact. Each item one sentence + link to section.

1. Enable XMP/EXPO in BIOS (memory bandwidth; 3× win for MoE models)
2. Run Linux (or tune Windows/WSL2)
3. Replace power-profiles-daemon with tuned-ppd
4. Build llama.cpp from source; keep updated
5. Use `--fit on` for automatic VRAM placement
6. Use `-ctk q8_0 -ctv q8_0` (KV VRAM → extra GPU layers)
7. Set `--parallel 1` (reclaim KV VRAM for weights)
8. Pin to P-cores with `taskset -c`
9. Enable `--flash-attn on`
10. Enable `--no-mmap` and `--mlock`
11. Go headless for maximum VRAM
12. Use iGPU for display if available
13. Set `LLAMA_SET_ROWS=1` + `GGML_CUDA_GRAPH_OPT=1` (fixed-context servers)
14. iGPU for display if available

---

## Changes from V1 Plan

| What changed | Why |
| --- | --- |
| Glossary added as §1 | Users need shared vocabulary before any of §7–§14 makes sense |
| Full inference landscape §2 | Start from "should I even run locally?" and zoom in |
| OS section expanded to cover Linux/Windows/macOS | Give users the info to choose, not just a recommendation |
| Memory explained as bandwidth concept, not DDR5 specifically | Generalizes to any platform; the principle is what matters |
| Dense vs MoE distinction is now an organizing principle | Filters which sections are relevant before the user dives in |
| "Good enough" throughput table | Anchors expectations without prescribing hardware |
| Backend section (CUDA/Vulkan/Metal/CPU) | Flags are often backend-specific; this makes scope clear early |
| All CUDA-specific flags marked [CUDA] | Reader can skip if on Vulkan or Metal |
| ik_llama moved to §16 (advanced/optional) | It's not the baseline path |
| Link from gpt-oss-120b-post.md | That post's checklist section gets replaced with a link here |

---

## Open Questions

> [!NOTE]
> **Format**: pure markdown reference doc, or do you want the 11ty/blog template with frontmatter (making it a publishable post)? The content is dense enough to stand alone as a reference page rather than a dated blog post.

> [!NOTE]
> **Depth on ik_llama**: Given you have real bench data in the runbook, should §16 include the actual numbers or keep it at flag descriptions only?

> [!NOTE]
> **gpt-oss-120b-post.md edit**: When we write this doc, do you want me to simultaneously add a short note + link to that post's optimization section, or handle that separately?
