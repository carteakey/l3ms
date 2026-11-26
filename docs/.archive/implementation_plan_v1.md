# Local LLM Inference Optimization Guide — Structure Plan

Synthesized from: `bench-runbook.md`, `gpt-oss-120b-post.md`, `qwen3-coder-next-post.md`,
`gemma4-post.md`, `sarvam-local-post.md`, `model-onboarding-playbook.md`.

Target file: `docs/local-llm-optimization.md`

---

## Proposed Document Structure

### 0. Preamble / Philosophy

Short framing:
- Two numbers that matter: **pp** (prompt throughput) vs **tg** (generation speed)
- For interactive use, **tg dominates**. You feel every token of generation; prefill is one-shot.
- Memory hierarchy: `VRAM >> Unified RAM (Apple Silicon / Strix Halo) >> DDR5 >> DDR4 >>>> SSD`
- This guide targets: single consumer GPU (8–16 GB VRAM) + 64 GB+ DDR5, Linux.

---

### 1. Hardware

#### 1.1 GPU / VRAM
- VRAM is the #1 bottleneck for expert-layer models
- More VRAM = fewer experts on CPU = faster tg
- iGPU trick: use motherboard HDMI for display → free ~800 MiB VRAM (RTX 4070 example: 824 MiB → 27 MiB for display)
- Practical tiers: 8 GB / 12 GB / 24 GB and what each can run at what quality

#### 1.2 RAM — **The single biggest gotcha**
- Full section on the XMP/EXPO incident from `gpt-oss-120b-post.md`
- For MoE models: expert weights live in RAM; tg ∝ memory bandwidth
- `DDR5-2000 (auto/default) → 32 GB/s → ~10 t/s` vs `DDR5-6000 (XMP) → 96 GB/s → ~30 t/s`
- How to check:
  ```bash
  sudo dmidecode -t memory | grep -E "Speed|Configured"
  ```
- Minimum 64 GB for 120B-class models; SSD is infeasible

#### 1.3 CPU — Core pinning matters
- Intel hybrid arch: P-cores vs E-cores
- E-cores are slow for GEMM workloads (LLVM inference is core-count-sensitive)
- Pin with `taskset -c 0-11` (P-cores only on i5-12600K)
- Thread count: `--threads = P-core count - 1`; more ≠ better

---

### 2. OS Setup

#### 2.1 Linux over Windows
- ~20% TPS uplift (CUDA scheduler, driver overhead)
- WSL2: partial help, not full native

#### 2.2 CPU Governor + Power Profile
- Governor must be `performance` on all cores
- EPP (`energy_performance_preference`) must also be `performance` — not just governor
- **The `power-profiles-daemon` trap** (the root cause from `bench-runbook.md`): KDE default silently degrades HWP on some boots even when sysfs shows "performance"
- Fix: replace with `tuned-ppd` + `throughput-performance`
  ```bash
  sudo pacman -S tuned-ppd
  sudo systemctl enable --now tuned
  sudo tuned-adm profile throughput-performance
  ```
- Preflight checks before every bench

#### 2.3 Go Headless (When Squeezing Every MB)
- `sudo systemctl isolate multi-user.target` — frees 200–400 MB RAM + compositor VRAM
- Drop page cache too:
  ```bash
  sudo sync && sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
  ```
- Use `zellij` in TTY for split panes without a display server

#### 2.4 Transparent Huge Pages
- Recommended: `[always]` mode active (CachyOS default)
- Check: `cat /sys/kernel/mm/transparent_hugepage/enabled`

#### 2.5 CUDA / Driver
- Keep CUDA toolkit updated; MoE kernel improvements track upstream closely
- Build llama.cpp from source; don't use distro packages

---

### 3. Why llama.cpp

- Ollama is great for "just works"; llama.cpp gives you everything below
- Build from source → hardware-specific tuning, latest kernels
- `llama-server` = OpenAI-compatible endpoint; works with any tooling
- Key tools in the build target: `llama-server`, `llama-bench`, `llama-fit-params`, `llama-sweep-bench`
- Useful env vars:
  - `LLAMA_SET_ROWS=1` — CPU cache locality for MoE expert rows
  - `GGML_CUDA_GRAPH_OPT=1` — CUDA graph optimization (disable if context depth varies a lot)
- ik_llama.cpp fork: better tg for MoE-heavy workloads, worse pp; tradeoffs covered below

---

### 4. Model Layer Placement (The Core of It)

This is the single most impactful tuning dimension for models that don't fit in VRAM.

#### 4.1 `--n-gpu-layers` (`-ngl`)
- What it does: how many transformer blocks to keep on GPU
- Start at `99` (all); drop if CUDA OOM
- Tradeoff: fewer layers on GPU → slower pp, often slower tg

#### 4.2 `--n-cpu-moe`
- Integer count: keep first N MoE layers' expert weights on CPU
- Quick starting knob; coarse but safe
- ⚠️ RAM ceiling: for 60 GB models, `N_CPU_MOE=36` (all on CPU) will hard-crash a 64 GB system (~60 GB lands in RAM)

#### 4.3 `--override-tensor` (`-ot`) regex
- Fine-grained per-tensor per-layer placement
- Most control; most complexity
- ⚠️ Shared expert gotcha: models with `_shexp` tensors (Qwen3.5-122B, some gpt-oss variants) silently consume VRAM if pattern only matches `_exps`
  ```bash
  # Wrong (misses shared expert):
  --override-tensor ".ffn_.*_exps.=CPU"
  
  # Correct (matches both _exps and _shexp):
  --override-tensor ".ffn_(up|down|gate)_(ch|)exps=CPU"
  ```
- Partial-CPU pattern (recommended starting point):
  ```bash
  --override-tensor "blk\.(5|[6-9]|[0-9][0-9]+)\.ffn_(up|down|gate)_(ch|)exps=CPU"
  ```

#### 4.4 `--fit on` (Recommended "just works")
- Auto-probes free VRAM at startup, computes optimal `-ngl` + `-ot` placement
- Key params:
  - `--fit-ctx N` — minimum context to guarantee fits (this context's KV cache is accounted for)
  - `--fit-target M` — VRAM headroom in MiB; use ≥512 MiB to survive CUDA VMM pool growth
- Dry run without serving:
  ```bash
  llama-fit-params -m model.gguf -fitt 512 -fitc 65536
  ```
- Tradeoff: startup delay (seconds) vs zero manual tuning

---

### 5. Context and KV Cache

#### 5.1 `--ctx-size`
- Directly determines KV cache VRAM usage
- At 131k context + f16 KV: ~8 GB of the 12 GB card is consumed by KV alone
- Practical guidance:
  - Coding sessions: 64k is usually enough
  - Vision/multimodal: keep at 64k max on 12 GB with conservative fit margin
  - Long-context RAG: maximize but watch OOM

#### 5.2 KV Cache Quantization (`-ctk`, `-ctv`)
- `q8_0`: halves KV VRAM, effectively lossless, zero bench speed cost
  → liberates 2+ extra GPU layers at 64k context on 12 GB card
- `q4_0`: saves ~3 GB more but quality regression on long context; skip for coding
- At bench context (512 tokens): effect is near-zero — test at your real serving context
- **Default recommendation**: `-ctk q8_0 -ctv q8_0` always

#### 5.3 `--parallel`
- Number of concurrent inference slots
- Each slot multiplies KV cache VRAM by N
- For single-user homelab: `--parallel 1` reclaims KV VRAM → one more GPU layer
- Only raise if you need concurrent requests (e.g. agentic pipelines hitting multiple streams)

---

### 6. Batch Sizes

#### 6.1 `--batch-size` (`-b`)
- Tokens processed in one forward pass during prompt processing (pp)
- Higher → better pp throughput; more VRAM required
- Typical values: 512, 1024, 2048
- Vision models: reduce batch to avoid batch assertion OOM at image token boundaries

#### 6.2 `--ubatch-size` (`-ub`)
- Physical micro-batch; must be ≤ `batch-size`
- For vision: must be ≥ image token count or you get assertion failure
- Typical: match to `batch-size` or use 512

---

### 7. Sampling Parameters

Sampling parameters affect output quality and speed (top-k truncation cuts vocabulary search).

#### 7.1 `--temp`
- 0 = greedy (deterministic, good for coding agents)
- 1.0 = model-card default for most chat models (Gemma 4, Sarvam)
- Do not lower without reason; most GGUF model cards specify a tested value

#### 7.2 `--top-k`
- Keeps only top K most probable tokens
- `0` = full vocabulary (slowest)
- `100` = safe cap for speed without measurable quality loss (confirmed on gpt-oss-120b)
- Model-specific: Sarvam recommends `20`, Gemma recommends `64`

#### 7.3 `--top-p`
- Nucleus sampling; keep tokens whose cumulative probability ≥ P
- `0.95` = Gemma/Qwen default; `1.0` = Sarvam default

#### 7.4 `--min-p`
- Minimum probability floor: filters tokens below p × top_token_prob
- `0.0` = off; `0.01` = Qwen3-Coder-Next default

#### 7.5 `--repeat-penalty`
- 1.0 = no penalty (recommended for most models; penalizing can hurt code generation)

---

### 8. Thread and CPU Control

#### 8.1 `--threads` (`-t`)
- CPU threads for generation (expert compute lives here in hybrid setups)
- Recommended: P-core count, leave 1–2 free for OS
- i5-12600K: `--threads 10` (out of 12 P-core threads)

#### 8.2 `--threads-batch`
- CPU threads for prompt processing (batch/pp phase)
- Can be equal to or slightly higher than `--threads`; set to full core count

#### 8.3 `taskset -c` (P-core pinning)
- Most impactful CPU control on Intel 12th gen+
- `taskset -c 0-11` on i5-12600K: P-cores only
- E-cores are 25-40% slower for GEMM; never include them

#### 8.4 `--poll`
- 0 = yield/sleep, 100 = busy spin
- **Flat on hybrid CPU+GPU inference** — GPU kernel + PCIe transfer dominate; polling level doesn't change throughput
- Leave at default (50) or set 0 to reduce CPU load; don't tune this

#### 8.5 `--numa`
- NUMA modes: `distribute`, `isolate`
- **Single-socket systems: skip** — there is only one NUMA node, no benefit
- Only useful on dual-socket or AMD multi-die systems

---

### 9. Flash Attention

- `--flash-attn on` (`-fa`)
- Required for large context: reduces attention memory from O(n²) to O(n)
- No downside on CUDA; always enable
- Some model configs require it for long context to fit at all

---

### 10. Memory Control

#### 10.1 `--no-mmap`
- Loads entire model into RAM before inference starts
- Avoids page-fault latency during generation (expert access is non-sequential)
- **Recommended for hybrid CPU+GPU inference** — always use

#### 10.2 `--mlock`
- Pins model weights in RAM; prevents OS from swapping them out mid-session
- Important if vm.swappiness is high (CachyOS default: 150 with ZRAM)
- Requires sufficient RAM headroom; skip only if you're RAM-constrained

---

### 11. Vision / Multimodal Notes

- `--mmproj`: path to multimodal projector weights
- OOM failure modes on 12 GB:
  1. `FIT_TARGET` too aggressive → mmproj allocation fails
  2. Image token batch > `n_ubatch` → assertion failure
- Safe vision profile (12 GB): `FIT_TARGET=2048`, `CTX_SIZE=65536`, `BATCH_SIZE=256`, `UBATCH_SIZE=512`
- Run text and vision servers separately if VRAM-constrained

---

### 12. Priority / Process Settings

#### 12.1 `--prio`
- 0–3 priority levels for the inference process
- `--prio 2` (high) on homelab systems minimizes OS scheduling jitter

#### 12.2 `--no-warmup`
- Skips initial kernel warmup pass at startup
- Reduces startup time; safe for production use

---

### 13. CUDA-Specific

#### 13.1 `GGML_CUDA_GRAPH_OPT=1`
- Enables CUDA graph optimization: batches kernel launches → lower dispatch overhead
- **Disable if context depth changes frequently** (long agent sessions at varying depths) — graph re-capture triggers CUDA VMM pool growth that can OOM on tight-fit configs

#### 13.2 `LLAMA_SET_ROWS=1`
- Improves CPU cache locality for MoE expert weight access patterns
- Low-risk; enable by default for MoE models

#### 13.3 Build flags
- `GGML_CUDA_FA_ALL_QUANTS=ON` — flash-attention for all quantization types
- `GGML_CUDA_F16=ON` — f16 CUDA ops
- `GGML_CUDA_GRAPHS=ON` — enables CUDA graph optimization at build time
- `GGML_NATIVE=ON` — CPU-native tuning
- `GGML_LTO=ON` — link-time optimization
- `GGML_CUDA_FORCE_CUBLAS=OFF` (default) — keep off; GGML MMQ kernels outperform cuBLAS for mxfp4/MoE at consumer batch sizes

---

### 14. ik_llama.cpp Fork

When upstream llama.cpp tops out, the fork adds:

| Flag | Effect | Tradeoff |
| --- | --- | --- |
| `-fmoe` (fused-moe, **default on**) | +10–30% tg | pp cuts in half |
| `-muge` (merge-up-gate) | +0.5–1.0 t/s tg | +27 GB RAM, slow startup |
| `-mqkv` (merge-qkv) | +marginal tg | no cost |
| `-ger` (grouped routing) | variable | no cost |

**Rule of thumb**: use ik_llama if generation speed is the only metric and you have RAM headroom. Stay on upstream llama.cpp for prefill-heavy workloads (RAG, long prompts).

---

### 15. Diagnostic Checklist (Pre-Bench / Pre-Run)

```bash
# 1. RAM speed
sudo dmidecode -t memory | grep "Configured Memory Speed"
# Expected: your XMP/EXPO rated speed

# 2. CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# Expected: performance

# 3. EPP (must be performance, not just governor)
cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference
# Expected: performance

# 4. CPU frequency (P-cores near max boost)
grep "cpu MHz" /proc/cpuinfo | sort -rn | head -4
# Expected: 4500-4900 MHz on i5-12600K

# 5. Free VRAM
nvidia-smi | grep MiB
# Expected: near-empty before starting server

# 6. Thermal
cat /sys/class/thermal/thermal_zone*/temp
# Expected: < 70000 (70°C)

# 7. Background CPU hogs
ps aux --sort=-%cpu | head -8
```

---

### 16. Optimization Priority Order (Summary Checklist)

1. ✅ **Enable XMP/EXPO in BIOS** (biggest single win for MoE models; 3x memory bandwidth)
2. ✅ **Run Linux** (+~20% TPS over Windows)
3. ✅ **Replace power-profiles-daemon with tuned-ppd** (eliminates intermittent tg degradation)
4. ✅ **Build llama.cpp from source** (keep updated; MoE kernels improve regularly)
5. ✅ **Use `--fit on`** (automatic VRAM-optimal layer placement — no manual tuning)
6. ✅ **Use `-ctk q8_0 -ctv q8_0`** (free KV VRAM win; unlocks extra GPU layers)
7. ✅ **Set `--parallel 1`** (single homelab user; reclaims KV cache VRAM)
8. ✅ **Pin to P-cores** with `taskset -c 0-11` (E-cores drag tg down)
9. ✅ **Go headless** for maximum VRAM (200–400 MB freed)
10. ✅ **Use iGPU for display** if available (~800 MiB VRAM freed on RTX 4070 example)
11. ✅ **Env vars**: `LLAMA_SET_ROWS=1`, `GGML_CUDA_GRAPH_OPT=1` (only if context is fixed-depth)
12. ✅ **Flash attention**: `--flash-attn on` always
13. ✅ **`--no-mmap`** for hybrid inference (prevents page-fault jitter)

---

## Open Questions for You

> [!IMPORTANT]
> **Target audience and depth**: Should this be a practical quick-reference post (like the model posts) or a thorough deep-dive reference doc that you can link to from model posts? The bench-runbook has raw data; this would be the "why" companion.

> [!IMPORTANT]
> **Hardware scope**: Should this be scoped to your exact setup (i5-12600K + RTX 4070 + 64 GB DDR5) or should it generalize to any consumer GPU system with callouts for different configs?

> [!NOTE]
> **Sections to cut or defer**: The ik_llama fork section (§14) could be a separate post. Worth including here or link out?

> [!NOTE]
> **Windows coverage**: The sarvam and gemma posts are Linux-only. Should this guide include any Windows/WSL2 notes, or stay Linux-first?

> [!NOTE]
> **Existing post integration**: Should this doc replace or absorb the "Optimization checklist" section from `gpt-oss-120b-post.md`, or live as a standalone that that post links to?
