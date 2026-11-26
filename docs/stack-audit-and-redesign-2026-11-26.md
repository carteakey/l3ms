# L3MS Stack Audit, Website & Dashboard Redesign, and Repo Reorganization (2026-11-26)

Empirical comparison of our live `L3MS` stack on `yeti-cachy` (RTX 4070 12 GB `sm_89`, i5-12600K 6P+4E, 64 GB DDR5, CUDA 13.5, driver `620.73.04`) against the external Two-Tier Redesign & Audit report, plus concrete plans for:

1. **Host & `llama-swap.yaml` Two-Tier Cleanup (Phase 0)**
2. **`l3ms.carteakey.dev` Public Redesign — Stack + Engines + Roofline (Phase 1)**
3. **Local Live Control Dashboard (`:8085` on Tailscale) (Phase 2)**
4. **`l3ms` Monorepo Reorganization & `l2m2`/`l3ds` Purge (Phase 3)**

> **Confirmed Design Decision (2026-11-26):** Keep `qwen3-8-27b-iq3-xxs` (10.17 GiB, 100% VRAM + 262k context @ 36.7 t/s) as our pure-VRAM zero-RAM-traffic language tier instead of adding `Qwen3.5-9B`. Retire all standalone `l2m2` / `l3ds` mentions across the repo and site — everything lives as clean splits inside `l3ms`.

---

## 1. Current `L3MS` Stack vs. Suggested Stack

### A. Coverage & Performance Matrix

| Slot / Capability | Suggested Stack (External Audit) | Current `L3MS` Stack (Verified on Disk) | Verdict & Action |
| :--- | :--- | :--- | :--- |
| **Daily #1: Coding + Vision MoE** | Inkling 2 Mini (`UD-Q4_K_M`, PolyStrata `--tail-skip 0`, ~7.8 GiB VRAM cap -> **46–55 t/s**, 600–1,100 pp) | `inkling-2-mini` (`vendor/polystrata/polystrata-inkling2-256k.json`, `polystrata-4070` `v0.2.20-poly.5` + `#313`, `UD-Q4_K_XL`, 256k elastic FP8 KV, `STRATA_PREFILL_NVFP4=w4a8`, `--window 6 --lookup 3`, CPU `strata-vision`) -> **77.5 t/s fresh / 99.7 t/s edits, 1,833–2,296 t/s pp** + `inkling-2-mini-llama` (**68.7 t/s**) + `unbound-v2.1` | **L3MS beats the audit by +45–100%.** Real trade-off: active config uses 11.7 GiB VRAM, not 7.8 GiB. Optional: add a secondary `inkling-2-mini-cozy` (~8 GiB cap) only if desktop/Wayland headroom is needed while coding. |
| **Pure-VRAM Zero-RAM Language Tier** | `Qwen3.5-9B` (`UD-Q4_K_XL`, ~7.5–8.0 GiB VRAM) | `qwen3-8-27b-iq3-xxs` (`10.17 GiB`, **100% VRAM + 262k ctx @ 36.7 t/s**, zero DDR5 decode traffic) + `gemma-4-26b-qat-mtp` (**100.6 t/s**) + `humanizer-12b` (10.5 GiB rewrite) | **Confirmed: `qwen3-8-27b-iq3-xxs` is the right model here** instead of `Qwen3.5-9B`. Fits 100% in VRAM at 262k context with far stronger dense 27B capability and factual precision than a 9B model. |
| **Co-Resident E-Core Micro-Utilities** | `lfm3-d1` 2B + `embeddinggemma-2` + `Kokoro 82M` / `Supertonic 3` on E-cores | `embeddinggemma-2` (`Q8_0`, CPU-only, non-exclusive `embeddings` group in `llama-swap.yaml`) | **Partially covered.** Pin `embeddinggemma-2` to E-cores (`taskset -c 12-15`) in a `co-resident` group; optionally evaluate CPU-pinned tiny router (`lfm3-d1` or `nagi-smol-cpu`) + lightweight CPU TTS later. |
| **Burst Lang #1: Frontier MoE** | `Qwen3.8-Flash-Next` (`UD-IQ2_XS V3.1` + NVMe `Q8_0` PLE, `strata-lean` -> 64–70 t/s) | `qwen38-flash-next-plat` (`vendor/strata-lean/strata-ud-iq2_xs-ple_q8.json`, `UD-IQ2_XS` + NVMe `Q8_0` PLE, `--kv-grow`, K8V4, `STRATA_HC_Q8=1`, `STRATA_ADAPT_FETCH=1`, `STRATA_PREFILL_LEND_PCT=40`) -> **73.0 t/s steady, 1,136–1,888 t/s pp** + Vision + Abliterated + `gold`/`exp`/`mtp` fallbacks | **L3MS already beats the audit.** Update stale `dashboard-meta.json` (currently stuck at `59.7 t/s` from Nov 5). |
| **Burst Lang #2: Dense / Specialized** | `Qwen3.8-27B` (`UD-Q3_K_XL` 12.8 GB -> spills ~3.5 GiB RAM, 10–18 t/s) | `qwen3-8-27b-iq3-xxs` (100% VRAM @ 36.7 t/s) + `gemma-4-26b-qat-mtp` (100.6 t/s) + `nemotron-3.5-bolt` (74B-A9B VLM + DFlash) | **Covered with a superior 12 GB trade-off** (all-VRAM 36.7 t/s vs spilled 10–18 t/s). |
| **Burst Image** | `Qwen-Image-2.1-Turbo` (GGUF via `leejet`) + `MiniMax-H3-Image` (Pruned Q3) + `FLUX.3 Klein` | `media-runtimes.json`: `qwen-image-2.1-turbo`, `qwen-image-2.1`, `flux-3-klein`, `krea-2` via native ComfyUI **INT8 ConvRot** | **Mostly covered & cleaner** (native INT8 ConvRot avoids `ComfyUI-GGUF` node breakage). Optional future addition: `MiniMax-H3-Image`. |
| **Burst Video** | `MiniMax-H3` (`ComfyUI-H3-Multishot` + Qwen3-VL-32B auto-evict) + `LTX-2.5` + `Wan 2.2 A14B` | `minimax-h3` (`audio.cpp`, weights missing) + `ltx-2.5` (`ComfyUI`, weights missing — TODO `#22`/`#23`) | **Real gap (weights/workflows unverified).** Track H3 Video via `ComfyUI-H3-Multishot` + `Wan 2.2` Apache-2.0 fallback when reclaiming media disk. |
| **Burst Audio (ASR / TTS / Music)** | ASR: `Nemotron-3.5-ASR-0.6B` / `Qwen3-ASR` / `LFM3-Audio 2B` · TTS: `Fish S2 Pro` / `Qwen3-TTS 1.7B` · Music: `ACE-Step 1.5 Turbo` (`acestep.cpp`) + `BS-RoFormer` | ASR: `nemotron-3.5-transcribe` (30B-A4B via `speech.cpp`, unlisted) · TTS: `Breeze-TTS-2` (`audio.cpp` CUDA/CPU) · Music: `HeartMuLa-3B` + `YuE2` (weights missing) | **Mixed.** ASR is stronger than suggested; TTS is solid (`Breeze-TTS-2`); **Music is a real gap** (`ACE-Step 1.5 Turbo` via `acestep.cpp` beats `HeartMuLa`/`YuE2`). |
| **Burst Decisions (`/v1/systemone`)** | **27B** `Cloudflare Clef` / `vLLM-SR d3-27B` (`Artemis-Q3_K_S`) + 9B `d3-flash` (`Artemis-IQ4_XS`) | 9B `vllm-d3-flash` (`Q8_0`, 11.5 GiB) + 9B `cloudflare-clef-flash` (`Q4_K_M`, 8.9 GiB) + legacy Python `nagi`/`laya`/`semif` + hosted `jev`/`systemone` | **Real gap.** Plan to add 27B `Clef` / `d3-27B` for grounding and >50-label schemas; fix inaccurate "sub-10ms" claim in `README.md` (9B is ~65–110 ms; sub-10ms requires ≤2B). |
| **Remote Burst (Missed by Audit)** | *(Not mentioned)* | `docs/colab-burst-lane.md` (Colab L4/A100/G4/H100 remote burst + fine-tuning), Blender 5.2 MCP 3D, Abliterated tiers, Humanizer 12B | **Extra L3MS strength.** Gives a 3rd residency tier (Remote Burst) without evicting local VRAM. |

---

### B. Live Verification of the Audit's 7 Diagnostics on `yeti-cachy` (2026-11-26)

| # | Audit Claim | Live Reality on `yeti-cachy` | Action |
| :-: | :--- | :--- | :--- |
| 1 | **DDR5-4800 JEDEC (-20% BW)** | **CONFIRMED.** `inxi -m` shows **4 × 16 GiB DDR5 @ 4800 MT/s** (2DPC), matching measured `60.0 GB/s` CPU read floor in `docs/rtx4070-inference-ceilings.md`. Meanwhile `docs/dashboard-meta.json` inaccurately states `5600 MT/s`. | Fix spec label to `4×16 GB DDR5-4800` now; BIOS Gear-2 tuning (5200–5400 MT/s) or 2×32 GB swap is a manual UEFI step. |
| 2 | **CachyOS kernel: `ananicy-cpp`, `THP=always`, high `swappiness`** | **CONFIRMED.** Live readings: `ananicy-cpp` = **`active`**, `scx` = **`inactive`**, THP = **`[always]`**, `vm.swappiness` = **`150`** (**6.6 GiB already in zram swap** while 37 GiB RAM is free). CPU governor is already `performance`. | Document/apply `vm.swappiness=10` (and verify `ananicy-cpp` / `scx_bpfland` / `THP=madvise` carefully with A/B check on Strata since Strata uses 2 MiB hugepages). |
| 3 | **Alder Lake `-t 16` across 6P+4E** | **MOSTLY FALSE / ALREADY EMPIRICALLY TUNED.** `llama-swap.yaml` already pins `cpu_range: "0-11"` (excluding E-cores `12-15`, which regressed Strata `-9.8 t/s`). Inkling 2 Mini sweeps already proved `-t 12` (`44.1 t/s`) beat `-t 6` physical (`38.1 t/s`) and `-t 10` (`43.4 t/s`). | **Keep `0-11` for P-core GPU-offloaded LLMs.** Add `ecores: "12-15"` macro specifically for CPU-resident utilities (`embeddinggemma-2`). |
| 4 | **`-np 4` KV bloat & d3 Issue #30402** | **FALSE ALARM.** Every LLM and decision profile in `llama-swap.yaml` already passes `--parallel 1`. | None. |
| 5 | **9B Clef-flash grounding / schema cliff** | **VALID.** Only 9B decision models are served; `README.md` claims "sub-10ms". | Fix `README.md` latency copy; track 27B `d3-27B`/`Clef-27B` onboarding. |
| 6 | **PolyStrata `--tail-skip` / Strata SQ-10 lossiness** | **FALSE ALARM.** `strata-lean` enforces `STRATA_ROUTE_TAIL_SKIP=0` and `polystrata-4070` runs lossless GGUF routing without `--tail-skip`. | None. |
| 7 | **`llama-swap` groups + `ComfyUI-GGUF` + `audio.cpp` ACE-Step** | **HALF-VALID.** Native ComfyUI INT8 ConvRot is already cleaner than `ComfyUI-GGUF`, and ACE-Step isn't served yet. **However**, `llama-swap.yaml` lacks clean residency groups, has ~400+ lines of commented retired models, and duplicates alias stanzas instead of using native `aliases:`. | Refactor `llama-swap.yaml`: remove dead commented blocks, consolidate duplicate aliases into native `aliases:`, add `ecores: "12-15"` on `co-resident` group (`embeddinggemma-2`), and organize clean residency groups (`co-resident`, `burst-exclusive`). |

---

## 2. `l3ms.carteakey.dev` Public Redesign + Local Live Dashboard

### A. Public Static Site (`l3ms.carteakey.dev`)

Fix all stale data (`2026-11-05`, missing Inkling 2 Mini `77.5/99.7 t/s`, outdated Qwen3.8 `59.7` -> `73.0 t/s`, Gemma 26B wrongly listed as startup preload, wrong RAM `5600` & `CUDA 12.8`, leftover `L2M2`/`L3DS` strings, hardcoded "Portable llama.cpp command" for Strata/PolyStrata).

Restructure into **5 tabs** (keeping the dark phosphor/amber terminal aesthetic):

1. **Header & Topology Strip**
   - `L3MS /// LOCAL MODEL STACK` — `RTX 4070 12 GB · i5-12600K · 64 GB DDR5-4800 · CUDA 13.5`
   - Inline SVG 3-tier memory/residency topology (`Co-Resident E-cores` -> `Exclusive Burst 12 GB VRAM + 64 GB RAM + NVMe` -> `Remote Colab Burst`).
2. **`[1] Stack` (Unified Catalog — Replaces Language / Media / Decisions Split)**
   - **Featured Active Primary Stack** cards (Frontier MoE `Qwen3.8-Flash-Next`, Fast Coding + Vision `Inkling 2 Mini`, Pure-VRAM Dense `Qwen3.8-27B IQ3_XXS`, Image `Qwen-Image-2.1-Turbo`, Speech/ASR `Breeze-TTS-2` + `Nemotron 3.5 Transcribe`, Decisions `vLLM-SR d3-flash` / `Clef-Flash`, Embeddings `EmbeddingGemma-2`).
   - Filter pills by **Tier** (`All`, `Co-Resident`, `Burst`, `Remote`), **Modality** (`Language`, `Vision`, `Image`, `Video`, `Audio`, `Decisions`, `Embeddings`, `3D`), and **Engine** (`strata-lean`, `polystrata-4070`, `llama.cpp`, `ComfyUI`, `audio.cpp`, `speech.cpp`).
   - **Visual Memory Bar** per model (`VRAM` + `RAM` + `NVMe stream`).
   - **Engine-Aware Expandable Recipe**: portable Strata/PolyStrata JSON + `STRATA_*` env flags, `llama-server` CLI, or ComfyUI/audio.cpp workflow links (no false "llama.cpp command" label on Strata/PolyStrata).
3. **`[2] Engines` (New Dedicated View)**
   - Cards for `strata-lean`, `polystrata-4070`, `llama.cpp-inkling2-lean`, `llama.cpp-master`, `llama.cpp-d3`, `ComfyUI`, `audio.cpp`, `speech.cpp`, `llama-swap`.
   - Auto-surfaces **Promoted Runtime Switches** with measured gains (`STRATA_HC_Q8=1`, `STRATA_ADAPT_FETCH=1`, `STRATA_PREFILL_LEND_PCT=40`, `STRATA_PREFILL_NVFP4=w4a8`, `--window 6 --lookup 3`) and a collapsible **Rejected Switches Graveyard**.
4. **`[3] Roofline & Evolution` (SVG Charts + Shootouts)**
   - **Dual-Track Evolution SVG Chart**: Qwen3.8-Flash-Next (`6.5 -> 73.0 t/s`) & Inkling 2 Mini (`5.9 -> 77.5 / 99.7 t/s`).
   - **RTX 4070 Silicon Roofline SVG**: Measured `455.8 GB/s` VRAM / `25.0 GB/s` PCIe / `60.0 GB/s` DDR5 / `60.5 TFLOPS` vs `PolyStrata` and `Strata-lean` across `4K–256K` context.
   - **Head-to-Head Engine Shootouts** (Inkling 5-engine shootout, Qwen3.8 Strata vs `llama.cpp`, ComfyUI Turbo vs Base).
5. **`[4] Hardware & Tuning`**
   - Exact node microbench specs, CachyOS / Alder Lake hygiene notes, and Colab Burst Lane overview.
6. **`[5] Archive`**
   - Retired profiles with reasons & disk reclaimed (plus Community Runs only when non-empty).

### B. Local Live Control Dashboard (`:8085` on Tailscale)

A live companion web UI sharing the exact same visual design language as `l3ms.carteakey.dev`, running locally on `yeti-cachy:8085` (accessible at `http://yeti-cachy.llama-alnair.ts.net:8085/`):

- **Server-side auth**: Reads `LLAMA_SWAP_API_KEY` from `llama-swap.service` / env in-process (never exposed to browser JS).
- **Live Silicon Strip (SSE / 2s poll)**: RTX 4070 VRAM (`used / 12,282 MiB`), GPU util/temp/power, DDR5 RAM + zram swap gauge, CPU P-core vs E-core load, **CPU governor badge** (amber alert if `powersave`), and active inference process NVMe read MB/s (spill detector).
- **Router & Model Control**: Resident model(s) per group, one-click **Wake / Switch / Unload** with eviction confirmation, deep links to `:8089` Chat UI and `/comfyui/`.
- **Active Engine Inspector & Logs**: Displays loaded Strata/PolyStrata JSON flags & env vars, live tail of `llama-swap.service` and active engine log, and recent `logs/results/*.jsonl` sweep viewer.
- **One-Click Smoke Runner**: Run health/smoke checks directly from the dashboard.

---

## 3. Repository Reorganization Plan (Clean Splits Inside `l3ms`)

Goals: eliminate all `l2m2` / `l3ds` naming, remove the 153 root `.bak*`/`.log` clutter files, separate the GitHub Pages site from pure markdown `docs/`, group active vs archived benchmarks, and keep `llama-swap.service`, Rust tests, and CI green at every step.

### Target Structure (Incremental, Breaking Nothing)

```text
l3ms/
├── Cargo.toml, Cargo.lock, rust-toolchain.toml
├── README.md, AGENTS.md, CHANGELOG.md, TODO.md
├── llama-swap.yaml                 # stays at root (or symlinked) for llama-swap.service compatibility
├── config/                         # Centralized tracked engine/runtime configs & manifests
│   ├── engines/                    # strata/, polystrata/, audiocpp-server.json
│   └── manifests/                  # media.json, decisions.json (cleanly linked/migrated)
├── src/                            # Rust TUI/CLI workbench
├── runners/                        # Clean domain splits (zero l2m2/l3ds naming)
│   ├── media/                      # image/video/audio/3d wrappers + workflows/
│   ├── decisions/                  # systemone runner + suites/
│   ├── colab/                      # remote burst wrappers
│   └── downloader/                 # HF Python downloader boundary
├── bench-models/                   # Active harnesses at top; retired models in bench-models/archive/
├── maintenance/ (or ops/)          # Clean subdirs: bin/, checks/, setup/, download/, systemd/
├── tools/local-dashboard/          # Local live control dashboard (:8085) + systemd unit
├── docs/                           # Pure docs organized into subdirs + GitHub Pages entrypoint
│   ├── index.html, styles.css, app.js, generated-models.js, CNAME, favicon.svg
│   ├── dashboard-meta.json, engines-meta.json, stack-meta.json, generate_dashboard_data.py
│   ├── architecture/               # ceilings, colab-burst-lane, architecture
│   ├── engines/                    # strata-lean, strata-tuning-matrix, inkling-2-mini-optimization
│   ├── runbooks/                   # llama-swap, bench, media, decisions, onboarding
│   ├── incidents/                  # postmortems
│   ├── posts/                      # blog mirrors
│   └── archive/                    # old plans, code reviews, migration history, challenges
└── .local/ (gitignored)            # backups/*.bak*, logs/, scratch/
```

> **GitHub Pages compatibility note:** GitHub Pages currently serves `l3ms.carteakey.dev` from `/docs` on `main`. Keeping the static site assets in `docs/` (cleanly separated from markdown subdirs `docs/architecture/`, `docs/engines/`, `docs/runbooks/`, `docs/archive/`, OR migrating Pages to a GitHub Actions workflow deploying `site/`) avoids any downtime on `l3ms.carteakey.dev`.

---

## 4. Execution Workstreams

1. **Workstream A — Hygiene, `llama-swap.yaml` Cleanup & `l2m2`/`l3ds` Copy Purge**
   - Relocate 153 root `.bak*`/`.log` files into `.local/backups/` & `.local/logs/` (gitignored; update `.gitignore` and snapshot convention).
   - Clean `llama-swap.yaml`: remove commented retired models, consolidate duplicate alias blocks via `aliases:`, pin `embeddinggemma-2` to `ecores: "12-15"` in `co-resident` group, keep `qwen3-8-27b-iq3-xxs` as pure-VRAM dense tier, and pass `python3 maintenance/check-serving-configs.py --local-paths`.
   - Fix `README.md` (sub-10ms claim, `l2m2`/`l3ds` naming, updated primary tiers including Inkling 2 Mini).
2. **Workstream B — `l3ms.carteakey.dev` Public Redesign & Fresh Data**
   - Update `docs/dashboard-meta.json`, `docs/stack-meta.json`, add `docs/engines-meta.json`, and update `docs/generate_dashboard_data.py` + `docs/test_generate_dashboard_data.py` to support the new 5-tab **Stack / Engines / Roofline & Evolution / Hardware & Tuning / Archive** structure (with current Inkling 2 Mini `77.5/99.7 t/s`, Qwen3.8 `73.0 t/s`, `qwen3-8-27b-iq3-xxs`, `DDR5-4800`, `CUDA 13.5`, SVG charts, engine-aware recipes, and zero `L2M2`/`L3DS` strings).
   - Verify with `python3 docs/generate_dashboard_data.py`, `python3 -m unittest docs/test_generate_dashboard_data.py`, and `node docs/test_dashboard.cjs`.
3. **Workstream C — Local Live Control Dashboard (`:8085`)**
   - Build a clean, zero-bloat local dashboard server (`tools/local-dashboard/` + `maintenance/systemd/l3ms-dashboard.service`) sharing the redesigned `l3ms` visual theme, serving live SSE hardware telemetry (VRAM, GPU temp/power, RAM/zram, P/E-core CPU, governor alert, NVMe spill), `llama-swap` model status & Wake/Switch/Unload controls (server-side API key from systemd), active engine config & log tail, and recent `logs/results/*.jsonl` viewer, with unit tests.
4. **Workstream D — Repo Reorganization & Docs Tidying**
   - Organize `docs/` markdown into clean subdirectories (`architecture/`, `engines/`, `runbooks/`, `posts/`, `archive/`), archive retired `bench-models/` scripts into `bench-models/archive/`, move stray root benchmark scripts (`mtp_bench*.py`, `mtpmeter.py`, `sortling.py`) safely (updating references in `dashboard-meta.json`, `docs/README.md`, `AGENTS.md`, CI, and Rust tests), and remove residual `l2m2`/`l3ds` labels from UI/docs while keeping `cargo test` and all Python/bash checks 100% green.
