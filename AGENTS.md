# AGENTS

Agent and contributor guide for `L3MS` (RTX 4070 12 GB `sm_89` + i5-12600K + 64 GB DDR5).

## Scope & Lanes

One product, three lanes, served by shared engines (`l2m2` and `l3ds` folded in on 2026-10-15; historical guides in `docs/migration/history/`). Direct scripts are a last resort (only Blender headless); see `docs/migration/runtime-policy.md`.

- **Language:** Rust workbench (`src/`), `llama-swap`, `Strata` (`strata-lean`), `PolyStrata` (`polystrata-4070`), and `llama.cpp`. `llama-swap.yaml` is the serving source of truth.
- **Media:** `media-runtimes.json`, `media-models/`, `workflows/`, `maintenance/`. Images/video run on ComfyUI (`comfyui_auto`), speech/music on `audio.cpp` (`audiocpp_auto`) via `llama-swap` (start at `docs/media-models.md` and `docs/migration/media-through-llama-swap.md`):
  - Budget **~246 s** for ComfyUI cold start.
  - Requesting `/comfyui/` evicts the resident LLM (including one serving this agent). Treat a ComfyUI batch as one blocking call with no LLM requests until it returns (`NVRM: Out of memory`).
  - `TextEncodeQwenImage21` autogrow slots use dotted ids (`"images.image_1"`), not bare `image_1`.
  - ComfyUI input dir is `vendor/ComfyUI/input/`, not `models/input/`.
- **Decisions:** `decision-runtimes.json`, `decision-models/`, `suites/`. Local models (`vllm-d3-flash`, `cloudflare-clef-flash`) run on `vendor/llama.cpp-d3` (`d3_server`) via `POST /upstream/<id>/v1/systemone` with `-c = -b = -ub`. Verify `<arch>.decision.type` with `maintenance/gguf-decision-check.py` before downloading (`docs/decision-models.md`, `docs/decision-toolkit.md`).

## Context Pointers

- **Homelab ops wiki (KPC drives, mounts, L3MS endpoints, services):** `~/repos/homelabbr/wiki/` - read `~/repos/homelabbr/wiki/AGENTS.md` before editing.
- **Qwen3.8-Flash-Next tiers, Strata sweeps & tuning matrix:** `docs/qwen38-flash-next-internal.md`, `docs/strata-tuning-matrix.md`, `docs/strata-lean.md`, `docs/strata-perf-next-steps-2026-10.md`, `docs/incidents/2026-11-09-strata-kv-grow-oom.md`.
- **Inkling 2 Mini PolyStrata sweeps, 5-engine shootout & silicon roofline:** `docs/inkling-2-mini-optimization.md`, `docs/rtx4070-inference-ceilings.md`.
- **Benchmarks & CPU power state:** `docs/bench-runbook.md`, `bench-models/bench-env.sh`, `maintenance/cpu-power-state.sh`.
- **Blog posts & charts (`docs/*post*.md`, `carteakey.dev`):** Read `~/repos/carteakey.dev/.agents/skills/human-writer/SKILL.md`, `dataviz/SKILL.md`, and `sketches/SKILL.md`. Zero em dashes (`—`), first-person voice, use build-time SVG charts over pasted images/raw tables, and run `npm run check:content && npm run check:design` in `carteakey.dev` before pushing.
- **Classifier triage (Jev / pi codemode):** `docs/pi-jev-classifier-workflow.md` (`maintenance/codemode/`, `/jev-bench`, `/jev-log`).
- **NSFW prompts & grading:** Sibling checkout `~/repos/l2m2-nsfw` (uses `L2M2_ROOT`). Keep explicit strings out of this repo.

## Engineering Rules

- **Strata Anti-Bloat Policy:** Never blindly rebase or re-slice `carteakey/strata-lean` (`vendor/strata-lean`, branch `strata-lean`) onto upstream. Keep it lean Ada `sm_89` for Qwen3.8-Flash-Next (`vendor/polystrata` is separate for Inkling 2 Mini). Headroom hierarchy: 1. runtime switches; 2. isolated CUDA kernel cherry-picks; 3. re-slicing is rejected.
- **Serve / bench parity & guardrails:** Keep `llama-swap.yaml` and `bench-models/*.sh` aligned. Run `python3 maintenance/check-serving-configs.py --local-paths` after editing `llama-swap.yaml` or Strata/PolyStrata JSON configs.
- **Snapshots & downloads:** Snapshot `llama-swap.yaml` (`llama-swap.yaml.bak-YYYYmmdd-HHMMSS`) before edits. Preserve existing files by default on downloads.
- **Disk checks:** Use `df -hlT` (local filesystems only) on KPC so dead/unreachable NFS mounts never hang the check.

## llama-swap Operations

- Always pass `--user` to `systemctl` (`systemctl --user is-active llama-swap.service`). Router listens on `:8080`; all 32 mutually exclusive LLM profiles bind `0.0.0.0:8089` (`inference_port: "8089"`) so Strata/PolyStrata Web UI is always reachable at `http://yeti-cachy.llama-alnair.ts.net:8089/` (with `allowed_hosts` configured in every Strata/PolyStrata JSON).
- **Use `maintenance/l3ms-ctl.sh`** (`health`, `models`, `wake <id>`, `smoke <id>`, `ui-check`, `restart`) so `LLAMA_SWAP_API_KEY` is read in-process from the systemd unit and never echoed or passed inline on command lines.
- **Reload caveat on fixed port `8089`:** Prefer `systemctl --user restart llama-swap.service` over `kill -HUP $(pgrep '[l]lama-swap')` whenever a model is resident on `:8089` (`SIGHUP` can start the new preload before the old server releases port `8089`).
- **Full-VRAM experiments:** `systemctl --user stop llama-swap.service`, run the experiment, then `systemctl --user start llama-swap.service`.
- **Reasoning smoke tests:** Pass `max_tokens >= 64` (smaller budgets land entirely in `<think>` / `reasoning_content`).
- **Flag check:** Master builds use `--lazy-mode on`; older PR builds use `--tensor-read-lazy on`. Verify with `llama-server --help` before porting flags.

## Active Primary Tiers (Summary)

Model weights live on `/mnt/dev/models` (`ext4`, `nvme2n1p1`); smaller models/ComfyUI weights live in `~/models` (`~/models/SC117` and `~/models/unsloth/UD-IQ2_XS` symlink into `/mnt/dev/models/`).

- **`qwen38-flash-next-plat` (`strata`, startup preload):** `vendor/strata-lean` (`strata-ud-iq2_xs-ple_q8.json`), Unsloth `UD-IQ2_XS` + `Q8_0` PLE on NVMe, 128k `--kv-grow`, K8V4, MTP Q2_0, `STRATA_HC_Q8=1`, `STRATA_ADAPT_FETCH=1`, `STRATA_PREFILL_CPU_SHARE_MAX=3072`, `STRATA_PREFILL_LEND_PCT=40`. **73.0 t/s** steady decode, **1,136 / 1,439-1,888 t/s** pp2k/pp8k. Vision/abliterated peers: `qwen38-flash-next-plat-vision`, `qwen38-flash-next-plat-abliterated`, `abliterated-vision`. Fallback `llama.cpp` tiers: `gold`, `exp`, `mtp`, `vision`.
- **`inkling-2-mini` (`inkling-2-mini-128k`):** `vendor/polystrata` (`polystrata-4070`, `v0.2.20-poly.5` + `#313`, `polystrata-inkling2-256k.json`), `Inkling-2-Mini-UD-Q4_K_XL.gguf`, 256k elastic `--vram-pool --kv fp8 --loop-bf16 --prompt-batch 2048 --window 6 --lookup 3`, `STRATA_PREFILL_NVFP4=w4a8`, `STRATA_INK_FAST_CTRL=1`, `STRATA_INK_PRESTAGE=0`, pure-CPU `strata-vision`. **77.5 t/s** fresh / **99.7 t/s** edits, **1,833 / 2,296 / 2,270 t/s** pp2k/8k/20k. Fallback: `inkling-2-mini-llama`.

## Engine & Bench Gotchas

- **Vendor clones:** Check `git status` before trusting or rebuilding (`vendor/llama.cpp` is a stale bisect checkout; use `vendor/llama.cpp-master` for master; `vendor/llama.cpp-exp-stack-v2` holds both `llama-server` MTP and `llama-server-exp`).
- **PLE SSD rule:** Keep Qwen3.8 PLE table on NVMe via mmap (`-ot "per_layer_token_embd\.weight=CPU"`, never `--no-mmap` or `--mlock`).
- **Bench hygiene:** Gate on `maintenance/cpu-power-state.sh check` (`performance` governor via `cpupower.service`), capture frontmatter via `bench-models/bench-env.sh`, use `TG_WARMUP=6` / `TG_PROBES=4` (read `tg_steady_mean`), isolate `--expert-profile` per A/B arm, and verify SSD spill with `/proc/<pid>/io read_bytes` (not zram). Greedy output on Strata is non-deterministic across boots/passes.
- **Rust target cache:** If repo-detection tests fail with `left: None` due to a foreign `CARGO_MANIFEST_DIR`, run `cargo clean -p l3ms`.

## Structure

- `src/`: Rust TUI workbench (`app.rs`, `ui/`, `cli.rs`, `param_registry.rs`, `param_builder.rs`, `llama_swap.rs`, `config_store.rs`, `script_store.rs`)
- `l3ms/`, `model_downloader/`: Legacy Python TUI & downloader compatibility boundary
- `media-runtimes.json`, `media-models/`, `workflows/`, `audiocpp-server.json`: Media lane
- `decision-runtimes.json`, `decision-models/`, `suites/`: Decision lane
- `maintenance/`: Checkers (`check-serving-configs.py`, `gguf-decision-check.py`, `cpu-power-state.sh`), `l3ms-ctl.sh`, downloaders, `codemode/`
- `.venv*`, `vendor/`, `.toolkit/`: Ignored per-host runtime state (never commit)

## Versioning & Planning Authority

- Use SemVer (`MAJOR.MINOR.PATCH`) and update `CHANGELOG.md` on user-visible changes. Retain legacy Python TUI/downloader until Rust parity coverage is explicit.
- **GitHub Issues on `carteakey/l3ms`** is authoritative for committed work (`TODO.md` mirrors open issues plus labelled speculative notes; `docs/` holds evidence; Linear was retired 2026-11-07).
- **Notion `Forge`** is shared cross-repo context (`Agent Operating Guide`, L3MS `Projects` row, `Open Work — GitHub`), not the tracker. Do not mirror this repo or its backlog into Notion, and do not copy Notion-only material into public commits without authorization.
