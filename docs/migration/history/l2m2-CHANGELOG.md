# Changelog

## Unreleased

- Reconcile the shared ComfyUI pin with root workflow routing and the consolidated
  Turbo downloader. FLUX.3 setup no longer replaces unrelated VAE aliases; fixture
  tests cover preservation and consistency of the shared installer revisions.

- Add FLUX.3 Image Klein text-to-image and layout-guided generation profile
  (`flux-3-klein`) with official INT8 ConvRot weights (5.6B DiT and Qwen3-VL 4B
  encoder) and shared Flux2 VAE. Includes `maintenance/setup-flux-3-klein-comfyui.sh`,
  CLI generator `media-models/generate-flux-3-klein.sh`, Python module
  `media-models/flux_3_klein.py`, UI workflow `workflows/flux-3-klein/image_flux_3_klein_t2i.json`,
  unit tests `media-models/test_flux_3_klein.py`, and ComfyUI pin update to
  commit `20b46065ef7f34fa21e8abebe8d0cca90a904d6d` (official FLUX.3 support).

- Root LTX/YuE2 profiles now use shared ComfyUI workflows, not the standalone
  Python runners. Add native API recipes and documented preview/acceptance limits;
  standalone scripts remain compatible and model data is unchanged.

- Require a mounted NAS and checksum-identical archive before removing local media;
  restore through a verified staged copy and refuse existing local paths. Preserve
  both copies when verification is uncertain.
- Resolve the Qwen Turbo downloader from the monorepo root with standalone
  compatibility, and use a general image grading rubric with `--rubric-file`
  support for external grading criteria.

- Record real standalone CPU headless and software GUI/MCP checks separately
  from root/GPU acceptance. GUI scene mutation/read-back passed, but shutdown
  crashed; do not claim a clean interactive lifecycle or change the installed
  Blender/add-on to mask it.

- Add opt-in Blender headless-only wrapper execution with a single explicit
  code/script/render action, Python exception exit codes and a fresh fixed PNG
  target. Preserve standalone GUI/MCP behavior and the scene's renderer backend;
  this prepares root integration but does not establish GPU ownership or safety.

- Declare LTX-2.5's CUDA requirement and add opt-in prepared-only generation
  using the existing `.venv/bin/python` offline, without `uv` syncing/downloads.
  Default standalone `uv run` and synchronized audio/video/image arguments stay
  unchanged. Root workbench use is now through shared ComfyUI instead.

- Explicitly declare H3's existing CUDA requirement for shared root routing.
  Original standalone script behavior, arbitrary native request options and
  optional MP4 muxing remain unchanged.

- Explicitly declare HeartMuLa's existing CUDA requirement in the media manifest
  for root native routing. Original standalone script flags/defaults are unchanged;
  shared execution requires an operator-managed idle GPU window.

- Extend the explicit CPU Breeze profile to reference-audio cloning and chunked
  TTS output through the existing recipe and WAV validator. Cloning requires a
  reference transcript; chunks are returned after completion, not played live.
  Preserve separate CUDA defaults and standalone modes.

- Add offline voice design to the explicit CPU Breeze profile. The shared L3MS
  wrapper requires an instruction for `design`; standard TTS remains text-only.
  Preserve original CUDA defaults and native output conventions.

- Add an explicit CPU Breeze profile for standard offline speech and a validated
  `--backend cpu|cuda` native script option. Keep the CUDA default and existing
  profile unchanged; backend availability is not proof of inference acceptance.

- Declare Breeze-TTS-2's CUDA device explicitly in its native manifest so
  accepted routes can validate GPU admission without guessing from prose.
  This does not pin a physical GPU or prove live inference/release.

- Add a read-only Breeze-TTS-2 readiness check and offline contract tests for
  standard TTS, instruction-led design, cloning, and streaming output routing.

- Document monorepo-root build/run instructions and the compiled default-root
  and `--root` override behavior.
- Add Qwen Image 2.1 + Krea 2 explicit-content probe harness, then extract it to a sibling repo:
  the adult prompt packs, batch runners, contact-sheet tooling, grading rubric, exported probe
  workflows and experiment write-ups now live in `~/repos/l2m2-nsfw`, which reaches this repo
  only through `L2M2_ROOT` (default `~/repos/l2m2`) so L2M2 itself carries no explicit strings.
  What stayed here:
  - Add `maintenance/grade-images-with-vision-tier.py` so a text-only agent can grade renders
    through the L3MS vision tier (`qwen38-flash-next-plat-abliterated-vision` serves it without a
    model swap), with `--rubric`, `--reasoning-log`, and a 16384-token default so thinking tiers
    do not exhaust the budget before emitting verdicts.
  - Add `maintenance/export-qwen-image-workflows.py` to dump replayable API-format ComfyUI
    workflows per prompt-pack row with a `manifest.json`.
  - Add `.agents/skills/media-gpu-self-preemption/SKILL.md` documenting the self-pause/resume
    pattern for media tests when the agent's own inference runs on llama-swap: stop
    `llama-swap.service`, gate on both free VRAM and `MemAvailable` (ComfyUI segfaults under
    memory pressure, not just VRAM pressure), start ComfyUI as `l2m2-comfyui.service`, and hand
    the GPU back on `trap … EXIT INT TERM` so an aborted run still restores the model server.
  - Change the Qwen Image 2.1 base generator default `--cfg` from `1.0` to `2.5`
    (`media-models/qwen_image_21.py`): at cfg 1 ComfyUI skips the negative-conditioning pass, so
    pack negatives were inert, and cfg 1.0 measured worst for multi-person anatomy.
  - Add `--model` to `media-models/qwen_image_21.py` to swap the diffusion model while keeping the
    text encoder and VAE pinned, so alternate community quants of the same architecture A/B
    like-for-like instead of forking a generator per release.
  - Add repeatable `--extra-lora NAME[:STRENGTH]` to `media-models/krea_2.py`, chained after the
    distill and bypass LoRAs through `LoraLoaderModelOnly` nodes with strength validated to 0-10.
    Chaining is not numerically neutral on an int8 base — every merge requantizes, so LoRA order
    changes output — and unit tests cover the chain order.
  - The extracted runner gained `--generator` (run one pack against any generator script) and an
    optional `--cfg` that inherits the generator's default instead of overriding it with 1.0.

- Add Krea 2 TextFusion refusal-reduction bypass LoRA:
  - Download and pin official `Krea2_TextFusion_Refusal_Reduction.safetensors` in `models/krea-2-comfyui-files.txt` and symlink to `vendor/ComfyUI/models/loras/`.
  - Add `--refusal-bypass` / `--bypass`, `--refusal-bypass-lora`, and `--refusal-bypass-strength` options to `media-models/krea_2.py` and `media-models/generate-krea-2.sh`.
  - Add unit tests in `media-models/test_krea_2.py` verifying model-only LoRA chaining.
  - Update `docs/krea-2-setup.md` with refusal bypass instructions.
  - Pin `fedor_bypass.safetensors` (488 bytes, refusal knobs 9 and 10 only) from
    `CliffNodes/fedor_bypass` revision `024bf4cd96c824321807ce93574a57ca72867366` in
    `models/krea-2-comfyui-files.txt` and symlink it into `vendor/ComfyUI/models/loras/`.
  - Fix `media-models/krea_2.py` counting ComfyUI's own resident checkpoint as a foreign
    workload: it now whitelists the `l2m2-comfyui.service` MainPID per process the way
    `media-models/qwen_image_21.py` does, instead of failing every row with
    `GPU is busy (9882 MiB used)` whenever ComfyUI holds the model between jobs.
  - Record measured Krea 2 throughput in `docs/krea-2-setup.md`: 32-36 s per 1024x1024 12-step
    image on this card, because ComfyUI stages 12,864 MB for the INT8 DiT and 4,999 MB for the
    FP8 text encoder and re-stages the DiT for every job.

- Add YuE2 foundation music generation harness:
  - Add `yue2-music` profile to `media-runtimes.json` for full-song generation with symbolic planning.
  - Add `media-models/generate-yue2.sh` script-first CLI runner and `media-models/yue2_runner.py` parameter validator with FP8 quantization and memory budgeting options for 12 GB GPUs.
  - Add `maintenance/setup-yue2.sh` for checkout, Python 3.12 uv virtual environment setup, and model weights management.
  - Add pinned model manifest in `models/yue2-files.txt` for `YuE2-3B` and `YuE2-Vae` with verified SHA-256 checksums.
  - Add unit tests in `media-models/test_yue2.py`, sample song request in `prompts/yue2-city-lights.json`, and setup runbook in `docs/yue2-setup.md`.
  - Update `maintenance/setup-media-runtimes.sh` with `install-yue` command and readiness checks.

- Add Qwen Image 2.1 Turbo 8-step ComfyUI setup with INT8 ConvRot weights,
  custom official sampling sigmas, and L2M2 generation profile:
  - Add `qwen-image-2.1-turbo` profile to `media-runtimes.json` for 8-step text-to-image generation.
  - Add `media-models/generate-qwen-image-2.1-turbo.sh` and `qwen_image_21_turbo.py` CLI runner.
  - Add `maintenance/setup-qwen-image-2.1-turbo-comfyui.sh` and `models/qwen-image-2.1-turbo-comfyui-files.txt` manifest.
  - Add unit tests in `media-models/test_qwen_image_21_turbo.py` and runbook in `docs/qwen-image-2.1-turbo-setup.md`.

- Add audio.cpp runtime management and BreezeTTS 2 profile:
  - Add `vendor/audio.cpp` and `vendor/audio.cpp-master` vendor tree layout mirroring L3MS llama.cpp tiers.
  - Add `maintenance/refresh-audio-cpp.sh` to audit upstream PRs, fetch, fast-forward, compile CUDA binaries (`audiocpp_cli` & `audiocpp_server`), and run smoke tests.
  - Add `breeze-tts-2` profile to `media-runtimes.json` for instruction-conditioned speech and voice cloning.
  - Add `media-models/generate-breeze-tts-2.sh` script-first CLI runner supporting offline/streaming modes, voice design (`--instruction`), and voice cloning (`--voice-ref`).
  - Update `maintenance/setup-media-runtimes.sh` with `install-breeze` and readiness checks.

- Add pinned official Z-Image-Turbo ComfyUI setup with BF16 and INT8 weights,
  Qwen 3 4B text encoders, VAE, and verified SHA-256 manifests.

- Add pinned official FLUX.2 Klein 9B ComfyUI setup with distilled 4-step turbo
  and base models, Qwen 3 8B text encoder, VAE, and verified SHA-256 manifests.

- Expose ComfyUI and local Comfy MCP server via Tailscale Funnel on port 443
  and configure workflow discovery under `.toolkit/comfyui/user/default/workflows`.

- Add a pinned Krea 2 ComfyUI setup with official raw INT8 weights,
  Qwen3-VL 4B encoder, VAE, r128 Turbo distill LoRA (0.60 strength, 12 steps,
  CFG 1.5), workflows, checksums, and an L2M2 text-to-image profile.

- Add a pinned Qwen Image 2.1 ComfyUI setup with official quantized weights,
  workflows, checksums, and an L2M2 text-to-image profile.

- Add a pinned, resumable MiniMax-H3 ComfyUI Ref2VA baseline installer using
  the official INT8 ConvRot model, NVFP4 text encoder, and video/audio VAEs.

- Add a pinned official ComfyUI/local MCP setup and CPU/GPU launch scripts;
  default CPU setup does not interfere with LLM benchmarks.

- Capture sourced H3 reference-workflow notes and add a bounded Batman scene test.
- Fix explicit H3 video dimensions/frame counts being overwritten by defaults.
- Expand L2M2 as Local Large Media Models.

## 0.2.0 — 2026-09-08

- Added `blender-3d` profile for Blender 5.2.1 LTS with Model Context Protocol (`blender-mcp`) support.
- Added `media-models/generate-blender-3d.sh` supporting `--check`, `--launch`, `--code`, and `--render`.
- Added Blender executable, addon, and listener readiness checks to `maintenance/setup-media-runtimes.sh`.
- Added detailed setup runbook at `docs/blender-mcp-setup.md`.

## 0.1.0 — 2026-09-08

- Extracted L3MS media profiles, argv-safe Rust launcher, generation scripts,
  runtime setup, and media documentation into a standalone local repository.
- Moved Qwen Image/Edit and LTX-2.3 download configuration; retained existing
  paths and flags and copied the downloader compatibility boundary.
- Preserved historical media handoff notes and original MIT license.
