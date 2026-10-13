# Changelog

## Unreleased

- Declare LTX-2.5's CUDA requirement and add opt-in prepared-only generation
  using the existing `.venv/bin/python` offline, without `uv` syncing/downloads.
  Default standalone `uv run` and synchronized audio/video/image arguments stay
  unchanged. Shared execution uses a private MP4 target and idle GPU window.

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
