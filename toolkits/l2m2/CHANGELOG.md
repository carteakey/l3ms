# Changelog

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
