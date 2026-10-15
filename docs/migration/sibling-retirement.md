# Sibling repository retirement — 2026-10-15

`~/repos/l2m2` and `~/repos/l3ds` are retired as sources of truth. Their work
lives under `toolkits/l2m2` and `toolkits/l3ds`. The monorepo is the only place
to make new changes.

## Archived (outside Git, `~/repos/_backups/siblings-20261015/`)

- `l2m2.bundle`, `l3ds.bundle`: all refs, verified with `git bundle verify`.
- `*-tracked-wip.patch`, `*-untracked.tgz`: uncommitted and untracked files as of
  the scout, so nothing unique is lost if a directory is removed.

## Imported

L3DS: nothing left; every file was already present. L2M2: the Qwen Image 2.1
Turbo and YuE2 runtimes with their setup guides and tests, the ComfyUI MCP
Tailscale notes, media archive/restore and vision-grading scripts, the newer
Krea 2 and Qwen Image 2.1 wrappers, and the merged manifest, README and
CHANGELOG. The copied `model_downloader/` stays retired; `toolkits/l2m2/models_config.json`
runs under the root downloader.

## Deliberately not imported (kept in the archive)

- A scraped Reddit thread (`docs/sources/minimax-h3-batman-reddit-*.txt`) and a
  1.9 MB PNG fixture (`prompts/turbo-boat-input.png`): third-party content and
  a bulk artifact.
- `.agents/skills/media-gpu-self-preemption`: it documents the retired
  stop-and-hand-back GPU flow and a separate `l2m2-nsfw` runner.

## Still on disk because something live depends on it

| Path | Size | Dependency |
| --- | --- | --- |
| `l2m2/.venv-comfy-mcp` | 228 MB | three running `comfy-mcp` processes |
| `l2m2/.venv-comfy` | 6.5 GB | ComfyUI runtime |
| `l2m2/vendor` | 370 MB | `toolkits/l2m2/vendor` symlink; `audio.cpp` points at `~/repos/audio.cpp` |
| `l3ds/.venv` | 5.1 GB | local decision runner |
| `~/repos/l2m2-nsfw` runners | n/a | hardcode `~/repos/l2m2` paths |

Virtual environments embed absolute paths, so they must be rebuilt at the new
location rather than moved, and L3DS's setup installs unpinned packages. Finish
the retirement by rebuilding both environments under `toolkits/`, repointing
`l2m2-nsfw`, restarting the MCP servers, and only then removing the sibling
directories. Until then the siblings are read-only fallbacks.
