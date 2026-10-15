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

## Runtime state moved (2026-10-15)

On this btrfs host the environments were reflink-cloned instead of rebuilt (no
extra disk, no network), and the embedded interpreter paths were rewritten in
the clones only. The originals were not touched.

| Cloned to | From | Verified |
| --- | --- | --- |
| `toolkits/l2m2/.venv-comfy` | `l2m2/.venv-comfy` (6.5 GB) | torch 2.14+cu130 sees CUDA; ComfyUI `6bfaacc67` starts; `comfyui_auto` launched it through an isolated router (HTTP 200, `/system_stats`) |
| `toolkits/l2m2/.venv-comfy-mcp` | `l2m2/.venv-comfy-mcp` | `comfy-mcp --help`; MCP `initialize` handshake through `maintenance/run-comfy-mcp.sh` |
| `toolkits/l2m2/vendor/ComfyUI`, `vendor/comfy-mcp`, `.toolkit/comfyui/user` | `l2m2/vendor/*`, `.toolkit/comfyui/user` | same commit as the source |
| `toolkits/l3ds/.venv` | `l3ds/.venv` (5.1 GB) | torch 2.6+cu124 sees CUDA; L3DS Python suites 38 + 17 pass |

`llama-swap.yaml` now sets `comfy_root` to `${env.L3MS_ROOT}/toolkits/l2m2`.
Codex's `comfy-local` MCP server was repointed to the monorepo launcher (backup
in `_backups/siblings-20261015/codex-config.toml.bak`).

## Retired (2026-10-15)

`~/repos/l2m2` and `~/repos/l3ds` were moved, not deleted, to `~/repos/_retired/`.
Before the move every consumer was repointed and nothing was running from the
old paths:

- Codex's and `agy`'s `comfy-local` MCP servers now launch
  `toolkits/l2m2/maintenance/run-comfy-mcp.sh` (backups in
  `_backups/siblings-20261015/`).
- `~/repos/l2m2-nsfw` runners default `L2M2_ROOT` to `~/repos/l3ms/toolkits/l2m2`
  (uncommitted edits in that repo for the owner to review).
- The dev worktree's `.toolkit`, `vendor` and `.venv-comfy` symlinks point at the
  deployed clones.

Verified after the move: ComfyUI venv (torch 2.14 + CUDA), the MCP handshake,
L3DS suites, root tests (543), and both toolkit binaries listing their profiles.

To undo: `mv ~/repos/_retired/l2m2 ~/repos/l2m2` (and `l3ds`). To remove for good,
delete `~/repos/_retired/`; full history and the uncommitted files are in
`~/repos/_backups/siblings-20261015/`.

`l2m2/vendor/audio.cpp` was a dangling symlink; the pinned binaries live in
`vendor/audio.cpp-955c872/` in the deployed checkout.
