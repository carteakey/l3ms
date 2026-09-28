# L2M2 Phase 0/1 preparation

Recorded 2026-09-28 for the [consolidation plan](../monorepo-consolidation-plan.md).
This evidence covers L2M2 only; it does not complete the repository-wide Phase 0 gate.

## Decision

- **Local import: GO** using the scrubbed history merged at `4deece449968b2f0f35801d0b766ea1e7c0c48b0` on `feature/l3ms-monorepo`.
- **Public push: NO-GO pending path/privacy cleanup.** Retained legacy L2M2 docs and config contain `/home/kchauhan` paths, machine output paths, and local setup details. A repository-wide publication review is still needed.
- The source checkout remains untouched at `main`, `39e4d40a4e5ed993e99aa4ed70c9d183873a90ba`, with its original dirty state.

## Source and selected snapshot

Source: `/home/kchauhan/repos/l2m2`; branch `main`; HEAD
`39e4d40a4e5ed993e99aa4ed70c9d183873a90ba`. It has no configured remotes or
tags. Before and after preparation, the source had the same 10 modified tracked
files and 59 non-ignored untracked files. All 30 baseline tracked files and the
selected working-tree state were captured in an isolated clone; no ignored
runtime tree was copied.

The selected working-tree changes were:

- Modified tracked: `.gitignore`, `AGENTS.md`, `CHANGELOG.md`, `Cargo.toml`,
  `README.md`, `docs/media-generation-runbook.md`,
  `maintenance/setup-media-runtimes.sh`,
  `media-models/generate-heartmula-music.sh`,
  `media-models/generate-minimax-h3.sh`, and `media-runtimes.json`.
- New docs: `docs/2026-09-25-comfyui-mcp-tailscale.md`,
  `docs/audio-cpp-runtimes.md`, `docs/comfyui-setup.md`,
  `docs/experiments/{breeze-tts-2-baseline,h3-batman,h3-reference-baseline,qwen-vs-krea2-comparison}.md`,
  `docs/flux-2-klein-9b-setup.md`, `docs/gpu-inference-switching.md`,
  `docs/krea-2-setup.md`, `docs/minimax-h3-quality-notes.md`,
  `docs/qwen-image-2.1-setup.md`,
  `docs/sources/minimax-h3-batman-reddit-20260903.txt`, and
  `docs/z-image-turbo-setup.md`.
- New maintenance files: `maintenance/check-comfy-mcp.py`,
  `maintenance/comfy-cli.sh`, `maintenance/comfyui-service.sh`,
  `maintenance/refresh-audio-cpp.sh`, `maintenance/register-comfy-mcp.sh`,
  `maintenance/run-comfy-mcp.sh`, `maintenance/run-comfyui.sh`,
  `maintenance/run-h3-batman-isolated.sh`,
  `maintenance/setup-comfyui.sh`,
  `maintenance/setup-flux-2-klein-9b-comfyui.sh`,
  `maintenance/setup-krea-2-comfyui.sh`,
  `maintenance/setup-minimax-h3-comfyui.sh`,
  `maintenance/setup-qwen-image-2.1-comfyui.sh`,
  `maintenance/setup-z-image-turbo-comfyui.sh`,
  `maintenance/test-h3-batman.sh`, and `maintenance/test-h3-comfyui.py`.
- New media code: `media-models/generate-breeze-tts-2.sh`,
  `media-models/generate-krea-2.sh`, `media-models/generate-qwen-image-2.1.sh`,
  `media-models/{krea_2,qwen_image_21,test_krea_2,test_qwen_image_21}.py`.
- New data: five `models/*-comfyui-files.txt` inventory files, four prompt
  files under `prompts/`, two Python runtime locks under `runtime-locks/`, and
  11 ComfyUI workflow JSON files under `workflows/`.

## Filtered refs and backup

The full selected WIP snapshot is retained only for private recovery. A disposable
clone was rewritten with `git-filter-repo --to-subdirectory-filter
toolkits/l2m2`; its ref is `migration/l2m2-under-toolkit` at
`6d18d45d83860a33aecec797a7f140a10290c860` (3 commits, 89 files). It includes
the two docs excluded from the local integration candidate.

Durable selected-WIP bundle, outside `/tmp`:
`/home/kchauhan/repos/l2m2-migration-backup-20260928/l2m2-selected-private.bundle`
(mode `0600`; containing directory mode `0700`). `git bundle verify` passed and
reported a complete history with ref
`refs/heads/migration/l2m2-under-toolkit` at `6d18d45d83860a33aecec797a7f140a10290c860`.

The local import candidate is `/tmp/l2m2-phase1.PZPRZW/l2m2-public-safe`, ref
`migration/l2m2-public-safe`, HEAD
`c6154b7fb5b052b3a020c666513d597a60b5758b`. Its verified bundle is
`/tmp/l2m2-phase1.PZPRZW/l2m2-public-safe.bundle`. It retains all three source
commits, puts all 87 selected files under `toolkits/l2m2/`, and is clean. The
`git-filter-repo` commit maps are in each clone's `.git/filter-repo/commit-map`.

| Original snapshot commit | Full filtered ref | Local import candidate |
| --- | --- | --- |
| `e6d2d1dbd6886f6c998ec8afaf47b0001520f127` | `53e96de30af87889314872a6b980fbcbef1bfebe` | `53e96de30af87889314872a6b980fbcbef1bfebe` |
| `39e4d40a4e5ed993e99aa4ed70c9d183873a90ba` | `5be2d7d06817f0832445fd7a71cc035bcca54f94` | `5be2d7d06817f0832445fd7a71cc035bcca54f94` |
| Synthetic WIP snapshot `b3221e935b11e93ecbda495200411c3d9cd12972` | `6d18d45d83860a33aecec797a7f140a10290c860` | `c6154b7fb5b052b3a020c666513d597a60b5758b` |

The local candidate intentionally removes these paths from every reachable
commit; both remain in the private recovery bundle and original checkout:

- `toolkits/l2m2/docs/sources/minimax-h3-batman-reddit-20260903.txt` is a
  540-line transcript containing a public post and many verbatim third-party
  comments and handles. Keep the full transcript out of the public history.
- `toolkits/l2m2/docs/2026-09-25-comfyui-mcp-tailscale.md` records a public
  ComfyUI Funnel endpoint, host and hardware identifiers, absolute home paths,
  and live service setup. It contains no credential found by the scans below,
  but should remain private or be rewritten as sanitized operator guidance.

## Audit and runtime inventory

- History-wide scans covered the original two commits and the selected snapshot
  history. No common token patterns (`hf_`, `sk-`, GitHub/AWS tokens, bearer
  values, or credential assignments) were found. The scrubbed history passed
  `git fsck --no-reflogs --connectivity-only`.
- The largest blob in the scrubbed history is a 69,831-byte workflow JSON; the
  history packs to about 160 KiB. No model weights, large generated blobs, or
  binary weight files are in the import candidate. The five `models/*.txt`
  files are path inventories, not weights.
- Ignored local state left in the original checkout: `.venv-comfy/` (6.5 GiB),
  `.venv-comfy-mcp/` (228 MiB), `target/` (293 MiB), `vendor/` (324 MiB),
  `.toolkit/` (624 KiB), and Python `__pycache__` directories. `.toolkit/`
  contains config/script snapshots and backups, ComfyUI user database/settings,
  copied user workflows, dependency locks, and diagnostic logs. Do not import
  these files or venv/build trees.
- `vendor/ComfyUI/models/` has 21 ignored `.safetensors` symlinks: seven point
  into `/home/kchauhan/models/media/` and fourteen into
  `/run/media/kchauhan/Games/models/media/`. The external weights remain outside
  Git; the second volume's targets were not verified as mounted. Also ignored:
  `vendor/audio.cpp -> audio.cpp-master`, with `audio.cpp-master` targeting
  `/home/kchauhan/repos/audio.cpp`; venv interpreters point into uv's managed
  Python tree and each `lib64` points to its local `lib` directory.
- The source `.gitignore` WIP unignored `/models/`, which would expose later
  weights to Git alongside inventory files. After import, L3MS tightened the
  child rule to ignore `/models/**`, allow the `models/` directory, and unignore
  only direct `*-files.txt` inventories. Read-only `git check-ignore` confirmed
  example `.safetensors` and nested `.gguf` paths are ignored. Inventory `.txt`
  files remain visible to Git.
- `git diff --cached --check` on the exact source snapshot reports existing
  Markdown hard-break whitespace and a few extra blank lines at EOF. The
  import preserved source content without formatting cleanup.

## Phase 2 path check

The imported child README now starts its build instructions with `cd
toolkits/l2m2` from the monorepo root and explains Cargo's compile-time default
root plus the `--root` override. The release binary was run from the monorepo
root on 2026-09-28:

- `./toolkits/l2m2/target/release/l2m2 --list` used its compiled default and
  listed seven profiles.
- `./toolkits/l2m2/target/release/l2m2 --root /home/kchauhan/repos/l2m2 --list`
  loaded the separate original checkout and listed seven profiles.
- `git diff --check -- README.md CHANGELOG.md` passed. No inference job or
  runtime switch was started. No new test was needed; the compiled CLI was
  exercised directly.

## Remaining gates

- The safe local ref is ready and was merged at `4deece449968b2f0f35801d0b766ea1e7c0c48b0`.
- Before a public push, review and sanitize remaining absolute paths in legacy
  files such as `HANDOFF.md`, `DETAILED_SESSION.md`, `model_downloader/` docs,
  and experiment summaries. The credential scan used known patterns and is not
  a substitute for human review.
- Do not move, delete, or recreate the original ComfyUI/Audio vendor checkouts,
  external weight files, runtime venvs, or `.toolkit` state as part of import.
