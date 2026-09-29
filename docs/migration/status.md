# L3MS monorepo migration status

Working branch: `feature/l3ms-monorepo` in the isolated checkout
`/home/kchauhan/repos/l3ms-monorepo`.

Plan: [monorepo-consolidation-plan.md](../monorepo-consolidation-plan.md).
This record tracks implementation evidence. A phase is complete only after its
gate in the plan is met; a passed offline test does not stand in for a live
runtime check.

Linear tracking: [CAR-215](https://linear.app/carteakey/issue/CAR-215/consolidate-l3ms-l2m2-and-l3ds-into-the-local-3-mode-model-stack)
is the parent; [CAR-216](https://linear.app/carteakey/issue/CAR-216/import-l2m2-and-l3ds-into-the-l3ms-monorepo-with-verified-standalone)
covers Phases 0–3, CAR-217 Phases 4–5, CAR-218 Phases 6–7, CAR-219 Phase 8,
CAR-220 Phase 9, and CAR-221 Phase 10.

| Phase | State | Evidence / remaining gate |
| --- | --- | --- |
| 0. Baseline and protection | In progress | Original L3MS `68fd7bc48f803aecb5ae5d25a3f30967b46da3eb`; L2M2 `39e4d40a4e5ed993e99aa4ed70c9d183873a90ba`; L3DS `4716ddf3508d224645416f3ba70d5e34c138cedd`. Source inventories and backups in progress. |
| 1. Folder/history import | In progress | L3DS five-commit and L2M2 three-commit histories imported under `toolkits/`; source/WIP audit, private recovery, and README wordmark done. Final tree/secret/public-path review pending. |
| 2. Independent operation | In progress | Imported child builds, tests, profile lists, and explicit old-root lists pass; CI added locally, pre-push validation pending. |
| 3. Runtime cutover | In progress | Bounded L2M2 GPU-mode pilot from the branch passed and original service was restored. Representative model generation, L3MS router, and real L3DS inference still pending. |
| 4. Shared contracts | In development | Read-only catalog and offline `l3ms --list stack`, layered settings, and existing-store job metadata are on the branch. Full compatibility/live release gates remain; release stays gated by Phase 3. |
| 5. Runtime adapters | Pending | Manual, verified service transitions. |
| 6. Local coordinator | Pending | Cross-client queueing and request-driven switching. |
| 7. HTTP ingress | Pending | Gateway to the same coordinator. |
| 8. Unified workbench | Pending | Language, media/TTS, and decision UI/CLI flows. |
| 9. Public site | In development | Local static site now has separate language, media/TTS, and decision views generated from native manifests; six generator/site checks pass. Visual/browser review and all publication gates remain pending. |
| 10. Final acceptance | Pending | Live gates and old-path retirement review. |

## Current safeguards

- The original `~/repos/l3ms` checkout remains on `main` with its pre-existing
  tracked and untracked changes. The migration branch lives in a separate Git
  worktree; no runtime path or service has been switched.
- L2M2 and L3DS originals remain in place. Their source and ignored runtime
  state are being classified before history import.
- The source tree and plan are being reviewed for paths, secrets, and large
  generated files before the first push. The current
  [publication audit](publication-audit.md) is **NO-GO** for a public push:
  retained imported history still exposes a Funnel endpoint and host-specific
  paths. Keep this branch local until the exact outgoing history is sanitized
  and re-audited.

## Mutable runtime state (not Git migration inputs)

| State | Current location / writer | Pilot strategy and rollback |
| --- | --- | --- |
| L3MS vendor, weights, `.toolkit`, user state | Original `~/repos/l3ms` and external model roots; no active llama-swap service at baseline | Keep root deployment paths unchanged; reconcile dirty config before any deployment. Restore the saved config/service revision if the pilot fails. |
| ComfyUI vendor, user data, outputs, and model symlinks | Original `~/repos/l2m2/vendor/ComfyUI` and `.toolkit`; the active `l2m2-comfyui.service` is the writer | Snapshot mutable user/workflow state, verify symlink targets, and reconnect the imported child only in an exclusive pilot window. Stop old writer before starting new; preserve pilot writes and return the unit to the original checkout on rollback. |
| L2M2 Python environments | Original `.venv-comfy/` (6.5 GiB) and `.venv-comfy-mcp/` (228 MiB); no Git owner | Rebuild or reconnect by a verified path at the new child; do not copy venvs blindly because interpreter paths are absolute. Keep originals until acceptance. |
| L3DS Python environment and model cache | Original `.venv/` (5.1 GiB) and shared host Hugging Face cache (36 MiB, no snapshots) | Rebuild only after resolving dependency conflicts; leave the original venv/cache intact. No local model inference acceptance until a real model is available. |

## Evidence log

| Date | Check | Result |
| --- | --- | --- |
| 2026-09-28 | Created isolated L3MS worktree from `68fd7bc` | `feature/l3ms-monorepo`; deployed checkout untouched. |
| 2026-09-28 | Copied the user's revised plan into the branch | SHA-256 matched the source plan before the README-art amendment. |
| 2026-09-28 | L3MS baseline `cargo test --all-targets --locked` in original checkout | Passed: 258 tests, 1 ignored. |
| 2026-09-28 | L3MS baseline lint/Python checks | `cargo clippy --all-targets --locked -- -D warnings` passed; 14 downloader unit tests passed. `cargo fmt --all -- --check` failed on pre-existing formatting drift in clean L3MS source files. |
| 2026-09-28 | Branch formatting recovery | Formatted five L3MS source files (18 insertions, 17 deletions); root `cargo fmt --all -- --check` now passes on the branch. Original checkout remains unchanged. |
| 2026-09-28 | Read-only live host baseline | `l2m2-comfyui.service` active from the original L2M2 checkout in GPU mode; `http://127.0.0.1:8188/system_stats` responds. `llama-swap.service` inactive; ports 8080/8001 do not respond. RTX 4070 reports 12,282 MiB total and 362 MiB used. No service was switched. |
| 2026-09-28 | L3MS recovery snapshot | Verified Git history bundle, tracked WIP patch, and archive of 12 untracked files under `/home/kchauhan/repos/l3ms-migration-backup.Z0zIPT/`. Ignored runtimes and weights remain in the original checkout. |
| 2026-09-28 | Dashboard data baseline repair | Three generator unit tests pass. Regeneration initially differed from the checked-in `docs/generated-models.js` in four command examples' reasoning-effort settings; this was pre-existing source/payload drift, not caused by the monorepo import. The branch now commits the deterministic regenerated payload to keep existing dashboard CI green. |
| 2026-09-28 | L3DS import | Audited disposable five-commit history merged as `ee27a64`; 15 tracked paths and blob IDs match source under `toolkits/l3ds/`. Full-history high-signal pattern and size audit found no hits; dedicated secret scanner remains a pre-push safeguard. |
| 2026-09-28 | L3DS recovery bundle | Original complete history bundled and verified at `/home/kchauhan/repos/l3ms-migration-backup.Z0zIPT/l3ds-history.bundle`; source checkout remains clean. |
| 2026-09-28 | L3DS child operation | `cargo build --locked`, `cargo test --locked` (0 tests), `l3ds --list` (6 profiles), and `l3ds --root /home/kchauhan/repos/l3ds --list` passed from the imported child; child target output is ignored. No decision inference was claimed. |
| 2026-09-28 | Root regression after first child import | `cargo test --all-targets --locked` passed: 258 passed, 1 ignored. |
| 2026-09-28 | Root lint after first child import | `cargo clippy --all-targets --locked -- -D warnings` passed. |
| 2026-09-28 | L3DS child path follow-through | Documented child-directory commands and compile-time default root; added two offline tests for the default root and `--root` override. Child fmt, offline test (2 passed), Clippy, build, and `--list` pass. |
| 2026-09-28 | L2M2 import | Audited public-safe three-commit history merged as `4deece4`; 87 source/evidence files under `toolkits/l2m2/`. A third-party transcript and a host-specific Funnel note were excluded from every imported commit and retained in the verified private WIP backup at `/home/kchauhan/repos/l2m2-migration-backup-20260928/l2m2-selected-private.bundle`. No weights/bulk binaries in imported history. |
| 2026-09-28 | L2M2 child operation | `cargo fmt --check`, Clippy, Rust tests (4 passed), release build, 7-profile list, Python syntax, 14 downloader tests, 9 media-model tests, shell syntax, and explicit old-root list passed. No inference or runtime switch was attempted. |
| 2026-09-28 | Model ignore safeguard | Imported child `.gitignore` now ignores GGUF/safetensors and other `models/` contents while allowing only direct `*-files.txt` inventory files; verified with `git check-ignore --no-index`. |
| 2026-09-28 | L2M2 live pilot and rollback | After user approval, confirmed ComfyUI queue empty and llama-swap inactive; stopped original transient unit, archived 81 `.toolkit` entries to a verified mode-600 backup at `/home/kchauhan/repos/l2m2-migration-backup-20260928/l2m2-toolkits-prepilot.tar.gz`, and started a separate `l2m2-comfyui-monorepo-pilot` unit from the branch with ignored symlinks to original vendor, venv, and `.toolkit`. ComfyUI 0.36.0 served `127.0.0.1:8188` in GPU mode, exposed 11 saved workflows, and completed a 64×64 `EmptyImage` → `SaveImage` job (`65b9ccfe-6ce0-4ac5-acdb-6f8f497eeaaa`), producing `l3ms-monorepo-pilot_00001_.png` in the normal media output directory. This verifies service-path wiring, not a heavyweight model workflow. Pilot queue was empty before stop. Original `l2m2-comfyui.service` was restarted and is active/healthy with its original user-directory path; pilot unit is inactive. |
| 2026-09-28 | Isolated llama-swap probe | A loopback-only router with the branch config parsed and listened on port 18080, but its declared preload failed because the branch worktree has no `vendor/llama.cpp` (the deployed root still owns ignored vendor state). The probe was shut down; ComfyUI remained active. Branch L3MS `--quickstart` and `--list bench` passed; `--list run` still requires a live router. No language inference is claimed. |
| 2026-09-28 | Phase 4 offline development | Read-only three-lane manifest catalog, layered stack settings in the existing settings store, and cross-toolkit metadata in the existing job-history store are being implemented/tested. No new inference or service action is exposed by these contracts. |
| 2026-09-29 | Phase 4 offline catalog and CLI | `l3ms --list stack` lists 21 lane-qualified profiles from the three native manifests without contacting llama-swap; each is explicitly `manifest-only`, not installed/healthy. It works from another working directory with `L3MS_ROOT` set. Root `cargo test --all-targets --locked` passed 284 tests (1 ignored); root fmt, Clippy, and dashboard generator checks passed. |
| 2026-09-29 | Phase 4 settings and jobs | Existing settings store now has versioned optional stack preferences, declared path anchoring, built-in → host → user → environment → CLI precedence, snapshots on writes, and portable profile exports that omit machine roots/policy. Existing job history now supports queued cross-toolkit metadata, explicit pre-dispatch persistence, transitions/timestamps/artifacts, and no-replay interruption on restart while preserving legacy row shape. Targeted tests pass; final full-suite check pending after integration. |
| 2026-09-29 | Phase 9 site development | Added native-manifest-derived media/TTS and decision catalog payloads, separate site tabs, Breeze's four TTS modes, hosted/local decision labels, and explicit absence of measured results. Existing language ranking remains separate. Six Python tests, JavaScript syntax, and whitespace checks pass. No browser surface was available for visual review; no site was published. |

## Open decisions

- Choose the host admission mode for direct llama-swap/ComfyUI requests before
  claiming automatic cross-engine switching. The plan allows coordination
  only where the runtime can block and drain new direct work, or where callers
  use exclusive coordinated ingress.
- Live GPU and external-route cutovers require host access and recorded
  before/after state. They are not implied by a successful build.
