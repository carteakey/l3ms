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
| 1. Folder/history import | Pending | Import selected source histories and WIP under `toolkits/`; preserve originals; add README wordmark and navigation. |
| 2. Independent operation | Pending | Per-toolkit checks, path fixes, and CI. |
| 3. Runtime cutover | Pending | Live L3MS, L2M2, L3DS checks on the deployment host, with rollback. |
| 4. Shared contracts | Pending | Catalog, configuration, and job contracts. |
| 5. Runtime adapters | Pending | Manual, verified service transitions. |
| 6. Local coordinator | Pending | Cross-client queueing and request-driven switching. |
| 7. HTTP ingress | Pending | Gateway to the same coordinator. |
| 8. Unified workbench | Pending | Language, media/TTS, and decision UI/CLI flows. |
| 9. Public site | Pending | Separate evidence for all three lanes. |
| 10. Final acceptance | Pending | Live gates and old-path retirement review. |

## Current safeguards

- The original `~/repos/l3ms` checkout remains on `main` with its pre-existing
  tracked and untracked changes. The migration branch lives in a separate Git
  worktree; no runtime path or service has been switched.
- L2M2 and L3DS originals remain in place. Their source and ignored runtime
  state are being classified before history import.
- The source tree and plan are being reviewed for paths, secrets, and large
  generated files before the first push.

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
| 2026-09-28 | Dashboard data baseline | Three generator unit tests pass. Regeneration differs from the checked-in `docs/generated-models.js` in four command examples' reasoning-effort settings; this is pre-existing source/payload drift, not caused by the monorepo import. The generated file was returned to its checked-in state on this branch pending the site phase. |

## Open decisions

- Choose the host admission mode for direct llama-swap/ComfyUI requests before
  claiming automatic cross-engine switching. The plan allows coordination
  only where the runtime can block and drain new direct work, or where callers
  use exclusive coordinated ingress.
- Live GPU and external-route cutovers require host access and recorded
  before/after state. They are not implied by a successful build.
