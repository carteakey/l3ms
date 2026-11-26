# Workbench checkpoint — 2026-10-17

## Saved baseline

- Checkout: `/home/kchauhan/repos/l3ms`, branch `main`.
- Implementation HEAD: `be74d5d9deb1ed09c409f407233e8e14c99e4c85`.
- Working tree was clean before this checkpoint. The former feature worktree
  is no longer registered; its UI work was merged in `064abab`.
- `0ece7ff` committed the previously pending bench, media, decision and agent
  tooling changes. Do not restore the earlier dirty-tree snapshot over them.
- No services, models or runtime configuration were changed for this checkpoint.

## Implemented

- Source consolidation into one repository is complete.
- Workbench shell: sidebar, toolbar, single pane, compact status bar, Model
  details, engine grouping, router-state refresh and explicit load confirmation.
- Reload router utility: confirmation, job output and journal-based completion
  check (`be74d5d`). Implementation is committed; this checkpoint did not perform
  a live reload or validate restoration.

## Verification at this checkpoint

`cargo test --quiet` on the implementation HEAD failed:
**584 passed, 3 failed, 4 ignored** (591 tests).

Failures:

- `control::media_execution::tests::cancellation_and_deadline_clean_up_the_owned_child_group`:
  `OwnershipUncertain` instead of `ChildGroupReaped`.
- `download_preflight::tests::timeout_kills_descendant_that_keeps_output_pipes_open`:
  pipe-holding descendant still exists.
- `app::tests::cpu_decision_cancel_keeps_job_busy_until_cleanup_and_rejects_a_second_form`:
  `direct_native_admission_closed` remained set.

These are observed failures, not diagnosed regressions. The previous result of
585 passed / 4 ignored applied to the older feature-branch HEAD, not this baseline.
No fmt, clippy, Python or real-model acceptance gates were rerun here.

## Resume here

1. Diagnose the three process-cleanup test failures before calling the baseline
   green. Do not weaken ownership/admission safety to make tests pass.
2. Continue step 2 of `workbench-plan.md`: safe Bench, Tune with conflict-aware
   snapshot/revert, side-by-side engine updates, downloads, Health and media setup.
   Reuse existing commands and runners; Tune remains a disabled toolbar entry.
3. Replace redundant local telemetry with safe router monitoring where supported.
   Never automatically poll autoloading `/upstream/` routes.
4. Reconcile stale TODO acceptance notes with installed assets and the completed
   source fold. Installed weights are not proof of successful generation.
5. Retain real-model media acceptance as a separate gate; no new acceptance was
   performed for HeartMuLa/H3/LTX/YuE2 during this checkpoint.

Release scope remains steps 1–2: select a model, bench/tune/run media, inspect
outputs/history, safely revert. GPUI, splits and crate restructuring are optional.
Keep llama-swap responsible for engines; keep L3MS small. No push or publication
is authorized by this checkpoint.
