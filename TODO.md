# TODO

All intentionally committed TODO items were reconciled into the single Linear
`L3MS` project and its `Operations backlog completion — 2026-08-10` milestone.
Linear remains authoritative for committed work. This file is now only a concise
forward index; see the [simplification plan](docs/migration/simplification-plan.md),
[migration status](docs/migration/status.md), and [caller audit](docs/migration/upstream-first-audit.md)
for scope and evidence.

## Monorepo: next work

- [x] Native decision-image check: d3 and Clef text and image cases pass through
  isolated v265 `/v1/systemone` (sanity cases, not quality evaluation).
- [x] Comfy scoped cancellation and a learned Krea CUDA graph pass through the
  opt-in v265 router recipe ([live receipts](docs/migration/live-port-acceptance.md)).
- [x] Blender root CLI CPU render passes. Open: clean GUI shutdown
  ([Blender record](docs/migration/blender-acceptance.md)).
- [x] Live language → CUDA speech → language continuation passed in the approved
  window; service and model restored.
- [ ] Finish remaining cross-lane runtime and keyboard/visual checks, including
  Comfy and GPU workflows. Committed standalone-source parity, source-export
  builds and toolkit tests pass; remaining runtime checks are separate. Remote
  CI and comprehensive accessibility remain release/quality checks. Cached CPU
  speech and embedding acceptance cover those paths only,
  not full-port acceptance. See the [speech record](docs/migration/direct-speech-acceptance.md).
  The post-retirement 80×24 decision-image form, Chat capability selection and
  metadata-only history pass against a private synthetic endpoint; comprehensive
  accessibility checks remain open. The current
  site passes real desktop/mobile pointer, keyboard, focus and overflow checks
  with all 8 media and 9 decision cards; see the
  [browser record](docs/migration/site-browser-acceptance.md).
- [ ] Audit the exact outgoing history/export, endpoints, host paths, secrets and
  large files before publication. Publication remains **NO-GO**; no push or
  history rewrite without authorization.

## Retired prototype scope

The branch-added managed session, daemon/worker, private-engine transport and
duplicate chat dispatch were retired in the 2026-10-15 upstream-first reduction.
This is not a pending implementation task. Retained-ID observe/cancel/results
and history/artifact reads remain compatibility clients for explicitly supplied
older servers; direct-workflow safety and on-disk records remain. Older status
entries are historical implementation evidence, not a plan to revive the retired
platform.

## Speculative / research

- A native Rust/GPUI frontend remains optional and separate from the CLI. Only
  consider a bounded read-only prototype after full-port acceptance; preserve
  keyboard parity before replacing any existing UI.
