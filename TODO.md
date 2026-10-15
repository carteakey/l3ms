# TODO

All intentionally committed TODO items were reconciled into the single Linear
`L3MS` project and its `Operations backlog completion — 2026-08-10` milestone.
Linear remains authoritative for committed work. This file is now only a concise
forward index; see the [simplification plan](docs/migration/simplification-plan.md),
[migration status](docs/migration/status.md), and [caller audit](docs/migration/upstream-first-audit.md)
for scope and evidence.

## Monorepo: next work

- [ ] Finish the native decision-image check. Actual CPU text cases for d3 and
  Clef pass, and the d3 image case passes. The Clef image attempt used an invalid
  question and returned HTTP 400; its preflight is fixed, but that model request
  has not been rerun. These are sanity cases, not quality evaluation.
- [ ] Recheck Comfy cancellation through the opt-in v265 router recipe under a
  quiet CPU fixture. A bounded `EmptyImage` → `SaveImage` job and artifact passed;
  an earlier cancellation passed, but the latest run's cancel observation stayed
  `in_progress` under load. Do not claim cancellation parity, GPU acceptance, or
  learned-model acceptance from these runs.
- [ ] Finish Blender acceptance at the intended boundaries. Standalone CPU-only
  headless rendering and child cleanup pass; root CLI dispatch, GPU, external GUI
  and MCP handoff are not verified. Keep CPU/GPU claims explicit.
- [ ] With the operator, arrange any live GPU or language → media → language
  check after Pi's GPU window is clear. Read-only checks already confirmed the
  installed/running llama-swap v265 build; no model was dispatched. Verify
  cancellation, artifacts and restoration only in an approved window. This
  checklist does not authorize service changes or downloads.
- [ ] Finish remaining cross-lane product checks, including Comfy and GPU
  workflows, keyboard/visual/accessibility, standalone/import and
  remote CI. Cached CPU speech and embedding acceptance cover those paths only,
  not full-port acceptance. See the [speech record](docs/migration/direct-speech-acceptance.md).
  The post-retirement 80×24 decision-image form, Chat capability selection and
  metadata-only history pass against a private synthetic endpoint; real Clef
  image inference and comprehensive accessibility checks remain open. The current
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
