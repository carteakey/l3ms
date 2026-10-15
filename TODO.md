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
- [x] llama-swap owns language, embeddings, decisions, speech (`audiocpp_auto`)
  and images (`comfyui_auto`); a full chat → Breeze on CUDA → ComfyUI → chat cycle
  passed on the live router ([evidence](docs/migration/media-through-llama-swap.md)).
- [x] Sibling repos `l2m2` and `l3ds` imported, repointed and moved to
  `~/repos/_retired/` ([record](docs/migration/sibling-retirement.md)).

Optional follow-ups, none blocking:

- [ ] Route `minimax-h3` and `heartmula-music` through `audiocpp_auto` (they are
  audio.cpp families). Needs its own CUDA run before replacing the verified
  scripts. LTX, YuE2 and Blender stay direct by design (see the same doc).
- [ ] Measure `qwen38-flash-next-plat-abliterated-vision` so it can join the
  ranking; until then the site lists it as not measured.
- [ ] Delete `~/repos/_retired/` once you are satisfied; history is bundled in
  `~/repos/_backups/siblings-20261015/`.
- [ ] Comprehensive accessibility checks; the site already passes real
  desktop/mobile keyboard, focus and overflow checks
  ([browser record](docs/migration/site-browser-acceptance.md)).
- [ ] Audit the exact outgoing history/export, endpoints, host paths, secrets and
  large files before publication. Publication remains **NO-GO**; no push or
  history rewrite without authorization.

## Retired prototype scope

The branch-added managed session, daemon/worker, private-engine transport,
duplicate chat dispatch, retained-ID coordinator clients and the resident
pause/restore handoff were all removed in the 2026-10-15 upstream-first
reduction. None is a pending task. Older status entries and
`docs/migration/history/` are historical evidence, not a plan to revive them.

## Speculative / research

- A native Rust/GPUI frontend remains optional and separate from the CLI. Only
  consider a bounded read-only prototype after full-port acceptance; preserve
  keyboard parity before replacing any existing UI.
