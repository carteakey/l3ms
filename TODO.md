# TODO

Committed work is tracked as GitHub Issues on `carteakey/l3ms`, which is
authoritative. This file is the in-repo forward index that mirrors the open
issues, plus speculative or research notes that are deliberately not issues.
Linear was retired as the tracker on 2026-11-07; the eight `migrated-from-linear`
issues are the only surviving tickets from it, and `CAR-*` identifiers in older
notes are historical references. See the
[simplification plan](docs/migration/simplification-plan.md),
[migration status](docs/migration/status.md), and
[caller audit](docs/migration/upstream-first-audit.md) for scope and evidence.

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

## Remaining full-port acceptance gates

- [ ] ([#22](https://github.com/carteakey/l3ms/issues/22)) Verify HeartMuLa and H3 audio through the root CLI and `audiocpp_auto`
  with real weights; then verify language-model return. HTTP fixtures are not
  real-model acceptance. H3 video was removed from the root workbench.
  Blocked on this host (checked 2026-10-15): neither `~/models/media/HeartMuLa-GGUF`
  nor the H3 package is installed (the media root holds Breeze, FLUX and Qwen-Image
  only) and the NAS archive mount is unavailable. Nothing was downloaded; reinstall
  or restore the weights, then run the same four-step check used for Breeze
  (CLI run, decode the artifact, restore the language model, confirm no engine
  process remains).
- [ ] ([#23](https://github.com/carteakey/l3ms/issues/23)) Verify LTX and YuE2 ComfyUI workflow outputs with real weights;
  decode each artifact and retain settings/logs. Both share the existing ComfyUI
  engine; separate Python runtimes are not workbench prerequisites.
  Blocked on this host (checked 2026-10-15): no LTX or YuE2 weights exist under
  `~/models` or ComfyUI's model tree, and `/home` is 97% full (15 GB free), so
  they cannot be installed here without freeing space first.
  Do not call the full port complete until these advertised paths pass.

Optional follow-ups:
- [x] Decision (2026-10-15): keep the idle-GPU guard for Blender headless work.
  The wrapper runs arbitrary user Python, so the scene, not L3MS, picks the render
  backend; the guard is what keeps that from colliding with the router's VRAM. Revisit
  only if Blender is pinned to CPU in the wrapper itself.
- [ ] ([#24](https://github.com/carteakey/l3ms/issues/24)) Blender headless acceptance (the only profile still using the idle-GPU
  window) remains separate from the shared-engine workbench. Do not expand direct-script orchestration to fill
  unsupported model gaps by default; narrow the supported scope instead.
- [ ] ([#26](https://github.com/carteakey/l3ms/issues/26)) Finish source consolidation after shared-engine acceptance: keep one
  product/UI/config/history, relocate still-needed toolkit modules, audit callers
  before retiring standalone compatibility entry points. A folder rename alone
  is not runtime integration. Do not delete model/runtime data with source cleanup.
- [ ] ([#25](https://github.com/carteakey/l3ms/issues/25)) Measure `qwen38-flash-next-plat-abliterated-vision` so it can join the
  ranking; until then the site lists it as not measured.
- [ ] ([#30](https://github.com/carteakey/l3ms/issues/30)) Delete `~/repos/_retired/` once you are satisfied; history is bundled in
  `~/repos/_backups/siblings-20261015/`.
- [ ] ([#28](https://github.com/carteakey/l3ms/issues/28)) Comprehensive accessibility checks; the site already passes real
  desktop/mobile keyboard, focus and overflow checks
  ([browser record](docs/migration/site-browser-acceptance.md)).
- [ ] ([#27](https://github.com/carteakey/l3ms/issues/27)) Audit the exact outgoing history/export, endpoints, host paths, secrets and
  large files before publication. Publication remains **NO-GO**; no push or
  history rewrite without authorization.

## Strata tuning (plat tier)

Filed 2026-11-07 from `docs/strata-perf-next-steps-2026-10.md`. The blocking one is
[#13](https://github.com/carteakey/l3ms/issues/13): no tuning arm can be promoted or
rejected until warm-up is separated from steady state.

- [#13](https://github.com/carteakey/l3ms/issues/13) Run the blocking `kv_grow` ladder (p0)
- [#14](https://github.com/carteakey/l3ms/issues/14) Build an expert-cache-stable identity instrument (p0)
- [#15](https://github.com/carteakey/l3ms/issues/15) Quality-gate `STRATA_ROUTE_PRIOR=1.5` before serving (p0)
- [#16](https://github.com/carteakey/l3ms/issues/16) Build an expert profile from real traffic (p0)
- [#17](https://github.com/carteakey/l3ms/issues/17) Put the four E-cores to work (p1)
- [#18](https://github.com/carteakey/l3ms/issues/18) Prompt-side switches: prefill has not moved in a month (p0)
- [#19](https://github.com/carteakey/l3ms/issues/19) Reserve huge pages for the expert arena (p1)
- [#20](https://github.com/carteakey/l3ms/issues/20) Re-slice `strata-lean` onto upstream v0.2.x (p2)
- [#21](https://github.com/carteakey/l3ms/issues/21) Batching, only if two clients ever run at once (p3)
- [#31](https://github.com/carteakey/l3ms/issues/31) Verify and quality-gate `STRATA_ADAPT_FETCH` on the 4070 (p0)
- [#32](https://github.com/carteakey/l3ms/issues/32) Port the 0.2.11 server tool-call and conversation-slot fixes (p1)
- [#36](https://github.com/carteakey/l3ms/issues/36) Evaluate the DFlash2 drafter on the 4070 (p3)

Filed 2026-11-09 from the v0.2.12 intake: the CPU prefill share limit (3,072) and
`STRATA_ADAPT_FETCH=1` are served; the re-slice is a reference arm, not served.

## Serving stability

- [#33](https://github.com/carteakey/l3ms/issues/33) The CPU governor resets to `powersave` on resume (p1)

## Docs and publication

- [#34](https://github.com/carteakey/l3ms/issues/34) Add a prefill chart to the Strata post (p2)
- [#35](https://github.com/carteakey/l3ms/issues/35) Bring the self-tuning post up to the l3ms mirror and fix the bit-identical claim (p1)

## Workbench

- [#29](https://github.com/carteakey/l3ms/issues/29) Reconcile crate version, changelog and
  tags: `Cargo.toml` says 1.0.0, the changelog stops at 0.7.0, and there are no git tags.

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
