# L3MS interface direction — Signal / 03

First visual implementation: `docs/index.html`. This is the profile/evidence
website, not a live control panel. No model orchestration moved into the browser.

## Character

An inference instrument, not a hacker-movie terminal. Dark blue-graphite surfaces,
quiet grid registration, warm phosphor highlights, slanted L3MS lettering and
three lane signatures: language = phosphor, media = mineral cyan, decisions =
amber. The shared shell stays consistent; lane accents indicate location, never
invent runtime status. Hollow snapshot markers are not live-health indicators.

Use thin rules, square panels, compact monospace labels and generous grouping.
Reserve colour for selection, evidence, focus and actual state; state must also
have a readable label. Avoid falling code, pulsing/glowing status decorations,
scanline overlays, gratuitous animation and illegible dim secondary text.

User refinement: retain the atmosphere, but make the identity specific to L3MS,
not a Matrix imitation. Explore ink-black surfaces, phosphor mint and icy violet
as a bounded palette refinement, with fine routing lines that make the stack's
three lanes recognizable. A future live client may highlight an actual observed
route while it executes; the static profile website must not simulate that
activity. This direction is recorded, not yet a verified palette/UI change.

CSS custom properties in the dashboard are the initial palette source. Keep
text/secondary text and functional accent colours at 4.5:1 on the darkest and
lightest panel surfaces. Keyboard focus has a separate high-contrast token;
respect reduced motion. Do not declare comprehensive accessibility acceptance
from palette checks alone.

## Native Rust / GPUI direction

Recommendation, not an implemented pivot: consider a native frontend after the
workbench's full-port acceptance. GPUI is a GPU-rendered Rust UI framework
([official examples](https://gpui.rs/examples/)); that makes it worth a bounded
prototype, not a reason to replace verified orchestration.

Reuse the existing catalog, settings, history and concrete native operations.
llama-swap owns supported routing and swapping; direct scripts retain their
existing admission and cleanup safeguards. The retired coordinator is not a
frontend prerequisite. A GPUI application would be another caller alongside the
CLI/TUI, not a second runtime owner. UI code must
not spawn engines, rewrite manifests without snapshots, retry an ambiguous
submission, or hold the UI thread while waiting on inference.

Prototype sequence:

1. Pin an upstream GPUI revision and prove build/launch on the target Linux
   desktop in a separate optional package; do not add GPUI to the CLI build.
2. Port Signal / 03 tokens, lane navigation, command palette and keyboard focus.
3. Connect read-only profile discovery, job history and native result views
   through existing operations. Distinguish unavailable, queued,
   active, completed and uncertain states; no fabricated charts or health.
4. Reuse single-shot submissions and cancellation with the same route validation,
   native job identity, bounded results and uncertainty behavior as CLI/TUI.
5. Verify two-client behaviour, Unicode/IME, clipboard, keyboard navigation,
   text scaling and desktop resource usage. GPU rendering must not weaken the
   inference GPU ownership policy; measure its actual claimant/resource impact.
6. Keep the CLI/TUI and profile website available until native parity is proven.

The current change establishes a style on the website only. TUI palette
alignment and the native prototype remain future work. A GPUI migration is not
part of declaring the ongoing inference port complete.

## First-pass verification

On 2026-10-08, ten dashboard Python tests and the Node DOM fixtures passed.
The real Firefox probe passed pointer/tab/keyboard/focus checks at 1440px and
390px, with seven media cards, six decision cards, no reported app console
exceptions, and verified disposable-profile/process cleanup. Root visually
reviewed desktop media and mobile decisions screenshots from
`.toolkit/dashboard-probe-lqqvly/`. Palette tests cover text, secondary text,
all four functional accents and focus against the three surface tokens.
This is not a full accessibility audit or native-app acceptance.
