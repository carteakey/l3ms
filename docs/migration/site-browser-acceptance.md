# Local dashboard browser acceptance

The connected browser surface was unavailable, but read-only host discovery on
2026-10-07 found Firefox 157.0.1. Isolated headless Firefox profiles rendered the
actual local dashboard at desktop 1440×1600 and mobile 390×1400 without using the
operator's browser profile, publishing the site or changing inference services.

The root reviewer inspected both language-panel screenshots: the header,
three-lane navigation and model list rendered; mobile layout stacked rather
than overflowing the viewport. This is visual evidence for those two initial
views, not yet real tab interaction, keyboard/focus, every lane, full-page layout
or comprehensive accessibility acceptance.

The review found secondary text below the normal-text contrast threshold on
the dashboard's dark backgrounds. `--muted` now uses `#888`, with an offline
regression checking at least 4.5:1 against all three configured panel backgrounds.
See the [W3C contrast criterion](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html).
Ten data/contrast tests and existing JavaScript DOM fixtures pass. Screenshot
artifacts and disposable profiles remain in ignored branch-local storage.

Real tab/click/keyboard/focus/overflow checks in isolated Firefox are the next
gate. Browser screenshots do not prove installed engines, model accuracy,
coordinated GPU switching or public deployment.
