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

The opt-in dependency-free browser probe is now available:

```sh
node docs/probe_dashboard_browser.cjs --run
```

After root code review and corrections, the root rerun passed real pointer tab
clicks, ArrowRight/Home/End keyboard navigation and focus assertions. Media and
decision panels rendered 7 and 6 cards, respectively; neither the document nor
lane cards overflowed horizontally at 1440px or 390px. A controlled pre-dashboard
fixture proved that the native event subscription observes both `console.error`
and an uncaught exception; the dashboard then produced no such events.

The static server and browser automation endpoint were loopback-only. Exact live
child-handle shutdown and read-only process-tree/group checks verified cleanup
before removing the fresh profile. Root-run screenshots remain in ignored
`.toolkit/dashboard-probe-4RkvVF/`; earlier image review covered all four desktop
and mobile media/decision views. No original browser profile or inference service
was changed. The harness retains the disposable profile if cleanup is uncertain.

These checks close the specific multi-lane visual/keyboard/mobile gates exercised
here, not comprehensive screen-reader/accessibility certification or every public
link's deployment. Browser screenshots do not prove installed engines, model
accuracy, coordinated GPU switching or public deployment.
