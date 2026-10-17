---
description: Triage L3MS bench results with TypeSafe Jev (codemode)
argument-hint: "[glob or jsonl path]"
---

Read `maintenance/codemode/jev-bench-triage.js` with the read tool, then run its exact contents through the `codemode` tool, inserting this as the first statement after the `// @options:` line:

```js
const ARGS = "${@:-bench-models/logs/results/*.jsonl}";
```

Report verbatim from the script output: `VERDICT`, `flags`, the raw answer probabilities, and `usage`. Do not soften or re-interpret Jev's numbers. If `next_action.confidence` is below 0.6, say the call is weak and name which evidence is missing instead of guessing.
