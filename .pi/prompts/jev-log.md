---
description: Triage llama-swap / llama-server logs with TypeSafe Jev (codemode)
argument-hint: "[log path]"
---

Read `maintenance/codemode/jev-log-triage.js` with the read tool, then run its exact contents through the `codemode` tool, inserting this as the first statement after the `// @options:` line:

```js
const ARGS = "${@:-llama.log}";
```

Report verbatim from the script output: `VERDICT`, `failure_class`, `serving_ready`, the raw answer probabilities, and `usage`. Do not run any router command from the verdict without confirming it with me first; the script only recommends an action.
