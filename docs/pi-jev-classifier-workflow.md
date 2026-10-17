# Jev classifier workflow (pi + codemode)

Operational workflow for using TypeSafe's **Jev** classifier model inside pi through
the `codemode` tool, applied to L3MS bench and serving decisions.

Jev does not chat. It answers **typed questions about JSON state** — `choice`
(pick one option), `bool`/noul (probability of yes), `score` (position on ordered
levels) — each with probabilities. Code owns evidence gathering, thresholds, and
actions; the model supplies the judgment only.

## One-time setup (done on KPC, 2026-09-30)

1. Credential in `~/.pi/agent/auth.json` (keep mode `600`, never commit):

   ```json
   { "typesafe": { "type": "api_key", "key": "apikey_…" } }
   ```

   Equivalent environment form: `export TYPESAFE_API_KEY=…` before starting `pi`.

2. Enable the `codemode` tool in `~/.pi/agent/settings.json`:

   ```json
   {
     "defaultTools": ["+codemode"],
     "codemode": { "mode": "on" }
   }
   ```

   `read`, `bash`, `edit`, `write` stay enabled. Run `/reload` (or restart pi) after
   editing settings or adding prompt templates.

3. TypeSafe skill: `~/.agents/skills/typesafe-ai/SKILL.md` (matches upstream
   `typesafe-ai/skills@main`). Pi discovers `~/.agents/skills/` automatically.

Verify:

```js
const avail = await models.getAvailableOfType("classifier");
// -> typesafe/jev-latest
```

## Commands

| Command | What it does |
|---|---|
| `/jev-bench [glob\|jsonl]` | Triage newest rows in `bench-models/logs/results/*.jsonl` plus the system frontmatter of the newest bench log. |
| `/jev-log [log path]` | Triage a server log (`llama.log` by default) plus router status/journal and a GPU/RAM snapshot. |

Both are thin prompt templates (`.pi/prompts/`) over two scripts:

- `maintenance/codemode/jev-bench-triage.js`
- `maintenance/codemode/jev-log-triage.js`

`.pi/prompts/` is a trust-protected project resource: if `/jev-bench` and
`/jev-log` do not appear after `/reload`, accept the project-trust prompt for
`/home/kchauhan/repos/l3ms` and reload again. The templates contain no secrets.

Each script is plain `codemode` JavaScript: gather evidence with `tools.bash`,
ask narrow questions, then apply thresholds in code and print a `VERDICT`,
`flags`, and `usage`.

Manual form (no template): read the script file, paste its contents into
`codemode`, and prepend the arguments:

```js
const ARGS = "bench-models/logs/results/qwen3-6-mtp-q6.jsonl";
const MODEL_OVERRIDE = "jev-latest";   // optional
const PROVIDER_ID = "typesafe";        // optional
```

## Judgments asked

`/jev-bench`: `steady_state` (warm-up transient vs settled), `evidence_strength`
(repetitions/probes/caveats), `comparable` (system frontmatter captured and
comparable), `memory_spill` (thrash or SSD re-faulting), `next_action`
(report / rerun / investigate / rebench_context / archive).

`/jev-log`: `serving_ready`, `failure_class` (oom / memory_pressure /
config_flag / crash_abort / thermal / none), `hot_reload_enough`, `next_action`
(none / sighup / restart_router / free_ram / fix_flags / investigate). The action
options mirror the llama-swap operations documented in `AGENTS.md`.

## Verified runs (2026-09-30, KPC)

Server triage (`llama.log` + router journal + GPU/RAM snapshot, 1,894 tokens):

```
serving_ready 0.89 | failure_class none 0.75 | hot_reload_enough 0.23
VERDICT: NONE (p=0.45, conf=0.35)
```

Bench triage (3 newest `*.jsonl` + bench log frontmatter, 3,325 tokens):

```
steady_state 0.58 | evidence_strength 0.15 | comparable 0.13 | memory_spill 0.30
VERDICT: REBENCH CONTEXT (p=0.60, conf=0.51)
flags: not steady state, weak evidence, context not comparable
```

The low `comparable` is a true signal, not a model error: those archived Gemma /
Qwen3-6 runs predate the `bench-models/bench-env.sh` frontmatter standardization,
so Jev has no governor/RAM/VRAM context to compare against.

## Cost, limits, and gotchas

- `typesafe/jev-latest` has no catalog price, so pi reports its tokens at **$0**
  while still counting them in `/session`. Other classifier routes
  (`openrouter/typesafe/jev-1.13`, `vercel-ai-gateway/typesafe-ai/jev`,
  `cloudflare-workers-ai/typesafe/jev`, `opencode/jev-1.13-free`) appear in the
  catalog once their provider credentials exist.
- Keep state small. These runs used 1.7k–3.3k input tokens; feeding whole logs
  costs tokens and blurs the judgment. Tail, don't cat.
- Confidence is distribution concentration, not correctness. `next_action`
  confidence under ~0.6 means "ask for more evidence", not "the answer is wrong".
- A `bool` near 0.5 means yes and no are equally supported — usually missing
  evidence, not medium intensity.
- Script gotcha: do not name a parameter lookup after its own `const`
  (`const MODEL_ID = (typeof MODEL_ID !== "undefined" …)`) — QuickJS throws
  `ReferenceError` from the temporal dead zone. Use `ARGS` / `PROVIDER_ID` /
  `MODEL_OVERRIDE` as the external names.
- llama-swap models are **not** classifier models here. `~/.pi/agent/models.json`
  declares llama-swap as a custom `openai-completions` provider, so it contributes
  chat models only. The "every chat model is also a classifier" behavior belongs
  to pi's built-in llama.cpp provider (`/login llama.cpp`, `llama-cpp-classify`).
  Local classification through this router would need a custom provider that
  registers a classifier API.

## Extending

Add a question when you would otherwise ask a human the same thing, keep it to one
narrow judgment, and give it explicit criteria with both poles described. Put new
thresholds in the script's code block, not in the prompt text, so weights and
cut-offs stay editable text. Snapshot the script before editing
(`jev-*.js.bak-YYYYmmdd-HHMMSS`) in line with the repo's snapshot rule.
