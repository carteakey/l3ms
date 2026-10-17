// @options: {"max_output_tokens": 3000}
// L3MS bench-result triage with TypeSafe Jev, run through pi's `codemode` tool.
//
// Run from a pi session:
//   /jev-bench                                  (newest rows under bench-models/logs/results)
//   /jev-bench bench-models/logs/results/qwen3-6-mtp-q6.jsonl
// or paste this file's contents into `codemode` with, as the first statement:
//   const ARGS = "<glob or path>"; const MODEL_OVERRIDE = "jev-latest";
//
// Code owns evidence gathering, thresholds, and the verdict. Jev only answers the
// typed questions. Keep questions narrow; add a question when you would otherwise
// ask a human "should I trust this run?".

const target = (typeof ARGS !== "undefined" && ARGS) || "bench-models/logs/results/*.jsonl";
const provider = (typeof PROVIDER_ID !== "undefined" && PROVIDER_ID) || "typesafe";
const modelId = (typeof MODEL_OVERRIDE !== "undefined" && MODEL_OVERRIDE) || "jev-latest";
const ROOT = "/home/kchauhan/repos/l3ms";

const model = await models.getModelOfType("classifier", provider, modelId);
if (!model) return `classifier ${provider}/${modelId} is not in the catalog or has no credentials`;

// Evidence: newest result rows plus the system frontmatter of the newest bench log.
const listing = await tools.bash({
  command: `cd ${ROOT} && ls -t ${target} 2>/dev/null | head -n 3`,
});
const files = listing.output.trim();
if (!files) return `no files matched: ${target}`;

const rows = await tools.bash({
  command: `cd ${ROOT} && for f in ${files.split("\n").join(" ")}; do echo "== $f"; tail -n 2 "$f"; done | head -c 4200`,
});
const frontmatter = await tools.bash({
  command: `cd ${ROOT} && f=$(ls -t bench-models/logs/*.log 2>/dev/null | head -n 1); [ -n "$f" ] && echo "== $f" && head -c 1200 "$f"`,
});

const state = {
  bench_result_rows: rows.output,
  bench_log_system_frontmatter: frontmatter.output || "(no bench log frontmatter found)",
};

const r = await models.classify(model, {
  state,
  questions: {
    steady_state: {
      type: "bool",
      instructions:
        "Do these rows represent steady-state decode rather than a warm-up transient from a fresh model load (ngram-mod pool, first probes)?",
      criteria: {
        true: "Numbers look settled across probes",
        false: "Numbers look like a warm-up transient or a single fresh-load probe",
      },
    },
    evidence_strength: {
      type: "bool",
      instructions:
        "Is the recorded evidence strong enough to quote as this model's number (repetitions, probe count, and stated caveats)?",
      criteria: {
        true: "Enough probes/repetitions and no unresolved caveat",
        false: "Single repetition, missing probes, or an unresolved caveat in the notes",
      },
    },
    comparable: {
      type: "bool",
      instructions:
        "Can these numbers be compared against other runs on this box as recorded, given the captured system frontmatter (CPU governor/EPP, RAM state, VRAM headroom)?",
      criteria: {
        true: "System context is captured and looks comparable",
        false: "System context is missing or looks different from a normal performance run",
      },
    },
    memory_spill: {
      type: "bool",
      instructions:
        "Does the record suggest memory spill or SSD re-faulting of expert/PLE pages during generation?",
      criteria: {
        true: "Signs of thrash, spill, fit/override pressure, or decode collapse",
        false: "No signs of spill or thrash",
      },
    },
    next_action: {
      type: "choice",
      instructions: "What is the most useful next step for this measurement?",
      criteria: {
        report: "Report it as the model's number",
        rerun: "Discard warm-up probes and re-run for steady state",
        investigate: "Investigate an anomaly before trusting it",
        rebench_context: "Re-run with corrected system context (governor/RAM/VRAM) before comparing",
        archive: "Archive it and move on",
      },
    },
  },
});

if (r.stopReason !== "stop") {
  return `classify failed: ${r.stopReason} ${r.errorMessage || ""}`;
}

const a = r.answers;
const yes = (k) => (a[k] && a[k].probability >= 0.6 ? true : a[k] ? false : null);
const flags = [];
if (!yes("steady_state")) flags.push("not steady state");
if (!yes("evidence_strength")) flags.push("weak evidence");
if (!yes("comparable")) flags.push("context not comparable");
if (yes("memory_spill")) flags.push("possible memory spill");

text(`target: ${target}  model: ${provider}/${modelId}`);
text(`files:\n${files}`);
text("answers: " + JSON.stringify(a));
text(
  "VERDICT: " +
    (a.next_action.choice.toUpperCase().replace(/_/g, " ")) +
    " (p=" +
    a.next_action.probabilities[a.next_action.choice].toFixed(2) +
    ", conf=" +
    a.next_action.confidence.toFixed(2) +
    ")"
);
text("flags: " + (flags.length ? flags.join(", ") : "none"));
text("usage: " + JSON.stringify(r.usage));
