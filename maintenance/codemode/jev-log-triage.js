// @options: {"max_output_tokens": 3000}
// L3MS llama-swap / llama-server log triage with TypeSafe Jev, via pi's `codemode` tool.
//
// Run from a pi session:
//   /jev-log                                   (triages llama.log plus router journal)
//   /jev-log srv-ad427-20836.log
// or paste this file's contents into `codemode` with, as the first statement:
//   const ARGS = "<log path>"; const MODEL_OVERRIDE = "jev-latest";
//
// The action options mirror the documented llama-swap operations in AGENTS.md:
// hot reload (SIGHUP), full router stop/start, free RAM before a long session,
// or fix flags. Jev picks; you run the command.

const target = (typeof ARGS !== "undefined" && ARGS) || "llama.log";
const provider = (typeof PROVIDER_ID !== "undefined" && PROVIDER_ID) || "typesafe";
const modelId = (typeof MODEL_OVERRIDE !== "undefined" && MODEL_OVERRIDE) || "jev-latest";
const ROOT = "/home/kchauhan/repos/l3ms";

const model = await models.getModelOfType("classifier", provider, modelId);
if (!model) return `classifier ${provider}/${modelId} is not in the catalog or has no credentials`;

const log = await tools.bash({
  command: `cd ${ROOT} && tail -n 60 ${target} 2>/dev/null | tail -c 3000`,
});
const router = await tools.bash({
  command:
    "pgrep -a llama-swap | head -n 3; systemctl --user is-active llama-swap.service 2>/dev/null; " +
    "journalctl --user -u llama-swap.service -n 25 --no-pager 2>/dev/null | tail -c 1500",
});
const vram = await tools.bash({
  command: "nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null; free -g | head -n 2",
});

const state = {
  server_log_tail: log.output || "(empty log)",
  router_status_and_journal: router.output || "(no router status)",
  gpu_and_memory: vram.output || "(no gpu/memory snapshot)",
};

const r = await models.classify(model, {
  state,
  questions: {
    serving_ready: {
      type: "bool",
      instructions: "Is the router/served model in a state that can answer requests right now?",
      criteria: {
        true: "Server is listening and ready to serve",
        false: "Failed, crashed, still loading, or never became ready",
      },
    },
    failure_class: {
      type: "choice",
      instructions: "Which failure class best fits what is shown?",
      criteria: {
        oom: "Out of memory or VRAM exhaustion",
        memory_pressure: "RAM pressure, mmap/PLE or expert paging pressure, thrash",
        config_flag: "Bad, unsupported, or mismatched flags or config",
        crash_abort: "Crash, abort, assertion, or engine bug",
        thermal: "Thermal throttle or hardware fault",
        none: "No problem visible",
      },
    },
    hot_reload_enough: {
      type: "bool",
      instructions:
        "Would a llama-swap hot reload (kill -HUP on the router) plausibly fix this without a full router stop?",
      criteria: {
        true: "A config hot reload is enough",
        false: "Needs a full router stop/start, RAM reclaim, or a flag fix",
      },
    },
    next_action: {
      type: "choice",
      instructions: "What is the most useful next operator action?",
      criteria: {
        none: "Nothing; leave it running",
        sighup: "Hot reload the router (kill -HUP $(pgrep '[l]lama-swap'))",
        restart_router: "Stop and start llama-swap.service for a clean swap",
        free_ram: "Reclaim RAM (close desktop stack or headless isolate) before retrying",
        fix_flags: "Edit the model entry flags/context settings and reload",
        investigate: "Gather more evidence before acting",
      },
    },
  },
});

if (r.stopReason !== "stop") return `classify failed: ${r.stopReason} ${r.errorMessage || ""}`;

const a = r.answers;
text(`target: ${target}  model: ${provider}/${modelId}`);
text("answers: " + JSON.stringify(a));
text(
  "VERDICT: " +
    a.next_action.choice.toUpperCase() +
    " (p=" +
    a.next_action.probabilities[a.next_action.choice].toFixed(2) +
    ", conf=" +
    a.next_action.confidence.toFixed(2) +
    ")"
);
text("failure_class: " + a.failure_class.choice + " | serving_ready: " + a.serving_ready.probability.toFixed(2));
text("usage: " + JSON.stringify(r.usage));
