# Direct CPU decision acceptance — 2026-10-10

This is the first replacement slice under the [simplification plan](simplification-plan.md),
not full media/TTS/GPU or monorepo acceptance.

## Real native CLI evidence

Built the branch debug CLI and invoked `--run-profile --task decision --input`
with a public synthetic choice question. The existing Nagi environment and
cached `nagisanzeninz/nagi-smol-v0` weights were used offline; CUDA visibility
was empty and CPU thread counts were bounded to two. No daemon, socket,
service stop/start, new model download or inference retry was used.

Two deliberately separate acceptance runs succeeded:

- Private copied CPU manifest fixture: native latency **1,629,339,970 ns**.
- Actual branch profile `decisions:nagi-smol-cpu`: **1,607,733,271 ns**.

Both returned native choice/probabilities; the declared-profile run selected
`accept` with confidence approximately 0.5897. This tests execution and native
metadata, not semantic accuracy or a latency guarantee. Latency is reported by
the native runner, not whole CLI wall time.

The existing bounded acceptance harness observed process birth identities and
confirmed their release after each run. The mode-0700 private test fixture was
removed only after both checks passed. Original checkouts, cached weights and
live services were untouched. The branch manifest was snapshotted before adding
the separate CPU profile; the original `nagi-smol` CUDA profile is unchanged.

## Scope and remaining work

CLI and Stack native CPU task forms share a typed request builder and direct
runner, reusing native recipes and bounded process-group supervision. Input
stays on stdin; diagnostics are fixed and payload-free. No automatic replay,
CUDA-to-CPU fallback or runtime installation is introduced.

CLI intentionally writes native answers and optional latency/usage to stdout.
The TUI retains answers in its existing in-memory result panel and safe job
metadata in history; native usage-panel parity and manual interactive acceptance
remain separate work. GPU profiles and other task lanes still use the old
opt-in path until replacement. No platform modules have been removed yet.

Offline UI fixtures cover socket-free keyboard forms and execution, redacted
history, save-failure refusal before launch, cancellation remaining busy until
cleanup, shutdown cancellation/join, coordinator overlap refusal and result
profile gating. Existing process-runner tests cover deadline and process-group
cleanup. The shutdown join avoids detaching a child; its process cleanup has a
five-second recovery budget, but that is not a wall-time guarantee for local
catalog/filesystem preflight I/O.
