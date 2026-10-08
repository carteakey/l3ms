# Breeze native acceptance — 2026-10-07

The opt-in `toolkits/l2m2/maintenance/probe-breeze-acceptance.py` defaults to a
plan. `--self-test` performs only offline fixtures. `--run` requires an explicit
empty current-user mode-0700 output directory and the approved bounded
maintenance window. This harness is native-wrapper acceptance, not proof of
coordinator GPU switching.

The planned sequence is standard speech, voice design, streaming chunks, then
voice cloning using only the first generated synthetic speech WAV. Each model
invocation has seed 42, at most 256 tokens and a 120-second limit, with no retry.
Logs are capped at 16 MiB per child; WAVs/chunks have format and size checks.
Script copies, exact commands, timings, artifacts and service/GPU observations
are retained privately. Ownership uncertainty blocks restoration rather than
starting overlapping GPU owners. Process-group cleanup does not prove an
escaping descendant is contained or establish production GPU admission.

Seventeen offline regressions currently pass: PCM/finite-float WAVs, bounded logs,
watchdog joining, no signals after leader reap, exact service-generation
revalidation and cgroup/GPU cleanup/restoration guards. Toolkit CI runs
`--self-test`, never `--run`.

## Live preflight outcomes

Two explicit runs refused before any service stop or model invocation:

1. ComfyUI's configured user unit was absent (`LoadState=not-found`, inactive,
   PID zero), rather than installed and inactive. The harness now accepts that
   known inactive state while rejecting unknown/error states and requiring
   ComfyUI to remain unchanged.
2. NVIDIA's compute-app listing included KDE's existing `kwin_wayland`
   compositor (PID 6813, 10 MiB), outside the llama-swap cgroup. Total desktop
   GPU use remained 44 MiB. The current strict claimant policy refused the test.
   The owner was asked whether a verified existing compositor may remain as
   explicit background GPU use; no allowance was silently inferred and the
   desktop was not stopped or reconfigured.

`llama-swap.service` remains active at its prior PID 138426; no restoration was
needed. Original YAML and checkouts were unchanged. These outcomes are safe
preflight refusals, **not TTS inference passes**.

The toolkit's `.toolkit` path links to the original checkout. The first newly
created preflight audit was moved intact into the branch root's ignored
`.toolkit/`; subsequent audits use that branch-local directory. No original
artifact was removed or overwritten. Audit directories remain private and are
not committed.
