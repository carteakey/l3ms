# L3MS monorepo migration status

Working branch: `feature/l3ms-monorepo` in the isolated checkout
`/home/kchauhan/repos/l3ms-monorepo`.

Plan: [monorepo-consolidation-plan.md](../monorepo-consolidation-plan.md).
This record tracks implementation evidence. A phase is complete only after its
gate in the plan is met; a passed offline test does not stand in for a live
runtime check.

## HeartMuLa native music integration — 2026-10-12

Declared `media:heartmula-music` now routes through the existing native command
recipe, process runner, foreground UI owner and explicit idle-GPU window. Its
existing CUDA requirement is explicit in the manifest (snapshotted first).
Standalone script flags/defaults are unchanged; no model or environment installed.

Strict JSON `{"args":[...]}` retains native argv boundaries. Shared flag-shape
validation refuses caller output/backend/help controls and conflicting/missing
values; numeric interpretation remains native. Both native output file and
directory are pinned to the existing private job root, including environment
defaults. Existing WAV validation is reused. `Music` results have artifact paths,
not a speech-mode or raw-log payload; prompt/lyrics/argv stay out of history/logs.
No inference retry, fallback, service takeover, new module, worker or daemon.

Five model-free HeartMuLa fixtures and the CLI input/launcher regression pass.
The real CLI accepts the declared music route but, with GPU config unset, exits
1 before native dispatch. Fixture: `/tmp/l3ms-heartmula-cli-ciXBqq`; no artifacts
or service changes. The smoke caught and corrected a duplicated CLI task list;
spec validation and launch now share the same lane selector.

Root build/strict Clippy/fmt pass. Full suite **798 passed, 102 socket failures,
1 ignored**; failure names match the preceding idle-GPU checkpoint's socket-only
run. Existing GPU error-category fixture flake remains open, not fixed or hidden.
Logs: `/tmp/l3ms-heartmula-{focused,cli-tests,tests-final,clippy,build}.log`.
Standalone L2M2: six Rust, 17 media and 14 downloader tests plus lint/fmt pass.
Site: 11 generator tests and DOM fixtures pass; source links regenerated without
changing language ranking or publishing the branch.

Source delta **+783 lines**, including fixtures; cumulative **+4,646** against
`702b4cd`. Control source **61,291 lines** versus 60,729 baseline: the reduction
target remains unmet. This adds one native leaf route, not a new engine framework.
Real HeartMuLa/WAV/GPU and interactive acceptance remain unverified. Next: H3,
LTX and scoped Blender contracts, live handoff/restoration, then broader deletion.

## Native idle-GPU path and CPU worker retirement — 2026-10-12

CUDA-local Nagi/Laya and exact Breeze speech profiles now share CLI/TUI native
dispatch and the existing foreground owner. A strict `L3MS_NATIVE_GPU` operator
configuration supplies physical UUID/index/idle allowance and the shared private
lock directory, with explicit client quiescence. Existing advisory locks and
bounded NVIDIA idle observations surround native execution. No model/device
fallback, new worker, daemon, service adoption or automatic stop/start.

One payload-free `native-gpu-recovery-required` marker remains after interruption,
uncertain child cleanup or failed release observation. Subsequent cooperating
CUDA work refuses it; only verified release clears its unchanged inode. Recovery
is operator-led, not a replay journal or external-client fence. Outputs are not
returned on unconfirmed release. Cancellation uses a fresh bounded release context.
UI blocks active Chat/model operations against native work in both directions.
See [exact operator contract](../native-gpu-workflows.md).

Deleted the retired CPU daemon loop/helper closure and `decision_worker.rs`.
Retained managed late-ready/uncertain-completion lifecycle tests pass with a small
fixture independent of the removed CPU path. Original commands/history/services
and remaining managed-session workflows are preserved.

Build/strict Clippy/fmt/diff checks pass. First root suite: **792 passed, 102 failed,
1 ignored**. That failure-name comparison has only the six removed CPU-daemon
transport tests; remaining failures still hit restricted sockets. Final repeat:
**791 passed, 103 failed, 1 ignored**, additionally resurfacing the previously
recorded `malformed_n_a_and_missing_or_extra_rows_fail_closed` GPU fixture error-
category flake (`QueryFailed` vs `MalformedOutput`). `gpu_probe.rs` is unchanged;
no assertions are weakened or speculative fix claimed. These are not transport
fixes. New guard/config/form/cancellation/recovery fixtures pass. Logs:
`/tmp/l3ms-idle-gpu-{tests,tests-final,clippy,build}.log`.

Actual CLI `decisions:nagi-smol` with GPU config unset exits 1 before model launch.
Actual `decisions:nagi-smol-cpu` still succeeds offline with cached weights,
CUDA hidden and two CPU threads: **1,756,774,111 ns native latency**, empty stderr.
Synthetic choice output is execution evidence, not accuracy, GPU acceptance or a
latency guarantee. Fixture: `/tmp/l3ms-idle-gpu-cli-hFdCl0` (private input/result).
This regression did not repeat the earlier process-birth-identity acceptance audit.
No live service was stopped, no CUDA inference/provider/network request made,
and no model installed/downloaded.

Whole-slice source delta: **−1,111 lines**, cumulative **+3,863** against `702b4cd`.
Control source: **60,751 lines**, versus 60,729 baseline. The substantial reduction
target remains unmet, but this deletes an entire worker rather than relocating it.
Remaining media scripts, live CUDA/HTTP acceptance, automatic scoped handoff and
verified restoration, broader coordinator deletion and final UI/site gates remain.

## Native model-control correction — 2026-10-12

The existing `SwapClient` now targets `POST /api/models/unload/{encoded-id}`
instead of posting a model body to the unload-all route. Load sends one routed
`GET /upstream/{encoded-id}/health` for llama-server-compatible backends instead
of forwarding unsupported `/load`. Endpoint prefixes are retained; empty/dot
identifiers are refused and redirects disabled. No inference, service changes,
automatic retry or GPU ownership/readiness claim is introduced.

Source authority: [pinned native router routes](https://github.com/mostlygeek/llama-swap/blob/40027d62fdc0cf08131973c02db2d4a7de172df1/internal/server/server.go).
Eight focused client tests, build, strict Clippy, formatting and diff checks pass.
Full suite: **790 pass, 108 unchanged socket-restricted failures, 1 ignored**;
failure names match the CPU-retirement checkpoint. Logs:
`/tmp/l3ms-native-controls-{tests,clippy,build}.log`.
URL construction tests are not live router acceptance, especially for unusual
model identifiers or non-llama-server backends. Live scoped unload, activation
and GPU release/restoration remain open.

This slice adds **32 net source lines**, including regression tests, but adds no
module or worker. Cumulative source delta is **+4,974** against `702b4cd`; the
substantial simplification target remains unmet. Next implement only the small
operator-window GPU path through these native controls and existing runners.

## CPU daemon retirement — 2026-10-12

Removed branch-added `--coordinator-cpu-daemon`, its CPU-only configuration and
launcher. CPU decisions already use the accepted direct CLI/UI runner, so there
is no replacement listener or daemon. Retained managed-session setup keeps its
private-file security checks and parser tests; existing tools, services and
history are untouched. The old CPU guide is clearly historical, not setup advice.

Build/strict Clippy/fmt and four retained launch tests pass. Full root suite:
**788 passed, 108 failed, 1 ignored**; failure names exactly match the preceding
hosted checkpoint's socket-restricted failures. Actual binary rejects the retired
flag with exit 2 before reading its nonexistent config. Logs:
`/tmp/l3ms-retire-cpu-{tests,clippy,build}.log`.

**300 net source lines removed**; cumulative root source delta is **+4,942**
against `702b4cd`. Control source is **62,114 lines**, still above the 60,729-line
baseline: the substantial reduction target is not met. This slice deletes code,
not moves it or suppresses tests for retained behavior.

GPU audit: native router ready/loaded states do not prove request-idle. Do not
adopt the existing deployed unit as an owned generation or treat an advisory lock
as external-client fencing. The next small path must reuse native unload,
bounded idle checks, single native execution and verified restoration during an
explicitly quiesced operator window. Reconcile the existing load client with the
pinned health-based activation contract before relying on restoration. No live
switching or new platform implementation occurred in this slice.

## Hosted decision checkpoint — 2026-10-12

Jev/SystemOne now reuse the native recipe/process/decoder and foreground UI owner.
The explicit hosted Python or `.venv-hosted/bin/python` is separate from Nagi;
no runtime is installed automatically. The existing toolkit runner's new
`--single-attempt` disables SDK retries/payload logs, with no simulation fallback.
Standalone retry defaults remain. Both profile IDs retain their identity and use
TypeSafe's verified `jev-latest` alias, not an OpenRouter protocol adapter.
Questions/answers remain private in memory; history contains metadata only.

After any child spawn, failure/timeout/cancellation or invalid output remains
remotely uncertain even if the child was reaped. Pre-dispatch failure stays
NoDispatch; successful answers preserve native usage/latency. The app serializes
Chat against all active/unconfirmed foreground native work; no global client
fencing is claimed. No new module, worker, framework or persistent authority.

Actual pinned SDK plus full CLI passed success and synthetic 503 cases through
socket-denying mock transport: **one call each**, no retries and no paid provider.
See [exact offline evidence](direct-hosted-acceptance.md). Fresh hosted installation,
approved paid provider/live cancellation and GPU switching acceptance remain open.

Build/strict Clippy/fmt, 21 toolkit runner tests and site data/DOM checks pass.
Full root suite: **789 passed, 108 unchanged transport failures, 1 ignored**.
Hosted UI fixture errors were corrected without changing privacy assertions.
An intermittent existing GPU-probe assertion could not be reproduced in bounded
stress/full-library loops; it remains documented, not fixed or suppressed.

This slice adds **411 net root source lines**, cumulative **+5,242** against
`702b4cd`; control scope 62,281. No platform module is removed. Next prioritize
the small native GPU lifecycle and remaining script/CUDA callers, then delete
their coordinator closure. The larger LOC target and full port remain unfinished.

## Previous CPU cloning/chunked TTS checkpoint — 2026-10-12

CPU Breeze now covers Standard, VoiceDesign, VoiceClone and Streaming through
the same native recipe, foreground worker and WAV/chunk validator. Strict CLI
cloning requires reference audio and its nonempty transcript; TTS accepts
optional `mode: streaming` (offline default). UI fields follow the selected task
and hidden retained fields are omitted from unrelated drafts. Shared field
validation was consolidated rather than adding another execution layer.

Real cached-weight CPU acceptance passed: cloning **14.20 seconds whole CLI**,
61,440-frame WAV; streaming **11.52 seconds**, final 46,080-frame WAV plus two
chunks. All outputs were 24 kHz mono PCM16, with owned process identities released.
Reference audio was earlier synthetic speech. No service changes, environment
installation or model downloads occurred. Chunks arrive after completion, not
live playback; voice quality/similarity and GPU lifecycle are not evaluated.
Exact retained commands/settings/copies/receipts:
[speech evidence](direct-speech-acceptance.md).

Build, strict all-target Clippy and formatting pass. Full root suite:
**784 passed, 108 unchanged transport failures, 1 ignored**. Local clone/stream UI
fixtures preserve privacy; standalone media checks pass (6 Rust, 14 downloader,
17 media Python), as do 11 site data checks and DOM fixtures. Generated payload is
deterministic; live/browser/publication acceptance remains open. This slice adds
**409 net root source lines**, cumulative **+4,831** against `702b4cd`; control
scope 61,981. No platform module is removed here; broad LOC reduction remains.

Hosted audit found the current SDK defaults to two retries, the Nagi-only
environment lacks that SDK, and reaping a hosted child cannot prove remote
cancellation. Before enabling Jev/SystemOne: pin its runtime, explicitly disable
SDK retries, and map post-spawn failures to remote uncertainty with no replay.
Both declared aliases call TypeSafe System One; `OPENROUTER_API_KEY` is a key
fallback, not an OpenRouter protocol adapter. No provider call or credential read
was performed. GPU/service switching, remaining media/decision callers and their
coordinator dependency closure still need implementation and real acceptance.

## Previous gateway retirement checkpoint — 2026-10-12

Removed the branch-added `--coordinator-http-gateway` action, launcher and custom
HTTP transport (`http_launch.rs`, `http_gateway.rs`). Callsite tracing found only
the gateway's CLI entry; native adapters, remaining private coordinator callers
and original standalone tools do not depend on it. No replacement proxy or
service mutation was introduced. Removed source is recoverable from Git.

Compile/build, strict all-target Clippy, formatting and retired-flag regression
pass; the built executable rejects the flag before opening its configuration.
Full suite: **781 passed, 108 transport failures, 1 ignored**. The count decreased
because the gateway and its tests were removed, not because sockets now work;
failure-name comparison shows exactly the 13 retired gateway failures disappeared
and all other failures are unchanged. Native transport/live acceptance stays open.

This cut removes **1,762 net root source lines**, including tests, and two modules;
cumulative delta against `702b4cd` is **+4,422**. Control scope is **61,898 lines**
versus the 60,729 baseline: the larger half-size target remains unfinished.
Remaining coordinator forms still cover native GPU/hosted tasks not yet moved;
replace those callers and GPU lifecycle before retiring their dependency closure.

Caller audit identifies CUDA Nagi/Laya/OpenJev and hosted Jev/SystemOne decisions,
script-backed MiniMax H3/LTX/HeartMuLa/Blender, and CUDA Breeze cloning/streaming
as remaining native replacement work. Existing native scripts remain available,
but that alone is not integrated UI or GPU-switching acceptance. Active language
chat already hands off to Chat; embeddings need an explicitly enabled endpoint
profile (the stock embedding YAML entry remains commented out).

## Previous native Comfy checkpoint — 2026-10-12

Declared Comfy workflow forms and image/video CLI tasks now share the existing
adapter and foreground worker, without coordinator submission. A fresh native
UUID is persisted in UI history (printed before CLI dispatch); submission is
single-attempt and cancellation targets only that ID. Unknown state, a missing
job or cancel acknowledgement never proves completion. Client cancellation
racing native completion remains cancelled with terminal state known, without
claiming that the backend itself cancelled or freed its GPU models.

Completed output descriptors use the configured server's escaped `/view`
request. Downloads stream into flat, exclusive 0600 files in an existing empty,
owned 0700 directory, with 64-file/512-MiB bounds and suffix/header sanity checks.
Only successful artifact references are returned; invalid or interrupted copies
can leave private partial files but are not published as results. Graphs and
credentials stay out of history. No new worker, service, registry or execution
journal is added, and external services/clients are not stopped or fenced.

Compile, strict all-target Clippy, formatting, CLI/form/privacy/Chat guards and
injected filesystem download fixtures pass. Full root suite: **785 passed,
121 transport failures, 1 ignored** under the existing socket restriction.
Actual submit/poll/cancel/download and real-model Comfy acceptance remain open;
there is no GPU switching or release acceptance. Original checkouts and live
services remain unchanged. See README for explicit server-contract configuration.

This slice adds **1,591 net root source lines**, including tests; cumulative
delta against `702b4cd` is **+6,184**. No platform module was removed. The fewer-LOC
target is not yet delivered: validate these replacements, finish the remaining
native/GPU callers, then delete displaced coordinator machinery inward. Keep the
wrapper direction; do not add another platform while deletion remains pending.

## Previous CPU voice-design checkpoint — 2026-10-12

Offline CPU voice design now reuses the existing Breeze recipe, foreground owner
and validated WAV path. CLI uses strict `text`/`instruction`; the form exposes
declared tts/design tasks, omits hidden design instructions from standard TTS,
and persists metadata only. A duplicate route predicate was removed rather than
adding another execution layer. Original CUDA/standalone defaults are unchanged.

Final native CPU acceptance passed in **6.09 seconds wall time**: 24 kHz mono
PCM16, 15,360 frames, observed process identities released. No GPU service was
stopped and no weights downloaded. See [speech evidence](direct-speech-acceptance.md).
Voice quality, cloning/streaming, Comfy and GPU switching are not thereby accepted.

Build/lint/fmt and CPU/design/CLI/builder fixtures pass. Full root suite remains
**777 passed, 121 transport failures, 1 ignored** under the socket restriction.
Standalone media checks also pass: 6 Rust tests, 14 downloader and 17 media
Python tests, strict lint/format and shell syntax; 11 site checks/DOM fixtures pass.
This slice adds **602 net root source lines**, including tests; cumulative delta
against `702b4cd` is **+4,593**, control scope 62,659. No platform module removed.
The larger simplification/LOC-reduction target is still unfinished.

## Previous site consistency checkpoint — 2026-10-12

The dashboard audit found manifest/display metadata drift from the new CPU
profiles. Added CPU Breeze/Nagi metadata and regenerated the payload: **8 media
and 7 decision cards**, with the existing **10 served language profiles** intact.
CPU Breeze shows standard-only mode; CPU Nagi's latency remains a manifest target,
not a benchmark claim. Their portable setup/evidence summary omits host paths,
endpoints and credentials, and distinguishes execution smoke tests from quality
or deadline guarantees. README scope now reflects the actual partial port.

Dashboard Python checks and JavaScript DOM fixtures pass. Two generator runs
produced the same SHA-256 (`b37339830e1d4f8cdedbd80a42b00f0aafe28ffdcc36b82fd5a37919fcb2150f`).
No root source LOC changed; cumulative source delta stays +3,991. No browser/live
link check or publication was performed; the publication NO-GO remains.

Parallel native-media audits found CPU voice design can reuse the existing
offline WAV recipe without waiting for GPU switching. Comfy's pinned checkout
supports scoped native UUID cancellation, but acknowledgement/404 is not terminal
proof; external-service artifacts need normal scoped file retrieval rather than
the old owned-engine release/pinned-FD registry. These are next implementation
steps, not evidence of completed media execution or GPU release.

## Previous embedding implementation checkpoint — 2026-10-12

CLI and Stack embedding batches now reuse the existing native language adapter
and foreground job/result owner, without coordinator submission or caller IDs.
Strict CLI input is `{"input":[...]}`; vectors and native usage have the existing
4 MiB output cap. Decision/TTS output limits and CPU backend semantics remain
unchanged. UI validates the exact enabled model on its committed Chat connection,
including captured credentials, and stores metadata only in history.

Remote HTTP dispatch uncertainty is separate from local process ownership.
Cancellation after dispatch never claims server termination or GPU release;
foreground admission closes and overlapping Chat sends are refused without
losing the draft. No request replay, service management or model installation
is added. Older coordinator UI integration tests now use retained CUDA-decision
forms so their submit/observe/cancel/results coverage stays intact.

**Acceptance pending:** the resumed environment prohibits local socket creation
(`PermissionError: Operation not permitted`). Full suite: **772 passed, 121
transport-related failures, 1 ignored**. Build, strict all-target Clippy, fmt and
whitespace checks pass; all 13 task-form tests, 9 direct CLI tests and focused
local cancellation/connection guards pass. New loopback runner/UI success,
captured-auth and remote-cancel cases are enabled but not verified here. Tests
were not weakened or skipped to report a green suite. The temporary full-CLI
fixture drafted before interruption did not survive the session reset and was
never reported as passed. Real-model embedding acceptance also remains pending:
no cached embedding weights were found in the earlier read-only audit.

Whole-`src/` net change for this slice is **+1,260 lines** including tests/comments/
blanks; cumulative change against `702b4cd` is **+3,991**. Control scope is
**62,361 lines**. No platform module was removed; remaining media/GPU callers
still block broad deletion. Original checkouts/services/models remain unchanged.
Next unrestricted socket acceptance, real embeddings, then native media callers
and inward coordinator deletion. The full port is not complete.

## Previous Chat replacement checkpoint — 2026-10-10

Stack `f` on a chat-only language profile now selects the exact enabled model
on the current committed Chat connection and opens the original Chat tab.
It refreshes/compares the captured catalog profile before selection, checks
connection/model identity and active-work gates, and never connects, sends or
loads a model. Draft endpoint edits, conversation, prompt and sessions remain
unchanged. The handoff works without coordinator configuration; embeddings
retain their batch form. Chat now uses original streaming/history/cancel controls,
not raw-turn coordinator submission with caller-ID observation.

Removed the duplicate chat form fields and typed builder branch. Coordinator
submit/observe/cancel/result tests were retargeted to native embeddings rather
than discarded. New keyboard fixtures cover exact selection, preserved state,
missing/disabled routes, catalog drift, disconnected/mismatched/pending
connections, active chat/session work and coordinator conflict refusal.
This is fixture evidence; manual interactive/live-server acceptance is not claimed.

Verification: **879 root tests passed, 1 ignored**, strict all-target Clippy,
formatting, whitespace checks and CLI build pass. No live services, model runs
or original checkouts were changed. Net whole-`src/` change for this slice is
**+306 lines**, including tests/comments/blanks; cumulative change against
`702b4cd` is **+2,731**. Control scope is **61,823 lines**. A duplicate caller
concept was retired, but no platform module was removed and overall reduction
remains unfinished. Next native embedding/media callers, then inward deletion.

## Previous deletion checkpoint — 2026-10-10

Retired only generic coordinator envelope-file submission: CLI
`--request-submit`, Stack `s`, file preparation and the `submission_file` module.
Typed forms (`f`), observe/cancel/results, native CPU tasks and original
llama-swap/standalone workflows remain. The keyboard integration test now uses
typed forms and verifies the retired key does not dispatch. The historical
file-submit Nagi probe refuses `--run` before any daemon/model work.

This cut removes **714 net whole-`src/` lines** (including tests/comments/blanks),
and deletes a parser/duplicate entry-point concept rather than relocating it.
Cumulative source change against `702b4cd` is still **+2,425**; control scope is
**61,799 lines**. This is the first removal, not the target platform reduction.
Verification: **877 root tests passed, 1 ignored**, 17 decision-maintenance
Python tests, formatting, strict Clippy and CLI build pass. No live service
action or model run was needed for this cut.

Next reuse the original Chat owner for explicit Stack language selection;
embedding/media form callers still block broader coordinator deletion. Remove
those dependencies only after their native workflows are replaced.

## Previous speech replacement slice — 2026-10-10

Standard CPU Breeze speech now uses the same direct typed operation and
foreground UI owner as CPU decisions, not a second worker or IPC result store.
The explicit `breeze-tts-2-cpu` profile selects `--backend cpu` with empty CUDA
visibility. The original CUDA profile/default and standalone selection remain
unchanged. CLI accepts strict text JSON and a caller-owned private output
directory; the TUI allocates fresh directories and persists only safe metadata
and relative artifact references.

The final real offline CLI run passed in **4.81 seconds wall time**, producing
a validated 24 kHz mono PCM16 WAV. Process birth identities were released;
no GPU service was stopped. See [speech evidence](direct-speech-acceptance.md).
This is execution/artifact evidence, not speech-quality or interactive acceptance.

Verification: **884 root tests passed, 1 ignored**; six standalone media Rust
tests, 14 downloader and 17 media Python tests, strict Clippy, formatting, shell
syntax and whitespace checks pass. This slice adds **1,233 net source lines**;
whole `src/` change against `702b4cd` is **+3,139**, including tests/comments/blanks.
Control scope is **61,959 lines** versus 60,729 baseline. No platform modules
were removed. Remaining native media/GPU callers and inward dependency deletion
are next; overlap is still temporary, not the promised LOC reduction.

Config/script rollback snapshots were saved under the existing `.toolkit`
symlink (therefore in the original toolkit's ignored snapshot directory).
Original runtime scripts/configs, weights and live services were not changed.

## Previous direct decision checkpoint — 2026-10-10

The first direct CPU decision path now bypasses coordinator IPC in the CLI and
Stack native task form. Both call the same typed request builder and native
runner. A separate declared `nagi-smol-cpu` profile preserves existing CUDA
profiles. Execution uses installed cached weights, stdin payloads and bounded
process-group cleanup, with no runtime installation, inference retry or fallback.

Two real offline CLI runs passed, including the actual branch CPU profile;
native latency was approximately 1.61 seconds. The harness verified child
identity release. See [direct-decision-acceptance.md](direct-decision-acceptance.md)
for exact evidence and limits. No live services or originals were changed.

The old platform is not yet retired: other task lanes still have old opt-in
callers. This is temporary replacement overlap and currently **adds LOC**, not
the promised reduction. Next replace native TTS/media callers, consolidate the
foreground job/result owner and remove the displaced coordinator dependency
closure. Manual interactive acceptance and GPU switching remain unfinished.

Verification: **876 root tests passed, 1 ignored**; two standalone decision
toolkit tests, strict all-target Clippy, formatting and whitespace checks pass.
CLI build/catalog smoke passed. Whole `src/` net change against `702b4cd` is
**+1,906 lines**, including tests/comments/blanks and the new module. Control
scope is now **61,432 lines** versus 60,729 baseline. No relocation or test
deletion is counted as reduction; no concepts/platform modules were retired yet.

## Previous planning checkpoint — 2026-10-09

**Direction changed by the user:** simplify to an operator-owned native-engine
wrapper and reduce LOC. The [simplification plan](simplification-plan.md) governs
next work and supersedes hardened-platform requirements in older checkpoints.
The existing code/test evidence below remains valid for that code, but does not
justify continuing to expand the managed daemon or automatic crash recovery.
No code has yet been retired under this plan; control LOC baseline is 60,729
including tests/comments/blanks. Next trace and consolidate native job execution,
verify user workflows, then delete displaced infrastructure rather than adding
a parallel permanently supported stack.

Three parallel Luna Max audits have now completed that trace. The broad platform
retirement candidate is 41,161 gross existing LOC, not immediately dead code.
The [execution plan](simplification-plan.md#execution-plan--parallel-audit-resolved-2026-10-09)
selects existing native adapters/process supervision and a first CPU decision
CLI/TUI slice, followed by media/TTS, GPU switching and caller-first deletion.
No runtime implementation or service behavior changed during this planning pass.

Recovery now preflights every retained runtime/role and manager observation before
the first Stop, rather than discovering a later unsupported intent after partial
cleanup. This does not enable mixed-runtime recovery. The remaining durable
Stop-stage, legacy-record and pre-dispatch job-association requirements are captured
in [crash-recovery-contract.md](crash-recovery-contract.md). The updated checkpoint
passes **859 root tests, 1 ignored**, strict all-target Clippy, formatting and
whitespace checks. No live service action was performed.

The current slice adds bounded artifact bytes to managed Comfy/Breeze publication.
Opaque pinned file capabilities move from the released worker to the main-thread
registry; reference append failure removes the new registration. Additive v3
private IPC and authenticated HTTP forwarding plus `--artifact-chunk` provide
one exact request/kind/reference read at a time, with 1–65536 decoded bytes and
mandatory version continuity after offset zero. Durable success and live result
retention both gate reads. No client path selects a file, no global artifact
root is exposed, and restart does not reconstruct capabilities from history.
Root/file drift, wrong jobs/kinds, expiry and malformed wire/query shapes fail
closed. The persistent descriptor quota includes pinned roots. Chunk delivery is
post-completion retrieval, not native audio streaming or full TUI/file-save parity.

Artifact-delivery all-target evidence: **858 root tests passed, 1 ignored**, strict all-target
Clippy, formatting and whitespace checks. A new quota fixture initially counted
global process descriptors and raced unrelated parallel tests; it now checks
only its retained/rejected file inodes. Joined private-daemon chunk paging,
success/retention/restart gates, failed-publication rollback, transport response
correlation, HTTP forwarding and explicit one-exchange CLI fixtures are covered.
No original services/checkouts were changed. Next: full injected mixed-worker/
lifecycle acceptance, crash-safe mixed-runtime cleanup/history association,
full-file/TUI artifact consumption and gated real GPU/media switching. Publication
and permanent cutover remain unapproved/unverified.

Previous checkpoint:

Managed Comfy worker composition is now implemented on the branch: explicit
optional configuration, separate allowlists and exclusive configured-GPU
binding, fresh per-job private directories, main-thread durable candidate/permit
exchange, and separately verified native backend acknowledgment persistence.
The worker releases language before launching Comfy, never repeats an uncertain
POST, and requires exact engine/cgroup/GPU/WAL cleanup before publishing checked
`Backend` artifact references. The output root is pinned before launch; unsafe
paths, aliases, duplicates and invalid files fail closed. No bytes are delivered
by the reference mapper.

Verified Python launch preserves the `.venv-comfy/bin/python` invocation rather
than executing its resolved target outside the environment. Its path/target,
venv/bin directories and `pyvenv.cfg` identities are checked through helper
handoff and before spawn; ordinary launch constructors remain unchanged.
Fixtures cover actual `sys.prefix`, target substitution, configuration/input
drift, protected-directory overlap, allocation collisions, candidate persistence
before a channel reply, lost replies/no replay, late cancellation, exact backend
association and stale-generation refusal. Optional model-path JSON uses native
newline-separated category strings, not unsupported arrays or environment
expansion.

The all-target checkpoint passes **838 root tests, 1 ignored**, strict all-target
Clippy, formatting and whitespace checks. Two new handoff fixtures initially
expected an empty channel after its sole sender exited; they now correctly
accept disconnected-with-no-message too. This was a fixture assertion correction,
not a submission retry or production policy change. This is still not full
injected owner/session/daemon acceptance or real GPU/media acceptance. Original
services/checkouts were not changed. Next: full worker/lifecycle fixtures,
artifact-byte access, durable mixed-runtime recovery/history cleanup association,
and gated real switching with directingress/background-claimant policy resolved.

The subsequent managed Comfy handoff/transport slice passes **812 root tests,
1 ignored**, strict all-target Clippy, formatting and whitespace checks. It
atomically persists candidate plus exact safe generation association and returns
a distinct opaque non-restorable permit. Legacy candidate-only rows/pages remain
unchanged. The owned session keeps managed jobs out of legacy HTTP, checks the
accepted workflow/route and exact generation, and uses fixed peer-verified
submit/status/scoped-cancel operations. No auth/TCP/global interrupt fallback or
uncertain POST replay exists. Scoped cancellation is one attempt, not terminal
evidence. Native transport compares owner generation before/after connection and
after response; response budgets are 2 MiB for Comfy and unchanged 512 KiB for
health. Cgroup validation now requires the exact final unit-name component,
not a suffix match, and persisted manager PIDs remain in the supported range.

Fixtures cover transactional persistence, generation mismatch, absent legacy
fields, strict nested DTOs/duplicate keys, retained no-replay references,
fixed HTTP operation paths and graph budgets, partial-write uncertainty,
response budgets and redaction. A new mock response fixture initially blocked
its sender after an intentional oversized-response rejection; closing the
receiver before joining and bounding fixture IO fixes the hang. The rerun passes.
This is not full injected owner/session/daemon dispatch or live Comfy acceptance.
Installed ComfyUI source at clean commit `6bfaacc67c2103481e5f0c84d75257cd0581d86a`
was read-only checked for canonical client IDs and mutex-guarded scoped running
job cancellation. No original checkout, service or GPU state was changed.

Next: wire concrete Comfy runtime configuration and main-thread permit/event
handoff into the worker, account for positive native backend IDs without replay,
define typed artifact mapping/byte retrieval, and implement crash-safe cleanup
and history association before real mixed-runtime acceptance.

2026-10-09 client/generation checkpoint: **801 root tests passed, 1 ignored**,
with strict all-target Clippy, formatting and whitespace checks. Explicit
bounded CLI exports now include optional native decision metadata separately
from the unchanged answers. The private TUI result preview labels native timing
and preserves native provider usage, with control-character sanitization and
existing preview bounds. Request resets/unavailable observations clear metadata;
absent values stay omitted from legacy exports. Focused CLI/preview fixtures
verify exact nanoseconds, native usage semantics and redacted debug output.

Opaque owned-generation identity includes the complete retained intent and
exact manager invocation/cgroup/PID, not ports or names alone. It can be obtained
only from an owned running handle after WAL/manager validation; restored
Stop-only handles cannot produce one. Health connections revalidate that same
generation after the verified socket connection and before native HTTP. Focused
fixtures reject missing WAL/substituted invocation and distinguish launch and
coordinator associations. This is not yet durable Comfy generation binding or
mixed-runtime recovery/real GPU acceptance. No live service was changed.

The real offline CPU Nagi coordinator rerun also passes: one submission,
fourteen read-only status queries, one result page/three events, one typed answer
and exact native latency `1638821625` ns in CLI export. One matching successful
durable history row was verified. Daemon/process groups stopped and the exact
temporary fixture was removed; no service manager, GPU work, download or original
checkout mutation occurred. Nine offline harness safety tests pass. See
[Nagi acceptance](nagi-acceptance.md) for scope; real GPU/mixed-runtime acceptance
remains open.

The subsequent native decision metadata slice passes **798 root tests, 1 ignored**,
strict all-target Clippy, formatting and whitespace checks. CPU and managed
local publication now preserve optional native latency/usage in one transactional
append. Nanoseconds round-trip exactly across minimum-sized pages; overflow
rejects the whole append without publishing answers or advancing the cursor.
Usage retains native field semantics and redacted debug output. Existing
answer-only pages stay unchanged; additive metadata events require upgraded
clients. Typed consumer access is implemented, while TUI display/answer-only
exports remain unchanged. Joined CPU daemon coverage checks the actual fixture's
12.5 ms native value, not fabricated coordinator timing. This is offline evidence.

The subsequent managed-stack composition passes **796 root tests, 1 ignored**,
strict all-target Clippy, formatting and whitespace checks. It adds opt-in `local_job_profiles` and
`job_directory_root` to the private managed-session JSON. Omitting both keeps
the existing language-only behavior. The root must already exist with owner-only
0700 permissions, remain pinned and not overlap coordinator/standalone history,
transport or intent roots. Only explicit local Breeze CUDA and local Nagi/Laya
CPU/CUDA routes are accepted; there is no hosted or generic-script fallback.
This remains experimental branch functionality, not an installation/cutover.

The local worker owns a fresh per-job service, releases the resident language
router before CUDA work, preserves it for CPU work and can construct a fresh
language launch after proven local cleanup. Every CUDA activation requires a
fresh physical-device idle observation. Finalization sends Stop once and retains
its exact-generation receipt across read-only retries. Every retry freshly
checks cgroup release, then the selected GPU, result packet and Breeze artifacts;
publication requires successful proof-bound intent retirement. Regression
coverage includes a cgroup becoming populated after a failed GPU check, without
another Stop or premature output. Native uncertainty remains interrupted even
when resource cleanup is proven. Relative Breeze artifact references also reach
durable completion history; they do not expose the private absolute job root.

Remaining gates: full injected mixed-worker lifecycle tests; crash-safe
multi-runtime recovery and durable cleanup/history association; Comfy
generation-bound dispatch/transport and artifact integration; optional native
decision metadata end-to-end live acceptance; HTTP artifact-root integration; and
real model/GPU switching, cancellation and exact service restoration acceptance.
Current startup still refuses retained foreign/job intents rather than adopting
or replaying them. No live service or original checkout changed in this slice.

The owned-job handoff passes **779 root tests, 1 ignored**, strict all-target
Clippy, formatting and whitespace checks. It composes pinned private packets, explicit accepted
routes/devices, payload-free schema-4 service intents and an internal one-shot
native helper. Existing schema-1/2/3 record shapes remain compatible. A durable
exclusive execution claim refuses duplicate/restarted attempts; publication
uncertainty never grants retry. Request packets are bounded to 2 MiB including
the route/envelope and native request validation; result packets stay at 2 MiB.
Recovery bounds round-trip without truncating subsecond deadlines. Packet
parsers reject duplicate keys, unknown fields and tuple-shaped DTOs.

The owner publishes the validated request before preparing/recording the exact
runtime launch; helper argv carries job/request, coordinator birth identity and
the captured private directory device/inode. Parent loss cancels native work;
the wrapper stays parked, even after uncertain result publication, for exact
owned-service Stop. A model-free native subprocess fixture proves typed CPU
output does not exit its wrapper before Stop. Joined fake-manager fixtures
check the service WAL before dispatch and Stop-only restoration/retirement with
one Start and one Stop. This is not a live systemd/model/GPU acceptance claim.

Native candidates are deliberately not final published inference results.
Remaining integration: coordinator/worker per-job dispatch, exact Stop plus
cgroup/GPU release, fresh artifact validation and typed publication, then
multi-runtime language/Comfy/TTS/decision switching. The current managed-language
startup still refuses foreign runtime/job intents; it must not adopt or replay
them. A deliberate multi-runtime recovery path is required before enabling
those lanes. No original checkout or live service changed for this slice.

The physical-device execution slice passes **757 tests, 1 ignored**, strict
all-target Clippy, formatting and whitespace checks. A typed CUDA selection
accepts only one complete canonical physical GPU UUID. The managed-language
launch derives it from the same validated physical GPU probe configuration;
helper argument parsing rejects duplicate, malformed or reordered selection.
Explicitly selected resident launches clear ambient environment and supply
only the declared stack root and CUDA visibility. Legacy unselected launches
retain their previous environment contract. Runtime dependencies that require
additional environment need an explicit contract before real acceptance.
Model-free child fixtures verify the exact UUID for resident engines, local
decisions and Breeze, and reject CPU substitution/refreshed route drift.
`CUDA_VISIBLE_DEVICES` is selection, not a device fence or release proof.
Per-job owned-service handoff, Comfy worker integration and cross-runtime live
acceptance remain unfinished. No live service was stopped for this slice.
The subsequent fresh Breeze artifact interface passes **758 root tests, 1
ignored**, nine media execution tests and strict all-target Clippy, including rejection of a WAV changed after
native execution. This provides a revalidation step for post-stop publication;
it does not itself prove the service stopped or remove its reservation.

The subsequent Comfy candidate/history slice passes **747 tests, 1 ignored**,
strict all-target Clippy, formatting and whitespace checks. Joined coordinator/
native HTTP fixtures verify durable candidate persistence before the first POST,
exact workflow/UUID permit binding, positive observation after malformed native
acknowledgment and no resubmission. Candidate intent is distinct from verified
backend identity; restart interrupts rather than rebuilding a submission permit.
Unused permits cannot survive coordinator drop/quarantine, reservation completion,
request cancellation or deadline expiry; socket fixtures verify no POST occurs.
Coordinator-only strict history loading rejects malformed, duplicate-key,
duplicate-candidate and truncated source without rewriting it. Legacy standalone
normalization and omitted metadata shape remain unchanged. Breeze's native
manifest now declares CUDA explicitly, with a retained private pre-edit snapshot.
Actual Comfy worker handoff, owned-generation recovery and cross-runtime GPU
acceptance remain open; physical-device selection is now implemented as above.

The native managed-language integration passes **734 tests, 1 ignored**, strict
all-target Clippy, formatting and whitespace checks. This is not a live
managed-runtime acceptance claim. The new concrete executor is limited to one
owned isolated llama-swap router and declared llama-server-compatible language
models; media/TTS/decision GPU switching remains unfinished.

New fixtures cover exact GPU/startup binding, fresh route/model readiness,
retained-history rejection, generic reconciliation compatibility, object-shaped
private configuration, bounded worker event delivery and cleanup on lost normal
completion delivery. Stop-receipt retries prove one Stop across delayed cgroup
observations and retain the receipt on cancellation or foreign-ledger rejection.
Joined daemon completion accounting retains uncertain reservations unless
cleanup is proven. Late-readiness shutdown coverage remains predicate-level;
full injected daemon/native-worker lifecycle tests and real managed acceptance
are still outstanding. No live service was stopped for this checkpoint.

Compatibility rerun at `0e94ede`: both toolkit Rust suites (4 media / 2 decision)
and strict Clippy pass; Python passes 14 root downloader, 14 media downloader,
16 media, 14 decision and 15 decision-maintenance tests. All 17 offline Breeze
harness safety tests pass, as do ten site data/contrast tests and DOM fixtures.
Read-only host revalidation still finds the original llama-swap service active
with its original invocation and an active Strata model plus KDE compositor on
the GPU. No inference, desktop stop or service maintenance was performed.

The site now uses the Signal / 03 visual direction, committed as `6cbef1b`.
The real Firefox desktop/mobile probe passed again after the theme change;
root reviewed desktop media and mobile decisions renders. Palette regression
checks now cover primary/secondary text, functional accents and focus on all
three surfaces. The GPUI frontend is a recorded recommendation, not a dependency
or a completed native migration. See [interface direction](../interface-direction.md).

Original source HEAD moved from `5792522` to `ed7538a`; the migration branch
remains separate, with discovery/forms/supervisor/source-sync committed at
`2bec24c` and the subsequent embedding/health/drain slice verified. Original
dirty files are preserved. Committed serving YAML is synchronized
without a live reload. See [source reconciliation](serving-baseline.md) for
imported benchmark provenance and missing harnesses. Dashboard regeneration,
nine Python data-contract tests and JavaScript DOM fixtures pass; browser/live
acceptance and publication remain gated. A stale coordinator can authorize only
exact-generation stop-only cleanup, not reuse or inference admission. This
checkpoint is not a full-port completion claim.

The single-supervisor model-free live probe passed on 2026-10-07: named FD
handoff, actual helper/backend namespace equality and host separation,
coordinator-only Unix transport, one native proof GET, exact-generation stop,
process birth-identity disappearance and empty/removed cgroup. Existing live
services and GPU state were unchanged. See [measured launch gate](private-engine-probe.md).
Real native health, model/TTS execution and coordinated GPU switching remain open.

Combined verification after the embedding/health/drain slice:
**688 passed, 1 ignored**, strict all-target Clippy, formatting and whitespace
checks pass. Typed embedding request/vector/usage contracts, one-shot native
adapter, transactional paged results and keyboard forms now pass fixtures;
archived profiles and general managed dispatch remain disabled. Admission close
tokens are core-instance/generation-bound and targeted drain reservations cannot
select another runtime's pending work. Interrupted untracked history remains a
conservative drain blocker. Owned-engine health checks bind fixed native routes
to the exact supervisor peer and owner, not GPU readiness or release.

The owner approved isolated Nagi dependencies/weights and bounded live service
maintenance/restoration on 2026-10-07. Pinned dependencies are installed. The
first staged installation exposed stale activation/console-wrapper paths after
rename; supported relocatable venv creation and offline exact-pin reinstallation
repaired the installed environment. Direct Python, activation and `hf --help`
now work. Minimal revision/checksum-pinned model acquisition completed; six
offline acquisition safety tests pass. Real offline CPU Nagi execution and
the bounded ten-iteration native benchmark passed (P50 1.61 s, P95 1.64 s,
zero option-order flips), without claiming routing accuracy or GPU performance.
The real CPU foreground-coordinator probe also passed: one submission, matched
caller ID, native typed output, a unique durable successful terminal row and
verified daemon/process-group cleanup. Eight offline harness regressions pass.
See [Nagi evidence](nagi-acceptance.md). Explicit owned-engine stack-root
environment handoff and its actual-child fixtures also pass. No
existing service was stopped in this continuation; nothing was pushed/published.
The [managed foreground integration contract](managed-session-contract.md)
records the concrete remaining orchestration and ingress gates.

Root verification after accepted-route Breeze protection: **689 passed,
1 ignored**, strict all-target Clippy and formatting pass. Both toolkit Rust
suites (4 media, 2 decision), strict Clippy/formatting, 16 media Python,
14 downloader Python, 14 decision runner Python and 15 decision-maintenance
regressions pass. The bounded native TTS harness passed 17 offline safety tests;
two live preflights refused before any service stop or inference. The current
GPU listing includes KDE's existing compositor, so its background allowance
was asked explicitly rather than assumed. See [Breeze evidence](breeze-acceptance.md).

The incremental language/readiness/client slice passes **714 tests, 1 ignored**.
Native chat deltas, optional completion-token usage and native stream completion
are retained separately from durable outer job completion. Explicit workbench
observations resume the last verified cursor, with transactional rollback on a
failed later page; no automatic polling or inference replay is introduced.
Owned llama-swap readiness now checks plaintext router `/health` and the exact
selected model state `ready` in `/running`, with bounded chunked-response support
and object-only native JSON parsing. These are fixtures, not real managed-model
acceptance. Foreground owner integration is in progress; strict Clippy must be
rerun once its newly added ownership/readiness methods are actually wired.

Isolated headless Firefox is now available for local visual checks despite the
unavailable connected browser surface. The root reviewed desktop/mobile initial
language-panel renders and corrected secondary-text contrast; ten offline
data/contrast tests and DOM fixtures pass. Root reran the reviewed opt-in Firefox
probe: real pointer and keyboard/focus checks, media/decision card counts and
desktop/mobile horizontal layout checks pass. A controlled error fixture verifies
console capture; browser cleanup and disposable-profile removal are confirmed.
Comprehensive accessibility and publication remain pending. See [browser evidence](site-browser-acceptance.md).

Linear tracking: [CAR-215](https://linear.app/carteakey/issue/CAR-215/consolidate-l3ms-l2m2-and-l3ds-into-the-local-3-mode-model-stack)
is the parent; [CAR-216](https://linear.app/carteakey/issue/CAR-216/import-l2m2-and-l3ds-into-the-l3ms-monorepo-with-verified-standalone)
covers Phases 0–3, CAR-217 Phases 4–5, CAR-218 Phases 6–7, CAR-219 Phase 8,
CAR-220 Phase 9, and CAR-221 Phase 10.

| Phase | State | Evidence / remaining gate |
| --- | --- | --- |
| 0. Baseline and protection | In progress | Original L3MS `68fd7bc48f803aecb5ae5d25a3f30967b46da3eb`; L2M2 `39e4d40a4e5ed993e99aa4ed70c9d183873a90ba`; L3DS `4716ddf3508d224645416f3ba70d5e34c138cedd`. Source inventories and backups in progress. |
| 1. Folder/history import | In progress | L3DS five-commit and L2M2 three-commit histories imported under `toolkits/`; source/WIP audit, private recovery, and README wordmark done. Final tree/secret/public-path review pending. |
| 2. Independent operation | In progress | Imported child builds, tests, profile lists, and explicit old-root lists pass; CI added locally, pre-push validation pending. |
| 3. Runtime cutover | In progress | Bounded L2M2 GPU-mode pilot, real branch L3MS language request, and a Qwen Image 2.1 branch-script generation passed; original ComfyUI service was restored after language use. L3DS now fails closed instead of synthesizing answers, but real decision weights and full deployed-path reconciliation remain pending. |
| 4. Shared contracts | In development | Read-only catalog and offline `l3ms --list stack`, layered settings, existing-store job metadata, and typed lane request/result validation are on the branch. Adapter contracts and full compatibility/live release gates remain; release stays gated by Phase 3. |
| 5. Runtime adapters | In development | Concrete bounded language execution, ComfyUI job handling, CPU-local decision supervision, Breeze speech execution/artifact validation, and inert generic media recipes pass fixtures. Private language/Comfy Unix HTTP transport also passes native fixtures. General lifecycle/admission, GPU release, real TTS/decision pilots, and user-facing switching remain pending. |
| 6. Local coordinator | In development | Durable queue, ownership locks, v1 control/v2 native result pages, one-shot CLI Submit and opt-in foreground CPU-local decision dispatch pass offline fixtures. Current CUDA profiles remain gated. General managed-engine/media dispatch, installation, server-pushed streaming and real cross-client acceptance remain pending. |
| 7. HTTP ingress | Retired | Branch-added gateway and launcher removed on 2026-10-12 under the simplification plan; native endpoints remain. Historical pilot fixtures are not live acceptance. |
| 8. Unified workbench | In development | Three-lane Stack Catalog shares settings and has opt-in strict-file submit/observe/cancel with retained IDs and private native output. Full task-form parity and live managed routing remain pending. |
| 9. Public site | In development | Separate language, media/TTS and decision views derive from native manifests. Ten data/contrast tests, DOM fixtures and real Firefox desktop/mobile click/keyboard/focus/layout checks pass, including the Signal / 03 style. Comprehensive accessibility and publication/export gates remain pending. |
| 10. Final acceptance | Pending | Live gates and old-path retirement review. |

## Current safeguards

- The original `~/repos/l3ms` checkout remains on `main` with its pre-existing
  tracked and untracked changes. The migration branch lives in a separate Git
  worktree; bounded pilots have been restored, and no deployed runtime path or
  permanent service has been switched.
- L2M2 and L3DS originals remain in place. Their source and ignored runtime
  state are being classified before history import.
- The source tree and plan are being reviewed for paths, secrets, and large
  generated files before the first push. The current
  [publication audit](publication-audit.md) is **NO-GO** for a public push:
  retained imported history still exposes a Funnel endpoint and host-specific
  paths. Keep this branch local until the exact outgoing history is sanitized
  and re-audited.

## Mutable runtime state (not Git migration inputs)

| State | Current location / writer | Pilot strategy and rollback |
| --- | --- | --- |
| L3MS vendor, weights, `.toolkit`, user state | Original `~/repos/l3ms` and external model roots; no active llama-swap service at baseline | Keep root deployment paths unchanged; reconcile dirty config before any deployment. Restore the saved config/service revision if the pilot fails. |
| ComfyUI vendor, user data, outputs, and model symlinks | Original `~/repos/l2m2/vendor/ComfyUI` and `.toolkit`; the active `l2m2-comfyui.service` is the writer | Snapshot mutable user/workflow state, verify symlink targets, and reconnect the imported child only in an exclusive pilot window. Stop old writer before starting new; preserve pilot writes and return the unit to the original checkout on rollback. |
| L2M2 Python environments | Original `.venv-comfy/` (6.5 GiB) and `.venv-comfy-mcp/` (228 MiB); no Git owner | Rebuild or reconnect by a verified path at the new child; do not copy venvs blindly because interpreter paths are absolute. Keep originals until acceptance. |
| L3DS Python environment and model cache | Original `.venv/` (5.1 GiB) and shared host Hugging Face cache (36 MiB, no snapshots) | Rebuild only after resolving dependency conflicts; leave the original venv/cache intact. No local model inference acceptance until a real model is available. |

## Evidence log

| Date | Check | Result |
| --- | --- | --- |
| 2026-09-28 | Created isolated L3MS worktree from `68fd7bc` | `feature/l3ms-monorepo`; deployed checkout untouched. |
| 2026-09-28 | Copied the user's revised plan into the branch | SHA-256 matched the source plan before the README-art amendment. |
| 2026-09-28 | L3MS baseline `cargo test --all-targets --locked` in original checkout | Passed: 258 tests, 1 ignored. |
| 2026-09-28 | L3MS baseline lint/Python checks | `cargo clippy --all-targets --locked -- -D warnings` passed; 14 downloader unit tests passed. `cargo fmt --all -- --check` failed on pre-existing formatting drift in clean L3MS source files. |
| 2026-09-28 | Branch formatting recovery | Formatted five L3MS source files (18 insertions, 17 deletions); root `cargo fmt --all -- --check` now passes on the branch. Original checkout remains unchanged. |
| 2026-09-28 | Read-only live host baseline | `l2m2-comfyui.service` active from the original L2M2 checkout in GPU mode; `http://127.0.0.1:8188/system_stats` responds. `llama-swap.service` inactive; ports 8080/8001 do not respond. RTX 4070 reports 12,282 MiB total and 362 MiB used. No service was switched. |
| 2026-09-28 | L3MS recovery snapshot | Verified Git history bundle, tracked WIP patch, and archive of 12 untracked files under `/home/kchauhan/repos/l3ms-migration-backup.Z0zIPT/`. Ignored runtimes and weights remain in the original checkout. |
| 2026-09-28 | Dashboard data baseline repair | Three generator unit tests pass. Regeneration initially differed from the checked-in `docs/generated-models.js` in four command examples' reasoning-effort settings; this was pre-existing source/payload drift, not caused by the monorepo import. The branch now commits the deterministic regenerated payload to keep existing dashboard CI green. |
| 2026-09-28 | L3DS import | Audited disposable five-commit history merged as `ee27a64`; 15 tracked paths and blob IDs match source under `toolkits/l3ds/`. Full-history high-signal pattern and size audit found no hits; dedicated secret scanner remains a pre-push safeguard. |
| 2026-09-28 | L3DS recovery bundle | Original complete history bundled and verified at `/home/kchauhan/repos/l3ms-migration-backup.Z0zIPT/l3ds-history.bundle`; source checkout remains clean. |
| 2026-09-28 | L3DS child operation | `cargo build --locked`, `cargo test --locked` (0 tests), `l3ds --list` (6 profiles), and `l3ds --root /home/kchauhan/repos/l3ds --list` passed from the imported child; child target output is ignored. No decision inference was claimed. |
| 2026-09-28 | Root regression after first child import | `cargo test --all-targets --locked` passed: 258 passed, 1 ignored. |
| 2026-09-28 | Root lint after first child import | `cargo clippy --all-targets --locked -- -D warnings` passed. |
| 2026-09-28 | L3DS child path follow-through | Documented child-directory commands and compile-time default root; added two offline tests for the default root and `--root` override. Child fmt, offline test (2 passed), Clippy, build, and `--list` pass. |
| 2026-09-28 | L2M2 import | Audited public-safe three-commit history merged as `4deece4`; 87 source/evidence files under `toolkits/l2m2/`. A third-party transcript and a host-specific Funnel note were excluded from every imported commit and retained in the verified private WIP backup at `/home/kchauhan/repos/l2m2-migration-backup-20260928/l2m2-selected-private.bundle`. No weights/bulk binaries in imported history. |
| 2026-09-28 | L2M2 child operation | `cargo fmt --check`, Clippy, Rust tests (4 passed), release build, 7-profile list, Python syntax, 14 downloader tests, 9 media-model tests, shell syntax, and explicit old-root list passed. No inference or runtime switch was attempted. |
| 2026-09-28 | Model ignore safeguard | Imported child `.gitignore` now ignores GGUF/safetensors and other `models/` contents while allowing only direct `*-files.txt` inventory files; verified with `git check-ignore --no-index`. |
| 2026-09-28 | L2M2 live pilot and rollback | After user approval, confirmed ComfyUI queue empty and llama-swap inactive; stopped original transient unit, archived 81 `.toolkit` entries to a verified mode-600 backup at `/home/kchauhan/repos/l2m2-migration-backup-20260928/l2m2-toolkits-prepilot.tar.gz`, and started a separate `l2m2-comfyui-monorepo-pilot` unit from the branch with ignored symlinks to original vendor, venv, and `.toolkit`. ComfyUI 0.36.0 served `127.0.0.1:8188` in GPU mode, exposed 11 saved workflows, and completed a 64×64 `EmptyImage` → `SaveImage` job (`65b9ccfe-6ce0-4ac5-acdb-6f8f497eeaaa`), producing `l3ms-monorepo-pilot_00001_.png` in the normal media output directory. This verifies service-path wiring, not a heavyweight model workflow. Pilot queue was empty before stop. Original `l2m2-comfyui.service` was restarted and is active/healthy with its original user-directory path; pilot unit is inactive. |
| 2026-09-28 | Isolated llama-swap probe | A loopback-only router with the branch config parsed and listened on port 18080, but its declared preload failed because the branch worktree has no `vendor/llama.cpp` (the deployed root still owns ignored vendor state). The probe was shut down; ComfyUI remained active. Branch L3MS `--quickstart` and `--list bench` passed; `--list run` still requires a live router. No language inference is claimed. |
| 2026-09-28 | Phase 4 offline development | Read-only three-lane manifest catalog, layered stack settings in the existing settings store, and cross-toolkit metadata in the existing job-history store are being implemented/tested. No new inference or service action is exposed by these contracts. |
| 2026-09-29 | Phase 4 offline catalog and CLI | `l3ms --list stack` lists 21 lane-qualified profiles from the three native manifests without contacting llama-swap; each is explicitly `manifest-only`, not installed/healthy. It works from another working directory with `L3MS_ROOT` set. Root `cargo test --all-targets --locked` passed 284 tests (1 ignored); root fmt, Clippy, and dashboard generator checks passed. |
| 2026-09-29 | Phase 4 settings and jobs | Existing settings store now has versioned optional stack preferences, declared path anchoring, built-in → host → user → environment → CLI precedence, snapshots on writes, and portable profile exports that omit machine roots/policy. Existing job history now supports queued cross-toolkit metadata, explicit pre-dispatch persistence, transitions/timestamps/artifacts, and no-replay interruption on restart while preserving legacy row shape. Targeted tests pass; final full-suite check pending after integration. |
| 2026-09-29 | Phase 9 site development | Added native-manifest-derived media/TTS and decision catalog payloads, separate site tabs, Breeze's four TTS modes, hosted/local decision labels, and explicit absence of measured results. Existing language ranking remains separate. Six Python tests, JavaScript syntax, and whitespace checks pass. No browser surface was available for visual review; no site was published. |
| 2026-09-29 | Branch language runtime pilot and restoration | With the approved maintenance window and an empty ComfyUI queue, stopped the original transient ComfyUI unit; linked the branch's ignored `vendor` path to the original verified builds; started branch-configured llama-swap on loopback `127.0.0.1:18080`. The preloaded Gemma 4 QAT+MTP model passed health, authenticated `/v1/models` listed eight profiles, and a bounded `/v1/chat/completions` request returned `READY` (89 total tokens). Stopped the temporary router and restarted the original ComfyUI transient unit via its original `maintenance/comfyui-service.sh start-gpu` launcher. Verified active original `ExecStart`, ComfyUI 0.36.0 GPU health, empty queue, and closed pilot port. No permanent service path changed. |
| 2026-09-29 | Branch media model execution | With the restored original ComfyUI GPU service, the branch's Qwen Image 2.1 script passed `--check` and submitted a bounded 512×512, four-step inference job (`d5c28456-a97e-4d5f-87ab-e40cdb277c11`). It wrote a verified 512×512 PNG at `/home/kchauhan/media-output/l3ms-monorepo-qwen-pilot-20260929.png`; visual inspection showed the requested blue circle. The ComfyUI queue returned empty and the service stayed active. This proves branch script/model invocation against the original service, in addition to the earlier branch-hosted service-path pilot; it is not an end-to-end branch-hosted heavyweight model run. |
| 2026-09-29 | Phase 4 lane request/result shapes | Added in-memory chat, media workflow/native script/Breeze speech, and decision request/result types. Validation checks lane-qualified profile ownership, native task declarations, required cloning/voice-design inputs, decision answer alignment, and a 1 MiB request payload budget; it cannot start engines or claim readiness beyond manifest presence. Five new tests pass. Root full suite: 289 passed, 1 ignored; root Clippy and formatting pass. Adapter execution contracts remain Phase 4 work. |
| 2026-09-29 | L3DS fail-closed acceptance safeguard | Removed silent synthetic fallback from normal Nagi, Laya, SemIf, and hosted decision requests. `--simulate` is an explicit demo path with synthetic engine labels and no model/hosted API loading; SemIf has no native forward pass yet. Missing native inference now exits nonzero, and the Rust benchmark fails if any profile lacks a valid result rather than counting raw/fabricated output as success. Five offline Python safety tests pass; Rust child tests, fmt, and Clippy pass. A normal SemIf one-shot and benchmark both exited 1; explicit simulation exited 0 with a `semif-simulated` label. This does not satisfy the real L3DS inference gate. |
| 2026-09-29 | Phase 5 offline lifecycle contract | Defined explicit lifecycle ownership, GPU resource claims, runtime state, admission/drain/health/release adapter methods, and a fail-closed in-process exclusive transition sequence. Six mock tests cover order, drain and admission failures, target-health rollback, unknown release, external ownership, and pre-cancellation. No concrete adapter or live/manual switch command is connected. Root full suite: 295 passed, 1 ignored; formatting and Clippy pass. |
| 2026-09-29 | Structured transition failures | The offline exclusive-GPU switch now returns a typed disposition (`NoTransition`, `PreviousRestored`, or `OwnershipUncertain`) instead of requiring a future coordinator to parse error text. Existing mock tests assert the disposition for failed admission/drain, restored target-health failure, unknown release, and external ownership. Root full suite remains 295 passed, 1 ignored; Clippy and formatting pass. |
| 2026-09-29 | Phase 9 keyboard semantics | Added tablist/tabpanel relationships, selected-state updates, roving tab focus, and Arrow/Home/End navigation to the three-lane static site while preserving its existing language views. Six site/generator tests and inline JavaScript syntax pass. Browser visual review remains unavailable in this session, and the site remains unpublished. |
| 2026-09-29 | Breeze-TTS-2 offline route/readiness work | The imported L2M2 wrapper now has a read-only `--check` for the local audio.cpp executable, parsed `breeze_tts` loader key, model, and model spec; cloning checks a nonempty reference WAV. Standard TTS, voice-design alias, native clone task, and streaming chunk output have fixture tests. Local preflight passed with installed files but did not load the model or allocate the GPU. L2M2: 16 media tests, 14 downloader tests, 4 Rust tests, fmt, Clippy, and shell syntax pass. Shared request validation now accepts an optional clone transcript, matching the native audio.cpp contract. Root: 295 passed, 1 ignored; fmt and Clippy pass. A real TTS smoke remains pending. |
| 2026-09-29 | External serving-mode change observed | At approximately 12:49 local time, the original ComfyUI transient unit stopped and the original `llama-swap.service` became active on port 8080; this was not a branch operation. At inspection, the original ComfyUI unit was absent/inactive and `/health` on the language router returned `OK`. No service was toggled for TTS; preserve current mode and recheck direct-ingress/admission before any GPU pilot. |
| 2026-09-29 | Nagi-Smol environment preparation | The L3DS child has a pinned Nagi-only Python/CUDA dependency input and a separate dry-run-by-default staged preparer that refuses to overwrite an existing `.venv`; the original `maintenance/setup-env.sh` is byte-identical to the imported source for standalone compatibility. The preparer does not acquire weights or run inference. Its no-write preview, shell syntax, eight Python safety/setup tests, two Rust child tests, fmt, and Clippy pass. Local uv resolves CPython 3.12.13, and the offline resolver found the 51 non-Git pins against the original environment; a full offline sync cannot resolve the uncached pinned Nagi Git source. The imported child still has no `.venv`, no Nagi-Smol snapshot, and no native decision result. |
| 2026-09-29 | Publication re-audit | A private NO-GO audit is pinned to commit `2287d9d`: the imported reachable history still discloses a Funnel endpoint and host-specific records. The owner confirmed an unchanged baseline high-entropy key-like value was only a documentation example; a public export should still replace it with a clear placeholder. No push or history rewrite occurred. Later commits and the settled worktree need an exact outgoing-ref re-scan before publication. |
| 2026-09-29 | Phase 4 native route metadata | A read-only resolver now preserves media script paths and decision entrypoint/model/type/tier from the native manifests, keeps profile/runtime identities separate, and uses a symbolic llama-swap endpoint reference without guessing URL or auth. Declared relative paths are bounded and lexically anchored under each canonical toolkit root; a future dispatcher must revalidate symlink containment before opening them. It does not check file existence, contact endpoints, or start engines. Root tests passed 299/1 before the TUI change and 301/1 after integration; fmt and Clippy pass. |
| 2026-09-29 | Phase 8 read-only TUI catalog | Added F8/Alt+8 Stack Catalog navigation, palette open/refresh/selection commands, declared task/runtime/device/source details, explicit manifest-only readiness, and missing-manifest errors. CLI and TUI now share settings/environment resolution; navigation exits previous editor input safely. Tests exercise navigation, configured roots, failed refresh, and wide/compact rendering. Combined verification: root 301 passed/1 ignored, fmt/Clippy; L2M2 4 Rust, 16 media Python, 14 downloader; L3DS 2 Rust, 8 Python; six site/generator tests and deterministic payload. The three localhost HTTP fixtures required the approved unsandboxed test run. No coordinated run action was enabled. |
| 2026-09-29 | Native decision payload preservation | Shared decision requests and answers now retain the native named question/answer objects, including choice criteria/options and score payloads, instead of flattening them to text. Validation bounds the complete state/question JSON and verifies answer IDs and declared choices. Six request-contract tests pass; no backend invocation is enabled. |
| 2026-09-29 | Bounded runtime operations | Every potentially blocking lifecycle adapter operation receives its deadline/cancellation context; successful admission opening is the commit point. Cleanup and restoration share one independently bounded recovery budget with caller cancellation masked. Twelve runtime tests cover cancellation during startup/health, slow preflight, and recovery exhaustion, in addition to existing transition failures. Combined root suite after queue integration: 314 passed, 1 ignored; fmt and Clippy pass. The three localhost fixtures required the approved unsandboxed run. |
| 2026-09-29 | Phase 6 offline coordinator core | Added a bounded queue using the existing job store: acceptance is persisted before dispatch, retained request IDs refuse replay, restart interrupts rather than replays accepted work, conflicting GPU work is FIFO while hosted work can bypass it, and cancellation/drop retains active reservations until acknowledged completion. Quarantine also blocks already-reserved dispatch. Queue/cold-start/execution timings are separate; prompts and native payloads are not persisted. Six queue tests pass. Request completion does not establish resident-model GPU release; concrete adapters must still prove it. |
| 2026-09-29 | Offline host ownership guard | Added lifetime advisory locks in an explicitly shared, private, local host lock directory, including a coordinator lock and sorted GPU locks. Existing lock inodes are never unlinked or replaced; symlink/shared/nonregular lock files fail closed. Four tests include a competing independent process and release/reacquisition. This guard is not installed, does not fence legacy direct clients, and cannot replace admission/reconciliation. |
| 2026-09-29 | Phase 6 IPC contract and reconnect | Added bounded length-prefixed v1 submit/cancel/status envelopes, explicit version/unknown-field rejection, lane-native inputs, and bounded serialization. Request-ID validation reuses the existing history rules. Status lookup survives restart using retained caller IDs rather than rehydrated numeric job IDs; terminal records cannot be relabelled cancelled. Twelve IPC and seven queue tests pass. Combined root suite: 331 passed, 1 ignored; fmt, Clippy, and whitespace checks pass. Transport deadlines, peer authentication, daemon installation, concrete dispatch, results, and streaming remain unimplemented. No engine/service action or public command was enabled. |
| 2026-09-30 | Concrete language execution | The branch has a caller-configured, bounded HTTP language adapter that preserves native chat turns, separates reasoning/content, requires terminal SSE completion, checks cancellation/deadlines while waiting for headers/body, and never retries uncertain dispatch. Errors do not expose credentials, endpoints, prompts, or response bodies. Eight adapter tests include isolated HTTP fixtures; shared chat results bound combined output and permit reasoning-only completion. This is not connected to CLI/UI inference or service switching. |
| 2026-09-30 | Native path/recipe verification | Catalog refresh preserves canonical configured roots even for empty lanes. Six dispatch-time path tests reject missing/empty/nonregular files, symlink escapes, and changed runtime identity. Twelve native recipe tests cover actual Breeze mode flags, clone-file checks/path anchoring, explicit decision devices, bounded/redacted stdin and argv, missing interpreters, and unsupported workflow/simulation dispatch. These preparations start no engines. |
| 2026-09-30 | Decision stdin compatibility | Added optional `--input-stdin` with a bounded UTF-8 `{state,questions}` JSON envelope; malformed input is rejected before engine construction. Legacy inline/benchmark flags remain supported. Eleven child Python safety/setup tests pass; no environment or model was installed. |
| 2026-09-30 | Supervised local decision execution | Added a Linux child-group runner with bounded nonblocking stdin/output, distinct stdout/stderr under one shared cap, deadline/cancellation checks, and independently bounded recovery. The leader stays unreaped until the final owned-group signal; after reaping only a non-signalling absence probe is used. Nine runner tests pass. The concrete local decision adapter prepares/executes/decodes harmless fixtures with offline model flags, validates native engine labels/answers, and preserves native latency/usage; five decoder/execution tests pass. Escaped/daemonized descendants remain outside the process-group proof; local cleanup cannot establish remote Comfy cancellation or GPU release. Hosted execution remains gated by SDK retry/auth verification; standalone hosted CLI is unaffected. |
| 2026-09-30 | Combined adapter checkpoint | Root full suite: **373 passed, 1 ignored**, plus fmt, Clippy, and whitespace checks. Changes are committed locally on `feature/l3ms-monorepo`; no migration service switch, real model pilot, endpoint cutover, push, or publication occurred. Pi delegation was attempted with a read-only task and timed out; further Pi use is deferred at the user's request. Implementation assumptions and release gates are documented in [execution-contracts.md](execution-contracts.md). |

## Open decisions

Latest offline adapter/transport/control/safety checkpoint: **429 root tests passed, 1
ignored**, with fmt, Clippy, and whitespace checks passing. A deterministic
fork-inheritance regression reproduced an ownership guard release defect;
explicit unlock now releases both complete and partially acquired guards even
while a child retains inherited descriptors. A second regression rejects child
success with truncated stdin. Backend IDs also have a restart-retention test:
they can be persisted before dispatch and are retained without automatic replay.
The concrete ComfyUI adapter passes nine tests: unchanged graph/UUID submission,
adapter-local single-use ownership, native terminal status, bounded artifact
references, scoped cancellation with no unsafe fallback, unknown/no-op states,
uncertain submission without replay, manifest runtime drift, response limits,
and deadline/cancellation waits. It requires an explicit installed-server
compatibility assertion; durable token recovery and lifecycle/GPU-release proof
are not implemented. Luna workers provided native protocol and focused safety
reviews; the root implemented the adapter after narrowing an overlong worker
task. No live service was changed. These are offline checks, not completion of
the lifecycle/live gates.

The Linux local transport passes 14 tests, including two independent process
clients, private directory/socket permissions, peer UID denial, retained host
locks, protected existing/replaced socket paths, frame limits, request/reply
ordering and ID matching, slow/stalled IO, cancellation/deadlines, and full-backlog
connect behavior. A targeted native-socket probe confirmed that EAGAIN with
zero SO_ERROR can still mean unconnected; the reviewer withdrew an initial
same-descriptor polling suggestion after that evidence. Client frames are never
replayed. The daemon/event loop, installation, coordinator dispatch and result
streams remain pending; no live service was changed or started.
After the connector initially blocked the export, the owner approved a brief
progress summary to CAR-218. It was posted successfully on 2026-10-01 UTC
(comment `9b7a3e80-9b90-40ca-a953-931aa04f57e8`).

2026-10-01 control-RPC checkpoint: fifteen queue/RPC tests pass, including seven
new regressions for typed admission/duplicate/full/invalid/unbound rejection,
persist-before-acknowledgment, active cancellation reservation retention, failed
acceptance/cancellation persistence, restart status without replay, incomplete
retained state, and lost socket acknowledgment followed by status/cancel/status.
The existing Rust submit interface wraps the same typed submission path; persistence
failures now quarantine admission. A focused Luna review found no blocker. No
daemon, engine, or live service was started; runtime reconciliation/dispatch and
peer-loop ownership remain pending. This checkpoint is recorded locally; it does
not imply any further Linear export beyond the separately approved earlier summary.

2026-10-01 control-loop checkpoint: nine daemon fixtures pass, covering ownership
before history access, startup failure cleanup, closed submission admission,
restart interruption without replay, malformed/stalled peer isolation, outer
cancellation/deadlines, private data paths, GPU-lock coverage and release on
exit. The consuming serial loop bounds peer time and connection attempts and
has no dispatch or admission-opening interface. No live service was touched;
installation, lifecycle reconciliation and result delivery remain pending.

2026-10-01 decision-route correction: the native Nagi Smol/Big bridge now passes
the declared repository ID to its loader, including CPU fallback; previously it
selected only a tier and silently loaded the upstream default repository. Three
new offline mock regressions cover repository forwarding and preserve Huge's
existing call shape. L3DS Python: fourteen tests pass. No weights or environment
were installed. Public-metadata-only acquisition preflight is recorded in
[l3ds-prep.md](l3ds-prep.md); installation approval remains pending.

2026-10-03 CLI/Breeze checkpoint: five new CLI regressions cover parser bounds,
private-socket status/cancel exchanges, wrong-operation replies, bounded JSON,
and explicit unavailable errors without fallback. Six Breeze fixtures cover all
four modes, explicit fresh private outputs, sorted chunks, missing/unsafe/bounded
artifacts, redacted failures, cancellation/deadline cleanup, and PCM container
validation rather than a bare WAVE marker. Root: 426 passed, 1 ignored; no real
model or live service was used. Dispatch, lifecycle reconciliation and full UI
integration remain incomplete.

2026-10-03 shared-control-settings checkpoint: optional socket/artifact paths
share existing host/user/environment/explicit precedence and declaring-directory
anchoring. Defaults remain absent, resolution creates no files, and portable
exports strip both paths without changing the operator's store. Three new
settings regressions include an isolated-process check of environment-variable
collection. Root: 429 passed, 1 ignored; no deployment or admission was enabled.

2026-10-03 common-client/NVIDIA checkpoint: Linux CLI status/cancel now uses the
shared one-shot client and opt-in socket configuration, with explicit flag
precedence and no inferred default or automatic startup. Client fixtures cover
submit/status/cancel reply shapes, correlation, fixed errors and no fallback;
CLI fixtures preserve parser, path-selection and bounded JSON behavior. Ten
NVIDIA fixtures cover explicit identity, bounded queries, malformed/unsupported
readings, changing capacity, claimants, memory limits and cancellation/deadlines.
Root: 444 passed, 1 ignored; Clippy, formatting and whitespace checks pass. All
GPU tests use fake executables. No live service, admission or inference dispatch
was enabled. The managed-service ownership primitive is still in development.

2026-10-03 managed-ownership/history-isolation checkpoint: ten controller
fixtures verify single-use pre-dispatch intents, bounded restored records,
uncertain-launch retention, transient generation/cgroup checks, rejection of
preexisting/non-transient units and refusal to stop changed generations. Stop
acknowledgment is not cgroup/GPU release; same-user replacement races remain an
explicit trust constraint. Twelve daemon fixtures now include canonical
directory aliases, symlink/hard-link history rejection and absent standalone
paths without creation. Root: 457 passed, 1 ignored; Clippy, formatting and
whitespace checks pass. No actual service command or GPU query was run. Durable
intent storage, owned cgroup cleanup and adapter/admission wiring remain pending.

2026-10-03 durable-service-owner checkpoint: eight store fixtures validate
strict bounded restored records, immutable duplicates, private inode/type checks,
FIFO rejection, corruption/unsafe entries, capacity and no automatic creation.
Seven owner fixtures compose the actual store/controller under retained host
ownership, asserting record-before-launch, lost-launch-result observation without
replay, persistence refusal, deadline refusal, post-persistence cancellation,
nonmember-handle rejection and corrupt-store stop refusal. Mutable owner methods
serialize one instance; shared-directory/controller identity remains a caller
precondition. A focused Luna review found no blocker under that contract.
Twelve controller fixtures now include explicit working directories in identity
and argv; environment expansion is disabled. Root: 474 passed, 1 ignored;
Clippy/all-targets, formatting and whitespace checks pass. No live manager/GPU
operation occurred; only installed `systemd-run --help` was read. Records remain
retained, with no retirement/release/admission proof or daemon dispatch wiring.

2026-10-05 owned-stop/dispatch checkpoint: generation-matched stop now returns
an opaque receipt; ten cgroup fixtures verify bounded fixed-root traversal,
empty/removed observations and fail-closed populated, unsafe, malformed,
missing-event and cancelled/expired cases. Four new owner fixtures compose
stop and observation while retaining durable intents and transferable receipts
after observation failure, without repeating stop. The dispatch gate snapshots
the accepted payload, refreshes native routes and rechecks binding, history,
cancellation and actual deadline before persisting running. Tampered/stale
tickets quarantine without dropping reservations; start is not replayable.
Root: 499 passed, 1 ignored; locked/offline all-targets tests, Clippy with denied
warnings, formatting and whitespace checks pass. No live manager/GPU operation
occurred. Result retention/delivery, concrete lifecycle/admission adapters,
intent retirement and end-to-end dispatch remain incomplete.

2026-10-05 dashboard behavior checkpoint: six Python generator/data-contract
tests pass. The shipped generated payload and inline dashboard JavaScript also
pass a Node DOM fixture for native lane card rendering, click selection,
Left/Right wrapping, Home/End, roving focus/ARIA selection and HTML escaping.
Dashboard CI now runs both checks; no remote CI or publication was triggered.
The browser inventory is empty and the in-app browser is unavailable, so the
loopback preview was stopped without claiming visual/mobile/browser acceptance.
The fixture deliberately does not emulate layout or an accessibility tree.

2026-10-05 results/runtime-binding checkpoint: twelve result-buffer fixtures
verify native lane separation, bounded transactional retention, exact pagination,
UTF-8/escaping, explicit terminal eviction and strict decoded-page validation.
V2-only result queries preserve v1 behavior, correlate ID/version/cursor/budget,
and keep durable status independent from output-cache loss. Real private-socket
fixtures cover queued/not-found lookup, lost-response retry and restart with
retained succeeded history but unavailable output. CLI results uses shared
settings/explicit socket precedence, compact bounded JSON and no runtime startup.
Runtime-bound service intents retain lane/name in strict schema 2; unbound
schema-1 records remain compatible. Manager description/recipe checks preserve
the association through dispatch/restart, without replay or process-name guesses.
Review corrected explicit-null deserialization and mixed-lane page acceptance.
Root: 530 passed, 1 ignored; locked/offline all-targets tests, denied-warning
Clippy, formatting and whitespace checks pass. No live manager/GPU or inference
operation occurred. Native dispatch/event production, result retention policy,
intent retirement, concrete lifecycle/admission and deployment remain incomplete.

Repeat-run follow-up: the all-targets suite subsequently exposed intermittent
fake-manager `SpawnFailed` errors. Captured errno 26 (`ETXTBSY`) identified
parallel fixture writes overlapping fork/exec, not a runtime-binding or deadline
failure. Test-only executable-write/spawn serialization corrected the race;
100 parallel managed-service suite repetitions passed. A new four-thread fresh
executable regression reproduced the failure with its write guard removed.
With the guard restored, the full suite passed 531 tests, 1 ignored.
Production spawn behavior and failure disposition are unchanged.

### 2026-10-05 — owned intent retirement, isolated transport and CPU worker

Root verification: 556 passed, 1 ignored; locked/offline all-targets tests,
strict denied-warning Clippy, formatting and whitespace checks pass without
lint suppression. Explicit intent retirement freshly checks stopped generation
and cgroup, validates exact private records, then unlinks/syncs with retained
metadata on uncertainty. Runtime identity and declared native device are now
part of accepted-route checks, including both recipe refreshes. Uncertain
completion retains its active reservation until separate verified-cleanup
acknowledgment; interrupted outcomes cannot be reclassified by ordinary completion.

The isolated byte transport pins coordinator PID/UID/start, pidfd and retained
nsfs identity. Real child-namespace fixtures verify the FD handoff, private Unix
peer, isolated loopback backend, early response/backpressure and uncertain partial
writes. Captured EACCES for cross-user-namespace `/proc` namespace lookup was
resolved by authoritative FD handoff, not a boolean bypass. Namespace loopback
setup is fixture-only; no real manager units, inference engines or GPU were used.
Production manager launch/property validation and native admission/drain remain.

A one-job CPU-only local-decision worker now checks accepted routes, disables
CUDA discovery and downloads, supports cancellation and returns typed native
outputs/cleanup categories over bounded channels. Its success fixture initially
returned an undeclared answer, correctly rejected by native validation; the
fixture's accepted choices were corrected and full verification rerun. Restricted
CPU startup refuses GPU/hosted bindings and unresolved retained history. Default
profiles remain CUDA/remote, not silently converted. Output quotas reserve each
job's maximum before dispatch rather than competing with other active output.
Daemon execution/event-loop wiring, public submit/UI/HTTP, general media mapping,
live acceptance and publication remain incomplete. No goal-completion claim.

- Choose the host admission mode for direct llama-swap/ComfyUI requests before
  claiming automatic cross-engine switching. The plan allows coordination
  only where the runtime can block and drain new direct work, or where callers
  use exclusive coordinated ingress.
- Live GPU and external-route cutovers require host access and recorded
  before/after state. They are not implied by a successful build.
- The opt-in fresh Nagi environment and multi-gigabyte model acquisition still
  await approval. No dependency environment or weights were acquired in this
  continuation; real local decision acceptance remains unsatisfied.

### 2026-10-05 — foreground CPU dispatch and native client path

Locked/offline all-targets verification passes 594 tests, with 1 ignored. The
opt-in CPU loop has 21 focused daemon tests covering durable submit/lost replies,
responsive status/cancel, result pages, timeout and active shutdown cleanup,
output-quota preflight, GPU refusal and restart without replay. Explicit foreground
startup reads a strict bounded private configuration and handles SIGINT/SIGTERM
cooperatively. The normal control-only loop and standalone defaults are unchanged.
See [foreground-coordinator.md](foreground-coordinator.md).

One-shot CLI Submit uses a shared bounded nofollow reader and caller-owned IDs.
Private language SSE and Comfy submission fixtures verify native HTTP over the
selected Unix socket. Shared transactional result reassembly and multi-page
private IPC reading preserve native fields and reject incomplete groups, stale
cursors, wrong lanes and unavailable output. Output is not persisted or replayed.

Strict denied-warning Clippy passes after narrowing managed-launch error storage,
without lint suppressions. Managed engine integration is still unfinished.
Keyboard-first strict-file submit/observe/cancel and the explicit reassembled-output
CLI now pass fixtures; full native task forms remain pending. No
model-serving manager unit, real engine/model, dependency environment or GPU was changed. A new
browser availability check still returned no enabled apps or browsers, so actual
site visual/mobile acceptance remains open. Full-port acceptance and publication
are not complete.

The subsequent [model-free manager namespace probe](private-engine-probe.md)
disproved the proposed two-unit namespace-join topology on this installed user
manager: declared join properties still left the bridge in the host namespace.
Removing the join produced two genuinely isolated namespaces; a startup delay
and manager-opened namespace descriptor did not correct joining. All probe units
were generation-checked, stopped and collected. Managed launch is being reworked
to one isolated service owning backend and bridge, with kernel checks retained.
This is a discovered deployment gate, not real inference or GPU acceptance.

### 2026-10-05 — resumed live-state read-only check

The original ComfyUI unit is currently inactive and llama-swap is active; the
RTX 4070 reports 44 MiB of 12,282 MiB in use. These are point-in-time observations,
not ingress/drain or GPU ownership proof. Breeze `--check` still succeeds against
the imported toolkit's existing audio.cpp executable, compiled loader, model and
model spec. No synthesis, output creation, service stop/start, environment install
or model download was performed. Recheck live state before any approved pilot;
do not reuse the earlier Comfy-active baseline.

Offline compatibility checks in this continuation also pass: 16 media Python
tests, 14 decision safety Python tests, 14 imported downloader tests, six site
data tests and shipped-JavaScript DOM fixtures. These do not load models or
replace real browser visual/mobile/accessibility acceptance.

### 2026-10-05 — authenticated loopback ingress checkpoint (retired 2026-10-12)

The explicit `--coordinator-http-gateway` action reads a strict private 0600
configuration, binds only loopback, and forwards one native v1/v2 request per
connection to private IPC. Fourteen gateway fixtures cover authentication,
framing, bounded deadlines/attempts, native correlation and lost Submit replies
without retry. The all-target suite passes 617 tests with one ignored; strict
denied-warning Clippy passes. CPU startup and gateway sessions share scoped
SIGINT/SIGTERM ownership without changing normal startup. See
[http-gateway.md](http-gateway.md).

The internal single-service supervisor entrypoint also compiles and passes five
argument/descriptor-contract tests. Its process-cleanup regressions, typed manager
role replacement, actual service/namespace/readiness and native engine health are
still being implemented or awaiting acceptance. This checkpoint does not enable
general managed GPU/media dispatch or certify full-port completion.
