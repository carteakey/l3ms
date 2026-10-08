# Explicit foreground CPU coordinator

This is an opt-in branch workflow, not a service installation or full GPU/media
cutover. Normal `l3ms`, standalone toolkit commands, and existing services keep
their behavior. The current production decision profiles declare CUDA; this
command does not silently convert them to CPU or download missing weights.

Use `l3ms --coordinator-cpu-daemon /absolute/path/coordinator.json` only with an
already prepared local Nagi/Laya runtime and a native profile explicitly declaring
`device: cpu`. Native manifests still select the model and runner. Hosted/GPU
bindings and retained interrupted work are refused at startup. Read and reconcile
retained history after an uncertain exit; do not replay submissions automatically.

The JSON file must be a current-user-owned, single-link regular file with mode 0600,
at most 16 KiB. Unknown fields, relative paths, duplicate runtime names, and
out-of-range bounds are rejected. Supply all fields explicitly:

```json
{
  "schema_version": 1,
  "repository_root": "/absolute/path/l3ms-monorepo",
  "socket_directory": "/absolute/private/coordinator-socket",
  "lock_directory": "/absolute/private/coordinator-locks",
  "data_directory": "/absolute/private/coordinator-history",
  "standalone_data_directory": "/absolute/path/standalone-history",
  "runtimes": ["nagi"],
  "queue_capacity": 8,
  "peer_timeout_ms": 100,
  "max_connections": 4096,
  "session_timeout_seconds": 3600,
  "recovery_timeout_seconds": 30
}
```

The socket, lock and coordinator-history directories must already exist, belong
to the current user and have mode 0700. The coordinator does not create them or
remove stale endpoints. Its history must be separate from standalone history;
aliases and linked stores are refused. Trusted directory ancestors and native
repository writers are required. Select only `nagi`, `laya`, or both. Limits are
1–128 queued jobs, 1–30,000 ms per peer, 1–4096 connections, 1–3600 seconds per
foreground session, and 1–300 seconds for independent shutdown recovery.

SIGINT/Ctrl-C and SIGTERM stop intake and request bounded child cleanup. The
listener/host owner remains held while that cleanup is attempted. Session and
connection limits also trigger shutdown. An uncertain worker does not become a
successful cleanup; durable interruption prevents optimistic restart admission.
Forced termination such as SIGKILL cannot provide cleanup acknowledgment.

From another terminal, `--request-submit PATH` reads one strict v1 Submit envelope
with a caller-selected stable request ID. Use `--coordinator-socket-dir DIRECTORY`
or the shared coordinator setting. Acceptance is durable before its reply; a
lost reply means query that same ID with `--coordinator-status ID`, not resubmit.
`--coordinator-cancel ID` requests cancellation, and `--request-results ID` reads
one bounded v2 native result page. Results are memory-only; durable status survives
restart, but output loss never causes automatic rerun. Explicit native output
contains sensitive prompts/answers/artifact names and should not enter logs.

For example, the Submit file for an operator-prepared native CPU decision profile
has this shape. Replace `your-native-cpu-profile` with its actual catalog ID;
this example does not install a profile or turn an existing CUDA profile into CPU.

```json
{
  "version": 1,
  "message": {
    "type": "submit",
    "request_id": "operator-request-001",
    "timeout_ms": 30000,
    "request": {
      "lane": "decisions",
      "profile_id": "your-native-cpu-profile",
      "state": {"ready": true},
      "questions": {
        "route": {
          "type": "choice",
          "criteria": {"yes": "Proceed when ready", "no": "Stop when not ready"}
        }
      }
    }
  }
}
```

`--request-output ID --request-lane decisions` assembles bounded native result
pages under one three-second deadline. Select `language`, `media`, or `decisions`
explicitly; the JSON keeps chat content/reasoning, decision answers, and media
references separate. `complete` means output retrieval caught up to a terminal
request, not that inference succeeded; inspect `terminal` and durable status.
Unknown, expired, malformed, or incomplete output fails explicitly. Neither
result action submits work or retries inference.

In the Stack Catalog (F8/Alt+8), configure the same private socket directory and
select a native profile. `s` opens a strict request-file input, `o` observes a
retained or entered ID and displays native output in a private panel, and `c`
requests cancellation. `f` opens a native task form for the captured catalog
profile: chat turns or an embedding text-batch JSON array, media workflow JSON
or an argv array, speech text/mode and
optional cloning/design fields, or decision state/questions. Supply the caller
ID and timeout explicitly; select the declared task and operation using the
arrow keys, move fields with Tab/Shift+Tab, and submit with Ctrl+Enter or Enter
on the Submit row. Enter within a text field inserts a newline. Escape clears
the private form. Fields and the total envelope are bounded; refreshed profiles
must still match before submission. Script argv is never shell-split.

The caller ID is retained before submission, including when its acknowledgment
is lost. An unresolved request blocks another submit until observation/cancel
establishes a terminal outcome; no inference retry is inferred from lost output.
These bounded background actions do not start engines or write payloads to job
history. Embedding contracts/forms/adapter fixtures exist, but production
language dispatch, incremental chat publication and task-aware artifact access
remain parity gaps; a JSON output panel is not proof that a media file is usable.

This CPU workflow does not establish GPU release, fence standalone engine
clients, launch a managed llama/Comfy engine, expose HTTP ingress, or complete the
managed-engine/GPU acceptance gates. A real offline CPU Nagi coordinator request
passed with a private copied CPU profile, native output, durable terminal history
and verified process cleanup; see [measured evidence](nagi-acceptance.md).
Original checkouts and runtime rollback paths remain in place.
