# Explicit foreground CPU coordinator

**Historical, retired.** `--coordinator-cpu-daemon` is no longer accepted.
Use [native CPU workflows](../cpu-native-workflows.md) through `--run-profile`
or the Stack typed form instead. The material below records the old branch
contract, not current setup instructions. Existing history is left untouched.

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

Generic CLI envelope-file submission has been retired; use the typed Stack form
for remaining coordinator tasks, or direct CPU tasks without a coordinator.
Use `--coordinator-socket-dir DIRECTORY` or the shared coordinator setting for
observation/cancellation. Acceptance is durable before its reply; a
lost reply means query that same ID with `--coordinator-status ID`, not resubmit.
`--coordinator-cancel ID` requests cancellation, and `--request-results ID` reads
one bounded v2 native result page. Results are memory-only; durable status survives
restart, but output loss never causes automatic rerun. Explicit native output
contains sensitive prompts/answers/artifact names and should not enter logs.

For a completed managed Comfy/Breeze artifact, use its exact published reference:

```sh
l3ms --coordinator-socket-dir /absolute/private/coordinator-socket \
  --artifact-chunk caller:your-request --artifact-kind backend \
  --artifact-path job-1/image.png --artifact-offset 0 --artifact-byte-budget 65536
```

This reads exactly one chunk and prints bounded JSON with `bytes_hex`,
`availability`, offsets, total size, `end_of_file` and `artifact_version`.
Use `local_job` for Breeze, and supply the returned version with
`--artifact-version` on every subsequent nonzero offset. No output file is
created and no inference is started/retried. Unknown/expired data returns no
bytes; durable success after a daemon restart does not restore access. Full-file
saving, TUI consumption and native streaming remain separate acceptance work.

For reference, the internal wire envelope for a typed CPU decision form has this
shape. This is not a supported CLI input file. Replace `your-native-cpu-profile`
with its actual catalog ID; this example does not install a profile or turn CUDA into CPU.

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
select a native profile. `o` observes a
retained or entered ID and displays native output in a private panel, and `c`
requests cancellation. For language chat, `f` now selects the exact advertised
model in the existing Chat tab; it does not create a coordinator request or send
a prompt. Connect Chat explicitly first. The current committed Chat connection
is authoritative, and draft endpoint edits are not applied by this handoff.
For remaining tasks, `f` opens a native task form for the captured catalog
profile: media workflow JSON
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

Embedding profiles now use the direct batch form and existing Chat connection,
not this coordinator submission path. Remote HTTP cancellation does not prove
server termination; see [the current checkpoint](status.md).
