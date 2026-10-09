# Mixed-runtime crash recovery: remaining implementation contract

This is a pending implementation contract within CAR-217, not permission for
live cutover or proof that recovery is complete. Current startup still refuses
retained Comfy/local-job intents and absent/unknown manager observations.

## Verified gaps

- `ManagedRuntimeOwner::stop` and `stop_stop_only` retain the start intent but
  do not persist Stop attempt/acknowledgment stages. An in-memory stopped receipt
  therefore cannot survive a coordinator crash.
- `reconcile_retained_services` currently accepts only the configured language
  runtime. Its complete retained batch is now preflighted before the first Stop;
  this avoids partially cleaning an earlier supported service before discovering
  a later unsupported or unknown intent. It does not add mixed-runtime recovery.
- Local-job intents carry exact job/request/directory identity metadata. Comfy
  candidate rows carry exact service-generation metadata, but that association
  is persisted after service launch, leaving a pre-candidate crash window.
- Intent schema versions 1–4 describe role/runtime shape, not cleanup stage.
  Missing cleanup metadata cannot mean Stop was never attempted: old software
  may already have dispatched it.

## Required durable Stop protocol

Use a separately versioned `stop_journal` inside the existing intent file.
Do not encode a cleanup phase by reinterpreting the existing intent schema.
Newly recorded starts must explicitly carry `NotAttempted`. Legacy records
without the field are `LegacyUnknown`, not new-record defaults.

1. Capture opaque exact-generation evidence from an existing owned/stop-only
   handle: full intent association, invocation ID, cgroup and manager main PID.
   It is comparison evidence, not authority to start, adopt, or infer success.
2. Persist `Attempted` through validated descriptor-relative atomic replacement,
   file sync and parent-directory sync **before** any name-targeted Stop.
   Failed or durability-uncertain persistence prevents dispatch. Partial writes,
   unexpected temporary files, inode replacement and conflicting stages must
   quarantine rather than be repaired optimistically.
3. Dispatch Stop once, retaining current fresh generation checks. An uncertain
   dispatch or crash leaves `Attempted`; restart never issues another Stop from
   that state. Even a pre-dispatch crash after the marker is conservatively
   unresolved. This intentionally prefers operator recovery to duplicate action.
4. Persist `Acknowledged` only from a genuine exact-generation stopped receipt.
   Failure to persist the acknowledgment stays unresolved, even if the native
   command may have succeeded.
5. Restore only read-only cleanup authority from a securely decoded acknowledged
   stage. Recheck manager generation/inactivity, exact cgroup release and the
   configured physical device before retiring the intent. An absent unit alone
   cannot mint a stop receipt. Do not reconstruct an owned execution handle.

The stage and intent must be one atomic record so crashes during retirement do
not leave independent sidecars that accidentally authorize another Stop.
LegacyUnknown and Attempted require an explicit bounded operator recovery path;
they must not silently become NotAttempted or Acknowledged.

## Job association and admission

Persist exact job/request/runtime/launch association before service dispatch,
including Comfy's pre-candidate window. Preserve the distinct later candidate
and positive backend acknowledgment; cleanup association is not POST authority.
On restart, match retained history and intents uniquely using explicit metadata,
never executable names, directory names or runtime availability.

Preflight all configured runtime/role/job associations and unresolved stages
before any destructive action. Unknown, ambiguous, stale, corrupt or unmatched
records keep admission closed. Cleanup completion must reach the main thread
through exact job-bound evidence before clearing a history reservation. Keep
interrupted inference interrupted; cleanup does not rerun it, restore output
bytes, or turn it into success. Empty WAL is not proof that every historical
reservation was safely released.

## Required verification

Inject crashes before/after each durable stage, Stop dispatch and acknowledgment,
cgroup/GPU observation, history cleanup acknowledgment and retirement sync.
Verify one Stop at most, no inference replay, exact generation matching, no
optimistic absence-based receipt, and admission remaining closed on ambiguity.
Cover legacy schemas, explicit null/duplicate/unknown fields, cross-job/runtime
associations, inode changes, expired bounds, lost replies and restart fixtures.
Only after these pass should concrete mixed-runtime recovery be enabled, followed
by separately approved real language/media/TTS switching and exact restoration.
