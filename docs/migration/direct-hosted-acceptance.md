# Hosted decisions: offline integration checkpoint

2026-10-12, `feature/l3ms-monorepo`. This is actual SDK/CLI integration evidence
with synthetic HTTP responses, **not** paid provider, model-quality or live
cancellation acceptance.

The existing TypeSafe SDK 0.7.1 under the original L3DS Python 3.12 environment
was selected explicitly with `L3MS_HOSTED_DECISIONS_PYTHON`; it was not adopted
as the default or modified. `PYTHONDONTWRITEBYTECODE=1` prevented bytecode writes.
The default monorepo `.venv-hosted` still needs separate preparation/verification;
its observed dependency closure is in `toolkits/l3ds/requirements/hosted.lock.txt`.

## Retained fixture and commands

Private 0700 fixture: `/tmp/l3ms-hosted-BcnXfJ`. It retains synthetic `input.json`,
`sitecustomize.py`, executed runner copy, one-call counters, stdout and stderr.
The startup fixture installs a socket-denying audit hook before SDK import and
injects `httpx2.MockTransport` into the actual SDK client. It asserts the client
uses `RetryPolicy(max_retries=0)` and that the sole POST preserves the declared
model and state/questions. Credential values below are offline dummy strings.

```sh
env PYTHONPATH=/tmp/l3ms-hosted-BcnXfJ PYTHONDONTWRITEBYTECODE=1 L3MS_HOSTED_DECISIONS_PYTHON=/home/kchauhan/repos/l3ds/.venv/bin/python L3MS_DATA_DIR=/tmp/l3ms-hosted-BcnXfJ/data L3MS_HOST_SETTINGS_DIR=/tmp/l3ms-hosted-BcnXfJ/settings TYPESAFE_API_KEY=offline-test-key TYPESAFE_BASE_URL=https://offline.invalid TYPESAFE_LOG_LEVEL=debug SDK_FIXTURE_MODE=success SDK_FIXTURE_COUNTER=/tmp/l3ms-hosted-BcnXfJ/success.count target/debug/l3ms --run-profile decisions:systemone --task decision --input /tmp/l3ms-hosted-BcnXfJ/input.json --task-timeout-ms 15000 > /tmp/l3ms-hosted-BcnXfJ/success.stdout 2> /tmp/l3ms-hosted-BcnXfJ/success.stderr
env PYTHONPATH=/tmp/l3ms-hosted-BcnXfJ PYTHONDONTWRITEBYTECODE=1 L3MS_HOSTED_DECISIONS_PYTHON=/home/kchauhan/repos/l3ds/.venv/bin/python L3MS_DATA_DIR=/tmp/l3ms-hosted-BcnXfJ/data L3MS_HOST_SETTINGS_DIR=/tmp/l3ms-hosted-BcnXfJ/settings TYPESAFE_API_KEY=offline-test-key TYPESAFE_BASE_URL=https://offline.invalid TYPESAFE_LOG_LEVEL=debug SDK_FIXTURE_MODE=503 SDK_FIXTURE_COUNTER=/tmp/l3ms-hosted-BcnXfJ/failure.count target/debug/l3ms --run-profile decisions:jev-api --task decision --input /tmp/l3ms-hosted-BcnXfJ/input.json --task-timeout-ms 15000 > /tmp/l3ms-hosted-BcnXfJ/failure.stdout 2> /tmp/l3ms-hosted-BcnXfJ/failure.stderr
```

Success exited 0, returned validated native choice output and native usage, with
one SDK POST and empty stderr. Synthetic 503 exited 1 with one SDK POST, empty
stdout, and fixed `remote completion is unconfirmed; stop before continuing`.
No retries, real provider, socket operations, model download or service change.
Native latency from the mock is not a WAN/model benchmark.

SDK source fingerprints: sync client SHA-256
`bf8f7633fff1a525106fc178f319448ba30748db2e27f2d9b431d439725cd933`;
retry implementation
`6a71a8b1c6b518d2f8489d397d22b20f5929d449c7a69dfae726ac902d6ae601`.
Public contracts checked against [SDK retries](https://docs.typesafe.ai/sdk/python/api/retries),
[client options](https://docs.typesafe.ai/sdk/python/api/clients/sync),
[Choice](https://docs.typesafe.ai/primitives/choice) and
[model aliases](https://docs.typesafe.ai/models).

## Remaining gates

Fresh hosted environment installation and an explicitly approved, budgeted paid
provider test remain open. No cancellation/status/idempotency capability is
invented: a stopped child cannot prove the provider stopped. Post-spawn failure,
timeout, malformed output or cancellation closes foreground admission as remote
uncertainty; no inference is replayed. No credentials belong in chat or history.

During root verification an existing GPU-probe fixture intermittently returned
`QueryFailed` rather than `MalformedOutput` (`gpu_probe.rs:508`). Captured in
`/tmp/l3ms-hosted-tests-final.log`; no speculative fix or weakened assertion.
The diagnosis did not reproduce it: exact test 0/40 failures, GPU module 0/40,
parallel exact test 0/160, control suite 0/5 and full library suite 0/5. The cause
remains unresolved, rather than being claimed fixed; recheck before GPU acceptance.
