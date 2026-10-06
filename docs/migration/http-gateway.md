# Explicit loopback HTTP gateway

The Linux foreground action `l3ms --coordinator-http-gateway /absolute/gateway.json`
forwards one authenticated HTTP request to an already running private coordinator.
It does not install/start a daemon or engine, choose a model, download weights,
or bypass coordinator admission. Standalone commands and normal startup are unchanged.

The strict configuration must be a current-user-owned, single-link regular file
with mode 0600, at most 16 KiB. Supply every field; unknown fields are refused:

```json
{
  "schema_version": 1,
  "bind": "127.0.0.1:8081",
  "coordinator_socket_directory": "/absolute/private/coordinator-socket",
  "bearer_token": "REPLACE_WITH_A_FRESH_RANDOM_SECRET",
  "request_timeout_ms": 3000,
  "session_attempts": 64,
  "peer_attempts": 16,
  "session_timeout_seconds": 600
}
```

Replace the placeholder with a fresh unpredictable credential; do not commit it,
place it in argv, or include it in logs/screenshots. Accepted token length is
32–256 visible ASCII bytes, which is a format check, not an entropy guarantee.
Bind must be loopback. The coordinator directory must be absolute and satisfy
the existing private IPC ownership/permission checks when contacted. There is
no default token, endpoint fallback, CORS support, or public bind.

Bounds are 1–30,000 ms per connection, 1–256 session attempts, 1–64 per-IP
attempts no greater than the session limit, and 1–3600 seconds per foreground
session. Attempts include rejected connections; clients sharing one loopback IP
share its limit. This is a bounded pilot session, not an installed unlimited
gateway. Ctrl-C/SIGTERM, deadline, or attempt exhaustion closes intake. Listener
failure is reported separately and exits nonzero.

Send `POST /v1/coordinator` using HTTP/1.1, `Content-Type: application/json`, one
`Content-Length`, and `Authorization: Bearer <credential>`. The body is the same
strict native v1 Submit/status/cancel or v2 results envelope used by private IPC.
Each connection carries exactly one request and is closed after its reply.
Chunked transfer, upgrades, duplicate headers, oversized bodies, folded/control
headers and buffered pipelined requests are refused. IO and IPC share one
absolute connection deadline within the session deadline.

An HTTP 200 means a valid coordinator response was transported; its body may
still contain a native rejection or failed job status. Inspect the versioned
response instead of treating HTTP status as inference success. Native output
can contain sensitive answers/artifact references and must not enter shared logs.

Any lost Submit reply is unconfirmed: query status with the same caller-owned
ID, never automatically resubmit. Where a reply can still be sent, the gateway
adds `X-L3MS-Submit-Outcome` guidance after an IPC failure; an exhausted deadline
or broken connection may prevent even that header arriving. The gateway performs
one IPC exchange and never retries inference. Cancellation acknowledgment is
not active-child cleanup proof; observe terminal status independently.

This ingress does not satisfy managed-engine fencing, asynchronous backend drain,
GPU release, full native task-form parity or real-model acceptance. Those gates
remain in the [migration status](status.md).
