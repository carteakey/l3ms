# Real native embedding acceptance — 2026-10-14

The rebuilt branch CLI completed one real two-text embedding batch through its
shared direct language adapter and an isolated CPU-only llama-server. This
verifies native HTTP execution/results, not GPU serving, live llama-swap routing,
embedding quality or installation from an empty cache.

Cached model: `ggml-org/embeddinggemma-2-GGUF/embeddinggemma-2-Q8_0.gguf`
(approximately 296 MiB), using the already installed llama.cpp master binary.
A private acceptance-only `embeddinggemma2-acceptance` profile declared the
embedding task. No tracked manifest, disabled Nomic entry or live router config
was enabled or changed. The temporary endpoint bound only loopback on a fresh
port; CUDA visibility was empty, GPU layers were zero, op offload was disabled,
and CPU thread counts were two. No weights or runtime were downloaded.

Exact fixture, executed probe, input, manifest, native server logs, result and
receipt are retained under ignored mode-0700
`.toolkit/embedding-acceptance-UHgzvP/`. Run command:

```sh
python3 .toolkit/embedding-acceptance-UHgzvP/probe.py
```

The two public synthetic sentences produced **2 × 768** vectors, all finite,
nonzero and distinct, with native usage **17 prompt / 17 total tokens**.
The existing bounded CLI/process-identity probe confirmed CLI and temporary
server cleanup. Do not rerun this case: exclusive log creation prevents reuse.
Use a fresh fixture for any deliberately separate acceptance request.

The first attempt contained an incorrect private media-manifest schema and the
CLI refused during catalog loading, before HTTP dispatch. Read-only stack listing
reproduced the schema error; after correcting it to `schema_version: 1`, listing
succeeded. The original failed probe/log remain retained. A subsequent fixture
directory-creation refusal was also before server launch. No inference was
automatically retried after uncertain remote completion.

Native vectors remain in the private result artifact, not ordinary job history.
Existing UI fixtures cover batch rendering, committed endpoint/authentication,
exact model gating and uncertain-cancellation admission closure. A real embedding
keyboard-UI run and live-router embedding-profile acceptance are separate gates.
