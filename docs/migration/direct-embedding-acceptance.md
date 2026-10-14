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

## Real keyboard UI follow-up

A deliberately separate real batch used another temporary CPU server at the
same model/alias and a fresh `ui-state` within this private fixture. In the actual
80×24 terminal UI, F3/Ctrl+G committed the loopback endpoint and advertised model;
F8/`f` opened the native embedding form. Tab/text entry/Enter on Run once submitted
the two public synthetic sentences. The visible result showed **2 × 768** and
native usage **17 / 17**, then succeeded. No Chat message was sent.

The metadata-only `ui-state/jobs.json` records one succeeded embeddings task;
neither text nor vectors nor usage are persisted there. The UI exited 0 and its
owned native server exited 0 after Ctrl+C; its native output confirmed the two
completed inputs (8 and 9 tokens). The private fixture deliberately lacks normal
downloader configuration, so its unrelated strict-load warning is not evidence
of production downloader failure or complete UI acceptance.

Native vectors remain in the private CLI result artifact/in-memory panel, not
ordinary job history. Existing fixtures additionally cover authentication, exact
model gating and uncertain-cancellation admission closure. Live-router embedding
profile acceptance and GPU serving remain separate gates.
