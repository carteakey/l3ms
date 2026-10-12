# llama-swap runbook

Single-port OpenAI-compatible endpoint that hot-swaps between every model in
`llama-swap.yaml`. Replaces the 28 per-model `run-models/*.sh` scripts.

## Install

```bash
./maintenance/install-llama-swap.sh
```

Installs the latest release to `~/bin/llama-swap`. Env overrides:

- `LLAMA_SWAP_VERSION=v143` pin a specific release tag
- `LLAMA_SWAP_BIN_DIR=/usr/local/bin` choose a different install dir
- `FORCE=1` overwrite an existing binary

The script auto-detects `linux`/`darwin` and `amd64`/`arm64`. Override with
`LLAMA_SWAP_OS` / `LLAMA_SWAP_ARCH` if the detection is wrong.

`L3MS_ROOT` must resolve to this repo's root at launch time. The systemd
unit sets it to `%h/repos/l3ms` by default; override via
`systemctl --user edit llama-swap.service` (see below).

## Validate config

```bash
cd ~/repos/l3ms
L3MS_ROOT=$(pwd) ~/bin/llama-swap -config ./llama-swap.yaml -watch-config
```

`--watch-config` does a dry load and surfaces macro/env errors. Kill with Ctrl-C once
you see "listening on …".

## Run as a systemd user service

The unit uses `%h` (home) and three env vars so it runs unmodified on any
account. Defaults:

- `L3MS_ROOT=$HOME/repos/l3ms`
- `LLAMA_SWAP_BIN=$HOME/bin/llama-swap`
- `LLAMA_SWAP_LISTEN=:8080`

```bash
mkdir -p ~/.config/systemd/user
cp maintenance/systemd/llama-swap.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llama-swap.service
journalctl --user -fu llama-swap
```

Override any of the defaults with a drop-in (no fork of the unit file):

```bash
systemctl --user edit llama-swap.service
# [Service]
# Environment=L3MS_ROOT=/srv/l3ms
# Environment=LLAMA_SWAP_BIN=/usr/local/bin/llama-swap
# Environment=LLAMA_SWAP_LISTEN=:9090
```

Stop / restart:

```bash
systemctl --user stop    llama-swap.service
systemctl --user restart llama-swap.service
```

## Using it

List every model (including aliases):

```bash
curl -s http://localhost:8080/v1/models | jq '.data[].id'
```

Chat with the production default (warm after startup preload):

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [{"role":"user","content":"say hi"}]
  }'
```

Switch reasoning effort on gpt-oss without restarting (aliases are created
automatically from `setParamsByID` keys):

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-oss-120b:low", "messages":[...]}'
```

Hot-swap to another model (evicts the current one, loads the new one):

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-coder-next", "messages":[...]}'
```

Manually load / unload:

```bash
curl http://localhost:8080/upstream/qwen3-coder-next/health
curl -X POST http://localhost:8080/api/models/unload/qwen3-coder-next
```

These are side-effectful native controls for the pinned v247 router and a
llama-server backend. The health request can start/swap the model; it is not a
read-only idle probe. The unsuffixed `/api/models/unload` unloads **all** models,
not a model named in a JSON body. Neither acknowledgment proves GPU release.
Do not automatically repeat an action whose outcome is uncertain.
See the [pinned routes](https://github.com/mostlygeek/llama-swap/blob/40027d62fdc0cf08131973c02db2d4a7de172df1/internal/server/server.go).

## Adding a new model

1. Append a new entry under `models:` in `llama-swap.yaml`. Minimum fields:
   `cmd:` (using `${PORT}` for the upstream port) and a unique key.
2. Reuse the existing macros (`${llama_server}`, `${chat_template}`,
   `${cpu_range}`) so per-host paths stay in one place.
3. Add `env:` if the model needs a runtime-specific toggle such as `GGML_CUDA_GRAPH_OPT`
   overrides.
4. Restart: `systemctl --user restart llama-swap.service`.
5. Verify: `curl -s http://localhost:8080/v1/models | jq '.data[].id'`.

## Embeddings

`nomic-embed-text-v1.5` is served through the same authenticated llama-swap
endpoint as chat models. It is not preloaded: the first embeddings request
starts it automatically, and its per-model `ttl: 300` unloads it after five
minutes without a request.

Install the F16 GGUF at:

```text
/mnt/lab/models/nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.f16.gguf
```

The disabled downloader profile can be invoked directly without enabling any
other configured downloads:

```bash
./model_downloader/download_hf_model.py \
  --repo-id nomic-ai/nomic-embed-text-v1.5-GGUF \
  --allow-patterns 'nomic-embed-text-v1.5.f16.gguf' \
  --local-dir /mnt/lab/models/nomic-ai/nomic-embed-text-v1.5-GGUF
```

Nomic requires task prefixes. Use `search_document: ` when building an index
and `search_query: ` for search input. It returns normalized 768-dimensional
vectors, so changing to or from another embedding model requires a complete
index rebuild.

Verify on the serving host (include the configured bearer token):

```bash
curl -s http://localhost:8080/v1/embeddings \
  -H "Authorization: Bearer $LLAMA_SWAP_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"nomic-embed-text-v1.5","input":["search_query: thoughtful science fiction"]}' \
  | jq '{model, dimensions: (.data[0].embedding | length)}'
```

Use `/running` to inspect the active model. To test unloading without waiting
for the TTL, use `POST /api/models/unload/nomic-embed-text-v1.5`.

## Ports

- `8080` — llama-swap listener (client-facing OpenAI endpoint)
- `10001+` — per-model upstream ports (auto-assigned via `startPort`)

If you previously had the L3MS TUI Chat tab pointed at `http://<host>:8001`,
update it to `http://<host>:8080/v1`.

## Troubleshooting

- **"env macro not set: L3MS_ROOT"** — export `L3MS_ROOT` or start via
  systemd (the unit sets it).
- **"model file not found"** — the per-model path in `cmd:` is absolute; a
  missing mount (`/mnt/lab/models`) is the usual cause.
- **ik_llama / puzzle / sarvam builds missing** — those models reference
  `${ik_server}`, `${puzzle_server}`, `${sarvam_server}`. Build them via
  `maintenance/build-ik-llama-cpp.sh`, `maintenance/build-gpt-oss-puzzle-llama-cpp.sh`,
  `maintenance/build-sarvam-llama-cpp.sh` respectively.
- **Model stays loaded forever** — `globalTTL: 600` unloads after 10 min
  idle. Pin with `ttl: 0` per-model; evict now via `/models/unload`.
- **Concurrent models** — the default is single-model swap (LRU). To allow
  concurrent small+vision models, add a `matrix:` block (see
  `configuration.md`).
