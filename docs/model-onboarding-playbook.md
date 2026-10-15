# Model Onboarding Playbook (End-to-End)

Use this when adding a new llama.cpp model family to L3MS with full workflow support.

## 1) Confirm upstream/runtime source

Decide what binary source is required:

- upstream `vendor/llama.cpp/build/bin/*`
- or a PR-merge test build (recommended when architecture support is pending merge)

If PR-merge is needed, add a thin wrapper in `maintenance/` that delegates to:

```bash
./maintenance/llama-test-pr.sh <pr-number>
```

Wrapper pattern:

- File: `maintenance/build-<model-family>-llama-cpp.sh`
- Env override for PR number
- No positional args

## 2) Add downloader profile

Edit `model_downloader/models_config.json` and add a **disabled-by-default** entry:

- `repo_id`
- `local_dir`
- `allow_patterns` for your primary quant
- `max_workers` (optional throttle)
- clear description noting compatibility requirements

Keep defaults non-destructive (`force_download: false`).

## 3) Add a `llama-swap.yaml` entry

Append a model entry under `models:` in the root `llama-swap.yaml`.
Conventions:

- unique key (e.g. `"<model-family>"`); use `unlisted: true` for variants
  kept around for regression comparison
- reuse the existing macros (`${llama_server}`, `${ik_server}`,
  `${chat_template}`, `${cpu_range}`) — do not hardcode paths
- `cmd:` must pass `--port ${PORT} --host 0.0.0.0` (llama-swap auto-assigns
  the upstream port; its listener stays on `:8080`)
- put runtime-specific toggles such as `GGML_CUDA_GRAPH_OPT` in `env:` per model
- safe serving defaults (`--fit` or static `-ngl`+`-ot`, `--parallel 1`,
  `-ctk/-ctv q8_0`, `--flash-attn on`)
- add reasoning-effort variants with `filters.setParamsByID` (aliases like
  `<model>:high` / `:low` are auto-generated)

Restart llama-swap and verify:

```bash
systemctl --user restart llama-swap.service
curl -s http://localhost:8080/v1/models | jq '.data[].id'
```

See `docs/llama-swap-runbook.md` for the full invocation / curl patterns.

## 4) Add bench scripts

Create:

- `bench-models/bench-llama-cpp-<model>.sh`
- `bench-models/bench-llama-cpp-<model>-strategies.sh`
- `bench-models/bench-llama-cpp-<model>-fit.sh`

Reuse shared runners:

- `bench-models/run-llama-bench.sh`
- `bench-models/run-llama-fit-bench.sh`

For PR-merge-only models, set runner binaries explicitly, e.g.:

```bash
LLAMA_BENCH="${LLAMA_BENCH:-${REPO_DIR}/vendor/llama.cpp-pr-test-<pr>/build/bin/llama-bench}"
LLAMA_FIT="${LLAMA_FIT:-${REPO_DIR}/vendor/llama.cpp-pr-test-<pr>/build/bin/llama-fit-params}"
```

## 5) Update docs

Update `docs/bench-results.md`:

- add model in Script Layout examples (if relevant)
- add usage snippets in Quick Reference
- add model section in Bench Results once you have measurements

If discoverability needs it, add a short mention in `README.md`.

## 6) Update changelog

Add entries under `CHANGELOG.md` → `## [Unreleased]` → `### Added` listing:

- new maintenance wrapper
- new run/bench scripts
- downloader config profile
- docs additions

## 7) Validate wiring

Run:

```bash
l3ms --list stack bench
bash -n bench-models/bench-llama-cpp-<model>.sh
bash -n bench-models/bench-llama-cpp-<model>-strategies.sh
bash -n bench-models/bench-llama-cpp-<model>-fit.sh

# llama-swap dry-load + model ID check
L3MS_ROOT=$(pwd) ~/bin/llama-swap -config ./llama-swap.yaml -watch-config &
sleep 2
curl -s http://localhost:8080/v1/models | jq '.data[] | select(.id | contains("<model>"))'
kill %1
```

## 8) Trigger model download

Use targeted download (preferred over enabling all config rows):

```bash
./model_downloader/download_hf_model.py \
  --repo-id <org/model-repo> \
  --allow-patterns "<primary-quant-pattern>" \
  --local-dir <target-dir> \
  --max-workers 2
```

## Worked example: gpt-oss-puzzle-88B

- Build wrapper: `maintenance/build-gpt-oss-puzzle-llama-cpp.sh` (PR `#21032`)
- llama-swap entry: `gpt-oss-puzzle-88b` in `llama-swap.yaml` (uses `${puzzle_server}` macro)
- Bench scripts:
  - `bench-models/bench-llama-cpp-gpt-oss-puzzle-88b.sh`
  - `bench-models/bench-llama-cpp-gpt-oss-puzzle-88b-strategies.sh`
  - `bench-models/bench-llama-cpp-gpt-oss-puzzle-88b-fit.sh`
- Downloader profile: `SamPurkis/gpt-oss-puzzle-88B-GGUF` with `*MXFP4_MOE*`
- Client call: `curl http://localhost:8080/v1/chat/completions -d '{"model":"gpt-oss-puzzle-88b", ...}'`
- Targeted download:

```bash
./model_downloader/download_hf_model.py \
  --repo-id SamPurkis/gpt-oss-puzzle-88B-GGUF \
  --allow-patterns '*MXFP4_MOE*' \
  --local-dir /home/kchauhan/models/SamPurkis/gpt-oss-puzzle-88B-GGUF \
  --max-workers 2
```
