# /add-decision-model

Serve a native decision model (typed choice / noul / score answers with
probabilities, no text generation) as a llama-swap endpoint backed by
llama.cpp's `/v1/systemone` (TypeSafe System One API), then register it in the
L3DS decision catalog.

First wired: `vllm-d3-flash` (vllm-sr Decision 3.0, 2026-10-12). Use that
entry in `llama-swap.yaml` as the template.

## Usage

```
/add-decision-model <entry-id> <hf-gguf-repo> <weights-file> [--mmproj <file>]
```

- `<entry-id>` — llama-swap model id, org-prefixed (e.g. `vllm-d3-flash`, `liquid-d1-3b`)
- `<hf-gguf-repo>` — e.g. `ggml-org/vllm-d3-flash-GGUF`
- `<weights-file>` — e.g. `vllm-d3-flash-Q8_0.gguf`
- `--mmproj <file>` — vision/audio projector from the same repo (prefer BF16;
  the projector is small and vision quality drops faster under quantization)

## Steps

### 1 — Confirm llama.cpp supports the decision type
- First prove the GGUF is a decision model at all, without downloading it:
  `maintenance/gguf-decision-check.py <hf-repo> [<file.gguf>]` (range-reads the header).
  `NO - plain LLM conversion` means the decision head was lost. That's common for
  generic community conversions (mradermacher, Livesport pplx-decider) of
  LoRA-plus-head models, and `/v1/systemone` would return 501. Stop and look for an
  official or ggml-org conversion.
  Leaderboard and checked repos: `docs/decision-models.md`.
- Supported types live in
  `common/common.cpp` → `COMMON_DECISION_TYPE_NAMES`.
- Check the decision build first:
  `grep -A15 COMMON_DECISION_TYPE_NAMES vendor/llama.cpp-d3/common/common.cpp`
- If the type is missing, find the support PR
  (`gh pr view <n> -R ggml-org/llama.cpp --json state,mergeCommit`).
  Merged: move the worktree forward:
  `git -C vendor/llama.cpp-d3 fetch origin && git -C vendor/llama.cpp-d3 checkout --detach <merge-sha>`.
  Not merged: build it with `maintenance/llama-test-pr.sh <pr>` and add its own macro.
  Moving the worktree changes the binary for **every** decision entry, so step 6
  must re-test all of them.
- Never rebuild `vendor/llama.cpp-master` for this: it serves the gold Qwen tier.

### 2 — Build (only if step 1 moved the worktree)
`nvcc` is not on PATH, so pass it explicitly:
```bash
cd vendor/llama.cpp-d3 && rm -rf build && \
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF && \
cmake --build build -j12 --target llama-server
```
The binary is about 16 KB because it links the shared `.so` files in `build/bin`;
that is expected. Update the commit noted in the `d3_server` macro comment.

### 3 — Size and download
- List files and sizes first:
  `curl -s https://huggingface.co/api/models/<repo>/tree/main`.
- VRAM budget (12 GB card): weights + mmproj + about 1.3 GB for an 8k ubatch
  must stay under about 11.8 GB. d3-flash Q8_0 + BF16 mmproj measured 11.5 GB.
  If it doesn't fit, drop to Q4_K_M or a smaller size instead of partial offload:
  these are latency models.
- Download to `/home/kchauhan/models/<org>/<repo>/` with resumable curl
  (`curl -L -C - --fail -o <file> https://huggingface.co/<repo>/resolve/main/<file>`),
  in the background. Keep existing files.

### 4 — Standalone smoke test (outside the router)
Stop nothing if the GPU is idle. Otherwise
`systemctl --user stop llama-swap.service` first (AGENTS.md).
```bash
vendor/llama.cpp-d3/build/bin/llama-server -m <weights> [--mmproj <mmproj>] \
  -ngl 99 -fa on -c 8192 -b 8192 -ub 8192 -np 1 --host 127.0.0.1 --port 9931
```
- The log must show `srv init: decision model type: <type>`. If it doesn't,
  this is not a decision GGUF (or the build is too old), and `/v1/systemone` returns 501.
- **ubatch rule:** laya, clef, vllm-d3 and lfm2-d1-omni evaluate the whole
  prompt (images + state + question) in one batch, so set `-c = -b = -ub`.
  8192 is about 4 images at 1.6 MP. vllm-decision2 needs a question's options plus the
  end of the prompt in one batch. Other types (openjev, lev, kev, nimble, pplx-decider,
  lfm2-d1) can split across normal `-b/-ub` (from the server README `/v1/systemone` section).
- Record `nvidia-smi` memory used after load and after an image request.
- Stop it with `kill <pid>`. Do not use `pkill -f '<pattern>'` in the same shell
  command as the pattern: it matches and kills its own shell.

### 5 — Request checks
Text, all three question types in one call:
```json
{"state":"Customer message: I was charged twice for my order last week and nobody has replied.",
 "questions":{
  "route":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":null,"shipping":null,"technical":null}},
  "angry":{"type":"noul","instructions":"Is the customer angry?"},
  "urgency":{"type":"score","instructions":"How urgent is this?","criteria":["can wait","this week","today","right now"]}}}
```
Expect billing > 0.95. Image (if mmproj): render shapes with ImageMagick
(`magick -size 1024x768 xc:white -fill red -draw 'circle 300,384 300,184' ...`).
Don't test for text in the image: ImageMagick has no font installed here, so
text renders as empty boxes. Send `"state": null` (vllm-d3 / lfm2-d1 / lfm2-d1-omni only; other
types need a non-null state) and `"images": ["data:image/png;base64,..."]`.
Note latency (`curl -w '%{time_total}'`).

### 6 — Wire into llama-swap
- Snapshot first: `cp llama-swap.yaml llama-swap.yaml.bak-$(date +%Y%m%d-%H%M%S)`.
- Add the entry under the `# Decision models` banner, copying `vllm-d3-flash`:
  `${d3_server}`, `--alias <entry-id>`, the measured flags, and a `description`
  with the API path, ubatch limit, VRAM, and latency.
- Reload: `kill -HUP $(pgrep '[l]lama-swap')`.
- Smoke through the router. **llama-swap v247 does not route `/v1/systemone`**
  (404), so use the passthrough path:
  `POST http://127.0.0.1:8080/upstream/<entry-id>/v1/systemone` with
  `Authorization: Bearer $LLAMA_SWAP_API_KEY`. Read the key from
  `systemctl --user show llama-swap.service -p Environment` without echoing it.
- If step 1 moved the worktree, rerun step 5 against every other decision entry too.

### 7 — Register in the L3DS catalog
L3DS root: `toolkits/l3ds/` once the monorepo migration lands
(`docs/monorepo-consolidation-plan.md`), otherwise `~/repos/l3ds`.
- Add a profile to `decision-runtimes.json` with `type: "local"`, the HF repo in `model`,
  measured `vram_gb` and `expected_latency_ms`, and the router URL from step 6.
- `decision-models/runner.py` has no llama.cpp engine yet (`nagi`/`laya`/`semif`/`jev` only, and the
  local ones return simulated answers). Until a `llamacpp` engine that POSTs to
  the upstream URL exists, ask before adding the profile. Don't fake an engine name.
- Then run `suites/routing-gate.json` and `suites/latency-micro.json` against it and
  keep the results under the L3DS `docs/`.

### 8 — Record
- `CHANGELOG.md` → `[Unreleased] / Added`: model, quant, binary commit, endpoint,
  flags, VRAM, latency.
- Report: entry id, endpoint URL, VRAM, text/image latency, sample answers.
