# Native decision models through llama-swap

The preferred local integration candidates are `vllm-d3-flash` (accuracy) and
`cloudflare-clef-flash` (calibration and joint-question image work), following
the original checkout's 2026-10-12 runbook. These are source-reported preferences,
not a new monorepo quality evaluation. Nagi CPU and existing profiles remain.

The branch imports only their serving entries and `d3_server` macro, not unrelated
changes to the original checkout. Both use the existing decision-capable
`vendor/llama.cpp-d3` build at `efe3626b1`. Runtimes and weights are not bundled or
installed automatically. Declaring a profile does not establish readiness.

| Router ID | Existing recipe | Model input capability |
| --- | --- | --- |
| `vllm-d3-flash` | Q8_0, BF16 projector | Text and images |
| `cloudflare-clef-flash` | Q4_K_M, BF16 projector | Text and images |

Both recipes retain `-c 8192 -b 8192 -ub 8192 --parallel 1`: the whole decision
prompt must fit one microbatch. Clef evaluates questions jointly; d3 evaluates
them independently. A decision GGUF must retain its native decision head and
`<architecture>.decision.type` metadata; a generic chat-model conversion is not
an equivalent substitute.

## Shared CLI and keyboard form

The root CLI accepts the existing strict text/JSON input shape:

```json
{"state":{"message":"I was charged twice."},"questions":{"route":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Charges and refunds","technical":"Software faults"}}}}
```

```sh
l3ms --run-profile decisions:vllm-d3-flash --task decision --input input.json
l3ms --run-profile decisions:cloudflare-clef-flash --task decision --input input.json
```

Use `LLAMA_SWAP_URL` and `LLAMA_SWAP_API_KEY` as for existing router operations.
The keyboard form reuses the confirmed Chat endpoint/credentials and requires
the exact enabled model to be advertised. It does not send a chat completion.
The root HTTP integration is for loopback endpoints; it does not open a new
listener or run a Python model environment. Standalone L3DS uses its stdlib HTTP
bridge with the same environment variables or `--base-url`/`--api-key` options.
Prefer environment credentials over command-line keys.

Local decisions use one `POST /v1/systemone` with the selected router model ID
in the JSON `model` field. This requires llama-swap v262 or newer, which added
model-body routing for this endpoint. The configured base path and bearer
credential are retained. Native answers and usage remain typed. Cancellation,
timeout or malformed completion after dispatch is uncertain: inspect the
endpoint before retrying. Closing the client connection does not prove server
cancellation or GPU release.

These are router-owned CUDA models. The router performs its normal model swap;
do not apply the external-media service-stop path to a decision request served
by that same router. Requests can evict a loaded language model. Live switching
is parked while Pi works: **do not run these commands against the live router
until a quiet window is arranged**. No automatic restoration of the prior
language model or seamless LLM tool-call continuation is claimed by this port.

The current shared form/CLI carries `state` and `questions`, not image uploads.
The model's image capability remains accessible through the native API with its
top-level `images` data URLs; UI/CLI image-input wiring and real model acceptance
are separate unfinished gates. Fixture success is not hardware or quality proof.
