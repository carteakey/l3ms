# Native decision models through llama-swap: CLI and keyboard workflow

Model reference, latency and candidates: [decision-models.md](decision-models.md).

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

The root CLI accepts the existing strict JSON input shape and an optional
`images` array of base64 `data:image/...` URLs:

```json
{"state":{"message":"I was charged twice."},"questions":{"route":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Charges and refunds","technical":"Software faults"}}}}
```

For an image-capable d3/Clef profile, add up to eight image data URLs at the
top level (the full input remains limited to 1 MiB):

```json
{"state":{"message":"What is in this picture?"},"questions":{"route":{"type":"choice","instructions":"Classify the picture.","criteria":{"receipt":"A receipt","other":"Something else"}}},"images":["data:image/png;base64,..."]}
```

Image inputs are accepted only for the local `vllm-d3-flash` and
`cloudflare-clef-flash` profiles and are forwarded unchanged in the native
`images` field. URLs and filesystem paths are not fetched or opened. Nagi and
hosted Jev reject image-bearing requests before dispatch; their existing
text-only behavior is unchanged. The keyboard form exposes a JSON array field
for these profiles, and standalone L3DS accepts the same optional field in its
bounded `--input-stdin` envelope.

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
by that same router. Requests can evict a loaded language model. Run
them against the live router only when its loaded model can be evicted. No
automatic restoration of the prior language model or seamless LLM tool-call
continuation is claimed; an external client can resume a saved conversation
after the model returns (see [live acceptance](migration/live-port-acceptance.md)).

The form/CLI and standalone runner carry optional bounded image data URLs, but
do not decode, resize, or fetch them. A valid data URL does not establish that a
particular model/projector accepts its image format or content. Fixture success
is not hardware or quality proof.

Clef requires `instructions` on every question; supply a meaningful instruction
alongside native `criteria`. Missing/null instructions are refused before HTTP.

On 2026-10-15 both profiles returned real typed billing decisions through a
private v265 router with GPU offload disabled. d3 also identified a synthetic
green PNG through the image input path. A Clef image fixture lacked required
instructions and was rejected; its corrected image request remains unverified.
These are narrow native CPU sanity checks, not GPU, quality or latency evidence.
