# Decision models on llama.cpp

Native decision models answer typed questions about a `state` with a probability for
every allowed answer, in a forward pass with no text generation. There are three
question types: `choice` (pick one option), `noul` (yes/no probability) and `score`
(graded levels). llama.cpp serves them at `POST /v1/systemone` (TypeSafe System One
API). Images go in `images` as data URLs.

Snapshot date: 2026-10-12. Add new entries with `/add-decision-model`
(`.claude/commands/add-decision-model.md`).

## What is served

| llama-swap id | Model | Quant | VRAM | Inputs |
|---|---|---|---|---|
| `vllm-d3-flash` | vLLM-SR d3-flash 9B (Qwen3.5-9B) | Q8_0 + mmproj BF16 | 11.5 GB | text, image |
| `cloudflare-clef-flash` | Cloudflare Clef-Flash 9B (Qwen3.5-9B) | Q4_K_M + mmproj BF16 | 8.9 GB | text, image |

Both run on `vendor/llama.cpp-d3` (macro `d3_server`). It's a worktree of
`vendor/llama.cpp-master` detached at `efe3626b1` (#30383, vllm-d3 support, merged
2026-10-12), built separately so the gold Qwen tier binary is untouched.

Common flags: `-ngl 99 -fa on -c 8192 -b 8192 -ub 8192 --parallel 1`.
`ub = ctx` is required. vllm-d3, clef, laya and lfm2-d1-omni evaluate the whole
prompt (images + state + questions) in one batch, so an 8192 ubatch is the hard
prompt limit: about 4 full-size (1.6 MP) images for d3, more for Clef (see below).

### Calling it

llama-swap v247 does not route `/v1/systemone` (it returns 404). Use the passthrough path:

```bash
curl -s -H "Authorization: Bearer $LLAMA_SWAP_API_KEY" \
  localhost:8080/upstream/vllm-d3-flash/v1/systemone \
  -H 'Content-Type: application/json' -d '{
  "state": "Customer: I was charged twice and nobody replied.",
  "images": ["data:image/png;base64,..."],
  "questions": {
    "route":   {"type":"choice","instructions":"Which team?","criteria":{"billing":null,"shipping":null}},
    "angry":   {"type":"noul","instructions":"Is the customer angry?"},
    "urgency": {"type":"score","instructions":"How urgent?","criteria":["can wait","today","right now"]}
  }}'
```

- `state: null` is accepted only by vllm-d3, lfm2-d1 and lfm2-d1-omni (to ask about images only).
- Video: llama.cpp accepts video only on chat completions, not `/v1/systemone`, so
  these endpoints are text + image. (Both models' Python releases do take video.)
- A Clef server is decision-only; text generation is not available on it.
- Requesting either model swaps out whatever is loaded (12 GB card). `globalTTL: 600`.

## Measured latency (RTX 4070 12 GB, 2026-10-12)

Warm: first request discarded, 3 runs each, curl from localhost, standalone server.

| Request | d3-flash Q8_0 | Clef-Flash Q4_K_M |
|---|---|---|
| Text, 3 questions (≈325 tokens) | ~110 ms | ~85 ms (65 ms via router) |
| 1024×768 image + 4 questions | ~1.7 s, 3,461 tokens | ~0.5 s, 1,146 tokens |
| First request after unload (router) | ~4 s | ~13 s |

### Why images cost d3 so much more

- **d3 runs one forward pass per question.** Every pass includes the image, so 4 questions
  means the image is processed 4 times (3,461 ≈ 4 × (≈768 image + ≈100 text) tokens).
- **d3's work can't be shared between questions.** Its attention is non-causal
  (`attention_mode: noncausal_full_attention`, GGUF `causal_attention = false`), so the
  image tokens attend to the question that follows them. Their hidden states differ
  per question, and no prefix KV can be reused.
- **Clef answers jointly.** It puts all questions in one prompt and scores every option
  in one pass, so the image is processed once (1,146 ≈ 768 image + ≈380 text).
- **The gap grows with question count.** It's roughly even at 1 question per image and
  about N× at N questions. Q4 vs Q8 is a minor factor; prompt processing is compute-bound.
- **The trade-off:** d3's answers are independent and question-conditioned. Clef's are
  cheaper, but an answer can depend on which other questions share the request.

### Smoke checks used (not an evaluation)

- **Text:** the llama.cpp README `/v1/systemone` example (double-charged customer).
  Both models pick billing (d3 0.996, Clef 0.979). They disagree on "angry"
  (d3 0.92, Clef 0.43), which has no ground truth.
- **Image:** a synthetic red circle (left) and blue square (right). Both get
  circle and red, and both say no cat. The "INVOICE" text didn't render (ImageMagick
  has no font here, so it drew empty boxes); both said no word, d3 at 0.12 and Clef at 0.006.
  Check that test images actually render before trusting a text-in-image question.

## Leaderboard: Jev Decision Index

`https://huggingface.co/spaces/multimodalart/jev-decision-index` (data:
`data/index.json`, `data/methodology.json`). It's a third-party board, edition
"Decision Index 0.3.2", about 110k requests over 37 benchmarks in six text categories
(knowledge, language, retrieval, tools, arts, games). All models run at full precision
on one 96 GB RTX PRO 6000, with hosted TypeSafe Jev as the reference. Fetched 2026-10-12
(index generated 2026-10-12T13:25Z), 148 models.

**Limits:** it has no vision category (the d3 announcement's "71.6 vision" score comes
from another source). Its latency figures are single-request numbers on a datacenter
card and don't compare with the table above. Quantized local builds may score lower.

| Rank | Model | Skill | ECE | Median latency | Runs here? |
|---|---|---|---|---|---|
| 1 | vLLM-SR d3 27B | 66.3 | 0.043 | 97 ms | no: dense 27B, 19 GB at Q4 |
| 3 | Perplexity Decider v1.1 27B | 62.8 | 0.071 | 104 ms | no decision GGUF (Livesport conversion lost the head) |
| — | Jev (hosted reference) | 60.1 | — | 524 ms (network) | API |
| 12 | Kev 27B | 58.8 | **0.022** | 135 ms | offload only (15.8 GB Q4_K_M) |
| 25 | **vLLM-SR d3-flash 9B** | **55.8** | 0.056 | 54 ms | **served** |
| 31 | Bespoke Nimble 9B v3 | 54.7 | 0.123 | 74 ms | yes (text only) |
| 32 | Cloudflare Clef 27B | 53.1 | 0.040 | 102 ms | no |
| 33 | vLLM-SR d3-mini 4B | 52.6 | 0.056 | 50 ms | yes |
| 44 | Cloudflare Clef-Omni 30B-A3B | 48.3 | 0.040 | 191 ms | no GGUF yet |
| 47 | **Cloudflare Clef-Flash 9B** | **47.6** | **0.025** | 53 ms | **served** |
| 76 | LiquidAI d1-3B | 38.9 | 0.047 | 16 ms | yes |
| 77 | vLLM-SR d3-nano 2B | 38.9 | — | 38 ms | yes |
| 130 | LiquidAI d1-omni-600M | 9.1 | — | 11 ms | yes (audio) |
| 138 | Laya | 4.4 | 0.140 | 6 ms | yes, but not useful |

"Skill" is the board's `balanced_skill`. ECE is expected calibration error; lower means
the stated probabilities better match observed accuracy.

**Reading for this box:**
- **d3-flash is the accuracy pick.** It's about 8 points over Clef-Flash, ahead in
  language and retrieval.
- **Clef-Flash is the calibration and multi-question image pick.** It has the best ECE
  of everything here that fits.
- **Laya and Julia-1 aren't worth wiring.** They're very fast but near the bottom, with poor
  calibration and about 76% coverage.

## Which leaderboard models can llama.cpp actually serve

A GGUF is only a decision model if its metadata has `<arch>.decision.type`.
Generic community conversions (e.g. mradermacher) of LoRA-plus-head or custom-head
models keep the backbone and lose the head, so `/v1/systemone` returns 501.
Check before downloading:

```bash
maintenance/gguf-decision-check.py <hf-repo> [<file.gguf>]   # range-reads the header only
```

Supported types at `efe3626b1`: openjev, lev, kev, nimble, laya, clef, pplx-decider,
vllm-decision2, vllm-d3, lfm2-d1, lfm2-d1-omni.

Header checks run 2026-10-12:

| Repo | Result |
|---|---|
| ggml-org/vllm-d3-flash-GGUF, ggml-org/vllm-d3-mini-GGUF | decision (`vllm-d3`) |
| ggml-org/Clef-Flash-GGUF, ggml-org/Clef-GGUF | decision (`clef`) |
| ggml-org/OpenJev-GGUF (27B) | decision (`openjev`) |
| ggml-org/Bespoke-Nimble-9B-v3-GGUF | decision (`nimble`) |
| kdercksen/Kev-27B-GGUF | decision (`kev`), community upload with 0 downloads, Q4_K_M 15.8 GB / Q6_K 21.0 GB |
| Livesport/pplx-decider-v1.1-27b-GGUF (#3) | **no** — plain qwen35 conversion |
| Livesport/rune-26b-a4b-v3-GGUF (Surogate Rune, #21) | **no** — plain gemma4 MoE |
| mradermacher Xor-26B-A4B (#22), GEV-26B-Decide (#23), shisa-de-2 (#9) | **no** — plain gemma4 MoE |
| mradermacher decider-35b-a3b (#42) | **no** — plain qwen35moe |
| mradermacher torchcast-decision-27b / -12b (#8), jebadiah-27b, drex-v1.5 | **no** |
| frontier-infra/jebadiah-9b-v2-GGUF | **no** |
| Blink v0.3 26B-A4B (#19), Darwin-27B-ZTC (#4), Sieve (#7) | no GGUF found |

The 26B-A4B / 35B-A3B mixture-of-experts entries would suit this box (experts in RAM,
like the Qwen3.8 tiers), but none has a decision-type GGUF yet. Recheck as
ggml-org conversions appear.

## Untested candidates that fit

- **Kev 27B (Q4_K_M, 15.8 GB):** the only higher-ranked native-decision GGUF within
  reach (#12, 58.8, best ECE on the board, 0.022). It's dense, so about 4–5 GB of layers
  would sit on CPU. Decision calls are pure prompt processing, so CPU layers hurt
  directly. Expect seconds, not ms (not measured). Text only.
- **Bespoke Nimble 9B v3:** close to d3-flash on skill (54.7) with worse calibration
  (0.123), text only. Little reason to add beside d3-flash.
- **d3-mini 4B, LiquidAI d1-3B:** lower latency, lower skill. Add only if latency is the
  constraint. d1-omni-600M is the only local audio-input option.

## Open work

- **L3DS engine:** `~/repos/l3ds/decision-models/runner.py` (later `toolkits/l3ds/`) has
  no llama.cpp engine, and its local engines return simulated answers. Add a `llamacpp`
  engine that POSTs to `/upstream/<id>/v1/systemone`, then run `suites/routing-gate.json`
  against both served models for real accuracy and latency.
- **llama-swap routing:** check whether a newer llama-swap routes `/v1/systemone` by
  `model`, to drop the `/upstream/` path.
- **Vision board:** find the source of the vision scores in the d3 announcement.
