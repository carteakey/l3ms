# L3DS

Local Low-latency Decision System: a fast, script-first decision model toolkit.

- Keep orchestration explicit, script-first, and keyboard-friendly.
- Declare decision profiles in `decision-runtimes.json` and engine bridges in `decision-models/`.
- Focus strictly on non-autoregressive "System 1" decision models (typed choices, scores, probabilities in a single forward pass without token generation).
- Traditional generative LLMs remain in `l3ms`; latent image/audio/video models remain in `l2m2`.
- Preserve benchmarks and evaluation evidence under `suites/` and logs under `docs/`.
- Maintain standard `system_one(state, questions)` API parity across local (Nagi, Laya, SemIf) and hosted (Jev API) models.
- Track forward pass latency (P50/P95), option-flip stability, and tick deadline compliance.
