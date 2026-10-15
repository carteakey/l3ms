# Changelog

## Unreleased

- Correct the retired coordinator probe's `--run` help; it already refuses
  without model or service work. Use the direct native CLI for execution.

- Accept optional bounded image data URLs for the native d3/Clef profiles,
  forwarding top-level `images` without downloads or local file reads. Reject
  other image-bearing profiles before engine construction; check Clef question
  instructions before HTTP dispatch. Keep text-only generic router IDs usable.

- Add native llama.cpp decision profiles for d3-flash and Clef-Flash using
  `/v1/systemone` on llama-swap v262 or newer, with model selection in the JSON
  body. The stdlib bridge posts once with bounded, redacted failures and no
  redirects, proxies, simulated fallback, package installation or model
  downloads. Live acceptance remains separate.
- Clarify declared profile readiness, record the offline Nagi CPU acceptance and
  separate installed-cache evidence from clean-install, CUDA, and quality claims.
  Mark SemIf as lacking a native forward pass; `--simulate` remains demo-only.
