# MiniMax-H3: reference quality and a tiny Batman test

Captured 2026-09-09. This is a sourced experiment note, not a benchmark result.

## Source and evidence

[Original thread](https://www.reddit.com/r/StableDiffusion/comments/1w6nwp4/).
The user-provided post and comments are preserved verbatim in
[sources/minimax-h3-batman-reddit-20260903.txt](sources/minimax-h3-batman-reddit-20260903.txt).
We have not inspected the original video or an embedded workflow export.

## Author's reported recipe

- RTX 3070 with **8 GB VRAM and 32 GB system RAM**. The 64 GB RAM figure
  belongs to a different commenter, not the author.
- Standard ComfyUI reference template, **INT8 ConvRot reference model**, regular
  **20 steps**, no Turbo LoRA. Exact checkpoint revision and sampler settings
  were not supplied in the pasted author recipe.
- Film screenshots for character/environment references. Character sheets were
  also tried to preserve costume details from different angles; these are
  conditioning inputs, not evidence of character-model training.
- Roughly **15 seconds of clean dialogue** as a voice reference. Write the new
  spoken words in the prompt. Clean source audio, pauses, and room tone matter.
- Explicitly request no generated music; add music separately when editing.
- Test at about **0.2–0.3 MP**, finish around **0.4–0.5 MP**. Reference-image
  size also affects cost. The author reports 3–15 second clips and 5–20 minutes
  per render, not a reproducible speed guarantee for this machine.
- Short dialogue, close-ups, static scenes, repeated references, selection of
  good takes, and editing/retiming in Resolve. Some latent upscaling experiments,
  not one mandatory finishing recipe. Reported total: 4–5 hours over two days.
- Author says multiple references work, estimates nine and prefers six or fewer;
  treat this as a personal heuristic, not a documented hard limit.

## Other commenters' experiments — keep separate

- FastH3 FL2V with 3 steps for drafts; 4–6 for better audio, five a suggested
  compromise. One 3070/64 GB RAM user reports 220–250 seconds for 10 s at 0.5 MP
  with three steps, roughly 300 s with five. Not our hardware measurements.
- Larry + LightX2V LoRA combinations and a Larry v4 step600 suggestion with
  video shift 12 / audio shift 3. A warning about a pruned variant's audio is
  anecdotal. Do not combine these into the author's no-LoRA baseline.
- Windows launch flags are platform/version-specific. Do not paste them into
  this Linux audio.cpp runtime. Evaluate acceleration later with the same prompt,
  references, seed, resolution, and separate audio/visual judgements.

## What our installed runtime can test

L2M2 currently wraps audio.cpp commit `bb15edd`, a Q4_K FL2VA package, and text
input. It does not expose ComfyUI reference-image/voice conditioning. The current
wrapper cannot test voice identity or screenshot consistency from the post.
[ComfyUI's native reference node](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py)
provides image/video/audio references; that is the appropriate later route for
matching the author's setup. No ComfyUI installation or new weights were added.

## First experiment

Run `maintenance/test-h3-batman.sh`. Default: 640×384 (0.246 MP), 124 requested
frames (~5.17 s at 24 fps before runtime alignment), seed 42, **20 steps**.
The prompt uses the plant gag with generic generated voices; no voice reference
or film image is available in this test. One take, one camera, one short exchange.
The script refuses a busy GPU or active llama-swap service; it does not kill
another workload. Output goes to a fresh directory under ~/media-output.
`H3_TEST_STEPS=1` can check the pipeline only; do not judge quality from that run.

Record wall time, actual dimensions/duration, command, log, prompt and seed.
Judge: visible cowl ears/plant, stable faces/hands, readable joke, speaker order,
intelligible dialogue, pause, no unwanted music. A playable MP4 is only a pipeline
pass, not proof that the creative or audio quality succeeded.

For the later reference test, use one scene and one/two character references,
clean voice samples, a pinned Ref2VA workflow and checkpoint, and the same short
scene at 20 steps without acceleration. Save the workflow JSON and input hashes
beside each output, then compare acceleration one change at a time.
