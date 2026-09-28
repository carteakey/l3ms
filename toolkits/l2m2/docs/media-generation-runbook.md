# Media generation runbook

L2M2 exposes media generation as an explicit, script-first CLI surface. The
registry in [`media-runtimes.json`](../media-runtimes.json) is the discoverable
source of truth; the executable wrappers in `media-models/` keep each upstream
runtime's flags and credentials separate from the Rust launcher.

## Target profile: Yeti/Cachy

Yeti has an RTX 4070 with 12 GiB of VRAM, 64 GiB of RAM, and CUDA. The selected
profiles therefore use the smallest practical local variants:

- **MiniMax-H3:** audio.cpp, the normal `minimax_h3_q4_k` GGUF package, staged/layerwise weights, memory saver, and a 32×32 audio-first default. `--video` switches to 832×480 and 121 frames (about five seconds at 24 fps).
- **LTX-2.5:** the official 22B distilled BF16 transformer, quantized to FP8 while loading with `fp8-cast`, plus CPU offload. The convolutional video VAE avoids requiring the diffusion-VAE attention extra.
- **HeartMuLa:** audio.cpp with the published Q8_0 GGUF package, CUDA, and memory saver. This provides an offline lyrics/tags-to-song path.
- **Breeze-TTS-2:** audio.cpp native GGUF package (`breeze_tts_2_q8_0`), CUDA, T5Gemma conditioning, and Mimi waveform decoding. Supports standard TTS, instruction-conditioned voice design, reference-audio voice cloning, and low-latency streaming (~0.37 RTF on RTX 4070).

These are conservative starting points, not a claim that every route will fit
or be fast on a 12 GiB card. Keep llama-swap's GPU model unloaded while running
H3, HeartMuLa, Breeze-TTS-2, or LTX.

## Install and inspect

On Yeti, from the L2M2 checkout:

```bash
./maintenance/setup-media-runtimes.sh install-audio-cpp
./maintenance/setup-media-runtimes.sh install-breeze
./maintenance/setup-media-runtimes.sh install-ltx
./maintenance/setup-media-runtimes.sh install-music
./maintenance/setup-media-runtimes.sh check
```

To refresh upstream `audio.cpp` code, audit active PRs, and rebuild binaries:

```bash
./maintenance/refresh-audio-cpp.sh --update --build --smoke
```

`install-audio-cpp` updates/clones `audio.cpp`, builds the CUDA CLI/server for
`minimax_h3`, `heartmula`, and `breeze_tts`, and installs the model packages under
`${L3MS_MEDIA_ROOT:-$HOME/models/media}`. `install-breeze` installs the Q8_0
package and verifies loader readiness. It does not touch llama-swap or port 8080.
On CachyOS it discovers `/opt/cuda` even from a non-interactive SSH shell.

LTX-2.5 is a gated Hugging Face model. Accept the model terms and log in with a
Read-scoped token, then opt into the roughly 66 GiB download:

```bash
export HF_TOKEN='…'        # kept in the shell environment, never committed
LTX_DOWNLOAD=1 ./maintenance/setup-media-runtimes.sh install-ltx
```

The script downloads only the files needed by the distilled wrapper: the
transformer, Gemma 4 projection encoder, convolutional video VAE, audio VAE,
and spatial upscaler. If authentication is not ready, `install-ltx` still sets
up the checkout and reports the missing gated files.

The official LTX-2.5 repository currently lists BF16 split components and its
official pipeline documents load-time `fp8-cast`. Web/model-registry checks on
2026-08-15 did not find an official pre-quantized Ada checkpoint. A newly
reported 22B distilled NVFP4 transformer is about 18.7 GB before runtime
overhead and targets Blackwell's native NVFP4 path (`SM >= 10`), so it is not
a viable RTX 4070 12 GB selection. Yeti therefore keeps the BF16 source file,
stores eligible transformer linears in FP8 during inference, and offloads to
CPU. If a compatible artifact is released later, pass `--transformer PATH`
with its matching `--quantization` policy after validating it against the
official `ltx-pipelines` loader.

## CLI usage

List the profiles without contacting llama-swap:

```bash
cargo run --locked -- --list
```

Run a profile interactively, optionally filtering the picker:

```bash
cargo run --locked -- --media h3
cargo run --locked -- --media ltx
cargo run --locked -- --media music
```

The wrappers use argv-safe `--extra` values. Prompts, lyrics, and local file
paths remain one argument and are never evaluated as shell code.

MiniMax-H3 audio-first music/audio:

```bash
cargo run --locked -- --media minimax-h3 --extra \
  '--prompt "slow modular synth arpeggio, warm tape saturation" --steps 20'
```

MiniMax-H3 short video with synchronized audio (writes a playable MP4 when
`ffmpeg`, `jq`, and `xxd` are installed, while retaining the raw RGB24 JSON):

```bash
cargo run --locked -- --media minimax-h3 --extra \
  '--prompt "a rainy neon street, locked camera, reflections in puddles" --video'
```

Computer-assisted prompt input can use a local file instead of shell quoting:

```bash
cargo run --locked -- --media minimax-h3 --extra \
  '--prompt-file ./prompt.txt --video --video-output ./rainy-street.mp4'
```

The same `--prompt-file` option is available on LTX-2.5 and HeartMuLa;
HeartMuLa also accepts `--lyrics-file` for a locally authored lyric sheet.

LTX-2.5 text-to-video/audio:

```bash
cargo run --locked -- --media ltx-2.5 --extra \
  '--prompt "a paper boat crosses a puddle as thunder rolls in" --frames 121'
```

LTX-2.5 image-conditioned video (the image is passed as data, not shell code):

```bash
cargo run --locked -- --media ltx-2.5 --extra \
  '--prompt "the camera slowly pushes toward the boat" --image ./boat.png 0 0.8 --frames 121'
```

Local Q8_0 HeartMuLa music:

```bash
cargo run --locked -- --media heartmula-music --extra \
  '--prompt "dreamy ambient electronica with a gentle pulse" --instrumental'
cargo run --locked -- --media heartmula-music --extra \
  '--prompt "upbeat indie pop" --tags "pop,bright,drums,vocal" --lyrics-file ./lyrics.txt'
```

Local Q8_0 Breeze-TTS-2 speech generation, voice design, and voice cloning:

```bash
# Standard TTS / voice design:
cargo run --locked -- --media breeze-tts-2 --extra \
  '--text "Welcome to local text-to-speech with Breeze-TTS-2." --instruction "A warm female narrator with calm pacing."'

# Prompt-audio voice cloning:
cargo run --locked -- --media breeze-tts-2 --extra \
  '--task clon --text "Read this with the reference speaker tone." --voice-ref ./sample.wav --reference-text "Transcript of sample."'

# Low-latency streaming mode:
cargo run --locked -- --media breeze-tts-2 --extra \
  '--text "Streaming speech generation test." --mode streaming --stream-frames 16'
```

The default output directory is `$HOME/media-output`; set
`L3MS_MEDIA_OUTPUT_DIR` or pass `--output` to choose another destination.

## Inputs and computer-assisted workflows

The profiles expose text-file and lyrics-file inputs for computer-assisted
workflows, plus still-image conditioning for the LTX DistilledPipeline. H3's
official model family is multimodal, but the local audio.cpp `FL2VA` package
currently exposes the text-to-audio/video route only. LTX image paths use
explicit `--image PATH FRAME STRENGTH` triples; audio-to-video and
video-to-video work still require their corresponding upstream LTX pipeline and
model assets.

This keeps computer-use workflows safe: a UI or notebook can write a prompt or
lyrics file and invoke `l2m2 --media … --extra …`, while local paths are passed
as data. No browser automation or API key is required by L2M2 itself.

## Upstream references and research

- [MiniMax-H3 official repository](https://github.com/MiniMax-AI/MiniMax-H3) —
  local H3-Base variants, multimodal capabilities, 768p local limit, and the
  recommended SGLang/vLLM/diffusers/ComfyUI runtimes.
- [audio.cpp release-0.6](https://github.com/0xShug0/audio.cpp/tree/release-0.6)
  and its [MiniMax-H3 community model guide](https://github.com/0xShug0/audio.cpp/blob/release-0.6/docs/community_models/minimax_h3.md)
  — GGUF package layout, CLI options, and staged/layerwise memory controls.
- [LTX-2.5 official quick start](https://github.com/Lightricks/LTX-2#-quick-start)
  — gated components, distilled pipeline, image conditioning, and
  `fp8-cast`/offload guidance.
- [audio.cpp music-generation guide](https://github.com/0xShug0/audio.cpp/blob/release-0.6/docs/music_generation.md)
  and [Q8_0 HeartMuLa package](https://huggingface.co/audio-cpp/audio.cpp-gguf)
  — local lyrics/tags generation, package layout, and quantization status.
- Reddit reports from [the 12 GB DynamicVRAM thread](https://www.reddit.com/r/StableDiffusion/comments/1vghw05/minimax_h3_on_a_12gb_card_runaway_perstep/),
  [the H3 tips thread](https://www.reddit.com/r/StableDiffusion/comments/1vegtac/minimax_h3_tips_and_tricks_and_what_i_experienced/),
  and [the 16 GB timing thread](https://www.reddit.com/r/StableDiffusion/comments/1vjztod/26_sec_videos_on_16_gb_vram_rtx_5070_ti_and_only/) corroborate the
  practical advice to start at low resolution with substantial system RAM and
  expect multi-minute generation. Reddit is treated as field evidence, not as
  the source of runtime flags or model compatibility claims.
- A newer 12 GB H3 report gives a useful planning baseline: roughly 864×480,
  124 frames, and 20 steps completed in under nine minutes with 32 GB system
  RAM. Other reports describe reference-video/audio paths stalling or crashing
  on 12 GB, so this L2M2 profile intentionally keeps inputs text-first and
  exposes still-image conditioning only where the selected runtime supports it.
