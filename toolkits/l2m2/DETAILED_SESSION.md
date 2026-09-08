# Detailed L3MS Media Session Summary

Date: 2026-08-15  
Repository: `https://github.com/carteakey/l3ms.git`  
Local checkout: `/Users/kchauhan/repos/l3ms`  
Target: Yeti Cachy (`ssh kpc-cachy-ts`)  
Target checkout: `/home/kchauhan/repos/l3ms`  
Baseline before this documentation commit: `065c58a0edd40d18e60939ef486ec58a0b87d913`

> Follow-up note (2026-08-15): the hosted `minimax-music` profile described in
> this historical session record was subsequently replaced by local
> `heartmula-music` using audio.cpp and its Q8_0 GGUF package. LTX-2.5 remains
> on official BF16 source weights with FP8 quantization at load because no
> official pre-quantized Ada-compatible 2.5 artifact was found; the newly
> reported ~18.7 GB NVFP4 transformer targets the Blackwell path and does not
> fit Yeti's 12 GB GPU. The HeartMuLa package was installed at 7,659,762,592
> bytes and produced a verified 5.04-second stereo/48 kHz WAV in 2.56 seconds
> with a one-step smoke; llama-swap remained healthy with 13 listed models.

## 1. Objective

The session completed the L3MS media-runtime path on the Yeti homelab host:

1. Make local MiniMax H3 generation usable on the RTX 4070.
2. Add declarative LTX-2.5 and MiniMax Music profiles with safe, script-first
   entry points.
3. Provide media file inputs, prompt files, H3 video persistence, and
   playable MP4 output.
4. Verify the Rust build and deploy the exact Git revision to Yeti.
5. Record what is ready, what is blocked by external access, and how to resume.

This file is the detailed session record. `HANDOFF.md` is the short operator
handoff.

## 2. Target and operational constraints

Yeti is an NVIDIA RTX 4070 host with 12,282 MiB VRAM, Ada compute capability
8.9, driver `610.57.04`, and approximately 64 GiB RAM. The host also runs a
user-level `llama-swap.service`, so media generation and LLM serving compete
for the same GPU. The practical operating rule is to unload GPU-resident
llama-swap models before H3/LTX tests and restart the service afterward.

The deployment workflow uses a clean Git clone of the exact committed branch
revision, validates it, and swaps it into the live path while preserving the
two intentional host-local overrides:

```text
/home/kchauhan/repos/l3ms/llama-swap.yaml
/home/kchauhan/repos/l3ms/model_downloader/models_config.json
```

Secrets are deliberately excluded from this document. The Hugging Face token
and Music API key stay on Yeti or in their respective credential stores.

## 3. Work completed

### 3.1 Declarative profile layer

`media-runtimes.json` now describes three profiles:

| Profile | Runtime | Variant | Tasks | State |
| --- | --- | --- | --- | --- |
| `minimax-h3` | audio.cpp | Q4_K GGUF, CUDA, staged/layerwise | music, video, TTS | experimental-local, installed |
| `ltx-2.5` | LTX-2 Python | official BF16 components, fp8-cast, CPU/disk offload | video, music | gated-local, weights missing |
| `minimax-music` | MiniMax `mmx` | hosted Music API | music | hosted, CLI installed/auth missing |

The manifest remains the source of truth for profile names, runtime,
variants, supported tasks, and input types. The Rust CLI reads it instead of
maintaining a second hard-coded media catalog.

### 3.2 Rust CLI and shell boundary

`src/media.rs` provides manifest parsing/filtering and an argv-safe boundary to
the generation wrappers. `src/cli.rs` adds:

```text
--media <profile>
--list media
--extra <wrapper arguments>
```

When `--extra` is supplied and exactly one profile matches, the launcher
selects that profile without opening the interactive picker. This is useful
for SSH and automation while preserving the keyboard-first interactive flow
when no profile is specified.

### 3.3 MiniMax H3 wrapper

`media-models/generate-minimax-h3.sh` uses the audio.cpp release-0.6 CLI and
the installed Q4_K CUDA assets. It supports:

- text-to-audio and text-to-video;
- `--prompt-file` for file-backed text input;
- video dimensions and frame controls;
- raw RGB24 JSON persistence;
- `--video-output` MP4 muxing through ffmpeg, jq, and xxd;
- safe argv handling and explicit output paths.

The default video path is deliberately conservative for 12 GB VRAM:
`832x480`, `121` frames, and a staged/layerwise loading strategy. The default
audio path uses the corresponding small latent shape. The setup check now
reports whether all playable-MP4 mux tools are available.

### 3.4 LTX-2.5 wrapper

`media-models/generate-ltx-2.5.sh` follows the official split component layout
and supports:

- `--quantization` (default `fp8-cast`);
- CPU or disk offload;
- `--prompt-file`;
- repeatable image conditioning with `--image PATH FRAME STRENGTH`;
- normalized output paths so relative paths remain valid after the wrapper
  changes into the LTX checkout.

The wrapper does not pretend that an unsupported quantized artifact is
available. It expects the official gated BF16 files and casts at load time.

### 3.5 MiniMax Music wrapper

`media-models/generate-minimax-music.sh` calls the official hosted `mmx music
generate` command and supports prompt files, lyrics files, instrumental mode,
and the CLI's lyrics optimizer. It never prints or writes the API key into
repository files.

### 3.6 Runtime bootstrap and operator docs

`maintenance/setup-media-runtimes.sh` now supports `check`,
`install-audio-cpp`, `install-ltx`, `install-music-cli`, and `install`. It
prepares `$HOME/.local/bin` for noninteractive SSH, checks all H3 files,
checks MP4 mux tools, reports credential readiness without values, and gives a
specific gated-repository error for LTX.

`maintenance/run-l3ms-kpc.sh` also prepares the user-local binary path. The
README, ARCHITECTURE, CHANGELOG, and `docs/media-generation-runbook.md` now
describe the profiles, source-of-truth boundaries, Yeti commands, official
references, and the distinction between authoritative upstream documentation
and field reports.

## 4. Model-variant decision

The question raised during the session was whether a quantized LTX model could
be downloaded directly instead of the BF16 variants.

The deployed decision is to keep the official BF16 split files and use
`--quantization fp8-cast` at load time with CPU/disk offload. Upstream's LTX
optimization guidance describes FP8 casting as a load-time conversion from the
BF16 weights. The tempting NVFP4 route is not appropriate for this target:
upstream documents the NVFP4 cast/prequant path for Blackwell GPUs with
`SM >= 10`, while Yeti is Ada `SM 8.9`. The official INT8 ConvRot route is
documented in the ComfyUI integration, whose stated VRAM expectation is 32 GB
or more. Neither is a verified direct-download solution for this 12 GB Ada
host.

This can be revisited if Lightricks publishes a supported Ada artifact or the
runtime gains a tested conversion path. Until then, downloading an arbitrary
community quant would add compatibility and provenance risk without solving
the hardware constraint.

## 5. Asset installation and readiness

### 5.1 H3 assets installed on Yeti

```text
/home/kchauhan/models/media/MiniMax-H3-Q4-GGUF/dit.gguf                  15,502,530,720 bytes
/home/kchauhan/models/media/MiniMax-H3-Q4-GGUF/text_encoder_q4_k.gguf   15,270,376,000 bytes
/home/kchauhan/models/media/MiniMax-H3-Q4-GGUF/audio_vae_folded_f16.gguf    284,562,816 bytes
/home/kchauhan/models/media/MiniMax-H3-Q4-GGUF/video_vae.gguf            1,374,245,472 bytes
/home/kchauhan/repos/audio.cpp/build/linux-cuda-release/bin/audiocpp_cli
```

The setup check reports the GPU, CLI, all four weights, and the mux tools as
available.

### 5.2 LTX assets still missing

The checkout `/home/kchauhan/repos/LTX-2` exists and Hugging Face auth
material is detected, but the gated repository denied the download with:

```text
Error: Access denied. This repository requires approval.
```

The five expected files are:

```text
diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors
text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
vae/ltx-2.5-video-vae-conv-bf16.safetensors
vae/ltx-2.5-audio-vae-bf16.safetensors
latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
```

The failed attempts left only Hugging Face cache/lock material; no model
weights were copied or partially substituted. Approval must be granted by the
account owner in the Hugging Face UI before retrying:

```sh
cd /home/kchauhan/repos/l3ms
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
LTX_DOWNLOAD=1 ./maintenance/setup-media-runtimes.sh install-ltx
```

### 5.3 Music auth still missing

The `mmx` executable is installed at the user's local Node bin path, but
`mmx auth status --non-interactive` reports missing authentication. The user
must run `mmx auth login --api-key ...` directly on Yeti. No key was entered,
printed, or committed during this session.

## 6. Verified H3 outputs

The full standard smoke was run after deploying the current code. It used one
step, the default `832x480` video shape, `121` frames, layerwise Q4_K loading,
`L3MS_H3_THREADS=4`, `L3MS_H3_WEIGHT_CONTEXT_MB=256`, and
`L3MS_H3_MLP_CHUNK_TOKENS=512`.

Output directory:

```text
/home/kchauhan/media-output/h3-default-smoke-20250815-160249
```

The timestamped path above is preserved exactly as recorded by the run. The
output included:

```text
minimax-h3-20250815-160249.wav
minimax_h3_video_rgb24.json       297,124,004 bytes
default-480p.mp4                  366,264 bytes
```

The MP4 is H.264/AAC, `832x480`, 24 fps, and 5.166667 seconds. The one-step
run took approximately 73.6 seconds; the audio duration was 5,175 ms and the
reported real-time factor was 14.2137. A smaller five-frame smoke also
produced a valid 256x256 H.264/AAC MP4 at:

```text
/home/kchauhan/media-output/h3-mp4-smoke-20250815-144708/clip.mp4
```

After each smoke, llama-swap was restarted and its API returned healthy.

## 7. Verification matrix

The exact deployment clone passed these gates:

```text
cargo fmt --all -- --check                                      PASS
cargo clippy --all-targets --all-features --locked -- -D warnings PASS
LLAMA_SWAP_URL=http://192.0.2.1:81 cargo test --all-targets --all-features --locked PASS
cargo build --release --all-features --locked                    PASS
Python compile checks                                           PASS
14 model-downloader unit tests                                  PASS
l3ms --version                                                  PASS
l3ms --quickstart                                               PASS
l3ms --list bench                                                PASS
run-l3ms-kpc.sh --list media                                     PASS
H3 --help headless auto-selection                               PASS
```

The Rust test command used an unroutable placeholder URL so tests exercised
the local boundary without contacting the live service. The deployment smoke
then checked the real target separately.

## 8. Live deployment evidence

The handoff commit was intentionally not pushed to the GitHub origin because
these documents contain internal deployment paths and the Tailscale address.
Instead, the exact local branch revision was transferred to Yeti as a private
Git bundle, cloned into a clean staging checkout, validated, and swapped into
the live path. The previous checkout was moved to a timestamped backup rather
than deleted. The user-level `llama-swap.service` was restarted successfully.

The latest live health check was an authenticated `GET` of:

```text
http://127.0.0.1:8080/v1/models
```

It returned HTTP 200 with 13 models. The current Tailscale IPv4 is
`100.110.126.24`, so the service URL for a Tailscale client is:

```text
http://100.110.126.24:8080/v1
```

At one post-restart check, `nvidia-smi` reported only 44 MiB of 12,282 MiB
used, confirming that the service was healthy without a resident model. Do
not infer that media generation and an actively loaded large LLM will fit
simultaneously; unload/reload remains an operational requirement.

## 9. Computer-assisted input and research

The computer-use skill was used read-only to inspect local Finder and Voice
Memos state. This confirmed that GUI-produced files can be passed to the
script-first wrappers as explicit paths; no credentials were entered and no
external UI state was changed.

The session also reviewed upstream documentation and community reports. The
official sources are authoritative for supported model formats and hardware
constraints. Reddit reports were treated only as field evidence for H3 12 GB
settings, DynamicVRAM behavior, practical 480p/short-frame timings, and LTX
workflow ideas. They are linked in `docs/media-generation-runbook.md` and do
not override the upstream compatibility boundary.

## 10. Resume checklist

Run the following after the external access blockers are resolved:

```sh
ssh kpc-cachy-ts
cd /home/kchauhan/repos/l3ms
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# LTX: only after the account has accepted the gated model terms.
LTX_DOWNLOAD=1 ./maintenance/setup-media-runtimes.sh install-ltx

# Music: enter the user's private key at the target host.
mmx auth login --api-key '<key entered locally>'

./maintenance/setup-media-runtimes.sh check
./maintenance/run-l3ms-kpc.sh --list media
```

Then run one small LTX and one small Music generation, inspect their output,
and confirm that `llama-swap.service` is active afterward. Keep the two
host-local override files when deploying future commits. Use the deployment
skill's clean-clone/swap/rollback procedure; do not copy a working tree over
the live checkout with an archive or `scp`.

## 11. Relevant files and commits

Key files are listed in `HANDOFF.md`. The media implementation was developed
across these commits, ending at the pre-handoff baseline:

```text
b637014 Add media generation runtimes
e2ab2a8 CUDA 13 safe H3 setup
7a21ae5 Cachy CUDA media bootstrap
c2e7650 Harden hosted music CLI bootstrap
360f8cb Persist H3 video artifacts
5e48410 Add LTX image-conditioned media input
989a2cc Report media credential readiness
c824512 Document LTX quantization compatibility
6c21af6 Add media file inputs and H3 MP4 output
43b050d Report H3 MP4 tooling readiness
065c58a Make media tools available over SSH
```

This handoff documentation is intentionally the next commit. Git remains the
source of truth for the deployed implementation; this file records the
operational evidence and external blockers without embedding private
credentials or pretending that gated assets are installed.

## 12. Reference links

- [LTX-2 quick start](https://github.com/Lightricks/LTX-2#-quick-start)
- [LTX-2 optimization guide](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/docs/optimization.md)
- [LTX-2.5 gated model](https://huggingface.co/Lightricks/LTX-2.5)
- [Official ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo/)
- [audio.cpp MiniMax H3 guide](https://github.com/0xShug0/audio.cpp/blob/release-0.6/docs/community_models/minimax_h3.md)
- [H3 12 GB DynamicVRAM report](https://www.reddit.com/r/StableDiffusion/comments/1vghw05/minimax_h3_on_a_12gb_card_runaway_perstep/)
- [H3 tips and tricks](https://www.reddit.com/r/StableDiffusion/comments/1vegtac/minimax_h3_tips_and_tricks_and_what_i_experienced/)
- [H3 480p/short-video field report](https://www.reddit.com/r/StableDiffusion/comments/1vd9o0r/minimax_h3_1080p_25_seconds_text_to_video_in/)
- [LTX workflow field report](https://www.reddit.com/r/StableDiffusion/comments/1qnh696/ltx2_workflows/)
- [LTX audio/image-to-video field report](https://www.reddit.com/r/StableDiffusion/comments/1qbwc3c/ltx2_audio_image_to_video/)
- [LTX audio input and image-to-video field report](https://www.reddit.com/r/StableDiffusion/comments/1q6ythj/ltx2_audio_input_and_i2v_video/)
