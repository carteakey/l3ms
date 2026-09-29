# audio.cpp Runtimes & BreezeTTS 2 in L2M2

This document describes the `audio.cpp` local runtime setup in L2M2, mirroring the multi-tier operational design of `llama.cpp` in `L3MS`.

---

## 1. Architecture & Vendor Tier Layout

In `L3MS`, `llama.cpp` runtimes were organized under `vendor/` with upstream master tracking and dedicated maintenance scripts. L2M2 follows the exact same pattern for `audio.cpp`:

| Tier / Path | Role | Source Remote | Key Loaders & Capabilities |
|---|---|---|---|
| `vendor/audio.cpp-master` | **Gold / Master Tier** | `https://github.com/0xShug0/audio.cpp.git` (`main`) | `breeze_tts`, `heartmula`, `minimax_h3`, `marblenet_vad`, `silero_vad` |
| `vendor/audio.cpp` | Active Symlink | Symlink to `vendor/audio.cpp-master` | Default build search path |
| `$HOME/repos/audio.cpp` | Homelab Checkout | Upstream Git repository | Backing checkout for shared builds |

Binaries are compiled into:
`vendor/audio.cpp-master/build/linux-cuda-release/bin/`
- `audiocpp_cli`: Command-line multi-task audio runtime
- `audiocpp_server`: OpenAI-compatible local HTTP server with streaming support

---

## 2. Upstream Refresh & Build Management

Use `maintenance/refresh-audio-cpp.sh` to audit, update, compile, and smoke-test:

```bash
# Check upstream status, commit log, and PRs without modifying files
./maintenance/refresh-audio-cpp.sh --dry-run

# Query GitHub for active PRs on 0xShug0/audio.cpp
./maintenance/refresh-audio-cpp.sh --check-prs

# Update master branch to origin/main, compile with CUDA, and run loader smoke tests
./maintenance/refresh-audio-cpp.sh --update --build --smoke

# Compile custom model set
./maintenance/refresh-audio-cpp.sh --build --models "minimax_h3,heartmula,breeze_tts"
```

### Compiler & CUDA Specifications (RTX 4070 Ada Lovelace)
- **Target Architecture**: `sm_89-real`
- **Compiler**: GCC 16.2.1 with CUDA Toolkit 13.4 (`/opt/cuda/bin/nvcc`)
- **Key Flags**:
  - `-DENGINE_ENABLE_CUDA=ON`
  - `-DENGINE_ENABLE_NATIVE_CPU=ON`
  - `-DENGINE_ENABLE_LLAMAFILE=ON`
  - `-DAUDIOCPP_DEPLOYMENT_BUILD=ON`
  - `-DGGML_CUDA_CUB_3DOT2=ON`

---

## 3. BreezeTTS 2 Model Integration

BreezeTTS 2 is a native GGUF speech generation and voice cloning family featuring T5Gemma text conditioning, Qwen-style acoustic code generation, depth codebook decoding, and Mimi waveform decoding.

### Profile Declaration
Registered in `media-runtimes.json`:
- **ID**: `breeze-tts-2`
- **Variant**: `Breeze-TTS-2 Q8_0 GGUF, CUDA, T5Gemma conditioning + Mimi codec`
- **Tasks**: `tts`, `clon`, `design`
- **Inputs**: `text`, `audio`
- **Default Weights**: `$L3MS_MEDIA_ROOT/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf` (~5.08 GB)

### Generation CLI (`media-models/generate-breeze-tts-2.sh`)

Run from the L2M2 checkout root. The wrapper works in a standalone L2M2
checkout: it resolves `audio.cpp` from `vendor/audio.cpp`,
`L3MS_AUDIO_CPP_ROOT`, or the legacy `$HOME/repos/audio.cpp` location, and the
model from `L3MS_MEDIA_ROOT` (or an explicit model-path override).

#### Read-only readiness check

Check the selected CLI, compiled `breeze_tts` loader, Q8_0 model, and model
spec without requiring text or starting inference:

```bash
./media-models/generate-breeze-tts-2.sh --check
# The same wrapper through L2M2's profile runner:
./target/release/l2m2 --media breeze-tts-2 --extra '--check'
```

For a cloning-specific check, provide `--task clon --voice-ref PATH`. The check
only runs the CLI's loader-list command and reads local file metadata. It does
not download weights, create output directories, load Breeze-TTS, initialize a
GPU, or prove available VRAM/driver compatibility; use a later exclusive GPU
window for an actual inference smoke test.

#### 1. Standard TTS

Standard speech uses audio.cpp's native `tts` task and the default clear-speech
instruction:

```bash
./media-models/generate-breeze-tts-2.sh \
  --task tts \
  --text "Welcome to local voice synthesis with BreezeTTS 2 and audio.cpp." \
  --output "$HOME/media-output/welcome.wav"
```

Via `l2m2` unified runner:
```bash
./target/release/l2m2 --media breeze-tts-2 \
  --extra '--text "Welcome to local TTS."'
```

#### 2. Voice design

The wrapper's `--task design` is a friendly alias for native `tts`; the
voice/style direction is carried by the `instruction` request option:

```bash
./media-models/generate-breeze-tts-2.sh \
  --task design \
  --text "Welcome to local voice synthesis with BreezeTTS 2 and audio.cpp." \
  --instruction "A warm female narrator with calm pacing and studio clarity." \
  --output "$HOME/media-output/welcome-designed.wav"
```

#### 3. Prompt-audio voice cloning

Cloning uses the native `clon` task. The reference transcript is optional, but
include it when available to describe the reference speech accurately:

```bash
./media-models/generate-breeze-tts-2.sh \
  --task clon \
  --text "This speech was synthesized directly using the reference voice profile." \
  --voice-ref "/path/to/reference_speaker.wav" \
  --reference-text "Transcript of the reference audio speaker." \
  --instruction "Speak clearly and naturally."
```

#### 4. Streaming

Streaming emits incremental audio events. The wrapper keeps the merged WAV at
`--output` and directs chunk WAVs to a sibling `<output>.chunks/` directory.
Smaller frame counts can reduce time-to-first-audio at the cost of more chunk
overhead; defaults are 16 frames and a 12-frame lookahead:

```bash
./media-models/generate-breeze-tts-2.sh \
  --mode streaming \
  --stream-frames 16 \
  --stream-lookahead 12 \
  --text "Streaming voice test with low latency." \
  --output "$HOME/media-output/stream.wav"
```

---

## 4. Verification & Diagnostics

1. `--check` performs the non-generating, model-specific file/loader preflight.
2. Check native task/mode availability in the compiled CLI:
   ```bash
   ./vendor/audio.cpp/build/linux-cuda-release/bin/audiocpp_cli --list-loaders --json
   ```
   Confirm `breeze_tts` is listed. The profile exposes standard `tts` and `clon`;
   `design` is the wrapper's instruction-conditioned TTS alias.

3. Inspect overall runtime and model readiness across installed profiles:
   ```bash
   ./maintenance/setup-media-runtimes.sh check
   ```

---

## 5. Measured Baseline Performance

Benchmark runs on Yeti (RTX 4070 12 GB, Linux x86_64, CUDA 13.4):

| Task Mode | Wall Time | Audio Duration | Real-Time Factor (RTF) | Speedup | Peak Host RSS |
|---|---|---|---|---|---|
| **Standard TTS** | `1.96 s` | `5.20 s` | **0.377** | **2.65x real-time** | ~2,488 MB |
| **Voice Design** | `1.11 s` | `2.80 s` | **0.397** | **2.52x real-time** | ~2,493 MB |
| **Voice Cloning** | `3.47 s` | `7.28 s` | **0.476** | **2.10x real-time** | ~2,492 MB |
| **Streaming Mode** | Multi-chunk | Real-time | Low-latency TTFT | Incremental | ~2,490 MB |

Detailed run artifacts, logs, and seeds are recorded in [docs/experiments/breeze-tts-2-baseline.md](experiments/breeze-tts-2-baseline.md).
Sample prompt text files are saved under [prompts/](../prompts/).
