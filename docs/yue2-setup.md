# YuE2 Foundation Music Model Setup & Runbook

YuE2 (developed by Multimodal Art Projection / M-A-P) is an open-weights foundation model for full-song and long-form music generation, unifying symbolic planning (ABC score) and high-fidelity audio synthesis (48 kHz stereo).

## Architecture

![YuE2 Architecture](https://raw.githubusercontent.com/multimodal-art-projection/YuE/main/assets/architecture.png)

1. **Symbolic Planning (Chain-of-Thought)**:
   - Takes a `style` prompt and `lyrics` (with structural cues like `[Verse]`, `[Chorus]`).
   - Autoregressively plans a melody-and-chord composition expressed in ABC musical notation.
   - Modes: `full` (plans melody + chords), `melody` (melody only), or `off` (direct generation).
   - Alternatively, accepts an explicit `--abc-file` score for exact composition rendering and zero-shot covers.

2. **Mixture-of-Transformers (MoT) Backbone (3B)**:
   - Autoregressively predicts semantic music tokens conditioned on the symbolic plan and lyrics.
   - Synthesizes continuous acoustic latents via non-autoregressive (NAR) flow matching.

3. **Listening Decoder (YuE2-Vae)**:
   - High-fidelity variational autoencoder converting acoustic latents to 48 kHz stereo FLAC.

## Hardware & Quantization on Yeti (RTX 4070 12 GB)

- **FP8 AR Quantization**: On Ada Lovelace architectures (compute capability 8.9), YuE2 dynamically quantizes autoregressive linear projections to E4M3 FP8 via `torch._scaled_mm`, fitting generation inside 12 GB VRAM.
- **Memory Budgeting**: `--budget 12` reserves 2 GiB for system overhead and tunes VAE chunk frames to 512.
- **CPU Offload**: `--offload-ar` unloads the autoregressive weights from GPU to CPU before NAR flow matching begins.

> [!IMPORTANT]
> Always ensure llama-swap and competing GPU workloads are unloaded or paused before starting YuE2 generation.

## Pinned Model Files

The pinned model weights and manifests are documented in [`models/yue2-files.txt`](file:///home/kchauhan/repos/l2m2/models/yue2-files.txt):

| Component | Repository | Revision | Size | SHA-256 |
|---|---|---|---|---|
| YuE2-3B Model | `m-a-p/YuE2-3B` | `8f24312f...` | 7.26 GB | `1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9` |
| YuE2-3B Tiktoken | `m-a-p/YuE2-3B` | `8f24312f...` | 2.56 MB | `b2b1b8dfb5cc5f024bafc373121c6aba3f66f9a5a0269e243470a1de16a33186` |
| YuE2-Vae Decoder | `m-a-p/YuE2-Vae` | `152733a1...` | 530 MB | `807ce9d5149fa27c5ad3e6582058469852e908f6c5acc8c8aa338e7ab7751346` |

Weights reside under `L3MS_MEDIA_ROOT/YuE2/` (`$HOME/models/media/YuE2`).

## Installation & Maintenance

Verify environment and model readiness:
```bash
./maintenance/setup-yue2.sh check
```

Run environment diagnostics via the `yue2 doctor` protocol:
```bash
./maintenance/setup-yue2.sh doctor
```

Setup or refresh the checkout and Python 3.12 virtual environment:
```bash
./maintenance/setup-yue2.sh install-env
```

Download and verify pinned model weights:
```bash
./maintenance/setup-yue2.sh download-models
```

## Generation

### 1. Via L2M2 CLI Profile
```bash
# Style prompt and inline lyrics
./media-models/generate-yue2.sh --prompt "warm acoustic indie pop, female voice, acoustic guitar" --lyrics "[Verse]\nMorning sun through the glass\nWatch the shadows softly pass"

# Using a request JSON file
./media-models/generate-yue2.sh --request prompts/yue2-city-lights.json
```

### 2. Direct Script Invocation
```bash
./media-models/generate-yue2.sh \
  --prompt "cyberpunk synthwave, punchy bass, female vocals" \
  --lyrics-file prompts/lyrics.txt \
  --cot full \
  --quantization fp8 \
  --budget 12 \
  --output ~/media-output/my-first-song
```

### 3. Stage-Separated Inspection (Plan -> Audio)
YuE2 allows stopping after score generation to inspect and edit the ABC score before synthesis:
```bash
# Step 1: Generate only the symbolic score plan
./media-models/generate-yue2.sh \
  --prompt "jazz ballad, mellow saxophone, soft piano" \
  --lyrics-file prompts/lyrics.txt \
  --stage plan \
  --output ~/media-output/jazz-plan

# Step 2: Supply edited score back to synthesize audio
./media-models/generate-yue2.sh \
  --prompt "jazz ballad, mellow saxophone, soft piano" \
  --lyrics-file prompts/lyrics.txt \
  --abc-file ~/media-output/jazz-plan/score.abc \
  --output ~/media-output/jazz-song
```

## Output Artifacts

Each run outputs to a dedicated timestamped directory retaining complete reproducibility records:
- `audio.flac`: 48 kHz 24-bit stereo song file.
- `score.abc`: Generated or supplied ABC musical notation with melody, harmony, and lyrics alignment.
- `plan.json`: Symbolic plan tokens and request metadata.
- `result.json`: Generation receipt, timings, model hashes, and truncation flags.
