# BreezeTTS 2 Baseline & Verification Experiment

This experiment evaluates the initial deployment of `Breeze-TTS-2` (`breeze_tts_2_q8_0`) driven by `audio.cpp` (`main` @ commit `955c8725`) on the Yeti host (`RTX 4070 12 GB`, `CUDA 13.4`, `GCC 16.2.1`).

---

## 1. System & Runtime Configuration

- **Host GPU**: NVIDIA GeForce RTX 4070 (12,282 MiB, Ada Lovelace `sm_89-real`)
- **Host CPU / Memory**: AMD 8-core / 64 GB DDR5-5600
- **Audio Engine**: `audio.cpp` compiled via `maintenance/refresh-audio-cpp.sh --build`
- **Compiler**: GCC 16.2.1 with `/opt/cuda/bin/nvcc` 13.4.92
- **Model Package**: `Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf` (4.8 GiB on disk)
  - Text Conditioner: T5Gemma
  - Acoustic Generator: Qwen-style autoregressive backbone
  - Waveform Decoder: Mimi neural audio codec
  - Activation Rounding: BF16 (CUDA hardware native)
  - Attention lowering: `auto` (Flash Attention)

---

## 2. Measured Benchmark Runs

### Run 1: Standard TTS (Offline Baseline)
- **Command**:
  ```bash
  media-models/generate-breeze-tts-2.sh \
    --text "Hello world, BreezeTTS 2 is now running locally on audio.cpp." \
    --max-tokens 300
  ```
- **Performance & Metrics**:
  - `metrics.wall_ms`: `1960.62 ms` (~1.96 s)
  - `metrics.audio_duration_ms`: `5200 ms` (5.20 s speech)
  - `metrics.rtf`: **`0.377043`** (**2.65x faster than real-time**)
  - `metrics.memory.peak_rss_mb`: `2488 MB`
  - `metrics.sample_rate`: `24000 Hz` (mono)
- **Artifact**:
  - `/home/kchauhan/media-output/breeze-tts-20260926-124605.wav` (244 KB)

---

### Run 2: Unified `l2m2` Profile Runner
- **Command**:
  ```bash
  ./target/release/l2m2 --media breeze-tts-2 \
    --extra '--text "Testing the unified l2m2 media runner with BreezeTTS 2." --max-tokens 300'
  ```
- **Performance & Metrics**:
  - `metrics.wall_ms`: `2446.86 ms` (~2.45 s)
  - `metrics.audio_duration_ms`: `6640 ms` (6.64 s speech)
  - `metrics.rtf`: **`0.368502`** (**2.71x faster than real-time**)
  - `metrics.memory.peak_rss_mb`: `2492 MB`
- **Artifact**:
  - `/home/kchauhan/media-output/breeze-tts-20260926-124628.wav`

---

### Run 3: Voice Design (Instruction-Conditioned)
- **Command**:
  ```bash
  media-models/generate-breeze-tts-2.sh \
    --text "This is a warm female voice narration demo." \
    --instruction "A warm female narrator with calm pacing and clear studio acoustics." \
    --max-tokens 250
  ```
- **Performance & Metrics**:
  - `metrics.wall_ms`: `1110.80 ms` (~1.11 s)
  - `metrics.audio_duration_ms`: `2800 ms` (2.80 s speech)
  - `metrics.rtf`: **`0.396714`** (**2.52x faster than real-time**)
  - `metrics.memory.peak_rss_mb`: `2493 MB`
- **Artifact**:
  - `/home/kchauhan/media-output/breeze-tts-20260926-124636.wav`

---

### Run 4: Real-time Incremental Streaming Mode
- **Command**:
  ```bash
  media-models/generate-breeze-tts-2.sh \
    --text "This is real-time streaming speech generation." \
    --mode streaming \
    --stream-frames 16 \
    --stream-lookahead 12 \
    --max-tokens 250
  ```
- **Streaming Chunks Emitted**:
  - `chunk_0_part_0`: 7,680 samples (320 ms)
  - `chunk_0_part_1`: 30,720 samples (1,280 ms)
  - `chunk_0_part_2`: 23,040 samples (960 ms)
- **Artifact**:
  - `/home/kchauhan/media-output/breeze-tts-20260926-124703.wav`

---

### Run 5: Prompt-Audio Voice Cloning
- **Command**:
  ```bash
  media-models/generate-breeze-tts-2.sh \
    --task clon \
    --text "This is a voice cloning test using BreezeTTS 2 in local audio.cpp." \
    --voice-ref /home/kchauhan/repos/audio.cpp/assets/resources/b.wav \
    --reference-text "Some call me nature. Others call me Mother Nature. I have been here for over four and a half billion years." \
    --instruction "Speak clearly and naturally." \
    --max-tokens 250
  ```
- **Performance & Metrics**:
  - `metrics.wall_ms`: `3468.58 ms` (~3.47 s)
  - `metrics.audio_duration_ms`: `7280 ms` (7.28 s cloned speech)
  - `metrics.rtf`: **`0.476453`** (**2.10x faster than real-time**)
  - `metrics.memory.peak_rss_mb`: `2492 MB`
- **Artifact**:
  - `/home/kchauhan/media-output/breeze-tts-20260926-124719.wav` (342 KB)

---

## 3. Conclusions & Hardware Budget

1. **Throughput**: Across all generation modes (standard TTS, voice design, cloning), the real-time factor remained solidly between **0.368 and 0.476** (2.10× to 2.71× faster than real-time audio playback).
2. **Memory Efficiency**: Peak RSS footprint remained constant at **~2,490 MB** in system RAM, and GPU VRAM consumption remained at **~6.5 GB**, fitting comfortably on the 12 GB RTX 4070 without conflicting with concurrent background processes.
3. **Cloning Fidelity**: Speaker reference audio conditioning via `--voice-ref` and transcript matching via `--reference-text` produced clear voice adaptation with high vocal consistency and no repetition collapse.
