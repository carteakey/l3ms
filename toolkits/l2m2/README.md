# L2M2 — Local Large Media Models

A tiny, script-first home for local image, audio/music, speech, and video
experiments. Extracted from L3MS; vision-capable LLMs remain in L3MS.
No TUI or background service: one CLI, an editable manifest, and shell scripts.

## Start

From the monorepo root, enter the L2M2 child before building or running it:

```sh
cd toolkits/l2m2
cargo build --release --locked
./target/release/l2m2 --list
./target/release/l2m2 --media heartmula-music --extra '--prompt "a sleepy robot waltz" --instrumental'
# Interactive profile picker:
./target/release/l2m2 --media
```

The compiled binary defaults to this child directory through Cargo's
`CARGO_MANIFEST_DIR`. From the monorepo root, it also works as:

```sh
./toolkits/l2m2/target/release/l2m2 --list
```

Use `--root PATH` to select another L2M2 checkout's manifest and scripts:

```sh
./target/release/l2m2 --root /path/to/another/l2m2 --list
```

Prompts and paths are passed as argv values; shell substitutions in prompt
text are not executed.

## What is here

- `media-runtimes.json`: MiniMax-H3, HeartMuLa, Breeze-TTS-2, and LTX-2.5 profiles.
- `media-models/`: editable generation wrappers including `generate-breeze-tts-2.sh`.
- `maintenance/refresh-audio-cpp.sh`: operational script to audit, build with CUDA, and smoke-test audio.cpp.
- `maintenance/setup-media-runtimes.sh`: runtime install/readiness commands.
- `model_downloader/models_config.json`: extracted Qwen Image, Qwen Image Edit,
  and LTX-2.3 downloads. These are download entries, not new generation wrappers.
- `docs/media-generation-runbook.md`: setup and usage details.
- `HANDOFF.md` and `DETAILED_SESSION.md`: historical L3MS media records;
  old checkout paths and commands describe the original deployment.

Existing model, runtime, and output paths are preserved. The `L3MS_*` environment
variables in the extracted scripts remain supported under their original names.
No weights are bundled or downloaded during repository setup.

```sh
./maintenance/setup-media-runtimes.sh check
# Installs/builds runtimes and downloads weights when explicitly run:
./maintenance/setup-media-runtimes.sh install-audio-cpp

python3 -m venv .venv
.venv/bin/pip install -r requirements-downloader.txt
.venv/bin/python model_downloader/download_hf_model.py --config model_downloader/models_config.json
```

Review download destinations before running the downloader; the extracted config
preserves the original paths and enabled flags. The downloader is a small copied
compatibility boundary, so L2M2 does not require an L3MS checkout.
Before generation on a shared GPU, unload competing models as described in the
runbook. Runtime readiness and hardware requirements still apply.

## Checks

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --all-targets --locked
python3 -m unittest discover -s model_downloader -p 'test_*.py'
python3 -m unittest discover -s media-models -p 'test_*.py'
for script in media-models/*.sh maintenance/*.sh; do bash -n "$script"; done
```

MiniMax-H3 quality capture and the small video experiment: [notes](docs/minimax-h3-quality-notes.md).

ComfyUI and its local MCP connection: [setup and controls](docs/comfyui-setup.md).

Install the official MiniMax-H3 reference-quality model stack:

```sh
./maintenance/setup-minimax-h3-comfyui.sh
```

The selected baseline and first-run settings are documented in
[the H3 reference experiment](docs/experiments/h3-reference-baseline.md).

Qwen Image 2.1 ComfyUI setup, workflows, and the L2M2 CLI profile are documented
in [the Qwen Image 2.1 guide](docs/qwen-image-2.1-setup.md).

Krea 2 raw INT8 ComfyUI setup with Turbo distill LoRA and the L2M2 CLI profile
are documented in [the Krea 2 guide](docs/krea-2-setup.md).

audio.cpp multi-tier runtimes, compilation flags, and Breeze-TTS-2 speech generation/voice cloning are documented in [the audio.cpp runtimes guide](docs/audio-cpp-runtimes.md). Measured baseline runs are recorded in [the Breeze-TTS-2 experiment log](docs/experiments/breeze-tts-2-baseline.md).
