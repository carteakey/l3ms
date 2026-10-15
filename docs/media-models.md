# Media models

Script-first local image, audio/music, speech and video. This was the separate
`l2m2` repository until 2026-10-15; it now lives in the L3MS tree. Supported
profiles run through shared engines behind llama-swap (ComfyUI for images and
LTX/YuE2, audio.cpp for speech and music); the wrapper scripts below remain for
direct use and for Blender.

## Start

From the repository root:

```sh
l3ms --list stack                      # all profiles, including media
l3ms --run-profile media:breeze-tts-2 --task tts --input speech.json --output-dir /absolute/private/fresh-output
./media-models/generate-heartmula-music.sh --prompt "a sleepy robot waltz" --instrumental
```

Each profile's `script` field in `media-runtimes.json` names its wrapper. Prompts and
paths are passed as argv values; shell substitutions in prompt text are not executed.
The retired `l2m2` binary's `--media <id> --extra '<args>'` is now just
`./media-models/<script> <args>`.

## What is here

- `media-runtimes.json`: MiniMax-H3, YuE2, HeartMuLa, Breeze-TTS-2, and LTX-2.5 profiles.
- `media-models/`: editable generation wrappers including `generate-yue2.sh` and `generate-breeze-tts-2.sh`.
- `maintenance/setup-yue2.sh`: operational script for YuE2 checkout, Python 3.12 venv, and weights management.
- `maintenance/refresh-audio-cpp.sh`: operational script to audit, build with CUDA, and smoke-test audio.cpp.
- `maintenance/setup-media-runtimes.sh`: runtime install/readiness commands.
- `models_config.json`: extracted Qwen Image, Qwen Image Edit,
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
.venv/bin/python ../../model_downloader/download_hf_model.py --config models_config.json
```

Review download destinations before running the downloader; the extracted config
preserves the original paths and enabled flags. The downloader is a small copied
compatibility boundary, so L2M2 does not require an L3MS checkout.
Before generation on a shared GPU, unload competing models as described in the
runbook. Runtime readiness and hardware requirements still apply.

For a focused Breeze-TTS preflight that does not load the model or request GPU
work, run `./media-models/generate-breeze-tts-2.sh --check` from this child
directory. It verifies the compiled audio.cpp loader and local model/spec files;
CUDA TTS/design/clone/streaming requires an exclusive GPU window. The separate
`breeze-tts-2-cpu` profile now covers all four modes without GPU service changes;
see [CPU execution evidence](cpu-native-workflows.md). Chunk paths are
returned after completion, not played live.

## Checks

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --all-targets --locked
python3 -m unittest discover -s media-models -p 'test_*.py'
for script in media-models/*.sh maintenance/*.sh; do bash -n "$script"; done
```

MiniMax-H3 quality capture and the small video experiment: [notes](minimax-h3-quality-notes.md).

ComfyUI and its local MCP connection: [setup and controls](comfyui-setup.md).

Install the official MiniMax-H3 reference-quality model stack:

```sh
./maintenance/setup-minimax-h3-comfyui.sh
```

The selected baseline and first-run settings are documented in
[the H3 reference experiment](experiments/h3-reference-baseline.md).

Qwen Image 2.1 ComfyUI setup, workflows, and the L2M2 CLI profile are documented
in [the Qwen Image 2.1 guide](qwen-image-2.1-setup.md).

Qwen Image 2.1 Turbo 8-step ComfyUI setup, INT8 ConvRot weights, and CLI profile
are documented in [the Qwen Image 2.1 Turbo guide](qwen-image-2.1-turbo-setup.md).

Krea 2 raw INT8 ComfyUI setup with Turbo distill LoRA and the L2M2 CLI profile
are documented in [the Krea 2 guide](krea-2-setup.md).

audio.cpp multi-tier runtimes, compilation flags, and Breeze-TTS-2 speech generation/voice cloning are documented in [the audio.cpp runtimes guide](audio-cpp-runtimes.md). Measured baseline runs are recorded in [the Breeze-TTS-2 experiment log](experiments/breeze-tts-2-baseline.md).
 
YuE2 foundation music model setup, FP8 quantization, memory budgeting, and symbolic ABC score planning are documented in [the YuE2 setup guide](yue2-setup.md).

