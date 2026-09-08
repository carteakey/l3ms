# L2M2 — Local Latent Media Machine

A tiny, script-first home for local image, audio/music, speech, and video
experiments. Extracted from L3MS; vision-capable LLMs remain in L3MS.
No TUI or background service: one CLI, an editable manifest, and shell scripts.

## Start

```sh
cargo build --release --locked
./target/release/l2m2 --list
./target/release/l2m2 --media heartmula-music --extra '--prompt "a sleepy robot waltz" --instrumental'
# Interactive profile picker:
./target/release/l2m2 --media
```

`--root PATH` selects another checkout's manifest/scripts. Prompts and paths
are passed as argv values; shell substitutions in prompt text are not executed.

## What is here

- `media-runtimes.json`: MiniMax-H3, HeartMuLa, and LTX-2.5 profiles.
- `media-models/`: editable generation wrappers.
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
for script in media-models/*.sh maintenance/*.sh; do bash -n "$script"; done
```
