# Nemotron 3.5 Transcribe

`nemotron-3.5-transcribe` is configured in `llama-swap.yaml` for the 12 GB RTX
4070. It uses the speech.cpp 0.9.1 Vulkan runtime and the IQ4_NL model. Some
experts run on the CPU because the 19 GB model does not fit wholly in VRAM.
The router starts the runtime on the first request and unloads it after ten
minutes idle or when another model needs the GPU.

Install the runtime, model, voice activity detector and diarization model:

```bash
bash maintenance/install-nemotron-transcribe.sh
```

The installer verifies the published checksums. If the separate parallel
download is interrupted, resume it with `python3 maintenance/download-nemotron-ranges.py`.
That command checks SHA-256 before renaming the `.part` file. The installer
refuses to treat its sparse partial file as a sequential download.

Check `speech devices` output
if the GPU changes: the configured device is `Vulkan1` on this host. After
installation, reload llama-swap when a model swap is acceptable, then test an
actual WAV file through the router:

```bash
kill -HUP "$(pgrep -f '[l]lama-swap -config' | head -1)"
curl --fail-with-body --max-time 900 \
  -H "Authorization: Bearer $LLAMA_SWAP_API_KEY" \
  -F model=nemotron-3.5-transcribe -F file=@recording.wav \
  -F response_format=verbose_json -F chunking_strategy=auto \
  http://127.0.0.1:8080/v1/audio/transcriptions
```

The runtime accepts WAV files up to 25 MB. `chunking_strategy=auto` uses VAD and
returns region timestamps. Add `response_format=diarized_json` to get speaker
labels; omit the earlier `response_format` field when doing so. The runtime
does not expose the model's word alignment or translation features. Lytte
converts uploads to WAV and splits larger recordings before sending them here.

The model download passed its SHA-256 check. A real 11-second sample sent
through the router returned timed speaker-labeled text. On this host, speech.cpp
placed 12 expert layers on the GPU and 11 on the CPU. With a one-second silence
threshold, the sample's main phrase stayed together and was transcribed correctly.
