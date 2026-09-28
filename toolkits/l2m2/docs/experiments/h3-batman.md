# H3 Batman plant experiment

## Attempt 1 — 2026-09-09

Output: `/home/kchauhan/media-output/h3-batman-ZwXfaEw4`.
640×384, 124 requested frames, 20 steps, seed 42, Q4_K audio.cpp.
Interrupted: no surviving process, no WAV/MP4 or result file. Cause not established;
user reports Blender had consumed substantial RAM. Do not label as model failure
or claim a completed quality result. Original command, prompt and log retained.

## Repeating safely

```sh
systemd-run --user --collect --unit="l2m2-h3-$(date +%Y%m%d-%H%M%S)" \
  /home/kchauhan/repos/l2m2/maintenance/run-h3-batman-isolated.sh
```

The supervisor pauses/restores llama-swap only if it was initially active.
The test refuses a busy GPU, limits inference to 30 minutes, and creates a unique
output directory. Every new attempt saves copies of both test/generation scripts,
prompt, exact command, repository commit and working diff, GPU/RAM preflight,
log, exit code/wall time, and ffprobe metadata on success. Large media stays outside
Git; keep the experiment summary and artifact paths here. Script edits are
snapshotted under `.toolkit/script_versions/` before writes.

## Retry preflight — 2026-09-09 10:31

Transient unit `l2m2-h3-batman-retry-20260909` exited with status 2 before
creating an output folder. The journal contains no detailed error. A new
`bench-llama-qwen38-flash-next-ik-campaign.sh` workload was found running
with `llama-spec-bench`, around 8 GB VRAM. No video generated. The benchmark
was left running; do not start H3 while this campaign owns the GPU.

## Completed Run — 2026-09-26 17:48

- Unit: `l2m2-h3-20260926-174322.service`
- Output: `/home/kchauhan/media-output/h3-batman-oa6suGUb/`
- Configuration: 640×384 (0.246 MP), 124 requested frames (5.17 s @ 24 fps), 20 steps, seed 42, guidance scale 1.0, layerwise Q4_K on RTX 4070 (12 GB).
- Result: `exit_code=0`, `wall_seconds=303` (~5 min 3 s).
- Artifacts:
  - Video: `video.mp4` (640×384, 24 fps, H.264 High profile, 224 KB)
  - Audio: `audio.wav` (32 kHz stereo, 647 KB) and synced AAC in `video.mp4`
  - Metadata: `probe.json`, `result.txt`, `run.log`, `command.txt`, `prompt.txt`

