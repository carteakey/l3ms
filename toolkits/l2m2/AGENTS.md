# L2M2

Local Large Media Models: a barebones local media toolkit.

- Keep orchestration explicit, script-first, and keyboard-friendly.
- Declare generation profiles in media-runtimes.json and commands in media-models/.
- Keep image generation/editing, audio/music, speech, and video here;
  vision-capable LLMs belong in L3MS.
- Snapshot configs and scripts under .toolkit/ before changing existing files.
- Preserve downloads unless explicitly told to overwrite them.
- Keep existing L3MS_* script environment variables compatible.
- Do not add a TUI, daemon, or shared framework without a concrete need.
- Update CHANGELOG.md for user-visible changes and run the README checks.

- Save reusable experiment runners under maintenance/, prompts under prompts/,
  and measured run summaries under docs/experiments/. Each run must retain its
  exact command, script copies, seed/settings, log, duration and artifact paths.
