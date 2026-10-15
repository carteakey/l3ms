# One workbench, shared engines

Approved direction (2026-10-15): one UI, llama-swap wherever supported, shared
engines where appropriate, and direct scripts only as an explicit last resort.
Prefer a smaller supported scope over another orchestration layer.

| Workload | Root execution path |
| --- | --- |
| Language, embeddings, native decisions | llama-swap native endpoints |
| Breeze speech, HeartMuLa music, H3 audio | Shared `audiocpp_auto` process |
| Images, LTX video/audio, YuE2 music | Shared `comfyui_auto` process |
| Blender headless | Existing opt-in direct workflow in an idle-GPU window; separate acceptance |
| H3 video, CUDA Nagi/Laya decisions | Retired from the root workbench; standalone toolkit scripts only |

LTX and YuE2 use the existing ComfyUI workflow interface, cancellation, downloads
and history. There are no model-specific root Python executors or fallback
to standalone scripts. Weights and compatible workflow nodes are still required;
an engine label does not prove those assets are installed.

The CLI defaults to `LLAMA_SWAP_URL` plus `/comfyui`, with the router's API key.
The TUI uses its confirmed Chat connection and captured key.
`L3MS_COMFY_SCOPED_JOBS=1` still asserts native job-ID/scoped-cancellation support.
A separate server can be selected with `L2M2_COMFY_URL` and its own
`L3MS_COMFY_API_KEY`; the router key is not forwarded to this override.
No automatic setup, retry or engine fallback is added.

Workflows are native API JSON wrapped as `{"graph": {...}}`. The profile selects
the shared engine and declared task; the graph selects actual model files and
settings. Use the toolkit's ComfyUI recipes with filenames matching installed
assets. Real-model/codec/VRAM acceptance remains separate from fixture coverage.

The former `toolkits/l2m2` and `toolkits/l3ds` trees were folded into the
repository root on 2026-10-15: manifests, wrappers, workflows and setup scripts sit
beside `src/`, and their separate Rust crates were retired in favour of the root CLI.
The root owns the shared UI, routing and history. Preserve weights, private state
and external callers (MCP client configs, `l2m2-nsfw`) when paths change; a
directory move alone is not integration.

The branch YAML labels describe all declared shared-engine capabilities.
Editing them does not reload the live router: reload can evict a resident model
and requires a quiet window. Downloads and public publication remain separate.
