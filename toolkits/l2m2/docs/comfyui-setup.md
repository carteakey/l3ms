# Local ComfyUI and MCP

Verified 2026-09-22. ComfyUI is at http://127.0.0.1:8188, initially in CPU
mode for setup while LLM work uses the GPU. Qwen Image 2.1 support is pinned to
the official ComfyUI revision documented in the Qwen setup guide. Native H3
reference nodes are present; usable Ref2VA weights and reference inputs still
need to be prepared before the Batman experiment.

## Installed versions

- Official ComfyUI, Qwen Image 2.1 support commit `6bfaacc67c2103481e5f0c84d75257cd0581d86a`.
- Official comfy-mcp 0.10.0, commit `eb8ea47397fb13cdcf0a818b83bfba48bd4ff236`.
- comfy-cli 1.20.0; Python 3.13.15; PyTorch 2.14.0+cu130.
- Sources: `vendor/ComfyUI`, `vendor/comfy-mcp` (ignored, upstream licenses apply).
- Environments: `.venv-comfy`, `.venv-comfy-mcp` (ignored).
- Exact resolved dependencies: `runtime-locks/`. Setup pins the source revisions.

## Repeatable commands

Run from L2M2:

```sh
./maintenance/setup-comfyui.sh
./maintenance/register-comfy-mcp.sh
./maintenance/comfyui-service.sh start-cpu
./maintenance/comfyui-service.sh status
./maintenance/comfyui-service.sh logs
./maintenance/comfyui-service.sh stop
```

Start refuses an already-running service. Stop before changing mode.
`start-gpu` uses low-VRAM mode with 1 GiB reserved and refuses active llama-swap
or a busy GPU. Arrange an exclusive GPU window before using it; it never stops
an LLM job. CPU mode is for setup/catalog checks, not practical H3 generation.
There is no automatic boot startup. The listener is loopback-only and hosted API
nodes are disabled. Outputs go to `~/media-output/comfyui`; local ComfyUI user
state goes to `.toolkit/comfyui/user`.

## MCP connection

Codex's global config has a STDIO server named `comfy-local` running
`maintenance/run-comfy-mcp.sh`. Registration saves a timestamped backup beside
`~/.codex/config.toml` before modification. The shim `maintenance/comfy-cli.sh`
selects this workspace and runtime Python explicitly; it does not change the
system Python or another ComfyUI installation.

A new Codex task/session may be needed to load the newly registered tool list.
The current task's static tool list does not prove hot reload. Registration was
verified with `codex mcp get comfy-local`; the separate protocol check below
successfully initialized, listed 40 tools, and called `server_info` against the
running local ComfyUI. No workflow submission was used as a connection test.

Use the saved service wrapper for lifecycle management. This instance was
started by systemd, so comfy-cli's own background-launch record is empty.
Do not launch another instance on port 8188 via the MCP lifecycle tools.

```sh
.venv-comfy-mcp/bin/python maintenance/check-comfy-mcp.py
```

Results: `.toolkit/comfyui/mcp-check.json` and `env-check.json`.
Both dependency environments passed `uv pip check`; `/system_stats` confirms
CPU mode and `/queue` was empty. The native catalog includes
`MiniMaxH3ReferenceToVideo`, `MiniMaxH3ImageToVideo`, and `MiniMaxH3SigmaShift`.
Custom-node Manager has not been installed; none is needed for these core nodes.

## MiniMax-H3 reference models

`maintenance/setup-minimax-h3-comfyui.sh` downloads the four-file official
Ref2VA baseline into `~/models/media/MiniMax-H3-ComfyUI` and links the files
into this ComfyUI workspace. Downloads are resumable and existing unrelated
models are preserved. The tracked file manifest records the exact upstream
paths; `SHA256SUMS` is written after all files finish.

## Sources

- [Official ComfyUI](https://github.com/Comfy-Org/ComfyUI)
- [Official local MCP server](https://github.com/Comfy-Org/comfy-mcp)
- [Codex MCP configuration](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)

## Qwen Image 2.1

See [Qwen Image 2.1 setup](qwen-image-2.1-setup.md) for the pinned model files,
workflow imports, and the exclusive-GPU generation procedure.

## GPU Inference Mode Switching

See [GPU Inference Switching](gpu-inference-switching.md) for the exact runbook
to switch exclusive GPU access and Tailscale Funnel port 443 between text
(`llama-swap`) and image (`ComfyUI`).

