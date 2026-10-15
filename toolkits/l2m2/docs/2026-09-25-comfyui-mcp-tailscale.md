# Working Notes: ComfyUI, MCP, Tailscale Funnel & API Setup

**Date:** 2026-09-25  
**Host:** Yeti (`yeti-cachy`, CachyOS Linux x86_64, Kernel 7.2.6)  
**GPU:** NVIDIA GeForce RTX 4070 12 GiB (Driver 615.71.09, CUDA 13.4, Ada Lovelace compute 8.9)  
**Public Endpoint:** `https://yeti-cachy.llama-alnair.ts.net/`  
**Repository:** `/home/kchauhan/repos/l2m2`  

---

## 1. Objective & Summary

This session configured and verified the operational stack for exposing local image generation from L2M2:
1. Verified the official ComfyUI Model Context Protocol (MCP) server integration (`comfy-mcp`).
2. Diagnosed network accessibility from remote cloud sandboxes (e.g. Claude Code cloud sessions) and configured Tailscale Funnel on standard HTTPS port 443.
3. Resolved the workflow discovery issue where ComfyUI's API and Web UI returned empty workflows.
4. Switched ComfyUI to full GPU hardware acceleration on the RTX 4070 and verified CUDA compute context.
5. Upgraded `comfyui-workflow-templates` and `comfyui-frontend-package` to current upstream releases.
6. Documented repeatable inference switching between local LLMs (`llama-swap`) and image diffusion (`ComfyUI`).

---

## 2. Comfy MCP Setup & Verification

- **Source Code**: [`vendor/comfy-mcp`](file:///home/kchauhan/repos/l2m2/vendor/comfy-mcp) (commit `eb8ea47`, `Comfy-Org/comfy-mcp`)
- **Virtual Environment**: `.venv-comfy-mcp` (Python 3.13.15, `comfy-mcp 0.10.0`, `comfy-cli 1.20.0`)
- **Execution Shim**: [`maintenance/run-comfy-mcp.sh`](file:///home/kchauhan/repos/l2m2/maintenance/run-comfy-mcp.sh) (sets `COMFY_BIN` and `COMFY_LOCAL_URL=http://127.0.0.1:8188`)
- **Transport**: Standard I/O (`stdio`, MCP schema version `2024-11-05`).

### Verification Test
Executed [`maintenance/check-comfy-mcp.py`](file:///home/kchauhan/repos/l2m2/maintenance/check-comfy-mcp.py) against running ComfyUI:
- **Result**: MCP handshake and `server_info` succeeded.
- **Tools Discovered**: 40 tools available, including `generate_image`, `run_workflow`, `job`, `fetch_outputs`, `search_models`, `search_templates`, `fetch_template`, `run_template`, `nodes`, `system_stats`.
- **Diagnostic Log**: Saved to [`.toolkit/comfyui/mcp-check.json`](file:///home/kchauhan/repos/l2m2/.toolkit/comfyui/mcp-check.json).

### Agent Client Registration
- **Antigravity IDE**: Registered in `~/.gemini/config/mcp_config.json`:
  ```json
  "comfy-local": {
    "command": "/home/kchauhan/repos/l2m2/maintenance/run-comfy-mcp.sh",
    "args": []
  }
  ```
- **Codex CLI**: Registered via `./maintenance/register-comfy-mcp.sh` (`codex mcp add comfy-local -- ...`).
- **Claude Code (Local)**: `claude mcp add comfy-local -- /home/kchauhan/repos/l2m2/maintenance/run-comfy-mcp.sh`.
- **Remote Agents (MacBook / Laptop)**: Drives Yeti across Tailscale without running ComfyUI locally:
  ```json
  "comfy-remote": {
    "command": "uvx",
    "args": ["comfy-mcp"],
    "env": {
      "COMFYUI_URL": "https://yeti-cachy.llama-alnair.ts.net/"
    }
  }
  ```

---

## 3. Tailscale Funnel & Cloud Egress Diagnostics

### The Cloud Sandbox Issue
When attempting to access ComfyUI from a Claude Code cloud container:
- Cloud runner environments (Anthropic, AWS Lambda, GitHub Actions) enforce strict egress security policies: **outbound TCP traffic is restricted exclusively to ports 80 and 443**.
- Ports `8443` and `10000` were dropped at the cloud sandbox firewall layer.
- `llama-swap` had previously worked because it was exposed on standard port **443**.

### Tailscale Funnel Routing on Port 443
To allow seamless access from cloud sandboxes without firewall drops:
1. Re-routed port 443 to ComfyUI:
   ```bash
   sudo tailscale funnel --https=443 --bg 8188
   ```
2. Cleaned up secondary test ports:
   ```bash
   sudo tailscale serve --https=8443 off
   sudo tailscale serve --https=10000 off
   ```
3. Verified clean single-port funnel status:
   ```
   https://yeti-cachy.llama-alnair.ts.net (Funnel on)
   |-- / proxy http://127.0.0.1:8188
   ```
4. Probed live: `curl -I https://yeti-cachy.llama-alnair.ts.net/` $\rightarrow$ `HTTP/2 200 OK`.

---

## 4. ComfyUI User Workflows Discovery Fix

### Root Cause
ComfyUI starts with `--user-directory .toolkit/comfyui/user`. When the web UI or API requests workflows (`GET /userdata?dir=workflows`), it reads from `.toolkit/comfyui/user/default/workflows/`. This folder was completely empty (`[]`).

Meanwhile, the 7 official tracked workflow graphs in L2M2 resided in `workflows/`:
- `workflows/qwen-image-2.1/image_qwen_image_2_1_t2i.json`
- `workflows/qwen-image-2.1/image_qwen_image_2_1_image_edit.json`
- `workflows/krea-2/image_krea2_turbo_t2i.json`
- `workflows/krea-2/image_krea2_turbo_t2i_int8.json`
- `workflows/krea-2/image_krea2_turbo_int8_image_style_reference.json`
- `workflows/minimax-h3/video_minimax_h3_r2v-official.json`
- `workflows/minimax-h3/video_minimax_h3_r2v-20step-no-lora.json`

### Fix Applied
Symlinked all 7 workflow files into `.toolkit/comfyui/user/default/workflows/`:
```bash
for f in workflows/*/*.json; do
  ln -sf "../../../../../$f" ".toolkit/comfyui/user/default/workflows/$(basename "$f")"
done
```
Verified via API: `curl -s "https://yeti-cachy.llama-alnair.ts.net/userdata?dir=workflows"` returns all 7 filenames. They now populate ComfyUI's **Browse Workflows** menu.

---

## 5. GPU vs CPU Mode & Mutual Exclusion Runbook

The 12 GiB RTX 4070 requires strict mutual exclusion between `llama-swap.service` (LLMs) and `l2m2-comfyui.service` (ComfyUI GPU).

- **Switch to Image Mode (ComfyUI GPU)**:
  ```bash
  systemctl --user stop llama-swap.service
  ./maintenance/comfyui-service.sh stop
  ./maintenance/comfyui-service.sh start-gpu
  sudo tailscale funnel --https=443 --bg 8188
  ```
- **Switch to Text Mode (llama-swap)**:
  ```bash
  ./maintenance/comfyui-service.sh stop
  systemctl --user start llama-swap.service
  sudo tailscale funnel --https=443 --bg 8080
  ```
- **Live GPU Verification**:
  - `curl -s http://127.0.0.1:8188/system_stats` confirms device: `cuda:0 NVIDIA GeForce RTX 4070 : cudaMallocAsync`.
  - `nvidia-smi` confirms PID `150607` (`.venv-comfy/bin/python`) active with compute context. Free VRAM: ~12.04 GiB.

Full reference guide: [`docs/gpu-inference-switching.md`](file:///home/kchauhan/repos/l2m2/docs/gpu-inference-switching.md).

---

## 6. Upstream Package Upgrades

Upgraded `.venv-comfy` packages from PyPI to pull in the latest official Comfy-Org templates and frontend UI improvements:
- `comfyui-workflow-templates`: `0.11.62` $\rightarrow$ **`0.11.70`**
- `comfyui-frontend-package`: `1.53.6` $\rightarrow$ **`1.54.7`**
- `comfyui-workflow-templates-core`: `0.3.350` $\rightarrow$ **`0.3.361`**
- `comfyui-workflow-templates-json`: `0.1.85` $\rightarrow$ **`0.1.96`**
- `comfyui-workflow-templates-media-assets-01`: `0.1.46` $\rightarrow$ **`0.1.48`**
- Added `comfyui-workflow-templates-media-assets-02`: **`0.1.6`**

Pinned lockfile [`runtime-locks/comfyui-py313-linux.txt`](file:///home/kchauhan/repos/l2m2/runtime-locks/comfyui-py313-linux.txt) was updated (pre-upgrade backup saved to `.toolkit/`).

---

## 7. Historical Generation Speeds Reference (RTX 4070 12 GiB)

Measured runs recorded under [`docs/experiments/qwen-vs-krea2-comparison.md`](file:///home/kchauhan/repos/l2m2/docs/experiments/qwen-vs-krea2-comparison.md):

| Model | Resolution | Steps & Sampler | Duration | Peak VRAM | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen Image 2.1** | 1024×1024 | 25 steps, CFG 1.0, Euler/simple | **23.2s** | 9.6 GiB | Fastest baseline (~0.9s/step); INT8 ConvRot + 8B encoder + BF16 VAE |
| **Krea 2 Raw INT8** | 1024×1024 | 12 steps, CFG 1.5, Turbo LoRA r128 | **36.1s** | 10.4 GiB | Includes r128 LoRA model patching overhead |
