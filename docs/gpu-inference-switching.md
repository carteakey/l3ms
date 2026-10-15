# GPU Inference Switching: Text & Image

Guide for switching exclusive GPU access between local LLM text generation (`llama-swap`) and image diffusion (`ComfyUI`) on Yeti (NVIDIA GeForce RTX 4070 12 GB), while keeping the public Tailscale Funnel endpoint on standard port 443 synchronized.

## Architecture

- **Public Endpoint**: `https://yeti-cachy.llama-alnair.ts.net/` (Port 443, HTTPS).
- **VRAM Constraint**: The 12 GB RTX 4070 requires exclusive VRAM windows. `ComfyUI` in GPU mode refuses to start if `llama-swap.service` is active, and vice versa.
- **Port Mapping**:
  - `llama-swap` listens on `http://127.0.0.1:8080`.
  - `ComfyUI` listens on `http://127.0.0.1:8188`.
  - Tailscale Funnel points port 443 to the active service.

---

## 1. Switch to Image Mode (ComfyUI on GPU)

Run on Yeti:

```bash
# 1. Stop llama-swap to release GPU VRAM
systemctl --user stop llama-swap.service

# 2. Stop any existing ComfyUI instance (e.g. CPU mode)
./maintenance/comfyui-service.sh stop

# 3. Start ComfyUI with GPU acceleration (lowvram mode, 1 GiB reserved)
./maintenance/comfyui-service.sh start-gpu

# 4. Route public port 443 to ComfyUI (8188)
sudo tailscale funnel --https=443 --bg 8188
```

- **Active Endpoint**: `https://yeti-cachy.llama-alnair.ts.net/` serves ComfyUI web UI & REST API.
- **MCP**: Local agents (Antigravity, Codex, Claude) and remote agents via `COMFYUI_URL` can drive workflows.

---

## 2. Switch to Text Mode (llama-swap)

Run on Yeti:

```bash
# 1. Stop ComfyUI to release GPU VRAM
./maintenance/comfyui-service.sh stop

# 2. Start llama-swap
systemctl --user start llama-swap.service

# 3. Route public port 443 back to llama-swap (8080)
sudo tailscale funnel --https=443 --bg 8080
```

- **Active Endpoint**: `https://yeti-cachy.llama-alnair.ts.net/` serves OpenAI-compatible text completions & Web UI.

---

## Quick Status & Diagnostics

```bash
# Check service states
systemctl --user is-active llama-swap.service l2m2-comfyui.service

# Inspect Tailscale Funnel mapping on port 443
tailscale funnel status

# Check VRAM allocation and active compute processes
nvidia-smi

# Check ComfyUI device status
curl -s http://127.0.0.1:8188/system_stats | jq .devices
```
