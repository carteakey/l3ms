# Blender 3D & Model Context Protocol (MCP) Setup

## 1. Overview

Blender 5.2.1 LTS has been integrated into **L2M2** as a local 3D modeling, rendering, and scene generation runtime. It connects directly to AI agents via the Model Context Protocol (MCP) using a two-part bridge:

1. **In-Blender Addon (`blender_mcp.py`)**: Runs inside Blender and opens a local socket listener on `localhost:9876`. It translates incoming JSON-RPC calls into Blender's internal Python API (`bpy`).
2. **MCP Server (`blender-mcp`)**: A lightweight process spawned by the MCP client (Antigravity, OpenCode, Claude) over stdio that proxies agent tool calls to the Blender socket.

In addition to interactive MCP sessions, L2M2 provides `media-models/generate-blender-3d.sh` for headless background Python execution and Cycles/Eevee image rendering.

---

## 2. Installation Architecture

### Blender Executable & Desktop Launcher
- **Version:** Blender 5.2.1 LTS (Linux x86_64 standalone release)
- **Install Path:** `~/.local/share/blender-5.2.1-linux-x64/`
- **Symlinks in PATH:**
  - `~/.local/bin/blender`
  - `~/.local/bin/blender-launcher`
- **Desktop Entry:** `~/.local/share/applications/blender.desktop` (with scalable icon at `~/.local/share/icons/hicolor/scalable/apps/blender.svg`)

### Blender Addon
- **File:** `~/.config/blender/5.2/scripts/addons/blender_mcp.py` (Protocol version 5)
- **State:** Pre-enabled in `~/.config/blender/5.2/config/userpref.blend`
- **Socket Port:** `9876` (TCP localhost)

### MCP Server CLI
- **Tool:** `blender-mcp` (v1.9.1) installed via `uv tool`
- **Binary:** `~/.local/bin/blender-mcp`

---

## 3. Client Configuration

### Antigravity (`~/.gemini/config/mcp_config.json`)
```json
{
  "mcpServers": {
    "blender": {
      "command": "/home/kchauhan/.local/bin/uvx",
      "args": ["blender-mcp"],
      "env": {
        "BLENDER_HOST": "localhost",
        "BLENDER_PORT": "9876"
      }
    }
  }
}
```

### OpenCode (`~/.config/opencode/opencode.jsonc`)
```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "blender": {
      "type": "local",
      "command": ["/home/kchauhan/.local/bin/uvx", "blender-mcp"],
      "environment": {
        "BLENDER_HOST": "localhost",
        "BLENDER_PORT": "9876"
      }
    }
  }
}
```

---

## 4. Operational Workflow

### Interactive AI Modeling
1. Launch Blender (`blender` or via desktop menu).
2. In the **3D Viewport**, press `N` to open the right sidebar.
3. Open the **MCP for Blender** tab.
4. Click **Start MCP Server** (starts listening on `localhost:9876`).
5. Prompt the AI agent in Antigravity or OpenCode to generate or modify 3D objects, materials, or scenes.

### CLI & L2M2 Invocation
```bash
# Check runtime and listener status
./target/release/l2m2 --media blender-3d --extra '--check'

# Launch interactive GUI with MCP
./target/release/l2m2 --media blender-3d --extra '--launch'

# Run headless Python code
./target/release/l2m2 --media blender-3d --extra '--code "import bpy; bpy.ops.mesh.primitive_monkey_add()"'

# Render scene to PNG
./target/release/l2m2 --media blender-3d --extra '--render scene.blend --output ~/media-output'
```

---

## 5. Smoke Test Verification

The live MCP integration was verified on 2026-09-08:
- Handshake with protocol v5 confirmed.
- Initial scene read (`Cube`, `Light`, `Camera`).
- Executed procedural addition of Suzanne monkey head (`MCP_Test_Suzanne`).
- Added Subdivision Surface modifier and assigned metallic gold BSDF material.
- Switched viewport shading to Material Preview.
- Captured and verified live viewport screenshot via `get_viewport_screenshot`.
