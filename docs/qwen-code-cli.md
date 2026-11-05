# Qwen Code CLI Setup

Guide for running Qwen Code CLI (`@qwen-code/qwen-code`) locally against L3MS and llama-swap.

## Overview

Qwen Code CLI is installed globally via npm and talks to the local llama-swap router at `http://127.0.0.1:8080/v1` through its OpenAI-compatible provider.

Settings file: `~/.qwen/settings.json`

## Configured Model Tiers

All tiers use AgentionAI's recommended sampling guardrails:
- `temperature`: 1.0
- `top_p`: 0.95
- `top_k`: 20
- `min_p`: 0.05

| Model ID | Runtime | Context | Purpose |
| --- | --- | --- | --- |
| `qwen38-flash-next-plat-abliterated-vision` | Strata v0.1.45 | 128k | Default driver: uncensored SC117 weights + CPU-offloaded vision + tool calling |
| `qwen38-flash-next-plat` | Strata v0.1.45 | 128k | Fast language tier: 60 to 90 t/s decode, K8V4 KV cache, 4-slot conversation parking |
| `qwen38-flash-next-plat-vision` | Strata v0.1.45 | 128k | Standard aligned weights with CPU vision encoder (`mmproj-F16.gguf`) |
| `qwen38-flash-next-plat-abliterated` | Strata v0.1.45 | 128k | Pure text SC117 weights |
| `qwen38-flash-next-gold` | llama.cpp master (b11241) | 96k | Upstream reference tier (20.5 to 22.1 t/s) |
| `qwen38-flash-next-mtp` | llama.cpp PR #28243 | 16k | MTP speculative decoding tier (25.3 to 27.1 t/s) |

## Usage

### Interactive Shell

Launch the interactive TUI in any project directory:

```bash
qwen
```

Inside the interactive session:
- Type `/model` to switch between tiers.
- Type `/help` for available slash commands.
- Type `/exit` or hit `Ctrl+C` to quit.

### Non-Interactive One-Shot Prompts

For quick queries or headless runs:

```bash
# Pure inquiry / read-only analysis
qwen -p "Explain the role of config_store.rs in this repository."

# Tool execution (run shell commands, edit files)
qwen -y -p "Run git status --short and summarize modified files."
```

Note: In non-interactive (`-p`) mode, Qwen Code defaults to read-only tool access unless `-y` (`--approval-mode yolo`) or `--approval-mode auto` is passed.

### Key Operational Differences vs Pi

- **Thinking Scratchpad Hygiene**: Unlike Pi (which preserves all prior `<think>` blocks across turns via `preserve_thinking: true`, causing context ballooning to 52k+ tokens and stylistic attractor loops), Qwen Code CLI strips prior `<think>` scratchpads after each turn, keeping context lean (typically 4k to 12k tokens) and avoiding hesitation contagion.
- **Reasoning Schema**: Qwen Code CLI's `capabilities.reasoning` with `profile: "qwen-chat-template"` is strictly a `toggleOnly` profile (`disableField: "enable_thinking"`). Do not configure `efforts` or `defaultEffort` for this profile.
