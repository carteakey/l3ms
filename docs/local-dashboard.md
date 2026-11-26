# L3MS Local Live Control Dashboard (`:8085`)

Live companion web console for `L3MS` on `yeti-cachy` (`http://yeti-cachy.llama-alnair.ts.net:8085/`), sharing the Signal/03 dark phosphor/amber visual theme of `l3ms.carteakey.dev`.

Built strictly on Python 3 stdlib (`http.server`, `ThreadingHTTPServer`, `urllib.request`, `subprocess`, `json`, `re`, `pathlib`, `threading`) with zero third-party packages required.

## Quick Start (Manual)

```bash
python3 tools/local-dashboard/server.py \
  --host 0.0.0.0 \
  --port 8085 \
  --repo-root /home/kchauhan/repos/l3ms \
  --swap-url http://127.0.0.1:8080
```

Then open `http://yeti-cachy.llama-alnair.ts.net:8085/` (or `http://127.0.0.1:8085/` locally).

## Installing the Systemd User Service

Copy or symlink the tracked unit file into `~/.config/systemd/user/` and enable it:

```bash
mkdir -p ~/.config/systemd/user
cp maintenance/systemd/l3ms-dashboard.service ~/.config/systemd/user/l3ms-dashboard.service
systemctl --user daemon-reload
systemctl --user enable --now l3ms-dashboard.service
systemctl --user status l3ms-dashboard.service
```

## Security & Safety Guarantees

- **Server-side API key handling:** `router_client.py` loads `LLAMA_SWAP_API_KEY` from the process environment, `systemctl --user show llama-swap.service --property=Environment`, or `<repo-root>/.env` (matching `maintenance/l3ms-ctl.sh`). The key is never returned in HTML, JSON payloads, SSE frames, or error messages.
- **Defensive log scrubbing:** All log tails (`llama-swap` journal and active Strata/PolyStrata logs) and upstream error strings scrub the active key, `Bearer ...` headers, `l3ms-...` tokens, and `LLAMA_SWAP_API_KEY=...` assignments before returning to the browser.
- **NFS-safe disk checks:** Disk telemetry always runs `df -hlT` (local filesystems only) for `/`, `/home`, and `/mnt/dev` so dead or unreachable NFS mounts never hang the server.
- **Reasoning-safe smoke checks:** Chat smoke tests send `max_tokens=64` so reasoning models have room to clear `<think>` / `reasoning_content`.

## Features & Endpoints

| Endpoint | Method | Purpose |
| :--- | :---: | :--- |
| `/` and `/static/*` | `GET` | Signal/03 dashboard UI (Live Silicon Strip, Active Resident Hero, Model Switcher Grid, Log Tailer & Bench History). |
| `/health` | `GET` | Lightweight dashboard liveness JSON (`{"ok": true}`). |
| `/api/telemetry` | `GET` | Live snapshot: RTX 4070 VRAM/util/temp/watts (`nvidia-smi`), DDR5 RAM & zram (`/proc/meminfo` + `/sys/block/zram0/mm_stat`), i5-12600K P-core (`0-11`) vs E-core (`12-15`) load (`/proc/stat`), CPU governor health check (flashes amber when not `performance`), `vm.swappiness` & THP status, active inference PID `/proc/<pid>/io` NVMe read MB/s spill detector, and local `df -hlT` usage. |
| `/api/events` | `GET` | Server-Sent Events (SSE) stream emitting `event: snapshot` every 2 seconds with live telemetry and `/running` router status. |
| `/api/router` | `GET` | Live `llama-swap` `/running` + `/v1/models`, parsed `llama-swap.yaml` catalog (groups, native `aliases:`, TTLs, `unlisted: true` filtering via `?include_unlisted=1`), and active Strata/PolyStrata JSON config inspection (`STRATA_*` env flags, CLI `args`, `vision` block, sampling). |
| `/api/models/<id>/wake` | `POST` | Wakes/switches to `<id>` via `llama-swap` `/upstream/<id>/` with bounded timeout and structured receipt. The UI shows an eviction confirmation banner naming the resident model before switching. |
| `/api/models/unload` | `POST` | Calls `llama-swap` `/unload` (or `/api/models/unload/<id>` when `{"model": "..."}` is passed) to free VRAM. |
| `/api/models/<id>/smoke` | `POST` | Runs a bounded health / embeddings / 64-token chat probe and returns latency, reasoning flag, and response snippet. |
| `/api/logs?source=llama-swap\|engine&lines=100` | `GET` | Bounded, secret-scrubbed tail of `llama-swap.service` journal or the active Strata/PolyStrata engine log. |
| `/api/benches` | `GET` | Parses `logs/results/*.jsonl` (`strata-tuning-ab.jsonl`, `inkling-refresh-ab.jsonl`, `qwen38-refresh-ab.jsonl`, etc.) into recent benchmark runs. |

## Verification

```bash
python3 -m unittest tools/local-dashboard/test_server.py -v
python3 -m py_compile tools/local-dashboard/server.py tools/local-dashboard/telemetry.py tools/local-dashboard/router_client.py
```
