---
name: gpu-handoff
description: Safely free the RTX 4070 from the llama-swap model server so another GPU job can run (ComfyUI/Qwen-Image sketches, benchmarks), then hand it back and verify the model server is healthy. Use before any task that needs more than ~1 GB of VRAM, and to recover when llama-swap is failed or VRAM is stuck.
---

# GPU handoff

**Status: draft.** Extracted from the `sketches` skill and the 2026-10-01
sessions, where this went wrong four ways: an orphaned `llama-server` kept 8-10 GB
of VRAM after the service was stopped, `llama-swap` crash-looped on a port clash,
`pkill -f` killed the shell, and a ComfyUI launch collided with the model server.

## Facts

- The card has 12 GB. The model server (Strata/llama.cpp via `llama-swap`) holds
  ~9-11 GB, so Qwen-Image-2.1 (INT8, ~11 GB) cannot run beside it.
- Idle VRAM is **~44 MiB**. That is the "free" signal.
- `llama-swap` is a systemd **user** unit (`llama-swap.service`), listening on
  `:8080`, API key required (an unauthenticated request returns 401 and that is
  healthy).
- ComfyUI: `bash /home/kchauhan/repos/l2m2/maintenance/run-comfyui.sh gpu`,
  serving on `127.0.0.1:8188`.

## Rules

1. **Ask the user before stopping `llama-swap`.** It interrupts anything they are
   serving. A request that obviously needs the GPU ("generate the sketches") is
   consent for that task, not for later ones.
2. **Never `pkill -f`.** Find processes by pid (`nvidia-smi --query-compute-apps`,
   `ss -ltnp`, `ps -eo pid,args`).
3. Do not start the dev server on `:8080` while `llama-swap` is stopped (see the
   `dev-server` skill).

## Hand the GPU over

```sh
systemctl --user stop llama-swap.service
sleep 4
# Stopping the unit can leave its llama-server child running. Kill orphans by exact binary, by pid:
for p in $(ps -eo pid,args | awk '$2 ~ /vendor\/llama.cpp\/build\/bin\/llama-server$/ {print $1}'); do kill $p; done
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader     # must be ~44 MiB before continuing
(nohup bash /home/kchauhan/repos/l2m2/maintenance/run-comfyui.sh gpu > $SCRATCHPAD/comfy.log 2>&1 &)
for i in $(seq 1 40); do curl -s -m 2 http://127.0.0.1:8188/system_stats >/dev/null && break; sleep 3; done
```

Do the GPU work. (A 1280x896 Qwen-Image generation takes ~20-30 s.)

## Hand it back

```sh
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  case "$(ps -o args= -p $p)" in *"port 8188"*) kill $p;; esac
done
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader     # ~44 MiB again
systemctl --user start llama-swap.service
sleep 6
systemctl --user is-active llama-swap.service                # active
journalctl --user -u llama-swap.service -n 20 --no-pager | grep -E 'listening|Health check passed'
```

Healthy looks like: `active`, `llama-swap listening on http://:8080`, and a
`Health check passed` line for the preloaded model. VRAM climbing back to ~10 GB
is the model loading. The `draft model` and `mlock` warnings in the log are
existing config noise, not failures.

## Recovery

- **`llama-swap` is `failed (start-limit-hit)`**: something holds `:8080` or an
  orphan `llama-server` is alive. Stop the dev server by pid, kill orphan
  `llama-server` processes (above), then
  `systemctl --user reset-failed llama-swap.service && systemctl --user start llama-swap.service`.
- **VRAM not freeing**: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
  and `ps -o args= -p <pid>` to see what holds it. ComfyUI's python process is
  the usual one (match on `port 8188`).
- **Another session may have the GPU**: if ComfyUI is running and you did not
  start it, leave it alone and do not restart `llama-swap` over it. Ask.

## Always report

State when you stopped and restarted `llama-swap` and that it came back healthy.
