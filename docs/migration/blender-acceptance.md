# Blender acceptance — 2026-10-15

## Standalone headless CPU render

The existing standalone wrapper produced a real 32×32 PNG using Blender
5.2.1 LTS, Cycles CPU, one sample and one thread. Owned child cleanup passed.
Private evidence: `.toolkit/blender-acceptance/run.noX0wA/`. This does not accept
root CLI/Stack dispatch or GPU rendering: those workflows retain the shared
idle-GPU guard and have not completed a successful real render through it.

## Interactive GUI and installed MCP bridge: partial pass

The actual standalone L2M2 CLI launched the installed Blender with factory
startup, private configuration/data/temp paths, two-thread software graphics
and the existing installed protocol-5 add-on. The installed `blender-mcp` 1.9.1
stdio bridge initialized, reported a matching add-on protocol, read the private
scene nonce, created one uniquely named cube after asserting that nonce inside
the mutation, and read the object back with `get_object_info` (8 vertices,
12 edges, 6 polygons). Telemetry and external provider tools were disabled.

This was an actual GUI: `bpy.app.background` was false, one window existed, and
the renderer was `llvmpipe (LLVM 23.1.1, 256 bits)`. The installed add-on source
was unchanged. A bootstrap-only constructor override selected a private
ephemeral loopback port during registration, then restored the constructor;
the normal port 9876 and existing clients were not used.

The round trip passed, but the overall check **failed**: Blender crashed during
shutdown, with `wl_egl_window_destroy` in its native backtrace, and the launcher
returned 1. The stdio bridge exited, the private listener was released and no
Blender child remained. Pi's Strata GPU allocation was unchanged. Do not label
this a clean interactive lifecycle pass or a GPU-rendering check.

Private receipts, hashes, exact argv/environment and logs:
`.toolkit/blender-gui-acceptance-lRalDpRDtV/runs/20261015T120159Z-51c6bd82/`.
Blender automatically wrote `tmp/quit.blend` and a crash report inside that
private directory. The original receipt's `blend_file_written: false` describes
the harness's intention incorrectly; no explicit save was requested, but this
automatic recovery file was observed. No original scene or preferences were
saved or changed. Retain that original receipt rather than rewriting the result.

## Shutdown isolation

A separate minimal factory-GUI test reproduced exit 1 and the same
`wl_egl_window_destroy` backtrace with **no add-on, MCP bridge, scene mutation or
L3MS launcher**. It again proved a real window and llvmpipe, then requested quit
through a Blender timer. Evidence:
`.toolkit/blender-gui-acceptance-lRalDpRDtV/quit-only-olmrq1n9/`.
No Blender process remained afterward, and the installed add-on hash and Pi's
GPU allocation were unchanged. This isolates the observed software-Wayland
timer-quit failure from the port; it does not establish its native root cause,
prove normal manual shutdown fails, or claim a fix. No installed runtime,
graphics driver or desktop setting was changed.

## Remaining gates

- Clean GUI shutdown in the intended installed display environment.
- Successful real root CLI/Stack headless execution in an approved idle window.
- GPU/device behavior, only if that backend is explicitly selected and tested.

External GUI/MCP remains standalone and externally owned. No new MCP server,
daemon, model download or persistent integration was added to L3MS.
