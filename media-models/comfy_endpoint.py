"""One rule for finding ComfyUI, shared by the media-models builders that talk through llama-swap.

ComfyUI is a llama-swap engine (comfyui_auto, routed at /comfyui/), so the normal endpoint is the
router, not a direct :8188. The rule lives here instead of in each builder because it has already
drifted: only one of five modules had the bare-key branch, and the others silently resolved
:8188 while llama-swap owned the card. Measured 2026-10-27 on the image lane, that fallback made
the GPU guard see the resident language server holding 11558 MiB and abort a six-row batch with
"GPU is busy" before the router ever had a chance to evict it.
"""

from __future__ import annotations

import os

ROUTER_DEFAULT = "http://127.0.0.1:8080"  # the house llama-swap listener (llama-swap.service, :8080)
DIRECT_DEFAULT = "http://127.0.0.1:8188"  # ComfyUI started directly by maintenance/comfyui-service.sh
COMFY_ROUTE = "/comfyui"


def resolve_endpoint(url: str | None = None, env: dict[str, str] | None = None) -> tuple[str, str]:
    """Return (base, api_key) for the ComfyUI request layer, base never ending in a slash.

    Precedence, first match wins:

    1. the ``url`` argument (a generator's ``--url``, or a runner's explicit endpoint);
    2. ``L2M2_COMFY_URL``, used verbatim, because whoever set it chose the endpoint;
    3. the router from ``LLAMA_SWAP_URL``, with ``/comfyui`` appended unless it is already there;
    4. the router implied by a bare ``LLAMA_SWAP_API_KEY``: agents in this tree get the key and
       not the URL, and a key with no URL means the house router on :8080;
    5. a direct ComfyUI on :8188.

    The key is returned whenever it is set, even for a direct URL: an extra Authorization header
    is ignored by ComfyUI and needed if that URL turns out to be the router.
    """
    env = os.environ if env is None else env
    base = (url or env.get("L2M2_COMFY_URL") or "").strip()
    key = (env.get("LLAMA_SWAP_API_KEY") or "").strip()
    if base:
        return base.rstrip("/"), key
    router = (env.get("LLAMA_SWAP_URL") or "").strip()
    if not router and key:
        router = ROUTER_DEFAULT
    if not router:
        return DIRECT_DEFAULT, key
    router = router.rstrip("/")
    if not router.endswith(COMFY_ROUTE):
        router = router + COMFY_ROUTE
    return router, key
