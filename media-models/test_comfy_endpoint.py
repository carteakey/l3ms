"""Unit tests for the shared ComfyUI endpoint rule in comfy_endpoint.py."""

import importlib.util
import pathlib
import unittest

MODULE = pathlib.Path(__file__).with_name("comfy_endpoint.py")
SPEC = importlib.util.spec_from_file_location("comfy_endpoint", MODULE)
EP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EP)

ROUTER = "http://127.0.0.1:8080/comfyui"
DIRECT = "http://127.0.0.1:8188"


class ResolveEndpoint(unittest.TestCase):
    def test_every_case(self):
        cases = [
            # (arg url, env, expected base, expected key, why)
            (None, {}, DIRECT, "", "nothing set: a ComfyUI started directly on 8188"),
            (None, {"LLAMA_SWAP_API_KEY": "k"}, ROUTER, "k",
             "a key with no URL is the house router, not a direct 8188"),
            (None, {"LLAMA_SWAP_API_KEY": "  "}, DIRECT, "", "a blank key is no key"),
            (None, {"LLAMA_SWAP_URL": "http://127.0.0.1:8080/"}, ROUTER, "",
             "the router route is built from LLAMA_SWAP_URL, slash normalized"),
            (None, {"LLAMA_SWAP_URL": "http://127.0.0.1:8080/comfyui"}, ROUTER, "",
             "an already-routed URL is not doubled"),
            (None, {"LLAMA_SWAP_URL": "http://box:8080/router/comfyui/"}, "http://box:8080/router/comfyui", "",
             "a routed URL under a subpath keeps its subpath"),
            (None, {"LLAMA_SWAP_URL": "http://127.0.0.1:8080", "LLAMA_SWAP_API_KEY": "k"}, ROUTER, "k",
             "URL plus key: the key still has to travel"),
            (None, {"L2M2_COMFY_URL": "http://127.0.0.1:8188/"}, DIRECT, "",
             "an explicit direct URL is used verbatim, slash stripped"),
            (None, {"L2M2_COMFY_URL": "http://127.0.0.1:8080"}, "http://127.0.0.1:8080", "",
             "L2M2_COMFY_URL is not rewritten, so a bare router root stays a bare router root"),
            (None, {"L2M2_COMFY_URL": "http://127.0.0.1:8080/comfyui", "LLAMA_SWAP_API_KEY": "k"},
             ROUTER, "k", "a routed L2M2_COMFY_URL still carries the key"),
            (None, {"L2M2_COMFY_URL": "", "LLAMA_SWAP_API_KEY": "k"}, ROUTER, "k",
             "an empty L2M2_COMFY_URL is absent, so the key implies the router"),
            (None, {"L2M2_COMFY_URL": "http://127.0.0.1:8188", "LLAMA_SWAP_URL": "http://127.0.0.1:8080"},
             DIRECT, "", "L2M2_COMFY_URL outranks LLAMA_SWAP_URL"),
            ("http://127.0.0.1:9000/", {"L2M2_COMFY_URL": "http://127.0.0.1:8188",
                                        "LLAMA_SWAP_URL": "http://127.0.0.1:8080"},
             "http://127.0.0.1:9000", "", "the argument outranks both env vars and is used verbatim"),
        ]
        for arg, env, want_base, want_key, why in cases:
            with self.subTest(why=why, arg=arg, env=env):
                self.assertEqual(EP.resolve_endpoint(arg, dict(env)), (want_base, want_key))

    def test_no_result_ever_keeps_a_trailing_slash(self):
        for arg, env in ((None, {"LLAMA_SWAP_URL": "http://h:8080/"}), ("http://h:9000//", {}),
                         (None, {"L2M2_COMFY_URL": "http://h:8188/"})):
            with self.subTest(arg=arg, env=env):
                base, _ = EP.resolve_endpoint(arg, dict(env))
                self.assertFalse(base.endswith("/"), base)

    def test_defaults_to_the_real_environment(self):
        import os

        original = dict(os.environ)
        try:
            os.environ.clear()
            os.environ["LLAMA_SWAP_API_KEY"] = "abc"
            self.assertEqual(EP.resolve_endpoint(), (ROUTER, "abc"))
        finally:
            os.environ.clear()
            os.environ.update(original)


if __name__ == "__main__":
    unittest.main()
