"""Unit tests for the MiniMax-H3 reference-to-video workflow builder.

These check graph shape and the wire rules that the engine enforces; they do not call
ComfyUI, so they run with no GPU and no router.
"""

import importlib.util
import json
import pathlib
import unittest

MODULE = pathlib.Path(__file__).with_name("minimax_h3_video.py")
SPEC = importlib.util.spec_from_file_location("minimax_h3_video", MODULE)
H3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(H3)

MANIFEST = MODULE.parents[1] / "models" / "minimax-h3-comfyui-files.txt"


def build(**overrides):
    kwargs = dict(
        prompt="a couple in a dim kitchen, handheld camera",
        width=608,
        height=352,
        frames=124,
        steps=20,
        seed=42,
        filename_prefix="h3/test",
    )
    kwargs.update(overrides)
    return H3.build_workflow(**kwargs)


class GraphMatchesTheOfficialWorkflow(unittest.TestCase):
    def test_model_names_match_the_download_manifest(self):
        wanted = [line.strip() for line in MANIFEST.read_text().splitlines() if line.strip() and not line.startswith("#")]
        graph = build()
        used = [
            f"diffusion_models/{graph['127']['inputs']['unet_name']}",
            f"text_encoders/{graph['128']['inputs']['clip_name']}",
            f"vae/{graph['119']['inputs']['vae_name']}",
            f"vae/{graph['120']['inputs']['vae_name']}",
        ]
        self.assertEqual(used, wanted)
        self.assertEqual(H3.DEFAULT_TE.split(".")[0], "qwen3vl_32b_minimax_h3_nvfp4_awq")

    def test_sampling_defaults_match_the_shipped_graph(self):
        graph = build()
        self.assertEqual(graph["124"]["inputs"]["scheduler"], "simple")
        self.assertEqual(graph["124"]["inputs"]["steps"], 20)
        self.assertEqual(graph["124"]["inputs"]["denoise"], 1.0)
        self.assertEqual(graph["123"]["inputs"]["sampler_name"], "res_multistep")
        self.assertEqual(graph["128"]["inputs"]["type"], "minimax")
        self.assertEqual(graph["130"]["inputs"]["fps"], 24.0)
        self.assertEqual(graph["92"]["inputs"]["format"], "auto")
        self.assertEqual(graph["92"]["inputs"]["codec"], "auto")
        self.assertEqual(graph["136"]["inputs"]["ref_image_size"], "match")
        # The official graph chains: reference -> guider + latent, scheduler off the model.
        self.assertEqual(graph["126"]["inputs"]["conditioning"], ("136", 0))
        self.assertEqual(graph["125"]["inputs"]["latent_image"], ("136", 1))
        self.assertEqual(graph["125"]["inputs"]["sigmas"], ("124", 0))
        self.assertEqual(graph["122"]["inputs"]["samples"], ("125", 0))
        self.assertEqual(graph["121"]["inputs"]["vae"], ("120", 0))

    def test_the_graph_has_no_negative_conditioning(self):
        # Verified against /object_info: MiniMaxH3ReferenceToVideo outputs are
        # ['positive','LATENT'] and BasicGuider has no negative input, so a negative prompt
        # is not expressible in this graph at all.
        graph = build()
        for node_id, node in graph.items():
            for key in node["inputs"]:
                self.assertNotIn("negative", key.lower(), f"node {node_id} unexpectedly has {key}")
        self.assertEqual(sorted(graph["126"]["inputs"]), ["conditioning", "model"])

    def test_turbo_lora_and_sigma_shift_rehook_the_model(self):
        graph = build(turbo_lora=(("minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", 1.0),))
        self.assertEqual(graph["126"]["inputs"]["model"], ("145", 0))
        self.assertEqual(graph["124"]["inputs"]["model"], ("145", 0))
        self.assertEqual(graph["145"]["inputs"]["model"], ("127", 0))
        graph = build(sigma_shift=(12.0, 3.0))
        self.assertEqual(graph["147"]["inputs"]["model"], ("127", 0))
        self.assertEqual(graph["126"]["inputs"]["model"], ("147", 0))
        self.assertEqual(graph["124"]["inputs"]["model"], ("147", 0))

    def test_audio_can_be_switched_off(self):
        graph = build(want_audio=False)
        self.assertNotIn("121", graph)
        self.assertNotIn("audio", graph["130"]["inputs"])


class ReferenceSlotsFollowTheAutogrowWireRule(unittest.TestCase):
    def test_references_use_dotted_finalized_slot_ids(self):
        graph = build(ref_names=("a.png", "b.png"))
        self.assertEqual(graph["136"]["inputs"]["ref_images.ref_image_0"], ("137", 0))
        self.assertEqual(graph["136"]["inputs"]["ref_images.ref_image_1"], ("138", 0))
        self.assertNotIn("ref_image_0", graph["136"]["inputs"])
        self.assertNotIn("ref_images", graph["136"]["inputs"])  # a nested dict arrives empty
        self.assertEqual(graph["137"]["inputs"]["image"], "a.png")

    def test_ninth_reference_is_still_a_single_node(self):
        graph = build(ref_names=tuple(f"r{i}.png" for i in range(9)))
        keys = sorted(k for k in graph["136"]["inputs"] if k.startswith("ref_images."))
        self.assertEqual(len(keys), 9)
        self.assertEqual(keys[-1], "ref_images.ref_image_8")
        loaded = [n for n in graph.values() if n["class_type"] == "LoadImage"]
        self.assertEqual(len(loaded), 9)

    def test_no_references_leaves_the_slot_absent(self):
        graph = build()
        self.assertFalse([k for k in graph["136"]["inputs"] if k.startswith("ref_images.")])

    def test_stage_reference_images_rejects_the_ninth_extra(self):
        with self.assertRaises(ValueError):
            H3.stage_reference_images([str(MODULE)] * 10)


class FrameAndDimensionRules(unittest.TestCase):
    def test_frames_for_seconds_reproduces_the_length_math_node(self):
        # max(5, round(a*24)) + (5 - (n % 17)) % 17, the expression in the shipped graph.
        self.assertEqual(H3.video_frames_for_seconds(5), 124)
        self.assertEqual(H3.video_frames_for_seconds(1), 39)
        self.assertEqual(H3.video_frames_for_seconds(10), 243)
        self.assertEqual(H3.normalize_frames(124), 124)
        self.assertEqual(H3.normalize_frames(100), 107)
        self.assertEqual(H3.normalize_frames(5), 5)

    def test_frame_counts_off_the_seventeen_step_round_up_then_are_validated(self):
        # normalize_frames is the forgiving knob (the CLI prints the adjustment); a bad
        # request is what validate_settings refuses, so an adjusted value always passes it.
        self.assertEqual(H3.normalize_frames(123), 124)
        self.assertEqual(H3.normalize_frames(130), 141)
        self.assertEqual(H3.max_legal_frames(), 3592)
        self.assertEqual(H3.normalize_frames(3592), 3592)
        for bad in (4, 0, -17):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    H3.normalize_frames(bad)
        for bad in (3601, 3600):  # 3600 would round up past the ceiling
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    H3.normalize_frames(bad)
        for frames in (123, 130, 3600):
            with self.subTest(frames=frames):
                with self.assertRaises(ValueError):
                    H3.validate_settings(width=608, height=352, frames=frames, steps=20, seed=42,
                                         scheduler="simple", sampler="res_multistep",
                                         ref_image_size="match", ref_count=0, timeout=3600)
        H3.validate_settings(width=608, height=352, frames=H3.normalize_frames(130), steps=20,
                             seed=42, scheduler="simple", sampler="res_multistep",
                             ref_image_size="match", ref_count=0, timeout=3600)

    def test_megapixel_ladder_matches_the_workflows_own_table(self):
        # The shipped workflow's note table: 0.2 -> 608x352, 0.3 -> 736x416, 0.4 -> 864x480.
        self.assertEqual(H3.dims_for_megapixels(0.2), (608, 352))
        self.assertEqual(H3.dims_for_megapixels(0.3), (736, 416))
        self.assertEqual(H3.dims_for_megapixels(0.4), (864, 480))

    def test_validate_settings_covers_dims_frames_and_names(self):
        good = dict(width=608, height=352, frames=124, steps=20, seed=42, scheduler="simple",
                    sampler="res_multistep", ref_image_size="match", ref_count=2, timeout=3600)
        H3.validate_settings(**good)
        for label, override in (
            ("width", {"width": 600}),
            ("height", {"height": 350}),
            ("frames", {"frames": 123}),
            ("steps", {"steps": 0}),
            ("seed", {"seed": -1}),
            ("scheduler", {"scheduler": "karrasbogus"}),
            ("sampler", {"sampler": ""}),
            ("ref_image_size", {"ref_image_size": "max-ish"}),
            ("ref_count", {"ref_count": 10}),
            ("timeout", {"timeout": 0}),
        ):
            with self.subTest(bad=label):
                with self.assertRaises(ValueError):
                    H3.validate_settings(**{**good, **override})

    def test_asset_names_must_be_bare_filenames(self):
        for bad in ("", "../other/thing.safetensors", "sub/dir/x.safetensors", ".hidden"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    H3.validate_asset_name("--model", bad)
        H3.validate_asset_name("--model", "minimax_h3_ref2va_pruned_int8_convrot.safetensors")


class EndpointResolution(unittest.TestCase):
    def test_router_route_is_built_from_environment(self):
        import os

        original = dict(os.environ)
        try:
            os.environ["LLAMA_SWAP_URL"] = "http://127.0.0.1:8080/"
            os.environ["LLAMA_SWAP_API_KEY"] = "k"
            os.environ.pop("L2M2_COMFY_URL", None)
            self.assertEqual(H3.configure_api(), "http://127.0.0.1:8080/comfyui")
            self.assertTrue(H3.via_router())
            os.environ["L2M2_COMFY_URL"] = "http://127.0.0.1:8188"
            self.assertEqual(H3.configure_api(), "http://127.0.0.1:8188")
            self.assertFalse(H3.via_router())
        finally:
            os.environ.clear()
            os.environ.update(original)

    def test_saved_video_lookup_ignores_the_history_key_name(self):
        record = {"outputs": {"92": {"videos": [{"filename": "h3/a.mp4", "subfolder": "h3", "type": "output"}]}}}
        self.assertEqual(H3.find_saved_video(record)["filename"], "h3/a.mp4")
        record = {"outputs": {"92": {"video": [{"filename": "a.webm"}]}}}
        self.assertEqual(H3.find_saved_video(record)["filename"], "a.webm")
        with self.assertRaises(LookupError):
            H3.find_saved_video({"outputs": {"92": {"anything": [{"nope": 1}]}}})

    def test_dry_run_graph_is_json_serialisable(self):
        graph = build(ref_names=("a.png", "b.png"), turbo_lora=(("turbo.safetensors", 0.8),))
        self.assertEqual(json.loads(json.dumps(graph))["136"]["inputs"]["ref_images.ref_image_0"], ["137", 0])


class PreflightAgainstAStubbedEngine(unittest.TestCase):
    """--check runs before any scheduler/sampler is chosen, so knob checks must stay silent."""

    def stub(self, info):
        def fake_request_json(path, payload=None, timeout=30):
            return {} if path == "/system_stats" else info
        return fake_request_json

    def test_check_passes_when_only_the_nodes_exist(self):
        info = {node: {} for node in H3.REQUIRED_NODES}
        original = H3.request_json
        try:
            H3.request_json = self.stub(info)
            H3.check_api()
            with self.assertRaises(RuntimeError):
                H3.check_api(extra_nodes=("MiniMaxH3SigmaShift",))  # not in the stub info
        finally:
            H3.request_json = original

    def test_missing_nodes_are_named(self):
        info = {node: {} for node in H3.REQUIRED_NODES}
        info.pop("MiniMaxH3ReferenceToVideo")
        original = H3.request_json
        try:
            H3.request_json = self.stub(info)
            with self.assertRaises(RuntimeError) as caught:
                H3.check_api()
            self.assertIn("MiniMaxH3ReferenceToVideo", str(caught.exception))
        finally:
            H3.request_json = original

    def test_knobs_are_validated_against_the_engine_only_when_requested(self):
        info = {node: {} for node in H3.REQUIRED_NODES}
        info["BasicScheduler"] = {"input": {"required": {"scheduler": ["COMBO", {"options": ["simple", "karras"]}]}}}
        info["KSamplerSelect"] = {"input": {"required": {"sampler_name": ["COMBO", {"options": ["res_multistep", "euler"]}]}}}
        original = H3.request_json
        try:
            H3.request_json = self.stub(info)
            H3.check_api()  # the --check path: no knobs requested, nothing to reject
            H3.check_api(scheduler="simple", sampler="res_multistep")
            for kwargs in ({"scheduler": "bogus"}, {"sampler": "bogus"}):
                with self.subTest(**kwargs):
                    with self.assertRaises(RuntimeError):
                        H3.check_api(**kwargs)
        finally:
            H3.request_json = original


if __name__ == "__main__":
    unittest.main()
