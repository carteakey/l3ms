#!/usr/bin/env python3
"""Unit tests for the FLUX.3 Image Klein ComfyUI pipeline."""

import unittest
from flux_3_klein import build_workflow, parse_args, validate_settings, DEFAULT_UNET


class TestFlux3Klein(unittest.TestCase):
    def test_build_workflow_basic(self):
        wf = build_workflow(
            prompt="A majestic eagle soaring over mountains",
            negative_prompt="blurry",
            width=1024,
            height=1024,
            steps=16,
            seed=123,
            cfg=1.0,
            filename_prefix="test_prefix",
        )
        self.assertIn("1", wf)
        self.assertEqual(wf["1"]["class_type"], "UNETLoader")
        self.assertEqual(wf["1"]["inputs"]["unet_name"], DEFAULT_UNET)

        self.assertIn("2", wf)
        self.assertEqual(wf["2"]["class_type"], "CLIPLoader")
        self.assertEqual(wf["2"]["inputs"]["type"], "flux3")

        self.assertIn("6", wf)
        self.assertEqual(wf["6"]["class_type"], "EmptyFlux2LatentImage")
        self.assertEqual(wf["6"]["inputs"]["width"], 1024)
        self.assertEqual(wf["6"]["inputs"]["height"], 1024)

        self.assertIn("7", wf)
        self.assertEqual(wf["7"]["class_type"], "KSampler")
        self.assertEqual(wf["7"]["inputs"]["steps"], 16)
        self.assertEqual(wf["7"]["inputs"]["cfg"], 1.0)
        self.assertEqual(wf["7"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(wf["7"]["inputs"]["scheduler"], "simple")

    def test_build_workflow_with_boxes(self):
        boxes = [
            {"id": "1", "desc": "red hot-air balloon", "bbox": [0.1, 0.1, 0.4, 0.4]},
            {"id": "2", "desc": "lighthouse", "bbox": [0.5, 0.5, 0.9, 0.9]},
        ]
        wf = build_workflow(
            prompt="hot air balloon and lighthouse",
            negative_prompt="",
            width=1024,
            height=1024,
            steps=16,
            seed=42,
            cfg=1.0,
            filename_prefix="test_box",
            boxes=boxes,
        )
        prompt_text = wf["4"]["inputs"]["text"]
        self.assertIn("obj_1", prompt_text)
        self.assertIn("red hot-air balloon", prompt_text)
        self.assertIn("obj_2", prompt_text)

    def test_validate_settings_valid(self):
        validate_settings(1024, 768, 16, 1.0, 300.0, DEFAULT_UNET)

    def test_validate_settings_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            validate_settings(1025, 1024, 16, 1.0, 300.0)

    def test_validate_settings_invalid_steps(self):
        with self.assertRaises(ValueError):
            validate_settings(1024, 1024, 0, 1.0, 300.0)

    def test_parse_args(self):
        args = parse_args(["a sunset over the ocean", "--steps", "20", "--cfg", "1.5"])
        self.assertEqual(args.prompt, "a sunset over the ocean")
        self.assertEqual(args.steps, 20)
        self.assertEqual(args.cfg, 1.5)


if __name__ == "__main__":
    unittest.main()
