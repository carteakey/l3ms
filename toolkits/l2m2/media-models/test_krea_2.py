import unittest

from krea_2 import build_workflow, validate_settings


class Krea2WorkflowTests(unittest.TestCase):
    def test_workflow_matches_krea2_nodes_and_defaults(self):
        workflow = build_workflow(
            prompt="a retro neon cityscape",
            negative_prompt="",
            width=1024,
            height=1536,
            steps=12,
            seed=42,
            cfg=1.5,
            lora_strength=0.60,
            lora_name="krea2_turbo_distill_r128.safetensors",
            filename_prefix="test",
        )

        self.assertEqual(
            workflow["1"]["inputs"]["unet_name"],
            "krea2_raw_int8_convrot.safetensors",
        )
        self.assertEqual(
            workflow["2"]["inputs"]["clip_name"],
            "qwen3vl_4b_fp8_scaled.safetensors",
        )
        self.assertEqual(workflow["2"]["inputs"]["type"], "krea2")
        self.assertEqual(
            workflow["3"]["inputs"]["vae_name"],
            "qwen_image_vae.safetensors",
        )
        self.assertEqual(
            workflow["4"]["class_type"],
            "LoraLoaderModelOnly",
        )
        self.assertEqual(
            workflow["4"]["inputs"]["lora_name"],
            "krea2_turbo_distill_r128.safetensors",
        )
        self.assertEqual(workflow["4"]["inputs"]["strength_model"], 0.60)
        self.assertEqual(workflow["5"]["inputs"]["text"], "a retro neon cityscape")
        self.assertEqual(workflow["6"]["class_type"], "ConditioningZeroOut")
        self.assertEqual(workflow["8"]["inputs"]["model"], ["4", 0])
        self.assertEqual(workflow["8"]["inputs"]["seed"], 42)
        self.assertEqual(workflow["8"]["inputs"]["steps"], 12)
        self.assertEqual(workflow["8"]["inputs"]["cfg"], 1.5)
        self.assertEqual(workflow["8"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(workflow["8"]["inputs"]["scheduler"], "simple")
        self.assertEqual(workflow["10"]["class_type"], "SaveImage")

    def test_workflow_explicit_negative_prompt(self):
        workflow = build_workflow(
            prompt="a portrait",
            negative_prompt="blurry, distorted",
            width=1024,
            height=1024,
            steps=12,
            seed=42,
            cfg=1.5,
            lora_strength=0.60,
            lora_name="krea2_turbo_distill_r128.safetensors",
            filename_prefix="test",
        )
        self.assertEqual(workflow["6"]["class_type"], "CLIPTextEncode")
        self.assertEqual(workflow["6"]["inputs"]["text"], "blurry, distorted")

    def test_workflow_no_lora_bypass(self):
        workflow = build_workflow(
            prompt="a portrait",
            negative_prompt="",
            width=1024,
            height=1024,
            steps=12,
            seed=42,
            cfg=1.5,
            lora_strength=0.0,
            lora_name="krea2_turbo_distill_r128.safetensors",
            filename_prefix="test",
        )
        self.assertNotIn("4", workflow)
        self.assertEqual(workflow["8"]["inputs"]["model"], ["1", 0])

    def test_validate_settings_accepts_valid_resolutions(self):
        validate_settings(1024, 1024, 12, 1.5, 0.60, 3600)
        validate_settings(1024, 1536, 12, 1.5, 0.60, 3600)
        validate_settings(1536, 1024, 12, 1.5, 0.60, 3600)
        validate_settings(768, 1344, 12, 1.5, 0.60, 3600)
        validate_settings(1280, 704, 12, 1.5, 0.60, 3600)

    def test_validate_settings_rejects_unsupported_dimensions(self):
        with self.assertRaises(ValueError):
            validate_settings(1000, 1024, 12, 1.5, 0.60, 3600)
        with self.assertRaises(ValueError):
            validate_settings(1024, 2048, 12, 1.5, 0.60, 3600)
        with self.assertRaises(ValueError):
            validate_settings(1536, 1536, 12, 1.5, 0.60, 3600)

    def test_validate_settings_rejects_invalid_sampling_settings(self):
        with self.assertRaises(ValueError):
            validate_settings(1024, 1024, 0, 1.5, 0.60, 3600)
        with self.assertRaises(ValueError):
            validate_settings(1024, 1024, 12, 0.0, 0.60, 3600)
        with self.assertRaises(ValueError):
            validate_settings(1024, 1024, 12, 1.5, -0.5, 3600)

    def test_workflow_with_refusal_bypass(self):
        workflow = build_workflow(
            prompt="a portrait",
            negative_prompt="",
            width=1024,
            height=1024,
            steps=12,
            seed=42,
            cfg=1.5,
            lora_strength=0.60,
            lora_name="krea2_turbo_distill_r128.safetensors",
            filename_prefix="test",
            refusal_bypass=True,
        )
        self.assertIn("11", workflow)
        self.assertEqual(workflow["11"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(workflow["11"]["inputs"]["lora_name"], "Krea2_TextFusion_Refusal_Reduction.safetensors")
        self.assertEqual(workflow["11"]["inputs"]["strength_model"], 1.0)
        # Verify chaining: UNet -> Node 4 (Turbo) -> Node 11 (Refusal Bypass) -> KSampler (Node 8)
        self.assertEqual(workflow["4"]["inputs"]["model"], ["1", 0])
        self.assertEqual(workflow["11"]["inputs"]["model"], ["4", 0])
        self.assertEqual(workflow["8"]["inputs"]["model"], ["11", 0])

    def test_workflow_with_refusal_bypass_without_turbo(self):
        workflow = build_workflow(
            prompt="a portrait",
            negative_prompt="",
            width=1024,
            height=1024,
            steps=25,
            seed=42,
            cfg=4.0,
            lora_strength=0.0,
            lora_name="krea2_turbo_distill_r128.safetensors",
            filename_prefix="test",
            refusal_bypass=True,
        )
        self.assertNotIn("4", workflow)
        self.assertIn("11", workflow)
        self.assertEqual(workflow["11"]["inputs"]["model"], ["1", 0])
        self.assertEqual(workflow["8"]["inputs"]["model"], ["11", 0])


if __name__ == "__main__":
    unittest.main()

