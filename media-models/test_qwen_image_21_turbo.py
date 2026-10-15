import unittest

from qwen_image_21_turbo import DEFAULT_TURBO_SIGMAS, build_workflow, validate_settings


class QwenImage21TurboWorkflowTests(unittest.TestCase):
    def test_workflow_matches_official_quantized_turbo_model_names(self):
        workflow = build_workflow(
            "a lighthouse", "", 1024, 1024, 42, 1.0, DEFAULT_TURBO_SIGMAS, "test"
        )

        self.assertEqual(
            workflow["1"]["inputs"]["unet_name"],
            "qwen-image-2.1-turbo-int8-convrot.safetensors",
        )
        self.assertEqual(
            workflow["2"]["inputs"]["clip_name"],
            "qwen3vl_8b_int8_convrot.safetensors",
        )
        self.assertEqual(
            workflow["3"]["inputs"]["vae_name"],
            "qwen_image_2.1_vae_bf16.safetensors",
        )
        self.assertEqual(workflow["4"]["inputs"]["prompt"], "a lighthouse")
        self.assertEqual(workflow["6"]["class_type"], "KSamplerSelect")
        self.assertEqual(workflow["6"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(workflow["7"]["class_type"], "ManualSigmas")
        self.assertEqual(workflow["7"]["inputs"]["sigmas"], DEFAULT_TURBO_SIGMAS)
        self.assertEqual(workflow["8"]["class_type"], "SamplerCustom")
        self.assertEqual(workflow["8"]["inputs"]["noise_seed"], 42)
        self.assertEqual(workflow["8"]["inputs"]["cfg"], 1.0)
        self.assertEqual(workflow["10"]["class_type"], "SaveImage")

    def test_workflow_rejects_unsupported_dimensions(self):
        with self.assertRaises(ValueError):
            validate_settings(1000, 1024, 1.0, 3600)
        with self.assertRaises(ValueError):
            validate_settings(1024, 2080, 1.0, 3600)

    def test_workflow_rejects_invalid_sampling_settings(self):
        with self.assertRaises(ValueError):
            validate_settings(1024, 1024, 0.0, 3600)
        with self.assertRaises(ValueError):
            validate_settings(1024, 1024, -1.0, 3600)

    def test_workflow_supports_input_image(self):
        workflow = build_workflow(
            "edit yacht", "", 1024, 1024, 42, 1.0, DEFAULT_TURBO_SIGMAS, "test",
            image_filename="boat.png", use_image_latent=True,
        )
        self.assertIn("11", workflow)
        self.assertEqual(workflow["11"]["class_type"], "LoadImage")
        self.assertEqual(workflow["11"]["inputs"]["image"], "boat.png")
        self.assertEqual(workflow["4"]["inputs"]["images.image_1"], ["11", 0])
        self.assertEqual(workflow["4"]["inputs"]["vae"], ["3", 0])
        self.assertEqual(workflow["8"]["inputs"]["latent_image"], ["4", 2])


if __name__ == "__main__":
    unittest.main()
