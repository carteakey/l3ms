import unittest

from qwen_image_21 import build_workflow, validate_settings


class QwenImage21WorkflowTests(unittest.TestCase):
    def test_workflow_matches_official_quantized_model_names(self):
        workflow = build_workflow("a lighthouse", "", 1024, 1024, 25, 42, 1.0, "test")

        self.assertEqual(
            workflow["1"]["inputs"]["unet_name"],
            "qwen_image_2.1_int8_convrot.safetensors",
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
        self.assertEqual(workflow["6"]["inputs"]["seed"], 42)
        self.assertEqual(workflow["8"]["class_type"], "SaveImage")

    def test_workflow_rejects_unsupported_dimensions(self):
        with self.assertRaises(ValueError):
            validate_settings(1000, 1024, 25, 1.0, 3600)
        with self.assertRaises(ValueError):
            validate_settings(1024, 2080, 25, 1.0, 3600)

    def test_workflow_rejects_invalid_sampling_settings(self):
        with self.assertRaises(ValueError):
            validate_settings(1024, 1024, 0, 1.0, 3600)
        with self.assertRaises(ValueError):
            validate_settings(1024, 1024, 25, 0.0, 3600)


if __name__ == "__main__":
    unittest.main()
