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

    def test_references_enter_through_dotted_autogrow_slots(self):
        workflow = build_workflow(
            "a woman",
            "",
            896,
            1216,
            25,
            42,
            2.5,
            "test",
            ref_names=("ref_a.png", "ref_b.png"),
            ref_resolution=0,
        )
        text_encode = workflow["4"]["inputs"]

        self.assertEqual(text_encode["vae"], ["3", 0])
        self.assertEqual(text_encode["resolution"], 0)
        # Autogrow slots are addressed by their finalized (dotted) input id; a bare
        # "image_1" key reaches execute() as an unexpected keyword argument.
        self.assertEqual(text_encode["images.image_1"], ["20", 0])
        self.assertEqual(text_encode["images.image_2"], ["21", 0])
        self.assertNotIn("image_1", text_encode)
        self.assertEqual(workflow["20"]["class_type"], "LoadImage")
        self.assertEqual(workflow["20"]["inputs"]["image"], "ref_a.png")
        # Native-size references set the output frame themselves.
        self.assertEqual(workflow["6"]["inputs"]["latent_image"], ["4", 2])

    def test_resized_references_keep_the_requested_output_frame(self):
        workflow = build_workflow(
            "a woman",
            "",
            896,
            1216,
            25,
            42,
            2.5,
            "test",
            ref_names=("ref_a.png", "ref_b.png", "ref_c.png"),
            ref_resolution=512,
        )

        self.assertEqual(workflow["4"]["inputs"]["resolution"], 512)
        self.assertEqual(workflow["6"]["inputs"]["latent_image"], ["5", 0])
        self.assertEqual(workflow["5"]["inputs"]["width"], 896)
        self.assertEqual(workflow["5"]["inputs"]["height"], 1216)

    def test_loras_chain_after_the_base_model_in_order(self):
        workflow = build_workflow(
            "a woman",
            "",
            896,
            1216,
            25,
            42,
            2.5,
            "test",
            loras=(("first.safetensors", 1.0), ("second.safetensors", 0.5)),
        )

        self.assertEqual(workflow["12"]["inputs"]["model"], ["1", 0])
        self.assertEqual(workflow["12"]["inputs"]["lora_name"], "first.safetensors")
        self.assertEqual(workflow["13"]["inputs"]["model"], ["12", 0])
        self.assertEqual(workflow["13"]["inputs"]["strength_model"], 0.5)
        self.assertEqual(workflow["6"]["inputs"]["model"], ["13", 0])

    def test_edit_path_reencodes_the_image_and_lower_denoise_survives(self):
        workflow = build_workflow(
            "keep the pose, change the face",
            "",
            896,
            1216,
            25,
            42,
            2.5,
            "test",
            edit_name="scene.png",
            denoise=0.45,
        )

        self.assertEqual(workflow["24"]["class_type"], "LoadImage")
        self.assertEqual(workflow["25"]["inputs"]["pixels"], ["24", 0])
        self.assertEqual(workflow["25"]["inputs"]["vae"], ["3", 0])
        self.assertEqual(workflow["6"]["inputs"]["latent_image"], ["25", 0])
        self.assertEqual(workflow["6"]["inputs"]["denoise"], 0.45)

    def test_validate_settings_covers_reference_and_edit_limits(self):
        with self.assertRaises(ValueError):
            validate_settings(1024, 1024, 25, 1.0, 3600, ref_resolution=5000)
        with self.assertRaises(ValueError):
            validate_settings(1024, 1024, 25, 1.0, 3600, edit_name_present=True, denoise=1.5)
        with self.assertRaises(ValueError):
            validate_settings(1024, 1024, 25, 1.0, 3600, edit_name_present=True, denoise=0.0)
        validate_settings(1024, 1024, 25, 1.0, 3600, edit_name_present=True, denoise=0.45)


if __name__ == "__main__":
    unittest.main()
