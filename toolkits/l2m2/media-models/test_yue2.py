import unittest
from pathlib import Path

from yue2_runner import (
    DEFAULT_BUDGET_GIB,
    DEFAULT_COT,
    DEFAULT_QUANTIZATION,
    DEFAULT_SEED,
    build_cli_args,
    prepare_request,
    resolve_runtime_paths,
    validate_settings,
)


class YuE2RunnerTests(unittest.TestCase):
    def test_validate_settings_valid(self):
        # Should not raise any exception
        validate_settings(
            style="pop song",
            lyrics="[Verse]\nHello world",
            cot="full",
            quantization="fp8",
            budget=12.0,
            cfg_scale=1.5,
        )

    def test_validate_settings_rejects_invalid_cot(self):
        with self.assertRaises(ValueError):
            validate_settings(
                style="pop song",
                lyrics="lyrics",
                cot="invalid_mode",
                quantization="fp8",
                budget=12.0,
            )

    def test_validate_settings_rejects_invalid_quantization(self):
        with self.assertRaises(ValueError):
            validate_settings(
                style="pop song",
                lyrics="lyrics",
                cot="full",
                quantization="int4",
                budget=12.0,
            )

    def test_validate_settings_rejects_non_positive_budget(self):
        with self.assertRaises(ValueError):
            validate_settings(
                style="pop song",
                lyrics="lyrics",
                cot="full",
                quantization="fp8",
                budget=0,
            )
        with self.assertRaises(ValueError):
            validate_settings(
                style="pop song",
                lyrics="lyrics",
                cot="full",
                quantization="fp8",
                budget=-5.0,
            )

    def test_validate_settings_rejects_non_positive_cfg(self):
        with self.assertRaises(ValueError):
            validate_settings(
                style="pop song",
                lyrics="lyrics",
                cot="full",
                quantization="fp8",
                budget=12.0,
                cfg_scale=0,
            )

    def test_validate_settings_requires_style_or_abc(self):
        with self.assertRaises(ValueError):
            validate_settings(
                style=None,
                lyrics="lyrics",
                cot="full",
                quantization="fp8",
                budget=12.0,
            )

    def test_prepare_request_structure(self):
        req = prepare_request(
            style="warm jazz",
            lyrics="[Chorus]\nStars shine bright",
            cot="melody",
            seed=12345,
            cfg_scale=1.2,
            req_id="test_song",
        )
        self.assertEqual(req["id"], "test_song")
        self.assertEqual(req["style"], "warm jazz")
        self.assertEqual(req["lyrics"], "[Chorus]\nStars shine bright")
        self.assertEqual(req["cot"], "melody")
        self.assertEqual(req["seed"], 12345)
        self.assertEqual(req["cfg_scale"], 1.2)

    def test_build_cli_args_generate(self):
        args = build_cli_args(
            command="generate",
            model="m-a-p/YuE2-3B",
            vae="m-a-p/YuE2-Vae",
            quantization="fp8",
            budget=12.0,
            backend="torch",
            offload_ar=True,
            cot="full",
            style="synthwave",
            lyrics="electric dreams",
            output_path=Path("/tmp/song"),
            seed=42,
            cfg_scale=1.0,
            stage="audio",
        )
        self.assertIn("generate", args)
        self.assertIn("--model", args)
        self.assertIn("m-a-p/YuE2-3B", args)
        self.assertIn("--vae", args)
        self.assertIn("m-a-p/YuE2-Vae", args)
        self.assertIn("--quantization", args)
        self.assertIn("fp8", args)
        self.assertIn("--budget", args)
        self.assertIn("12.0", args)
        self.assertIn("--offload-ar", args)
        self.assertIn("--cot", args)
        self.assertIn("full", args)
        self.assertIn("--style", args)
        self.assertIn("synthwave", args)
        self.assertIn("--lyrics", args)
        self.assertIn("electric dreams", args)
        self.assertIn("--output", args)
        self.assertIn("/tmp/song", args)
        self.assertIn("--seed", args)
        self.assertIn("42", args)

    def test_build_cli_args_doctor(self):
        args = build_cli_args(
            command="doctor",
            quantization="none",
            budget=24.0,
            backend="torch",
        )
        self.assertEqual(args[0], "doctor")
        self.assertIn("--quantization", args)
        self.assertIn("none", args)
        self.assertIn("--budget", args)
        self.assertIn("24.0", args)
        self.assertNotIn("--cot", args)

    def test_resolve_runtime_paths_structure(self):
        paths = resolve_runtime_paths()
        self.assertIn("yue_root", paths)
        self.assertIn("python_bin", paths)
        self.assertIn("model_root", paths)
        self.assertIn("media_root", paths)
        self.assertIn("output_dir", paths)
        self.assertIsInstance(paths["yue_root"], Path)


if __name__ == "__main__":
    unittest.main()
