from contextlib import redirect_stderr
import io
import os
import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from yue2_runner import (
    DEFAULT_BUDGET_GIB,
    DEFAULT_COT,
    DEFAULT_QUANTIZATION,
    DEFAULT_SEED,
    build_cli_args,
    convert_prepared_audio,
    main,
    prepare_request,
    resolve_runtime_paths,
    run_yue2,
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

    def test_prepared_paths_require_and_pin_existing_yue_venv(self):
        with tempfile.TemporaryDirectory() as directory:
            yue_root = Path(directory) / "YuE"
            python = yue_root / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            python.chmod(0o700)
            with patch.dict(
                os.environ,
                {
                    "L3MS_YUE_PREPARED_ONLY": "1",
                    "L3MS_YUE_ROOT": str(yue_root),
                    "L2M2_PYTHON": "/caller/override/python",
                },
            ):
                paths = resolve_runtime_paths()
            self.assertEqual(paths["yue_root"], yue_root.resolve())
            self.assertEqual(paths["python_bin"], python)

            python.unlink()
            with patch.dict(
                os.environ,
                {
                    "L3MS_YUE_PREPARED_ONLY": "1",
                    "L3MS_YUE_ROOT": str(yue_root),
                    "L2M2_PYTHON": "/caller/override/python",
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "Prepared YuE Python environment"):
                    resolve_runtime_paths()

    def test_prepared_invocation_sets_offline_flags_and_redacts_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            yue_root = Path(directory) / "YuE"
            python = yue_root / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            stderr = io.StringIO()
            with patch.dict(os.environ, {"L3MS_YUE_PREPARED_ONLY": "1"}):
                with patch(
                    "yue2_runner.subprocess.run",
                    return_value=SimpleNamespace(returncode=0),
                ) as run:
                    with redirect_stderr(stderr):
                        result = run_yue2(
                            ["generate", "--prompt", "private lyric prompt", "--offline"],
                            python,
                            yue_root,
                        )
            self.assertEqual(result, 0)
            command = run.call_args.args[0]
            environment = run.call_args.kwargs["env"]
            self.assertIn("--offline", command)
            self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
            self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
            self.assertNotIn("private lyric prompt", stderr.getvalue())

    def test_prepared_audio_conversion_uses_fixed_flac_and_pcm16_wav(self):
        try:
            import numpy as np
            import soundfile as sf
        except ImportError:
            self.skipTest("YuE runtime soundfile dependencies are unavailable")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            song = output / "song"
            song.mkdir()
            source = song / "audio.flac"
            sf.write(source, np.zeros((12, 2), dtype=np.float32), 48_000, format="FLAC")
            (output / "other").mkdir()
            sf.write(output / "other/audio.flac", np.ones((3, 1), dtype=np.float32), 8_000, format="FLAC")

            self.assertTrue(convert_prepared_audio(output))
            with sf.SoundFile(output / "result.wav") as wave:
                self.assertEqual((wave.samplerate, wave.channels, wave.subtype), (48_000, 2, "PCM_16"))
                self.assertEqual(wave.frames, 12)
            self.assertTrue(source.is_file())

    def test_prepared_audio_conversion_never_replaces_existing_result(self):
        try:
            import numpy as np
            import soundfile as sf
        except ImportError:
            self.skipTest("YuE runtime soundfile dependencies are unavailable")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            song = output / "song"
            song.mkdir()
            sf.write(song / "audio.flac", np.zeros((1, 1), dtype=np.float32), 8_000, format="FLAC")
            result = output / "result.wav"
            result.write_bytes(b"pre-existing")
            self.assertFalse(convert_prepared_audio(output))
            self.assertEqual(result.read_bytes(), b"pre-existing")

    def test_prepared_generation_rejects_preexisting_song_symlink_before_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            yue_root = root / "YuE"
            python = yue_root / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            python.chmod(0o700)
            output = root / "private-job"
            output.mkdir()
            outside = root / "outside"
            outside.mkdir()
            (outside / "marker").write_text("untouched")
            (output / "song").symlink_to(outside, target_is_directory=True)
            with patch.dict(
                os.environ,
                {
                    "L3MS_YUE_PREPARED_ONLY": "1",
                    "L3MS_YUE_ROOT": str(yue_root),
                    "L3MS_MEDIA_OUTPUT_DIR": str(root),
                },
            ):
                with patch("yue2_runner.run_yue2") as run:
                    self.assertEqual(
                        main(
                            [
                                "--prompt",
                                "private style",
                                "--lyrics",
                                "private lyrics",
                                "--output",
                                str(output),
                            ]
                        ),
                        1,
                    )
            run.assert_not_called()
            self.assertEqual((outside / "marker").read_text(), "untouched")
            self.assertFalse((output / "result.wav").exists())


if __name__ == "__main__":
    unittest.main()
