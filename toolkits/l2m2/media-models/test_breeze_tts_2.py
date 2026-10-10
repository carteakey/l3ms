import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("generate-breeze-tts-2.sh")


class BreezeTts2ScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cli = self.root / "fake-audiocpp-cli"
        self.call_log = self.root / "cli-calls.jsonl"
        self.model = self.root / "breeze.gguf"
        self.spec = self.root / "breeze_tts.json"
        self.reference = self.root / "reference voice.wav"
        self.output_dir = self.root / "outputs"
        self.model.write_bytes(b"test model fixture")
        self.spec.write_text('{"family":"breeze_tts"}\n', encoding="utf-8")
        self.reference.write_bytes(b"RIFF test fixture")
        self.cli.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "with open(os.environ['BREEZE_TEST_CALL_LOG'], 'a', encoding='utf-8') as f:\n"
            "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if '--list-loaders' in sys.argv:\n"
            "    print(os.environ.get('BREEZE_TEST_LOADERS', '{\"loaders\":{\"breeze_tts\":{\"tasks\":{\"tts\":[\"offline\"]}}}}'))\n",
            encoding="utf-8",
        )
        self.cli.chmod(0o755)
        self.env = os.environ.copy()
        self.env.update(
            {
                "L3MS_AUDIO_CPP_CLI": str(self.cli),
                "L3MS_BREEZE_TTS_MODEL": str(self.model),
                "L3MS_BREEZE_TTS_MODEL_SPEC": str(self.spec),
                "L3MS_MEDIA_OUTPUT_DIR": str(self.output_dir),
                "BREEZE_TEST_CALL_LOG": str(self.call_log),
            }
        )

    def tearDown(self):
        self.temp.cleanup()

    def run_script(self, *args, env=None):
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            check=False,
            capture_output=True,
            text=True,
            env=env or self.env,
        )

    def calls(self):
        if not self.call_log.exists():
            return []
        return [json.loads(line) for line in self.call_log.read_text().splitlines()]

    @staticmethod
    def option_value(args, option):
        index = args.index(option)
        return args[index + 1]

    def test_check_is_read_only_and_does_not_require_text(self):
        result = self.run_script("--check")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Breeze-TTS loader is compiled", result.stdout)
        self.assertIn("Read-only preflight", result.stdout)
        self.assertFalse(self.output_dir.exists())
        self.assertEqual(self.calls(), [["--list-loaders", "--json"]])

    def test_check_reports_missing_loader_model_and_spec_without_generation(self):
        env = self.env.copy()
        env["BREEZE_TEST_LOADERS"] = '{"loaders":[]}'
        env["L3MS_BREEZE_TTS_MODEL"] = str(self.root / "missing.gguf")
        env["L3MS_BREEZE_TTS_MODEL_SPEC"] = str(self.root / "missing.json")

        result = self.run_script("--check", env=env)

        self.assertEqual(result.returncode, 1)
        self.assertIn("Breeze-TTS loader is not listed", result.stderr)
        self.assertIn("Breeze-TTS model file", result.stderr)
        self.assertIn("Breeze-TTS model spec", result.stderr)
        self.assertFalse(self.output_dir.exists())
        self.assertEqual(self.calls(), [["--list-loaders", "--json"]])

    def test_check_does_not_accept_loader_name_in_unrelated_json_value(self):
        env = self.env.copy()
        env["BREEZE_TEST_LOADERS"] = '{"loaders":{},"note":"breeze_tts"}'

        result = self.run_script("--check", env=env)

        self.assertEqual(result.returncode, 1)
        self.assertIn("Breeze-TTS loader is not listed", result.stderr)
        self.assertFalse(self.output_dir.exists())

    def test_cloning_preflight_checks_reference_wav_without_synthesis_text(self):
        result = self.run_script("--check", "--task", "clon", "--voice-ref", str(self.reference))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cloning reference audio", result.stdout)
        self.assertFalse(self.output_dir.exists())
        self.assertEqual(self.calls(), [["--list-loaders", "--json"]])

    def test_standard_and_voice_design_use_native_tts_route(self):
        standard_output = self.root / "standard.wav"
        standard = self.run_script("--text", "A standard voice.", "--output", str(standard_output))
        self.assertEqual(standard.returncode, 0, standard.stderr)
        standard_args = self.calls()[-1]
        self.assertEqual(self.option_value(standard_args, "--task"), "tts")
        self.assertEqual(self.option_value(standard_args, "--backend"), "cuda")
        self.assertEqual(self.option_value(standard_args, "--text"), "A standard voice.")
        self.assertIn("instruction=Speak clearly and naturally.", standard_args)

        design_output = self.root / "design.wav"
        design = self.run_script(
            "--task",
            "design",
            "--text",
            "A designed voice.",
            "--instruction",
            "warm narrator",
            "--output",
            str(design_output),
        )
        self.assertEqual(design.returncode, 0, design.stderr)
        design_args = self.calls()[-1]
        self.assertEqual(self.option_value(design_args, "--task"), "tts")
        self.assertIn("instruction=warm narrator", design_args)

    def test_cpu_backend_is_explicit_and_invalid_backend_never_launches(self):
        result = self.run_script("--backend", "cpu", "--text", "CPU speech.")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.option_value(self.calls()[-1], "--backend"), "cpu")
        calls = self.calls()
        invalid = self.run_script("--backend", "automatic", "--text", "No fallback.")
        self.assertEqual(invalid.returncode, 2)
        self.assertEqual(self.calls(), calls)

    def test_voice_cloning_keeps_native_task_and_reference_arguments(self):
        output = self.root / "clone.wav"
        result = self.run_script(
            "--task",
            "clon",
            "--text",
            "Read in the reference voice.",
            "--voice-ref",
            str(self.reference),
            "--reference-text",
            "A short reference transcript.",
            "--output",
            str(output),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.calls()[-1]
        self.assertEqual(self.option_value(args, "--task"), "clon")
        self.assertEqual(self.option_value(args, "--voice-ref"), str(self.reference))
        self.assertEqual(self.option_value(args, "--reference-text"), "A short reference transcript.")

    def test_streaming_uses_native_options_and_dedicated_chunk_output(self):
        output = self.root / "stream.wav"
        result = self.run_script(
            "--text",
            "Incremental streaming.",
            "--mode",
            "streaming",
            "--stream-frames",
            "8",
            "--stream-lookahead",
            "4",
            "--output",
            str(output),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.calls()[-1]
        self.assertEqual(self.option_value(args, "--mode"), "streaming")
        self.assertIn("stream_frames_per_event=8", args)
        self.assertIn("stream_lookahead_margin=4", args)
        self.assertEqual(self.option_value(args, "--out-dir"), f"{output}.chunks")
        self.assertTrue(Path(f"{output}.chunks").is_dir())


if __name__ == "__main__":
    unittest.main()
