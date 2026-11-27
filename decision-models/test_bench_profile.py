import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).with_name("bench_profile.py")
SPEC = importlib.util.spec_from_file_location("bench_profile", PATH)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class BenchProfileTest(unittest.TestCase):
    def test_command_uses_the_manifest_engine_model_device_and_named_suite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "decision-runtimes.json"
            manifest.write_text(json.dumps({
                "version": "1.0",
                "profiles": [{
                    "id": "fixture-cpu",
                    "engine": "nagi",
                    "model": "fixture/model",
                    "device": "cpu",
                    "entrypoint": "decision-models/runner.py",
                }],
            }))
            command = bench.build_command("fixture-cpu", "routing-gate", 3, manifest_path=manifest)
        joined = " ".join(command)
        self.assertIn("decision-models/runner.py", joined)
        for flag, value in [("--engine", "nagi"), ("--device", "cpu"), ("--iterations", "3")]:
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertTrue(command[command.index("--bench-suite") + 1].endswith("suites/routing-gate.json"))

    def test_router_hosted_profiles_are_refused(self):
        for profile_id in ("vllm-d3-flash", "liquid-d1-2b", "vllm-d3-27b"):
            with self.subTest(profile=profile_id):
                with self.assertRaises(SystemExit) as raised:
                    bench.build_command(profile_id, "routing-gate", 1)
                self.assertIn("llama-swap", str(raised.exception))

    def test_unknown_profile_or_suite_fails_before_running_anything(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "decision-runtimes.json"
            manifest.write_text(json.dumps({
                "version": "1.0",
                "profiles": [{
                    "id": "fixture-cpu",
                    "engine": "nagi",
                    "model": "fixture/model",
                    "device": "cpu",
                    "entrypoint": "decision-models/runner.py",
                }],
            }))
            for profile, suite in [("nope", "routing-gate"), ("fixture-cpu", "no-such-suite")]:
                with self.assertRaises(SystemExit):
                    bench.build_command(profile, suite, 1, manifest_path=manifest)


if __name__ == "__main__":
    unittest.main()
