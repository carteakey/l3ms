import contextlib
import builtins
import importlib.util
import io
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


RUNNER = Path(__file__).with_name("runner.py")
TOOLKIT_ROOT = RUNNER.parents[1]
SETUP_SCRIPT = TOOLKIT_ROOT / "maintenance" / "setup-env.sh"
NAGI_SMOL_LOCK = TOOLKIT_ROOT / "requirements" / "nagi-smol.lock.txt"
SPEC = importlib.util.spec_from_file_location("l3ds_runner", RUNNER)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

STATE = {"alert": "test"}
QUESTIONS = {"route": {"criteria": {"one": "First", "two": "Second"}}}


class RunnerSafetyTest(unittest.TestCase):
    def test_unavailable_nagi_and_laya_fail_instead_of_fabricating_answers(self):
        for engine_type in (runner.NagiEngine, runner.LayaEngine):
            with self.subTest(engine=engine_type.__name__):
                engine = object.__new__(engine_type)
                engine.model_name = "missing-model"
                engine.load_err = "not installed"
                engine.allow_simulation = False
                if engine_type is runner.NagiEngine:
                    engine.model = None
                else:
                    engine.agent = None
                with self.assertRaises(runner.InferenceUnavailable):
                    engine.decide(STATE, QUESTIONS)

    def test_semif_requires_explicit_simulation(self):
        engine = object.__new__(runner.SemIfEngine)
        engine.allow_simulation = False
        with self.assertRaises(runner.InferenceUnavailable):
            engine.decide(STATE, QUESTIONS)
        engine.allow_simulation = True
        self.assertEqual(engine.decide(STATE, QUESTIONS)["engine"], "semif-simulated")

    def test_hosted_without_key_fails_closed(self):
        with patch.dict("os.environ", {"TYPESAFE_API_KEY": "", "OPENROUTER_API_KEY": ""}):
            engine = runner.MockOrApiJevEngine(allow_simulation=False)
        with self.assertRaises(runner.InferenceUnavailable):
            engine.decide(STATE, QUESTIONS)

    def test_explicit_simulation_does_not_load_models_or_call_hosted_api(self):
        original_import = builtins.__import__

        def guard_import(name, *args, **kwargs):
            if name in {"nagi", "laya", "typesafe_sdk"}:
                raise AssertionError(f"simulation attempted to import {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guard_import):
            nagi = runner.NagiEngine("nagi-smol", allow_simulation=True)
            laya = runner.LayaEngine("laya-en", allow_simulation=True)
            hosted = runner.MockOrApiJevEngine(api_key="unused", allow_simulation=True)
        self.assertIn("simulated", nagi.decide(STATE, QUESTIONS)["engine"])
        self.assertIn("simulated", laya.decide(STATE, QUESTIONS)["engine"])
        self.assertIsNone(hosted.client)

    def test_failed_benchmark_cannot_emit_result_marker(self):
        engine = object.__new__(runner.SemIfEngine)
        engine.allow_simulation = False
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(runner.InferenceUnavailable):
            runner.run_benchmark(engine, str(RUNNER.parents[1] / "suites" / "routing-gate.json"), 1)
        self.assertNotIn("BENCH_RESULT:", output.getvalue())


class NagiSmolSetupTest(unittest.TestCase):
    def test_setup_defaults_to_a_no_write_dry_run(self):
        venv = TOOLKIT_ROOT / ".venv"
        existed_before = venv.exists() or venv.is_symlink()
        result = subprocess.run(
            ["bash", str(SETUP_SCRIPT)],
            cwd=TOOLKIT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no changes", result.stdout)
        self.assertIn("--install", result.stdout)
        self.assertIn("Model weights: not part of this setup", result.stdout)
        self.assertEqual(venv.exists() or venv.is_symlink(), existed_before)

    def test_install_refuses_to_modify_an_existing_venv(self):
        with tempfile.TemporaryDirectory(prefix="l3ds-setup-test-") as temp:
            root = Path(temp)
            maintenance = root / "maintenance"
            requirements = root / "requirements"
            maintenance.mkdir()
            requirements.mkdir()
            setup_copy = maintenance / "setup-env.sh"
            shutil.copyfile(SETUP_SCRIPT, setup_copy)
            shutil.copyfile(NAGI_SMOL_LOCK, requirements / NAGI_SMOL_LOCK.name)
            existing_venv = root / ".venv"
            existing_venv.mkdir()
            sentinel = existing_venv / "preserve.txt"
            sentinel.write_text("keep", encoding="utf-8")

            result = subprocess.run(
                ["bash", str(setup_copy), "--install"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to modify existing environment", result.stderr)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_lock_is_pinned_to_observed_nagi_only_dependency_set(self):
        requirements = [
            line.strip()
            for line in NAGI_SMOL_LOCK.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertTrue(
            any(
                line
                == "nagi-decisions @ git+https://github.com/nagisanzenin/nagi.git@"
                "80f8c5e71832e227c8bf14cc52bbb448d3771c7a"
                for line in requirements
            )
        )
        self.assertIn("torch==2.6.0+cu124", requirements)
        self.assertIn("transformers==5.17.0", requirements)
        self.assertIn("peft==0.21.0", requirements)
        self.assertFalse(any("semif" in line.lower() for line in requirements))
        self.assertFalse(any("laya" in line.lower() for line in requirements))
        for line in requirements[1:]:
            self.assertIn("==", line, f"dependency is not exactly pinned: {line}")


if __name__ == "__main__":
    unittest.main()
