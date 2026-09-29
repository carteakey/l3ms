import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from unittest.mock import patch


RUNNER = Path(__file__).with_name("runner.py")
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

    def test_failed_benchmark_cannot_emit_result_marker(self):
        engine = object.__new__(runner.SemIfEngine)
        engine.allow_simulation = False
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(runner.InferenceUnavailable):
            runner.run_benchmark(engine, str(RUNNER.parents[1] / "suites" / "routing-gate.json"), 1)
        self.assertNotIn("BENCH_RESULT:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
