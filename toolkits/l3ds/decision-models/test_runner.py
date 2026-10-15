import contextlib
import builtins
import importlib.util
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, Mock, call, patch


RUNNER = Path(__file__).with_name("runner.py")
TOOLKIT_ROOT = RUNNER.parents[1]
SETUP_SCRIPT = TOOLKIT_ROOT / "maintenance" / "prepare-nagi-smol-env.sh"
NAGI_SMOL_LOCK = TOOLKIT_ROOT / "requirements" / "nagi-smol.lock.txt"
HOSTED_LOCK = TOOLKIT_ROOT / "requirements" / "hosted.lock.txt"
SPEC = importlib.util.spec_from_file_location("l3ds_runner", RUNNER)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

STATE = {"alert": "test"}
QUESTIONS = {"route": {"criteria": {"one": "First", "two": "Second"}}}


class RunnerSafetyTest(unittest.TestCase):
    def test_stdin_inputs_preserve_native_payload_without_loading_for_parse(self):
        engine = Mock()
        engine.decide.return_value = {"answers": {"route": {"choice": "one"}}, "engine": "fixture"}
        data = json.dumps({"state": STATE, "questions": QUESTIONS}).encode()
        with patch("sys.argv", [str(RUNNER), "--engine", "nagi", "--model", "fixture", "--input-stdin"]), \
             patch("sys.stdin", io.TextIOWrapper(io.BytesIO(data))), \
             patch.object(runner, "get_engine", return_value=engine), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 0)
        engine.decide.assert_called_once_with(STATE, QUESTIONS)

    def test_invalid_stdin_fails_before_engine_construction_without_echoing_input(self):
        for payload in [b"secret-malformed", b"\xff", b"{}", b"[]", b"x" * (runner.MAX_INPUT_BYTES + 1),
                        b'{"state":NaN,"questions":{"route":{"type":"score"}}}',
                        b'{"state":{},"questions":{}}', b'{"state":{},"questions":{"x":"invalid"}}',
                        b'{"state":{},"questions":{"x":{}},"secret":"credential"}']:
            with self.subTest(payload_size=len(payload)), \
                 patch("sys.argv", [str(RUNNER), "--engine", "nagi", "--model", "fixture", "--input-stdin"]), \
                 patch("sys.stdin", io.TextIOWrapper(io.BytesIO(payload))), \
                 patch.object(runner, "get_engine") as construct, contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(runner.main(), 2)
                construct.assert_not_called()
                self.assertNotIn("secret", errors.getvalue())

    def test_stdin_cannot_override_legacy_args_or_benchmark_mode(self):
        for extra in [["--state", "{}"], ["--question", "{}"], ["--bench-suite", "fixture.json"]]:
            with patch("sys.argv", [str(RUNNER), "--engine", "nagi", "--model", "fixture", "--input-stdin", *extra]), \
                 patch.object(runner, "get_engine") as construct, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                runner.main()
            self.assertEqual(error.exception.code, 2)
            construct.assert_not_called()

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
            engine = runner.MockOrApiJevEngine(allow_simulation=False, single_attempt=True)
        with self.assertRaisesRegex(
            runner.InferenceUnavailable,
            "^hosted decision API credentials are unavailable$",
        ):
            engine.decide(STATE, QUESTIONS)

    def test_single_attempt_hosted_success_uses_zero_retries_once_and_closes(self):
        response = Mock()
        response.model_dump.return_value = {
            "model": "jev-latest",
            "usage": {"total_tokens": 4},
            "answers": {"route": {"choice": "one"}},
        }
        client = Mock()
        client.system_one.return_value = response
        sdk = ModuleType("typesafe_sdk")
        retry_policy = object()
        sdk.RetryPolicy = Mock(return_value=retry_policy)
        sdk.TypeSafeClient = Mock(return_value=client)
        sdk_logger = logging.getLogger("typesafe_sdk")
        prior_disabled = sdk_logger.disabled
        with patch.object(sdk_logger, "disabled", prior_disabled), patch.dict(
            "sys.modules", {"typesafe_sdk": sdk}
        ), patch.dict("os.environ", {"TYPESAFE_LOG_LEVEL": "debug"}, clear=False):
            engine = runner.MockOrApiJevEngine(api_key="fixture-secret", single_attempt=True)
            self.assertEqual(os.environ["TYPESAFE_LOG_LEVEL"], "off")
            result = engine.decide(STATE, QUESTIONS)

        sdk.RetryPolicy.assert_called_once_with(max_retries=0)
        sdk.TypeSafeClient.assert_called_once_with(
            api_key="fixture-secret", base_url=None, retry=retry_policy
        )
        client.system_one.assert_called_once_with(
            state=STATE, questions=QUESTIONS, model="jev-latest"
        )
        client.close.assert_called_once_with()
        self.assertEqual(
            [entry for entry in client.mock_calls if entry[0] in {"system_one", "close"}],
            [
                call.system_one(state=STATE, questions=QUESTIONS, model="jev-latest"),
                call.close(),
            ],
        )
        self.assertEqual(result["answers"], {"route": {"choice": "one"}})
        self.assertEqual(result["usage"], {"total_tokens": 4})

    def test_single_attempt_hosted_failure_is_redacted_one_call_and_closed(self):
        client = Mock()
        client.system_one.side_effect = RuntimeError("secret credential and request payload")
        sdk = ModuleType("typesafe_sdk")
        sdk.RetryPolicy = Mock(return_value=object())
        sdk.TypeSafeClient = Mock(return_value=client)
        sdk_logger = logging.getLogger("typesafe_sdk")
        prior_disabled = sdk_logger.disabled
        with patch.object(sdk_logger, "disabled", prior_disabled), patch.dict(
            "os.environ", {"TYPESAFE_LOG_LEVEL": "debug"}, clear=False
        ), patch.dict("sys.modules", {"typesafe_sdk": sdk}):
            engine = runner.MockOrApiJevEngine(api_key="fixture-secret", single_attempt=True)
            with self.assertRaises(runner.InferenceUnavailable) as error:
                engine.decide(STATE, QUESTIONS)

        self.assertEqual(str(error.exception), "hosted decision request failed")
        self.assertIsNone(error.exception.__cause__)
        self.assertNotIn("secret", str(error.exception))
        client.system_one.assert_called_once_with(
            state=STATE, questions=QUESTIONS, model="jev-latest"
        )
        client.close.assert_called_once_with()
        self.assertEqual(len([entry for entry in client.mock_calls if entry[0] == "system_one"]), 1)

    def test_single_attempt_missing_sdk_fails_with_fixed_message_without_echoing_import_error(self):
        original_import = builtins.__import__

        def missing_sdk(name, *args, **kwargs):
            if name == "typesafe_sdk":
                raise ImportError("secret dependency failure")
            return original_import(name, *args, **kwargs)

        stderr = io.StringIO()
        sdk_logger = logging.getLogger("typesafe_sdk")
        prior_disabled = sdk_logger.disabled
        with patch.object(sdk_logger, "disabled", prior_disabled), patch.dict(
            "os.environ", {"TYPESAFE_LOG_LEVEL": "debug"}, clear=False
        ), patch("builtins.__import__", side_effect=missing_sdk), contextlib.redirect_stderr(stderr):
            engine = runner.MockOrApiJevEngine(api_key="fixture-secret", single_attempt=True)
            with self.assertRaises(runner.InferenceUnavailable) as error:
                engine.decide(STATE, QUESTIONS)

        self.assertEqual(str(error.exception), "hosted decision client is unavailable")
        self.assertEqual(stderr.getvalue(), "")
        self.assertNotIn("secret", str(error.exception))

    def test_single_attempt_mode_cannot_fall_back_to_simulation(self):
        engine = runner.MockOrApiJevEngine(
            api_key="unused", allow_simulation=True, single_attempt=True
        )
        with self.assertRaisesRegex(
            runner.InferenceUnavailable,
            "^single-attempt hosted mode cannot use simulation$",
        ):
            engine.decide(STATE, QUESTIONS)

    def test_single_attempt_cli_forwards_flag_for_one_live_jev_call(self):
        engine = Mock()
        engine.decide.return_value = {"answers": {"route": {"choice": "one"}}}
        output = io.StringIO()
        with patch(
            "sys.argv",
            [str(RUNNER), "--engine", "jev", "--model", "jev-latest", "--single-attempt"],
        ), patch.object(runner, "one_shot_inputs", return_value=(STATE, QUESTIONS)), patch.object(
            runner, "get_engine", return_value=engine
        ) as build_engine, contextlib.redirect_stdout(output):
            self.assertEqual(runner.main(), 0)

        build_engine.assert_called_once_with(
            "jev",
            "jev-latest",
            "cuda",
            api_key=None,
            base_url=None,
            allow_simulation=False,
            single_attempt=True,
        )
        engine.decide.assert_called_once_with(STATE, QUESTIONS)
        self.assertEqual(json.loads(output.getvalue()), engine.decide.return_value)

    def test_single_attempt_cli_rejects_non_jev_simulation_and_benchmark_modes(self):
        invalid_arguments = [
            ["--engine", "nagi", "--model", "fixture"],
            ["--engine", "jev", "--model", "fixture", "--simulate"],
            ["--engine", "jev", "--model", "fixture", "--bench-suite", "unused.json"],
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), patch(
                "sys.argv", [str(RUNNER), *arguments, "--single-attempt"]
            ), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                runner.main()
            self.assertEqual(error.exception.code, 2)

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


class LlamaCppSystemOneTest(unittest.TestCase):
    def setUp(self):
        self.answers = {
            "route": {
                "choice": "billing",
                "confidence": 0.987,
                "probabilities": {"billing": 0.987, "shipping": 0.013},
            }
        }

    def _response(self, payload, status=200):
        response = MagicMock()
        response.__enter__.return_value = response
        response.getcode.return_value = status
        response.read.return_value = json.dumps(payload, allow_nan=False).encode("utf-8")
        return response

    def test_one_post_uses_native_route_and_preserves_native_probabilities(self):
        body = {"model": "vllm-d3-flash", "answers": self.answers, "usage": {"tokens": 84}}
        response = self._response(body)
        opener = MagicMock()
        opener.open.return_value = response
        with patch.object(runner, "build_opener", return_value=opener) as build_opener:
            engine = runner.LlamaCppSystemOneEngine(
                "vllm-d3-flash",
                api_key="fixture-secret",
                base_url="http://127.0.0.1:8080/custom/proxy/",
            )
            result = engine.decide(STATE, QUESTIONS)

        build_opener.assert_called_once()
        handlers = build_opener.call_args.args
        self.assertEqual(handlers[0].proxies, {})
        self.assertIsInstance(handlers[1], runner._NoRedirect)
        opener.open.assert_called_once()
        request = opener.open.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "http://127.0.0.1:8080/custom/proxy/v1/systemone",
        )
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(
            json.loads(request.data),
            {"model": "vllm-d3-flash", "state": STATE, "questions": QUESTIONS},
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer fixture-secret")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(opener.open.call_args.kwargs, {"timeout": runner.LLAMACPP_TIMEOUT_SECONDS})
        response.read.assert_called_once_with(runner.MAX_HTTP_RESPONSE_BYTES + 1)
        self.assertEqual(result["answers"], self.answers)
        self.assertEqual(result["model"], "vllm-d3-flash")
        self.assertEqual(result["usage"], {"tokens": 84})
        self.assertEqual(result["engine"], "llamacpp-native")
        self.assertGreaterEqual(result["client_latency_ms"], 0)
        self.assertNotIn("latency_ms", result)

    def test_http_failure_is_one_attempt_redacted_and_never_simulated(self):
        opener = MagicMock()
        opener.open.side_effect = RuntimeError("fixture-secret request body: test alert")
        with patch.object(runner, "build_opener", return_value=opener):
            engine = runner.LlamaCppSystemOneEngine(
                "cloudflare-clef-flash",
                api_key="fixture-secret",
                base_url="http://localhost:8080",
            )
            with self.assertRaises(runner.InferenceUnavailable) as failure:
                engine.decide(STATE, QUESTIONS)

        self.assertEqual(str(failure.exception), "llama.cpp System One request failed")
        self.assertIsNone(failure.exception.__cause__)
        self.assertNotIn("fixture-secret", str(failure.exception))
        opener.open.assert_called_once()

    def test_non_local_or_non_http_endpoints_fail_before_open(self):
        for base_url in (
            "https://localhost:8080",
            "http://example.com:8080",
            "http://user@localhost:8080",
            "http://localhost:8080/prefix?token=hidden",
        ):
            with self.subTest(base_url=base_url):
                opener = MagicMock()
                with patch.object(runner, "build_opener", return_value=opener):
                    engine = runner.LlamaCppSystemOneEngine("vllm-d3-flash", base_url=base_url)
                    with self.assertRaisesRegex(
                        runner.InferenceUnavailable,
                        "^llama.cpp endpoint configuration is invalid$",
                    ):
                        engine.decide(STATE, QUESTIONS)
                opener.open.assert_not_called()

    def test_invalid_router_id_and_newline_credential_fail_closed(self):
        for kwargs in (
            {"model": "../v1/chat/completions"},
            {"model": "vllm-d3-flash", "api_key": "secret\nInjected: yes"},
        ):
            opener = MagicMock()
            with self.subTest(kwargs=kwargs), patch.object(runner, "build_opener", return_value=opener):
                engine = runner.LlamaCppSystemOneEngine(**kwargs)
                with self.assertRaises(runner.InferenceUnavailable):
                    engine.decide(STATE, QUESTIONS)
                opener.open.assert_not_called()

    def test_engine_uses_llama_swap_environment_connection(self):
        opener = MagicMock()
        with patch.dict(
            "os.environ",
            {
                "LLAMA_SWAP_URL": "http://localhost:8080/router-prefix",
                "LLAMA_SWAP_API_KEY": "environment-secret",
            },
        ), patch.object(runner, "build_opener", return_value=opener):
            engine = runner.get_engine("llamacpp", "cloudflare-clef-flash", "cuda")

        self.assertEqual(
            engine.endpoint,
            "http://localhost:8080/router-prefix/v1/systemone",
        )
        self.assertEqual(engine.api_key, "environment-secret")
        self.assertIs(engine.opener, opener)

    def test_redirect_handler_refuses_redirect_targets(self):
        request = runner.Request("http://localhost:8080")
        self.assertIsNone(
            runner._NoRedirect().redirect_request(
                request, None, 302, "Found", {}, "http://localhost:8081"
            )
        )

    def test_oversized_request_or_response_is_rejected_without_retry(self):
        opener = MagicMock()
        with patch.object(runner, "build_opener", return_value=opener):
            engine = runner.LlamaCppSystemOneEngine("vllm-d3-flash")
            with self.assertRaises(runner.InferenceUnavailable):
                engine.decide({"large": "x" * runner.MAX_INPUT_BYTES}, QUESTIONS)
            opener.open.assert_not_called()

        response = MagicMock()
        response.__enter__.return_value = response
        response.getcode.return_value = 200
        response.read.return_value = b"x" * (runner.MAX_HTTP_RESPONSE_BYTES + 1)
        opener = MagicMock()
        opener.open.return_value = response
        with patch.object(runner, "build_opener", return_value=opener):
            engine = runner.LlamaCppSystemOneEngine("vllm-d3-flash")
            with self.assertRaises(runner.InferenceUnavailable):
                engine.decide(STATE, QUESTIONS)
        opener.open.assert_called_once()
        response.read.assert_called_once_with(runner.MAX_HTTP_RESPONSE_BYTES + 1)

    def test_malformed_answers_and_nonfinite_json_fail_closed(self):
        invalid_responses = (
            b'{"answers":"invalid"}',
            b'{"answers":{}}',
            b'{"answers":{"other":{"choice":"billing"}}}',
            b'{"answers":{"route":"invalid"}}',
            b'{"choices":"invalid","scores":{"route":{"score":1}}}',
            b'{"answers":{"route":{"choice":"billing"}},"extra":NaN}',
            b'{"answers":{"route":{"choice":"billing"}},"extra":1e999}',
        )
        for body in invalid_responses:
            with self.subTest(response=body):
                response = MagicMock()
                response.__enter__.return_value = response
                response.getcode.return_value = 200
                response.read.return_value = body
                opener = MagicMock()
                opener.open.return_value = response
                with patch.object(runner, "build_opener", return_value=opener):
                    engine = runner.LlamaCppSystemOneEngine("vllm-d3-flash")
                    with self.assertRaisesRegex(
                        runner.InferenceUnavailable,
                        "^llama.cpp System One request failed$",
                    ):
                        engine.decide(STATE, QUESTIONS)
                opener.open.assert_called_once()

    def test_typed_native_answers_reject_wrong_values_without_retry(self):
        cases = [
            ("noul", {"noul": True}), ("noul", {"noul": 1.5}),
            ("score", {"score": "high"}),
            ("choice", {"choice": "undeclared"}),
            ("noul", {"noul": 0.5, "confidence": -1}),
            ("noul", {"noul": 0.5, "probabilities": {"yes": 2}}),
        ]
        for kind, answer in cases:
            with self.subTest(kind=kind, answer=answer):
                opener = MagicMock()
                opener.open.return_value = self._response({"answers": {"q": answer}})
                with patch.object(runner, "build_opener", return_value=opener):
                    engine = runner.LlamaCppSystemOneEngine("vllm-d3-flash")
                    with self.assertRaises(runner.InferenceUnavailable):
                        engine.decide(STATE, {"q": {"type": kind, "criteria": {"yes": "Yes"}}})
                opener.open.assert_called_once()

    def test_cli_accepts_llamacpp_and_forwards_explicit_connection_values(self):
        engine = Mock()
        engine.decide.return_value = {"answers": self.answers, "engine": "llamacpp-native"}
        output = io.StringIO()
        with patch(
            "sys.argv",
            [
                str(RUNNER),
                "--engine",
                "llamacpp",
                "--model",
                "vllm-d3-flash",
                "--base-url",
                "http://localhost:8080",
                "--api-key",
                "fixture-secret",
            ],
        ), patch.object(runner, "one_shot_inputs", return_value=(STATE, QUESTIONS)), patch.object(
            runner, "get_engine", return_value=engine
        ) as build_engine, contextlib.redirect_stdout(output):
            self.assertEqual(runner.main(), 0)

        build_engine.assert_called_once_with(
            "llamacpp",
            "vllm-d3-flash",
            "cuda",
            api_key="fixture-secret",
            base_url="http://localhost:8080",
            allow_simulation=False,
            single_attempt=False,
        )
        self.assertEqual(json.loads(output.getvalue()), engine.decide.return_value)

    def test_cli_rejects_simulation_for_local_http_engine(self):
        with patch(
            "sys.argv",
            [str(RUNNER), "--engine", "llamacpp", "--model", "vllm-d3-flash", "--simulate"],
        ), patch.object(runner, "get_engine") as build_engine, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            runner.main()
        self.assertEqual(error.exception.code, 2)
        build_engine.assert_not_called()


class NagiRepositoryTest(unittest.TestCase):
    def test_smol_and_big_loaders_receive_declared_repository(self):
        cases = [
            ("load_smol", "nagisanzeninz/nagi-smol-v0"),
            ("load_big", "nagisanzeninz/nagi-big-v3"),
        ]
        for loader_name, repository in cases:
            with self.subTest(loader=loader_name):
                fake_nagi = Mock()
                loaded_model = object()
                loader = getattr(fake_nagi, loader_name)
                loader.return_value = loaded_model

                with patch.dict("sys.modules", {"nagi": fake_nagi}):
                    engine = runner.NagiEngine(repository, device="cuda")

                self.assertIs(engine.model, loaded_model)
                loader.assert_called_once_with(repo=repository, device="cuda")

    def test_smol_and_big_cpu_fallback_reuses_declared_repository(self):
        cases = [
            ("load_smol", "nagisanzeninz/nagi-smol-v0"),
            ("load_big", "nagisanzeninz/nagi-big-v3"),
        ]
        for loader_name, repository in cases:
            with self.subTest(loader=loader_name):
                fake_nagi = Mock()
                loaded_model = object()
                loader = getattr(fake_nagi, loader_name)
                loader.side_effect = [RuntimeError("CUDA unavailable"), loaded_model]

                with patch.dict("sys.modules", {"nagi": fake_nagi}), \
                     contextlib.redirect_stderr(io.StringIO()):
                    engine = runner.NagiEngine(repository, device="cuda")

                self.assertIs(engine.model, loaded_model)
                self.assertEqual(
                    loader.call_args_list,
                    [
                        call(repo=repository, device="cuda"),
                        call(repo=repository, device="cpu"),
                    ],
                )

    def test_huge_loader_call_shape_remains_unchanged(self):
        fake_nagi = Mock()
        loaded_model = object()
        fake_nagi.load_huge.return_value = loaded_model

        with patch.dict("sys.modules", {"nagi": fake_nagi}):
            engine = runner.NagiEngine("nagisanzeninz/nagi-huge-v0", device="cuda")

        self.assertIs(engine.model, loaded_model)
        fake_nagi.load_huge.assert_called_once_with(device="cuda")


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
            setup_copy = maintenance / "prepare-nagi-smol-env.sh"
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

    def test_hosted_lock_pins_the_observed_sync_sdk_runtime_closure(self):
        requirements = [
            line.strip()
            for line in HOSTED_LOCK.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(
            set(requirements),
            {
                "typesafe-sdk==0.7.1",
                "httpx2==2.13.1",
                "httpcore2==2.13.1",
                "anyio==4.15.1",
                "idna==3.20",
                "truststore==0.10.4",
                "h11==0.16.0",
                "pydantic==2.13.5",
                "pydantic-core==2.46.5",
                "annotated-types==0.8.0",
                "typing-inspection==0.4.4",
                "typing-extensions==4.16.0",
                "tenacity==9.1.4",
            },
        )


if __name__ == "__main__":
    unittest.main()
