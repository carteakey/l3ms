#!/usr/bin/env python3
"""Network-free tests for the machine-readable downloader estimator."""

import io
import json
import logging
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


fake_hub = types.ModuleType("huggingface_hub")
fake_hub.snapshot_download = lambda **_: []
sys.modules.setdefault("huggingface_hub", fake_hub)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import download_hf_model as downloader


def file_info(
    filename: str,
    file_size: object,
    *,
    is_cached: bool,
    will_download: bool,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        filename=filename,
        file_size=file_size,
        is_cached=is_cached,
        will_download=will_download,
    )


class EstimateJsonTests(unittest.TestCase):
    def run_estimate(self, arguments, fake_snapshot):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(downloader, "snapshot_download", side_effect=fake_snapshot):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = downloader.main(["--estimate-json", *arguments])
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_selected_estimate_forwards_filters_and_partitions_cached_bytes(self):
        calls = []

        def fake_snapshot(**kwargs):
            calls.append(kwargs)
            print("hub stdout must not leak")
            print("hub progress must not leak", file=sys.stderr)
            return [
                file_info(
                    "weights-q4.gguf", 100, is_cached=True, will_download=False
                ),
                file_info(
                    "tokenizer.json", 250, is_cached=False, will_download=True
                ),
            ]

        exit_code, stdout, stderr = self.run_estimate(
            [
                "--repo-id",
                "org/model",
                "--revision",
                "release",
                "--allow-patterns",
                "*.gguf",
                "*.json",
                "--ignore-patterns",
                "*debug*",
                "--local-dir",
                "/models/custom",
                "--max-workers",
                "3",
                "--slow",
            ],
            fake_snapshot,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout.count("\n"), 1)
        self.assertEqual(
            json.loads(stdout),
            {
                "schema_version": 1,
                "models": [
                    {
                        "repo_id": "org/model",
                        "revision": "release",
                        "matched_files": 2,
                        "total_bytes": 350,
                        "download_bytes": 250,
                        "cached_bytes": 100,
                    }
                ],
                "totals": {
                    "models": 1,
                    "matched_files": 2,
                    "total_bytes": 350,
                    "download_bytes": 250,
                    "cached_bytes": 100,
                },
            },
        )
        self.assertEqual(
            calls,
            [
                {
                    "repo_id": "org/model",
                    "local_dir": "/models/custom",
                    "allow_patterns": ["*.gguf", "*.json"],
                    "ignore_patterns": ["*debug*"],
                    "revision": "release",
                    "force_download": False,
                    "dry_run": True,
                }
            ],
        )

    def test_config_estimate_aggregates_enabled_valid_models_and_ignores_workers(self):
        calls = []

        def fake_snapshot(**kwargs):
            calls.append(kwargs)
            if kwargs["repo_id"] == "org/one":
                return [
                    file_info("cached.gguf", 10, is_cached=True, will_download=False),
                    file_info("new.gguf", 20, is_cached=False, will_download=True),
                ]
            return [
                file_info("forced.gguf", 30, is_cached=True, will_download=True)
            ]

        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "models.json"
            config_path.write_text(
                json.dumps(
                    {
                        "base_models_dir": "/models",
                        "models": [
                            {
                                "repo_id": "org/one",
                                "revision": "v1",
                                "allow_patterns": "*Q4*",
                                "ignore_patterns": ["*old*"],
                                "max_workers": 99,
                            },
                            {"enabled": False, "repo_id": "org/disabled"},
                            {"enabled": True, "repo_id": ""},
                            {
                                "repo_id": "org/two",
                                "local_dir": "/custom/two",
                                "force_download": True,
                                "max_workers": 2,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            exit_code, stdout, stderr = self.run_estimate(
                [
                    "--config",
                    str(config_path),
                    "--max-workers",
                    "1",
                    "--slow",
                ],
                fake_snapshot,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(
            payload["models"],
            [
                {
                    "repo_id": "org/one",
                    "revision": "v1",
                    "matched_files": 2,
                    "total_bytes": 30,
                    "download_bytes": 20,
                    "cached_bytes": 10,
                },
                {
                    "repo_id": "org/two",
                    "revision": "main",
                    "matched_files": 1,
                    "total_bytes": 30,
                    "download_bytes": 30,
                    "cached_bytes": 0,
                },
            ],
        )
        self.assertEqual(
            payload["totals"],
            {
                "models": 2,
                "matched_files": 3,
                "total_bytes": 60,
                "download_bytes": 50,
                "cached_bytes": 10,
            },
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["local_dir"], "/models/org/one")
        self.assertEqual(calls[0]["allow_patterns"], ["*Q4*"])
        self.assertEqual(calls[0]["ignore_patterns"], ["*old*"])
        self.assertEqual(calls[1]["local_dir"], "/custom/two")
        self.assertTrue(calls[1]["force_download"])
        for call in calls:
            self.assertTrue(call["dry_run"])
            self.assertNotIn("max_workers", call)

    def test_model_limit_counts_only_rows_eligible_for_estimation(self):
        calls = []

        def fake_snapshot(**kwargs):
            calls.append(kwargs)
            return []

        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "models.json"
            config_path.write_text(
                json.dumps(
                    {
                        "models": (
                            [
                                {"enabled": False, "repo_id": f"org/disabled-{index}"}
                                for index in range(downloader.MAX_ESTIMATE_MODELS + 20)
                            ]
                            + [{} for _ in range(downloader.MAX_ESTIMATE_MODELS + 20)]
                            + [{"repo_id": "org/eligible"}]
                        )
                    }
                ),
                encoding="utf-8",
            )
            exit_code, stdout, stderr = self.run_estimate(
                ["--config", str(config_path)], fake_snapshot
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout)["totals"]["models"], 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["repo_id"], "org/eligible")

    def test_model_limit_rejects_too_many_eligible_rows_before_hub_calls(self):
        calls = []

        def fake_snapshot(**kwargs):
            calls.append(kwargs)
            return []

        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "models.json"
            config_path.write_text(
                json.dumps(
                    {
                        "models": [
                            {"repo_id": f"org/model-{index}"}
                            for index in range(downloader.MAX_ESTIMATE_MODELS + 1)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            exit_code, stdout, stderr = self.run_estimate(
                ["--config", str(config_path)], fake_snapshot
            )

        self.assertNotEqual(exit_code, 0)
        self.assertEqual(stdout, "")
        self.assertIn("257 eligible models", stderr)
        self.assertIn("limit is 256", stderr)
        self.assertEqual(calls, [])

    def test_hub_output_is_suppressed_and_error_is_stderr_only(self):
        hub_log = io.StringIO()
        hub_logger = logging.getLogger("huggingface_hub.estimator_test")
        hub_handler = logging.StreamHandler(hub_log)
        previous_level = hub_logger.level
        previous_propagate = hub_logger.propagate
        previous_logging_disable = logging.root.manager.disable
        hub_logger.addHandler(hub_handler)
        hub_logger.setLevel(logging.WARNING)
        hub_logger.propagate = False

        def fake_snapshot(**_kwargs):
            print("noisy stdout")
            print("noisy stderr", file=sys.stderr)
            hub_logger.warning("noisy logging handler")
            raise RuntimeError("hub\nfailure")

        try:
            exit_code, stdout, stderr = self.run_estimate(
                ["--repo-id", "org/broken"], fake_snapshot
            )
        finally:
            hub_logger.removeHandler(hub_handler)
            hub_logger.setLevel(previous_level)
            hub_logger.propagate = previous_propagate

        self.assertNotEqual(exit_code, 0)
        self.assertEqual(stdout, "")
        self.assertEqual(hub_log.getvalue(), "")
        self.assertEqual(logging.root.manager.disable, previous_logging_disable)
        self.assertEqual(stderr.count("\n"), 1)
        self.assertIn("estimate failed: dry-run failed for org/broken: hub failure", stderr)
        self.assertNotIn("noisy", stderr)
        self.assertNotIn("download backend", stderr)

    def test_estimate_does_not_weaken_a_stricter_global_logging_threshold(self):
        previous_logging_disable = logging.root.manager.disable
        strict_threshold = logging.CRITICAL + 10
        logging.disable(strict_threshold)
        try:
            exit_code, stdout, stderr = self.run_estimate(
                ["--repo-id", "org/model"], lambda **_kwargs: []
            )
            self.assertEqual(logging.root.manager.disable, strict_threshold)
        finally:
            logging.disable(previous_logging_disable)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout)["totals"]["models"], 1)

    def test_result_count_bound_fails_without_json(self):
        result = file_info("model.gguf", 1, is_cached=False, will_download=True)

        exit_code, stdout, stderr = self.run_estimate(
            ["--repo-id", "org/huge"],
            lambda **_kwargs: [result] * (downloader.MAX_ESTIMATE_FILES + 1),
        )

        self.assertNotEqual(exit_code, 0)
        self.assertEqual(stdout, "")
        self.assertIn(str(downloader.MAX_ESTIMATE_FILES), stderr)
        self.assertIn("files across all models", stderr)

    def test_negative_or_noninteger_file_size_is_rejected(self):
        for invalid_size in (-1, True, "10", None):
            with self.subTest(invalid_size=invalid_size):
                exit_code, stdout, stderr = self.run_estimate(
                    ["--repo-id", "org/invalid"],
                    lambda **_kwargs: [
                        file_info(
                            "bad.gguf",
                            invalid_size,
                            is_cached=False,
                            will_download=True,
                        )
                    ],
                )
                self.assertNotEqual(exit_code, 0)
                self.assertEqual(stdout, "")
                self.assertIn("expected a nonnegative integer", stderr)

    def test_missing_config_is_stderr_only_and_nonzero(self):
        exit_code, stdout, stderr = self.run_estimate(
            ["--config", "/definitely/missing/models.json"], lambda **_kwargs: []
        )

        self.assertNotEqual(exit_code, 0)
        self.assertEqual(stdout, "")
        self.assertIn("config file not found", stderr)

    def test_missing_hub_dependency_is_machine_safe_and_normal_mode_stays_friendly(self):
        with patch.object(downloader, "snapshot_download", None):
            with patch.object(
                downloader, "HUB_IMPORT_ERROR", ImportError("missing dependency")
            ):
                machine_stdout = io.StringIO()
                machine_stderr = io.StringIO()
                with redirect_stdout(machine_stdout), redirect_stderr(machine_stderr):
                    machine_exit = downloader.main(
                        ["--estimate-json", "--repo-id", "org/model"]
                    )

                normal_stdout = io.StringIO()
                normal_stderr = io.StringIO()
                with redirect_stdout(normal_stdout), redirect_stderr(normal_stderr):
                    normal_exit = downloader.main(["--repo-id", "org/model"])

        self.assertNotEqual(machine_exit, 0)
        self.assertEqual(machine_stdout.getvalue(), "")
        self.assertIn("huggingface_hub is not installed", machine_stderr.getvalue())
        self.assertNotEqual(normal_exit, 0)
        self.assertIn("Error: huggingface_hub is not installed.", normal_stdout.getvalue())
        self.assertIn("pip install", normal_stdout.getvalue())
        self.assertEqual(normal_stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
