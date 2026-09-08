#!/usr/bin/env python3
"""Regression tests for downloader concurrency precedence."""

import sys
import types
import unittest
from pathlib import Path


fake_hub = types.ModuleType("huggingface_hub")
fake_hub.snapshot_download = lambda **_: "unused"
sys.modules.setdefault("huggingface_hub", fake_hub)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_hf_model import resolve_max_workers


class WorkerPrecedenceTests(unittest.TestCase):
    def test_explicit_cli_override_wins(self) -> None:
        self.assertEqual(resolve_max_workers(12, 3, True), 3)

    def test_model_value_wins_over_slow_default(self) -> None:
        self.assertEqual(resolve_max_workers(8, None, True), 8)

    def test_null_model_uses_runtime_slow_default(self) -> None:
        self.assertEqual(resolve_max_workers(None, None, True), 4)

    def test_no_controls_leaves_hub_default(self) -> None:
        self.assertIsNone(resolve_max_workers(None, None, False))


if __name__ == "__main__":
    unittest.main()
