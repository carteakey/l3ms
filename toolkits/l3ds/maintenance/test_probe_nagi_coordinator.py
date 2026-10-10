"""Offline safety regressions for probe-nagi-coordinator.py."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("probe-nagi-coordinator.py")
SPEC = importlib.util.spec_from_file_location("probe_nagi_coordinator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def completed_row(request_id: str, result_state: str = "succeeded") -> dict[str, object]:
    return {
        "name": "l3ds:decision",
        "started": "00:00:00",
        "elapsed": "1s",
        "exit": "0" if result_state == "succeeded" else "…",
        "mode": "cross-toolkit",
        "script_path": "",
        "toolkit": "l3ds",
        "profile": "nagi-smol",
        "task": "decision",
        "request_id": request_id,
        "result_state": result_state,
        "finished_at": "2026-10-07T00:00:00Z",
    }


class ProbeNagiOfflineSafetyTests(unittest.TestCase):
    def test_retired_run_does_not_start_a_daemon_or_prepare_a_fixture(self) -> None:
        with mock.patch.object(sys, "argv", [str(SCRIPT), "--run"]), \
             mock.patch.object(probe, "validate_binary") as validate, \
             mock.patch.object(probe, "run_acceptance") as run, \
             mock.patch("builtins.print"):
            self.assertEqual(probe.main(), 1)
        validate.assert_not_called()
        run.assert_not_called()

    def test_native_metadata_requires_exact_positive_integer_nanoseconds(self) -> None:
        response = {
            "request_id": "fixture", "complete": True,
            "terminal": {"status": "succeeded"},
            "output": {"lane": "decisions", "answers": [{"route": {"choice": "accept"}}]},
            "native_decision_metadata": {"native_latency_ns": 12_500_001},
        }
        def fetch():
            return probe.fetch_and_validate_output(
                Path("/binary"), self.root, self.root, {}, "fixture", time.monotonic() + 5
            )
        with mock.patch.object(probe, "run_cli", return_value=response):
            self.assertEqual(fetch(), (1, 12_500_001))
            for latency in [True, 0, -1, 1.5, "17", 2**64, None]:
                response["native_decision_metadata"] = {"native_latency_ns": latency}
                with self.assertRaises(probe.ProbeFailure):
                    fetch()
            for metadata in [None, {}, {"native_latency_ns": 1, "native_usage": []},
                             {"native_latency_ns": 1, "coordinator_ms": 1}]:
                response["native_decision_metadata"] = metadata
                with self.assertRaises(probe.ProbeFailure):
                    fetch()

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="probe-nagi-offline-")
        self.root = Path(self.temp.name)
        self.history = self.root / "history"
        self.history.mkdir(mode=0o700)
        probe.UNVERIFIED_CLIENT_CLEANUP = False

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_history(self, text: str) -> None:
        path = self.history / "jobs.json"
        path.write_text(text, encoding="utf-8")
        os.chmod(path, 0o600)

    def test_durable_terminal_history_requires_exact_caller_profile_and_state(self) -> None:
        request_id = "nagi-smol-accept-fixture"
        self.write_history(json.dumps([completed_row(request_id)]))
        self.assertEqual(
            probe.validate_durable_history(self.history, request_id), (1, 1)
        )

        self.write_history(json.dumps([completed_row(request_id, "running")]))
        with self.assertRaises(probe.ProbeFailure):
            probe.validate_durable_history(self.history, request_id)

    def test_durable_history_rejects_duplicate_caller_ids_and_duplicate_json_keys(self) -> None:
        request_id = "nagi-smol-accept-fixture"
        self.write_history(
            json.dumps([completed_row(request_id), completed_row(request_id)])
        )
        with self.assertRaises(probe.ProbeFailure):
            probe.validate_durable_history(self.history, request_id)

        self.write_history(
            '[{"request_id":"nagi-smol-accept-fixture",'
            '"request_id":"nagi-smol-accept-fixture"}]'
        )
        with self.assertRaises(probe.ProbeFailure):
            probe.validate_durable_history(self.history, request_id)

    def test_durable_history_rejects_symlink_and_oversized_file(self) -> None:
        target = self.root / "outside.json"
        target.write_text("[]", encoding="utf-8")
        os.symlink(target, self.history / "jobs.json")
        with self.assertRaises(probe.ProbeFailure):
            probe.validate_durable_history(self.history, "caller")
        (self.history / "jobs.json").unlink()

        path = self.history / "jobs.json"
        with path.open("wb") as output:
            output.truncate(probe.MAX_HISTORY_FILE_BYTES + 1)
        with self.assertRaises(probe.ProbeFailure):
            probe.validate_durable_history(self.history, "caller")

    def test_proc_stat_parser_uses_final_parenthesis_and_exact_fields(self) -> None:
        fields = ["S", "12", "34", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "98765"]
        record = probe.parse_proc_stat(11, "11 (python (worker)) " + " ".join(fields))
        self.assertEqual(record, probe.ProcRecord(11, 12, 34, 98765))
        self.assertIsNone(probe.parse_proc_stat(11, "malformed"))

    def test_unknown_proc_access_is_not_treated_as_a_gone_identity(self) -> None:
        with mock.patch.object(Path, "stat", side_effect=PermissionError):
            with self.assertRaises(probe.ProbeFailure):
                probe.current_record(os.getpid())
        with mock.patch.object(Path, "stat", side_effect=FileNotFoundError):
            self.assertIsNone(probe.current_record(os.getpid()))

    def test_owned_daemon_stop_refuses_without_a_captured_identity(self) -> None:
        fake = mock.Mock()
        fake.poll.return_value = None
        self.assertFalse(probe.stop_owned_daemon(fake, None, set()))
        fake.send_signal.assert_not_called()

    def test_cli_stdout_is_drained_with_a_fixed_memory_cap(self) -> None:
        result = probe.run_cli(
            Path(sys.executable),
            ["-c", "print('{\"ok\":true}')"],
            self.root,
            {"PATH": os.environ.get("PATH", "")},
            5,
        )
        self.assertEqual(result, {"ok": True})

        oversized = probe.run_cli(
            Path(sys.executable),
            ["-c", f"import sys; sys.stdout.write('x' * {probe.MAX_CLI_OUTPUT_BYTES + 1})"],
            self.root,
            {"PATH": os.environ.get("PATH", "")},
            5,
        )
        self.assertIsNone(oversized)
        self.assertFalse(probe.UNVERIFIED_CLIENT_CLEANUP)

    def test_fixture_removal_only_accepts_this_owned_prefix_and_mode(self) -> None:
        exact = Path(tempfile.mkdtemp(prefix=probe.TEMP_PREFIX, dir="/tmp"))
        os.chmod(exact, 0o700)
        (exact / "private").write_text("fixture", encoding="utf-8")
        self.assertTrue(probe.safe_remove_fixture(exact))
        self.assertFalse(exact.exists())

        other = self.root / "not-owned"
        other.mkdir()
        self.assertFalse(probe.safe_remove_fixture(other))
        self.assertTrue(other.is_dir())


if __name__ == "__main__":
    unittest.main()
