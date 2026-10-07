"""Offline safety tests for the explicitly pinned Nagi-Smol fetcher.

All Hub calls are replaced with local fixtures. These tests never contact the
network, load model code, or acquire checkpoints.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).with_name("fetch-nagi-smol.py")
SPEC = importlib.util.spec_from_file_location("fetch_nagi_smol", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
FETCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCHER)


class FakeDigest:
    def __init__(self, value: str):
        self.value = value

    def hexdigest(self) -> str:
        return self.value


class FetcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "hub-cache"
        self.cache.mkdir()

    def install_hub_fakes(
        self,
        *,
        current_revisions: dict[str, str] | None = None,
        download_paths: dict[tuple[str, str, str], Path] | None = None,
        downloads: list[tuple[str, str, str]] | None = None,
        metadata_queries: list[str] | None = None,
    ) -> None:
        current_revisions = current_revisions or {
            repository: revision for repository, revision, _ in FETCHER.FILES
        }
        download_paths = download_paths or {}
        downloads = downloads if downloads is not None else []
        metadata_queries = metadata_queries if metadata_queries is not None else []

        class FakeApi:
            def model_info(self, repository: str) -> types.SimpleNamespace:
                metadata_queries.append(repository)
                return types.SimpleNamespace(sha=current_revisions[repository])

        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = FakeApi

        def fake_download(repository: str, filename: str, revision: str) -> str:
            downloads.append((repository, filename, revision))
            return str(download_paths[(repository, filename, revision)])

        hub.hf_hub_download = fake_download
        constants = types.ModuleType("huggingface_hub.constants")
        constants.HF_HUB_CACHE = str(self.cache)
        patcher = mock.patch.dict(sys.modules, {
            "huggingface_hub": hub,
            "huggingface_hub.constants": constants,
        })
        patcher.start()
        self.addCleanup(patcher.stop)

    def fixture_downloads(
        self, *, reference_revision: dict[str, str] | None = None
    ) -> dict[tuple[str, str, str], Path]:
        reference_revision = reference_revision or {}
        paths: dict[tuple[str, str, str], Path] = {}
        for repository, revision, files in FETCHER.FILES:
            repo_dir = self.cache / repository.replace("/", "--") / "snapshots"
            for filename, _ in files:
                path = repo_dir / revision / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"fixture:{repository}:{filename}".encode())
                paths[(repository, filename, revision)] = path
            config_name = files[0][0]
            ref = reference_revision.get(repository, revision)
            ref_path = repo_dir / ref / config_name
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            ref_path.write_bytes(f"reference:{repository}".encode())
            paths[(repository, config_name, "main")] = ref_path
        return paths

    def run_main(self, argv: list[str]) -> str:
        stdout = io.StringIO()
        with mock.patch.object(sys, "argv", [str(SCRIPT_PATH), *argv]):
            with contextlib.redirect_stdout(stdout):
                FETCHER.main()
        return stdout.getvalue()

    def allow_space_check(self) -> mock._patch:
        return mock.patch.object(
            FETCHER.shutil,
            "disk_usage",
            return_value=types.SimpleNamespace(free=10 * 1024**3),
        )

    def test_default_is_plan_only_and_does_not_import_hub_client(self) -> None:
        with mock.patch.dict(sys.modules, {"huggingface_hub": None}):
            output = self.run_main([])

        plan = json.loads(output)
        self.assertFalse(plan["download"])
        self.assertEqual(
            [(entry[0], entry[1]) for entry in plan["repositories"]],
            [(repository, revision) for repository, revision, _ in FETCHER.FILES],
        )

    def test_download_uses_only_pinned_revisions_and_emits_non_inference_evidence(self) -> None:
        paths = self.fixture_downloads()
        calls: list[tuple[str, str, str]] = []
        self.install_hub_fakes(download_paths=paths, downloads=calls)
        expected_hashes = {
            filename: digest
            for _, _, files in FETCHER.FILES
            for filename, digest in files
            if digest is not None
        }

        def fake_file_digest(handle, algorithm: str) -> FakeDigest:
            self.assertEqual(algorithm, "sha256")
            filename = Path(handle.name).name
            return FakeDigest(expected_hashes.get(
                filename, hashlib.sha256(handle.read()).hexdigest()
            ))

        with self.allow_space_check(), mock.patch.object(
            FETCHER.hashlib, "file_digest", side_effect=fake_file_digest
        ):
            output = self.run_main(["--download"])

        evidence = json.loads(output)
        self.assertFalse(evidence["inference_verified"])
        self.assertEqual(
            len(evidence["acquired"]),
            sum(len(files) for _, _, files in FETCHER.FILES),
        )
        expected_calls = [
            (repository, filename, revision)
            for repository, revision, files in FETCHER.FILES
            for filename, _ in files
        ] + [
            (repository, files[0][0], "main")
            for repository, _, files in FETCHER.FILES
        ]
        self.assertCountEqual(calls, expected_calls)
        for repository, revision, files in FETCHER.FILES:
            self.assertIn((repository, files[0][0], "main"), calls)
        self.assertTrue(
            all(row["bytes"] > 0 and len(row["sha256"]) == 64 for row in evidence["acquired"])
        )

    def test_upstream_revision_change_refuses_before_any_download(self) -> None:
        calls: list[tuple[str, str, str]] = []
        revisions = {repository: revision for repository, revision, _ in FETCHER.FILES}
        revisions[FETCHER.FILES[0][0]] = "different-upstream-commit"
        self.install_hub_fakes(current_revisions=revisions, downloads=calls)

        with self.allow_space_check(), self.assertRaises(SystemExit) as raised:
            self.run_main(["--download"])

        self.assertEqual(str(raised.exception), "Acquisition refused: upstream main changed; review the pinned revision")
        self.assertEqual(calls, [])

    def test_low_disk_space_refuses_before_hub_metadata_or_download(self) -> None:
        calls: list[tuple[str, str, str]] = []
        metadata: list[str] = []
        self.install_hub_fakes(download_paths={}, downloads=calls, metadata_queries=metadata)
        with mock.patch.object(
            FETCHER.shutil,
            "disk_usage",
            return_value=types.SimpleNamespace(free=1024**3),
        ), self.assertRaises(SystemExit) as raised:
            self.run_main(["--download"])

        self.assertEqual(
            str(raised.exception),
            "Acquisition refused: require 8 GiB free on the Hub cache filesystem",
        )
        self.assertEqual(calls, [])
        self.assertEqual(metadata, [])

    def test_weight_checksum_failure_preserves_cached_file(self) -> None:
        paths = self.fixture_downloads()
        calls: list[tuple[str, str, str]] = []
        self.install_hub_fakes(download_paths=paths, downloads=calls)
        first_repo, first_revision, first_files = FETCHER.FILES[0]
        weight_path = paths[(first_repo, first_files[1][0], first_revision)]
        original = weight_path.read_bytes()

        with self.allow_space_check(), mock.patch.object(
            FETCHER.hashlib,
            "file_digest",
            return_value=FakeDigest("0" * 64),
        ), self.assertRaises(SystemExit) as raised:
            self.run_main(["--download"])

        self.assertEqual(
            str(raised.exception),
            "Acquisition failed: weight checksum mismatch; cache preserved",
        )
        self.assertEqual(weight_path.read_bytes(), original)
        self.assertNotIn((FETCHER.FILES[1][0], FETCHER.FILES[1][2][0][0], FETCHER.FILES[1][1]), calls)

    def test_main_reference_race_refuses_and_keeps_cache(self) -> None:
        paths = self.fixture_downloads(
            reference_revision={FETCHER.FILES[0][0]: "newer-main-commit"}
        )
        calls: list[tuple[str, str, str]] = []
        self.install_hub_fakes(download_paths=paths, downloads=calls)
        before = {path: path.read_bytes() for path in paths.values()}

        with self.allow_space_check(), mock.patch.object(
            FETCHER.hashlib,
            "file_digest",
            side_effect=lambda handle, algorithm: FakeDigest(
                next(
                    (
                        digest
                        for _, _, files in FETCHER.FILES
                        for filename, digest in files
                        if filename == Path(handle.name).name and digest is not None
                    ),
                    hashlib.sha256(handle.read()).hexdigest(),
                )
            ),
        ), self.assertRaises(SystemExit) as raised:
            self.run_main(["--download"])

        self.assertEqual(
            str(raised.exception),
            "Acquisition refused: upstream main raced acquisition; cache preserved",
        )
        self.assertTrue(any(revision == "main" for _, _, revision in calls))
        self.assertEqual({path: path.read_bytes() for path in before}, before)


if __name__ == "__main__":
    unittest.main()
