"""Unit tests for L3MS serving config and model inventory guardrails."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit_models = _load_module(
    "audit_models",
    ROOT / ".agents/skills/model-inventory-manager/scripts/audit_models.py",
)
check_serving = _load_module(
    "check_serving_configs",
    ROOT / "maintenance/check-serving-configs.py",
)


class AuditModelsShardProtectionTests(unittest.TestCase):
    def test_multi_shard_gguf_siblings_are_marked_served(self) -> None:
        served = {"/mnt/dev/models/unsloth/UD-IQ2_XS/Qwen3.8-UD-IQ2_XS-00001-of-00003.gguf"}
        keys = {audit_models.shard_group_key(p) for p in served}
        keys.discard(None)

        shard2 = "/mnt/dev/models/unsloth/UD-IQ2_XS/Qwen3.8-UD-IQ2_XS-00002-of-00003.gguf"
        shard3 = "/mnt/dev/models/unsloth/UD-IQ2_XS/Qwen3.8-UD-IQ2_XS-00003-of-00003.gguf"
        unrelated = "/mnt/dev/models/unsloth/Other-00002-of-00003.gguf"

        self.assertTrue(audit_models.is_path_served(shard2, shard2, served, keys))
        self.assertTrue(audit_models.is_path_served(shard3, shard3, served, keys))
        self.assertFalse(audit_models.is_path_served(unrelated, unrelated, served, keys))

    def test_scan_roots_include_mnt_media_not_stale_run_media(self) -> None:
        roots = [str(p) for p in audit_models.SCAN_ROOTS]
        self.assertIn("/mnt/media/models", roots)
        self.assertIn("/mnt/dev/models", roots)
        self.assertNotIn("/run/media/kchauhan/Games/models", roots)


class ServingConfigValidatorTests(unittest.TestCase):
    def test_repo_llama_swap_passes_invariants(self) -> None:
        errors = check_serving.validate_configs(root=ROOT, local_paths=False)
        self.assertEqual(errors, [])

    def test_missing_host_and_allowed_hosts_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            troot = Path(tmp)
            cfg = troot / "strata.json"
            cfg.write_text(json.dumps({"args": []}), encoding="utf-8")
            (troot / "llama-swap.yaml").write_text(
                'macros:\n'
                '  "inference_port": "8089"\n'
                'models:\n'
                '  "broken-tier":\n'
                '    proxy: http://127.0.0.1:${inference_port}\n'
                '    cmd: |\n'
                f'      python server.py --engine strata --config {cfg}\n'
                '      --port ${inference_port}\n',
                encoding="utf-8",
            )
            errors = check_serving.validate_configs(root=troot, local_paths=False)
            joined = "\n".join(errors)
            self.assertIn("--host 0.0.0.0", joined)
            self.assertIn("allowed_hosts", joined)


if __name__ == "__main__":
    unittest.main()
