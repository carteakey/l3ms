import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docs" / "generate_dashboard_data.py"
SPEC = importlib.util.spec_from_file_location("generate_dashboard_data", MODULE_PATH)
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


def valid_community_run():
    return {
        "id": "test-24gb-run",
        "submittedAt": "2026-07-17",
        "sourceUrl": "https://example.com/raw",
        "hardware": {"gpu": "GPU", "vramGb": 24, "cpu": "CPU", "ramGb": 64},
        "software": {
            "os": "Linux",
            "backend": "CUDA",
            "llamaCppCommit": "0123456789abcdef",
        },
        "model": {"name": "org/model", "quant": "Q4_K_M"},
        "benchmark": {
            "command": "llama-bench -m model.gguf -p 512 -n 128 -r 5",
            "testedContext": 65536,
            "cacheState": "cold",
            "promptTokens": 512,
            "generatedTokens": 128,
            "repetitions": 5,
        },
        "metrics": {"pp": 1000.0, "tg": 50.0, "draftAcceptance": None},
    }


class DashboardDataTest(unittest.TestCase):
    def test_served_generation_profiles_match_dashboard_metadata(self):
        parsed = dashboard.parse_models((ROOT / "llama-swap.yaml").read_text())
        active = {
            key for key, model in parsed.items()
            if model.get("cmd") and not model.get("unlisted", False)
            and "--embedding" not in model["cmd"]
        }
        metadata = json.loads((ROOT / "docs" / "dashboard-meta.json").read_text())
        self.assertEqual(active, set(metadata["models"]))

    def test_every_local_profile_has_complete_evidence_shape(self):
        meta = json.loads((ROOT / "docs" / "dashboard-meta.json").read_text())
        for profile_id, display in meta["models"].items():
            with self.subTest(profile=profile_id):
                evidence = dashboard.profile_evidence(display)
                self.assertEqual(set(dashboard.EVIDENCE_FIELDS), set(evidence) - {"scope"})
                self.assertEqual("local", evidence["scope"])

    def test_new_engine_macros_require_explicit_portable_paths(self):
        snippet = dashboard.portable_command({"cmd":
            "${strata_python} ${strata_server} --config ${strata_iq3_config} "
            "--port ${PORT} --model ${env.L3MS_ROOT}/weights/model.gguf"})
        for variable in ("STRATA_PYTHON", "STRATA_SERVER", "STRATA_IQ3_CONFIG"):
            self.assertIn(f'{variable}="/path/to/{variable.lower()}"', snippet)
            self.assertIn(f'"${variable}"', snippet)
        self.assertNotIn("${strata_", snippet)
        self.assertNotIn("${env.", snippet)
        self.assertIn('"$MODEL_PATH"', snippet)

    def test_published_commands_do_not_include_host_paths_or_router_macros(self):
        parsed = dashboard.parse_models((ROOT / "llama-swap.yaml").read_text())
        meta = json.loads((ROOT / "docs" / "dashboard-meta.json").read_text())
        for profile_id in meta["models"]:
            with self.subTest(profile=profile_id):
                snippet = dashboard.portable_command(parsed[profile_id])
                self.assertNotIn("/home/", snippet)
                self.assertNotIn("${env.", snippet)
                self.assertNotIn("${qwen", snippet)
                self.assertNotIn("${strata", snippet)

    def test_community_run_requires_reproducibility_fields(self):
        run = valid_community_run()
        self.assertEqual([run], dashboard.validate_community_runs({"runs": [run]}))

        run["software"]["llamaCppCommit"] = None
        with self.assertRaises(SystemExit):
            dashboard.validate_community_runs({"runs": [run]})

    def test_community_ids_do_not_enter_local_profile_source(self):
        parsed = dashboard.parse_models((ROOT / "llama-swap.yaml").read_text())
        community = valid_community_run()
        self.assertNotIn(community["id"], parsed)

    def test_media_and_decision_views_keep_separate_evidence(self):
        metadata = json.loads((ROOT / "docs" / "stack-meta.json").read_text())
        media = dashboard.media_public_profiles(
            json.loads((ROOT / "toolkits" / "l2m2" / "media-runtimes.json").read_text()),
            metadata["media"],
        )
        decisions = dashboard.decision_public_profiles(
            json.loads((ROOT / "toolkits" / "l3ds" / "decision-runtimes.json").read_text()),
            metadata["decisions"],
        )
        self.assertEqual(len(media), 7)
        self.assertEqual(len(decisions), 6)
        breeze = next(profile for profile in media if profile["id"] == "breeze-tts-2")
        self.assertEqual(
            breeze["modes"],
            ["Standard TTS", "Voice design", "Voice cloning", "Streaming"],
        )
        h3 = next(profile for profile in media if profile["id"] == "minimax-h3")
        self.assertEqual(h3["status"], "experimental-local")
        hosted = next(profile for profile in decisions if profile["id"] == "systemone")
        self.assertEqual(hosted["kind"], "hosted")
        self.assertIsNone(hosted["experimentUrl"])
        for profile in media + decisions:
            self.assertIn("#L", profile["sourceUrl"])
            self.assertNotIn("tps", profile)

    def test_stack_metadata_must_match_native_manifests_and_safe_paths(self):
        manifest = json.loads(
            (ROOT / "toolkits" / "l2m2" / "media-runtimes.json").read_text()
        )
        metadata = json.loads((ROOT / "docs" / "stack-meta.json").read_text())["media"]
        missing = dict(metadata)
        missing.pop("breeze-tts-2")
        with self.assertRaises(SystemExit):
            dashboard.media_public_profiles(manifest, missing)
        unsafe = {key: dict(value) for key, value in metadata.items()}
        unsafe["breeze-tts-2"]["guide"] = "../private-note.md"
        with self.assertRaises(SystemExit):
            dashboard.media_public_profiles(manifest, unsafe)

    def test_site_has_separate_lane_panels_and_explicit_evidence_labels(self):
        html = (ROOT / "docs" / "index.html").read_text()
        for lane in ("models", "media", "decisions"):
            with self.subTest(lane=lane):
                self.assertIn(f'data-tab="{lane}"', html)
                self.assertIn(f'id="{lane}-panel"', html)
                self.assertIn(f'id="{lane}-tab" role="tab" aria-controls="{lane}-panel"', html)
                self.assertIn(f'id="{lane}-panel" role="tabpanel" aria-labelledby="{lane}-tab"', html)
        self.assertIn('id="media-list"', html)
        self.assertIn('id="decision-list"', html)
        self.assertIn('role="tablist"', html)
        self.assertIn('event.key === "ArrowRight"', html)
        self.assertIn("not a claim that its model is installed", html)
        self.assertIn("No measured decision benchmark published.", html)
        self.assertIn("not measured here", html)


if __name__ == "__main__":
    unittest.main()
