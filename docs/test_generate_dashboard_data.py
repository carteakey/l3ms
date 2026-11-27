import importlib.util
import json
import re
import tempfile
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
    def test_theme_text_and_accents_meet_contrast_on_all_panel_backgrounds(self):
        html = (ROOT / "docs" / "index.html").read_text()
        colors = dict(re.findall(r"--([a-z-]+):\s*(#[0-9a-fA-F]{3,6});", html))

        def luminance(color):
            hex_color = color.lstrip("#")
            if len(hex_color) == 3:
                hex_color = "".join(character * 2 for character in hex_color)
            channels = [int(hex_color[index:index + 2], 16) / 255 for index in (0, 2, 4)]
            linear = [channel / 12.92 if channel <= 0.04045
                      else ((channel + 0.055) / 1.055) ** 2.4 for channel in channels]
            return sum(weight * channel for weight, channel in zip((0.2126, 0.7152, 0.0722), linear))

        for foreground in ("text", "muted", "green", "blue", "yellow", "pink", "focus"):
            for background in ("bg", "panel", "panel-light"):
                with self.subTest(foreground=foreground, background=background):
                    levels = sorted((luminance(colors[foreground]), luminance(colors[background])))
                    self.assertGreaterEqual((levels[1] + 0.05) / (levels[0] + 0.05), 4.5)

    def test_served_generation_profiles_match_dashboard_metadata(self):
        parsed = dashboard.parse_models((ROOT / "llama-swap.yaml").read_text())
        decisions = json.loads((ROOT / "decision-runtimes.json").read_text())
        media = json.loads((ROOT / "media-runtimes.json").read_text())
        native_ids = {p["model"] for p in decisions["profiles"] if p["engine"] == "llamacpp"}
        native_media_ids = {p["id"] for p in media["profiles"]}
        self.assertTrue(native_ids <= set(parsed))
        active = {
            key for key, model in parsed.items()
            if model.get("cmd") and not model.get("unlisted", False)
            and "--embedding" not in model["cmd"]
            and key not in native_ids
            and key not in native_media_ids
            and not key.endswith("_auto")
        }
        metadata = json.loads((ROOT / "docs" / "dashboard-meta.json").read_text())
        self.assertEqual(active, set(metadata["models"]))

    def test_parse_models_handles_native_aliases_blocks_and_inline_lists(self):
        sample = """
macros:
  "cpu_range": "0-11"

models:
  "qwen38-flash-next-plat":
    name: "Qwen Plat"
    aliases:
      - "strata"
      - "plat"
    cmd: |
      taskset -c ${cpu_range} ${strata_python} --engine strata --config ${strata_ud_iq2_config}
  "inkling-2-mini":
    name: "Inkling 2 Mini"
    aliases: ["inkling-2-mini-128k", "inkling"]
    cmd: >-
      taskset -c ${cpu_range} ${strata_python} --engine strata --config ${polystrata_inkling2_config}
"""
        parsed = dashboard.parse_models(sample)
        self.assertEqual(parsed["qwen38-flash-next-plat"]["aliases"], ["strata", "plat"])
        self.assertEqual(parsed["inkling-2-mini"]["aliases"], ["inkling-2-mini-128k", "inkling"])
        self.assertIn("--engine strata", parsed["inkling-2-mini"]["cmd"])

    def test_unmeasured_profiles_are_listed_but_never_ranked(self):
        meta = json.loads((ROOT / "docs" / "dashboard-meta.json").read_text())
        unmeasured_ids = {k for k, v in meta["models"].items() if v.get("tps") is None}
        self.assertGreaterEqual(len(unmeasured_ids), 3)
        text = (ROOT / "docs" / "generated-models.js").read_text()
        payload = json.loads(text.split("window.L3MS_DASHBOARD = ", 1)[1].rsplit(";", 1)[0])
        ranked = {m["id"] for m in payload["models"]}
        listed = {m["id"] for m in payload["unmeasured"]}
        self.assertEqual(listed, unmeasured_ids)
        self.assertFalse(ranked & listed)
        self.assertTrue(all(isinstance(m["tps"], (int, float)) for m in payload["models"]))
        for item in payload["unmeasured"]:
            self.assertNotIn("tps", item)
            self.assertNotIn("/home/", item["command"])
            self.assertNotIn("/mnt/", item["command"])

    def test_every_local_profile_has_complete_evidence_shape(self):
        meta = json.loads((ROOT / "docs" / "dashboard-meta.json").read_text())
        for profile_id, display in meta["models"].items():
            if display.get("tps") is None:
                continue  # declared but unmeasured profiles carry no evidence
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

    def test_engine_aware_portable_recipes_for_strata_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cfg_path = tmp / "poly.json"
            cfg_path.write_text(json.dumps({
                "exe": "/home/user/vendor/polystrata/build/strata-poly",
                "args": ["--model", "/mnt/dev/models/inkling.gguf", "--kv", "fp8", "--vram-pool"],
                "env": {"STRATA_PREFILL_NVFP4": "w4a8"},
                "allowed_hosts": ["yeti-cachy.llama-alnair.ts.net", "100.110.126.24"],
                "log": "/home/user/polystrata.log",
            }))
            model = {
                "cmd": "taskset -c ${cpu_range} ${strata_python} ${polystrata_server} --engine strata --config ${poly_cfg} --port ${inference_port}",
                "env": [],
            }
            kind, label, snippet, excerpt = dashboard.portable_recipe(
                model,
                macros={"poly_cfg": str(cfg_path)},
                root=tmp,
            )
            self.assertEqual(kind, "polystrata")
            self.assertEqual(label, "Portable Strata / PolyStrata Recipe")
            self.assertIsNotNone(excerpt)
            self.assertIn('export STRATA_PREFILL_NVFP4="w4a8"', snippet)
            for forbidden in ("/home/", "/mnt/", ".ts.net", "100.110."):
                self.assertNotIn(forbidden, snippet)

            # Fallback when JSON config does not exist locally (e.g. CI without vendor/)
            kind2, label2, snippet2, excerpt2 = dashboard.portable_recipe(
                model,
                macros={"poly_cfg": str(tmp / "missing.json")},
                root=tmp,
            )
            self.assertEqual(kind2, "polystrata")
            self.assertEqual(label2, "Portable Strata / PolyStrata Recipe")
            self.assertIsNone(excerpt2)
            self.assertIn('"$POLY_CFG"', snippet2)

    def test_published_commands_do_not_include_host_paths_or_router_macros(self):
        config_text = (ROOT / "llama-swap.yaml").read_text()
        parsed = dashboard.parse_models(config_text)
        macros = dashboard.parse_macros(config_text)
        meta = json.loads((ROOT / "docs" / "dashboard-meta.json").read_text())
        for profile_id in meta["models"]:
            with self.subTest(profile=profile_id):
                kind, label, snippet, _ = dashboard.portable_recipe(parsed[profile_id], macros=macros, root=ROOT)
                for forbidden in ("/home/", "/mnt/", ".ts.net", "${env.", "${qwen", "${strata", "${polystrata"):
                    self.assertNotIn(forbidden, snippet)
                if kind in {"strata", "polystrata"}:
                    self.assertEqual(label, "Portable Strata / PolyStrata Recipe")
                else:
                    self.assertEqual(label, "Portable llama.cpp command")

    def test_confirmed_stack_facts_and_zero_legacy_naming(self):
        html = (ROOT / "docs" / "index.html").read_text()
        js = (ROOT / "docs" / "generated-models.js").read_text()
        for label, text in (("index.html", html), ("generated-models.js", js)):
            with self.subTest(file=label):
                self.assertIsNone(re.search(r"l2m2|l3ds", text, re.IGNORECASE))
                self.assertNotIn("/home/", text)
                self.assertNotIn(".ts.net", text)

        payload = json.loads(js.split("window.L3MS_DASHBOARD = ", 1)[1].rsplit(";", 1)[0])
        by_id = {m["id"]: m for m in payload["models"]}
        self.assertEqual(by_id["qwen38-flash-next-plat"]["tps"], 73.0)
        self.assertTrue(by_id["qwen38-flash-next-plat"]["preload"])
        self.assertEqual(by_id["inkling-2-mini"]["tps"], 77.5)
        self.assertEqual(by_id["inkling-2-mini"]["editTps"], 99.7)
        self.assertEqual(by_id["inkling-2-mini-llama"]["tps"], 68.7)
        self.assertEqual(by_id["qwen3-8-27b-iq3-xxs"]["ramGb"], 0.0)
        self.assertNotIn("Preload", by_id["gemma-4-26b-qat-mtp"]["tags"])
        self.assertNotIn("startup preload", by_id["gemma-4-26b-qat-mtp"]["description"].lower())
        self.assertEqual(len(payload["engines"]["engines"]), 10)
        self.assertGreaterEqual(len(payload["inklingProgression"]), 8)
        self.assertEqual(payload["roofline"]["hardwareBaselines"][0]["measured"], 455.8)

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
            json.loads((ROOT / "media-runtimes.json").read_text()),
            metadata["media"],
        )
        decisions = dashboard.decision_public_profiles(
            json.loads((ROOT / "decision-runtimes.json").read_text()),
            metadata["decisions"],
        )
        self.assertEqual(len(media), 14)
        self.assertEqual(len(decisions), 4)
        for profile_id in ("liquid-d1-2b", "vllm-d3-27b", "vllm-d3-flash", "cloudflare-clef-flash"):
            native = next(profile for profile in decisions if profile["id"] == profile_id)
            self.assertEqual(native["engine"], "llamacpp")
            self.assertIsNone(native["experimentUrl"])
        breeze = next(profile for profile in media if profile["id"] == "breeze-tts-2")
        self.assertEqual(
            breeze["modes"],
            ["Standard TTS", "Voice design", "Voice cloning", "Streaming"],
        )
        cpu_breeze = next(profile for profile in media if profile["id"] == "breeze-tts-2-cpu")
        self.assertEqual(cpu_breeze["modes"], ["Standard TTS (CPU)", "Offline voice design (CPU)",
                                             "Voice cloning (CPU)", "TTS chunks after completion (CPU)"])
        self.assertEqual(cpu_breeze["status"], "experimental-local")
        cpu_d1 = next(profile for profile in decisions if profile["id"] == "liquid-d1-2b")
        self.assertEqual(cpu_d1["kind"], "local")
        self.assertEqual(cpu_d1["device"], "cpu")
        self.assertEqual(cpu_d1["tier"], "co-resident")
        self.assertEqual(cpu_d1["declaredLatencyMs"], 1800)
        h3 = next(profile for profile in media if profile["id"] == "minimax-h3")
        self.assertEqual(h3["status"], "experimental-local")
        for profile in media + decisions:
            self.assertIn("#L", profile["sourceUrl"])
            self.assertNotIn("tps", profile)

    def test_stack_metadata_must_match_native_manifests_and_safe_paths(self):
        manifest = json.loads((ROOT / "media-runtimes.json").read_text())
        metadata = json.loads((ROOT / "docs" / "stack-meta.json").read_text())["media"]
        missing = dict(metadata)
        missing.pop("breeze-tts-2")
        with self.assertRaises(SystemExit):
            dashboard.media_public_profiles(manifest, missing)
        unsafe = {key: dict(value) for key, value in metadata.items()}
        unsafe["breeze-tts-2"]["guide"] = "../private-note.md"
        with self.assertRaises(SystemExit):
            dashboard.media_public_profiles(manifest, unsafe)

    def test_cpu_evidence_summary_is_portable_and_not_a_quality_claim(self):
        text = (ROOT / "docs" / "cpu-native-workflows.md").read_text()
        for private_marker in ("/home/", "/tmp/", ".ts.net"):
            self.assertNotIn(private_marker, text)
        self.assertNotRegex(text, r"\bsk-[A-Za-z0-9]{20,}")
        self.assertIn("not an accuracy, deadline", text)
        self.assertIn("execution and artifact delivery only", text)
        self.assertIn("voice-quality evaluation", text)
        self.assertIn("standard text-only speech", text)

    def test_site_has_five_crisp_tabs_and_explicit_evidence_labels(self):
        html = (ROOT / "docs" / "index.html").read_text()
        for tab in ("stack", "engines", "roofline", "hardware", "archive"):
            with self.subTest(tab=tab):
                self.assertIn(f'data-tab="{tab}"', html)
                self.assertIn(f'id="{tab}-panel"', html)
                self.assertIn(f'id="{tab}-tab" role="tab" aria-controls="{tab}-panel"', html)
                self.assertIn(f'id="{tab}-panel" role="tabpanel" aria-labelledby="{tab}-tab"', html)
        self.assertIn('id="media-list"', html)
        self.assertIn('id="decision-list"', html)
        self.assertIn('id="engines-grid"', html)
        self.assertIn('id="evolution-svg"', html)
        self.assertIn('id="roofline-bars-svg"', html)
        self.assertIn('id="roofline-ctx-svg"', html)
        self.assertIn('role="tablist"', html)
        self.assertIn('event.key === "ArrowRight"', html)
        self.assertIn("not a claim that its model is installed", html)
        self.assertIn("No measured decision benchmark published.", html)
        self.assertIn("not measured here", html)


if __name__ == "__main__":
    unittest.main()
