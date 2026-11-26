"""Comprehensive stdlib unittest suite for the L3MS Local Live Control Dashboard."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

DASHBOARD_DIR = Path(__file__).resolve().parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import router_client  # noqa: E402
import server  # noqa: E402
import telemetry  # noqa: E402


SAMPLE_NVIDIA_CSV = "NVIDIA GeForce RTX 4070, 11562, 12282, 87, 64.50, 152.40, 165.00\n"

SAMPLE_MEMINFO = """\
MemTotal:       65536000 kB
MemFree:        10240000 kB
MemAvailable:   24576000 kB
Buffers:          512000 kB
Cached:         12288000 kB
SwapTotal:      65536000 kB
SwapFree:       58720256 kB
"""

SAMPLE_STAT_1 = """\
cpu  1000 0 500 8500 0 0 0 0 0 0
cpu0 100 0 50 850 0 0 0 0 0 0
cpu1 100 0 50 850 0 0 0 0 0 0
cpu2 100 0 50 850 0 0 0 0 0 0
cpu3 100 0 50 850 0 0 0 0 0 0
cpu4 100 0 50 850 0 0 0 0 0 0
cpu5 100 0 50 850 0 0 0 0 0 0
cpu6 100 0 50 850 0 0 0 0 0 0
cpu7 100 0 50 850 0 0 0 0 0 0
cpu8 100 0 50 850 0 0 0 0 0 0
cpu9 100 0 50 850 0 0 0 0 0 0
cpu10 100 0 50 850 0 0 0 0 0 0
cpu11 100 0 50 850 0 0 0 0 0 0
cpu12 100 0 50 850 0 0 0 0 0 0
cpu13 100 0 50 850 0 0 0 0 0 0
cpu14 100 0 50 850 0 0 0 0 0 0
cpu15 100 0 50 850 0 0 0 0 0 0
"""

# P-cores (0-11): +80 busy, +20 idle => 80.0%
# E-cores (12-15): +20 busy, +80 idle => 20.0%
SAMPLE_STAT_2 = """\
cpu  1900 0 640 9060 0 0 0 0 0 0
cpu0 170 0 60 870 0 0 0 0 0 0
cpu1 170 0 60 870 0 0 0 0 0 0
cpu2 170 0 60 870 0 0 0 0 0 0
cpu3 170 0 60 870 0 0 0 0 0 0
cpu4 170 0 60 870 0 0 0 0 0 0
cpu5 170 0 60 870 0 0 0 0 0 0
cpu6 170 0 60 870 0 0 0 0 0 0
cpu7 170 0 60 870 0 0 0 0 0 0
cpu8 170 0 60 870 0 0 0 0 0 0
cpu9 170 0 60 870 0 0 0 0 0 0
cpu10 170 0 60 870 0 0 0 0 0 0
cpu11 170 0 60 870 0 0 0 0 0 0
cpu12 115 0 55 930 0 0 0 0 0 0
cpu13 115 0 55 930 0 0 0 0 0 0
cpu14 115 0 55 930 0 0 0 0 0 0
cpu15 115 0 55 930 0 0 0 0 0 0
"""

SAMPLE_DF_OUTPUT = """\
Filesystem     Type   Size  Used Avail Use% Mounted on
/dev/nvme0n1p1 btrfs  462G  385G   76G  84% /
/dev/nvme0n1p1 btrfs  462G  385G   76G  84% /home
/dev/nvme2n1p1 ext4   238G  236G  1.4G 100% /mnt/dev
tmpfs          tmpfs   32G   12M   32G   1% /tmp
"""

SAMPLE_SWAP_YAML = """\
globalTTL: 600

macros:
  "inference_port": "8089"
  "strata_python": "${env.L3MS_ROOT}/vendor/strata/.venv/bin/python"
  "strata_server": "${env.L3MS_ROOT}/vendor/strata-lean/serve/server.py"
  "strata_ud_iq2_config": "${env.L3MS_ROOT}/vendor/strata-lean/strata-ud-iq2_xs-ple_q8.json"
  "polystrata_server": "${env.L3MS_ROOT}/vendor/polystrata/terminal/serve/server.py"
  "polystrata_inkling2_config": "${env.L3MS_ROOT}/vendor/polystrata/polystrata-inkling2-256k.json"
  "cpu_range": "0-11"
  "ecores": "12-15"

models:
  # "retired-model":
  #   name: "Retired Model"
  #   cmd: echo ignored

  "qwen38-flash-next-plat":
    name: "Qwen3.8-Flash-Next Plat (Strata)"
    description: "Platinum Strata tier 73.0 t/s"
    ttl: 3600
    aliases:
      - "strata"
      - "plat"
    proxy: http://127.0.0.1:${inference_port}
    cmd: |
      taskset -c ${cpu_range} ${strata_python}
      ${strata_server}
      --engine strata
      --config ${strata_ud_iq2_config}
      --host 0.0.0.0 --port ${inference_port}

  "inkling-2-mini":
    name: "Inkling 2 Mini (PolyStrata 256k)"
    description: "Fast coding + vision MoE"
    ttl: 3600
    aliases: ["inkling-2-mini-128k"]
    proxy: http://127.0.0.1:${inference_port}
    cmd: |
      taskset -c ${cpu_range} ${strata_python}
      ${polystrata_server}
      --engine strata
      --config ${polystrata_inkling2_config}
      --host 0.0.0.0 --port ${inference_port}

  "legacy-unlisted-alias":
    name: "Legacy Unlisted Alias"
    unlisted: true
    cmd: >-
      echo unlisted

  "embeddinggemma-2":
    name: "EmbeddingGemma 2 (CPU)"
    ttl: 300
    metadata:
      task: embeddings
      dimensions: 768
    cmd: |
      taskset -c ${ecores} llama-server -m embed.gguf --embedding --port ${PORT}

groups:
  co-resident:
    swap: false
    exclusive: false
    members:
      - "embeddinggemma-2"

hooks:
  on_startup:
    preload:
      - "qwen38-flash-next-plat"
"""

SAMPLE_STRATA_JSON = {
    "exe": "/tmp/strata",
    "args": ["--kv", "k8v4", "--kv-grow", "--max-context", "131072"],
    "model_name": "qwen3.8-flash-next-ud-iq2_xs-ple_q8",
    "port": 8089,
    "fit_max_tokens": True,
    "reasoning_budget_tokens": 8192,
    "sampling": {"temperature": 1.0, "top_p": 0.95},
    "env": {
        "STRATA_HC_Q8": "1",
        "STRATA_ADAPT_FETCH": "1",
        "STRATA_PREFILL_LEND_PCT": "40",
    },
    "vision": {"exe": "/tmp/strata-vision", "gpu": False, "threads": 6},
}


class TelemetryParserTests(unittest.TestCase):
    def test_parse_nvidia_smi_csv(self) -> None:
        gpu = telemetry.parse_nvidia_smi_csv(SAMPLE_NVIDIA_CSV)
        self.assertTrue(gpu["available"])
        self.assertEqual(gpu["name"], "NVIDIA GeForce RTX 4070")
        self.assertEqual(gpu["vram_used_mib"], 11562.0)
        self.assertEqual(gpu["vram_total_mib"], 12282.0)
        self.assertAlmostEqual(gpu["vram_pct"], 94.1, places=1)
        self.assertEqual(gpu["util_pct"], 87.0)
        self.assertEqual(gpu["temp_c"], 64.5)
        self.assertEqual(gpu["power_w"], 152.4)
        self.assertEqual(gpu["power_limit_w"], 165.0)

        na_gpu = telemetry.parse_nvidia_smi_csv("RTX 4070, 500, 12282, [N/A], 35, [Not Supported], 165\n")
        self.assertTrue(na_gpu["available"])
        self.assertIsNone(na_gpu["util_pct"])
        self.assertIsNone(na_gpu["power_w"])

    def test_parse_meminfo_and_zram(self) -> None:
        mem = telemetry.parse_meminfo(SAMPLE_MEMINFO)
        self.assertEqual(mem["total_mib"], 64000.0)
        self.assertEqual(mem["available_mib"], 24000.0)
        self.assertEqual(mem["used_mib"], 40000.0)
        self.assertEqual(mem["used_pct"], 62.5)
        self.assertEqual(mem["swap_total_mib"], 64000.0)
        self.assertEqual(mem["swap_used_mib"], 6656.0)

        zram = telemetry.parse_zram_mm_stat("1073741824 268435456 300000000 0 0 0 0 0\n")
        self.assertIsNotNone(zram)
        assert zram is not None
        self.assertEqual(zram["orig_mib"], 1024.0)
        self.assertEqual(zram["compr_mib"], 256.0)

    def test_parse_proc_stat_p_and_e_cores(self) -> None:
        s1 = telemetry.parse_proc_stat(SAMPLE_STAT_1)
        s2 = telemetry.parse_proc_stat(SAMPLE_STAT_2)
        usage = telemetry.compute_cpu_usage(s1, s2)
        self.assertEqual(usage["p_core_count"], 12)
        self.assertEqual(usage["e_core_count"], 4)
        self.assertEqual(usage["p_core_pct"], 80.0)
        self.assertEqual(usage["e_core_pct"], 20.0)
        self.assertEqual(usage["per_core"]["cpu0"], 80.0)
        self.assertEqual(usage["per_core"]["cpu12"], 20.0)

    def test_governor_and_thp(self) -> None:
        ok = telemetry.evaluate_governor(["performance", "performance"])
        self.assertTrue(ok["healthy"])
        self.assertIsNone(ok["alert"])

        bad = telemetry.evaluate_governor(["powersave", "powersave"])
        self.assertFalse(bad["healthy"])
        self.assertIn("powersave", bad["alert"])

        self.assertEqual(telemetry.parse_thp_enabled("[always] madvise never"), "always")
        self.assertEqual(telemetry.parse_thp_enabled("always [madvise] never"), "madvise")

    def test_parse_df_output_and_local_flag(self) -> None:
        rows = telemetry.parse_df_output(SAMPLE_DF_OUTPUT)
        self.assertEqual([r["mount"] for r in rows], ["/", "/home", "/mnt/dev"])
        self.assertFalse(rows[0]["warning"])
        self.assertTrue(rows[2]["warning"])
        self.assertTrue(rows[2]["critical"])

        captured_cmds: list[list[str]] = []

        def fake_cmd(cmd: list[str], _timeout: float) -> tuple[int, str, str]:
            captured_cmds.append(cmd)
            if cmd[0] == "df":
                return 0, SAMPLE_DF_OUTPUT, ""
            return 1, "", "unsupported"

        collector = telemetry.TelemetryCollector(run_cmd=fake_cmd, df_cache_ttl_s=0.0)
        disks = collector.collect_disks()
        self.assertEqual(len(disks), 3)
        self.assertIn(["df", "-hlT"], captured_cmds)

    def test_proc_io_spill_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp) / "proc"
            pid_dir = proc_root / "4242"
            pid_dir.mkdir(parents=True)
            (pid_dir / "cmdline").write_bytes(b"/home/kchauhan/repos/l3ms/vendor/strata-lean/build/strata\x00--kv\x00k8v4\x00")
            (pid_dir / "status").write_text("Name:\tstrata\nVmRSS:\t4194304 kB\n", encoding="utf-8")
            (pid_dir / "io").write_text("rchar: 1000\nread_bytes: 0\nwrite_bytes: 0\n", encoding="utf-8")

            now = [100.0]
            collector = telemetry.TelemetryCollector(proc_root=proc_root, clock=lambda: now[0])
            first = collector.collect_inference_io()
            self.assertEqual(first["active_pid"], 4242)
            self.assertEqual(first["active_engine"], "strata")
            self.assertEqual(first["read_mb_s"], 0.0)
            self.assertEqual(first["spill_state"], "ok")

            # Simulate 250 MiB read in 2 seconds -> 125 MB/s -> spill alert!
            now[0] = 102.0
            (pid_dir / "io").write_text(f"rchar: 999999\nread_bytes: {250 * 1024 * 1024}\n", encoding="utf-8")
            second = collector.collect_inference_io()
            self.assertEqual(second["read_mb_s"], 125.0)
            self.assertEqual(second["spill_state"], "spill")
            self.assertTrue(second["spill_alert"])


class RouterParserAndScrubTests(unittest.TestCase):
    def test_parse_llama_swap_yaml_aliases_groups_unlisted(self) -> None:
        parsed = router_client.parse_llama_swap_yaml(SAMPLE_SWAP_YAML, repo_root=Path("/repo"))
        self.assertEqual(parsed["global_ttl"], 600)
        self.assertEqual(parsed["preload"], ["qwen38-flash-next-plat"])
        self.assertNotIn("retired-model", parsed["models_by_id"])

        qwen = parsed["models_by_id"]["qwen38-flash-next-plat"]
        self.assertEqual(qwen["aliases"], ["strata", "plat"])
        self.assertEqual(qwen["engine"], "strata")
        self.assertTrue(qwen["preload"])
        self.assertTrue(qwen["exclusive"])
        self.assertEqual(qwen["ttl"], 3600)
        self.assertEqual(qwen["config_path"], "/repo/vendor/strata-lean/strata-ud-iq2_xs-ple_q8.json")

        ink = parsed["models_by_id"]["inkling-2-mini"]
        self.assertEqual(ink["aliases"], ["inkling-2-mini-128k"])
        self.assertEqual(ink["engine"], "polystrata")

        unlisted = parsed["models_by_id"]["legacy-unlisted-alias"]
        self.assertTrue(unlisted["unlisted"])
        self.assertEqual(unlisted["ttl"], 600)

        emb = parsed["models_by_id"]["embeddinggemma-2"]
        self.assertEqual(emb["group"], "co-resident")
        self.assertFalse(emb["exclusive"])
        self.assertFalse(emb["swap"])
        self.assertEqual(emb["metadata"]["task"], "embeddings")

    def test_parse_bench_jsonl_variants(self) -> None:
        inkling_jsonl = json.dumps(
            {
                "ts": "2026-11-25T16:19:44+00:00",
                "arm": "v5_pb2048",
                "tg_steady": 77.47,
                "edit_tg": 99.7,
                "pp2k": 1833.3,
                "pp8k": 2296.1,
                "pp20k": 2269.5,
                "vram_peak_mib": 11722,
                "load_s": 7.77,
            }
        )
        strata_jsonl = "\n".join(
            [
                "{malformed json line",
                json.dumps(
                    {
                        "label": "hc_q8_adapt",
                        "ts": "2026-11-21 10:58:51",
                        "tg_steady_mean": 73.0,
                        "tg_probes": [{"tg": 73.0, "draft_acc": 0.78}],
                        "pp": {"2048": [{"pp": 1136.0}], "8192": [{"pp": 1888.0}], "32768": [{"pp": 1920.0}]},
                        "vram_mb": 11582,
                        "needle_4k": True,
                    }
                ),
            ]
        )

        ink_rows = router_client.parse_bench_jsonl("inkling-refresh-ab.jsonl", inkling_jsonl)
        self.assertEqual(len(ink_rows), 1)
        self.assertEqual(ink_rows[0]["label"], "v5_pb2048")
        self.assertEqual(ink_rows[0]["tg_steady"], 77.47)
        self.assertEqual(ink_rows[0]["edit_tg"], 99.7)
        self.assertEqual(ink_rows[0]["pp_long_label"], "20k")

        st_rows = router_client.parse_bench_jsonl("strata-tuning-ab.jsonl", strata_jsonl)
        self.assertEqual(len(st_rows), 1)
        self.assertEqual(st_rows[0]["tg_steady"], 73.0)
        self.assertEqual(st_rows[0]["draft_acc_pct"], 78.0)
        self.assertEqual(st_rows[0]["pp2k"], 1136.0)
        self.assertEqual(st_rows[0]["pp8k"], 1888.0)
        self.assertEqual(st_rows[0]["pp_long"], 1920.0)
        self.assertEqual(st_rows[0]["pp_long_label"], "32k")

    def test_secret_loading_and_scrubbing(self) -> None:
        key = router_client.parse_systemd_env_for_key(
            'L3MS_ROOT=/home/kchauhan/repos/l3ms LLAMA_SWAP_API_KEY="l3ms-supersecret999" FOO=bar'
        )
        self.assertEqual(key, "l3ms-supersecret999")

        loaded = router_client.load_api_key(
            env={},
            systemd_reader=lambda: "LLAMA_SWAP_API_KEY=l3ms-fromsystemd123",
        )
        self.assertEqual(loaded, "l3ms-fromsystemd123")

        dirty = (
            "Authorization: Bearer tok_abcdef12345 "
            "key=l3ms-abcdef123456 "
            "LLAMA_SWAP_API_KEY=custom_raw_secret "
            "direct=my_exact_secret_value"
        )
        clean = router_client.scrub_secrets(dirty, extra_secrets=["my_exact_secret_value"])
        self.assertNotIn("tok_abcdef12345", clean)
        self.assertNotIn("l3ms-abcdef123456", clean)
        self.assertNotIn("custom_raw_secret", clean)
        self.assertNotIn("my_exact_secret_value", clean)
        self.assertIn("[REDACTED]", clean)


class LiveHTTPIntegrationTests(unittest.TestCase):
    """End-to-end tests against a local mock llama-swap + ThreadingHTTPServer bound to 127.0.0.1:0."""

    API_KEY = "l3ms-test-secret-do-not-leak"

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self._tmpdir.name)

        # Write mock repo files: llama-swap.yaml, Strata JSON + log, JSONL bench results
        (self.repo_root / "llama-swap.yaml").write_text(SAMPLE_SWAP_YAML, encoding="utf-8")
        strata_dir = self.repo_root / "vendor" / "strata-lean"
        strata_dir.mkdir(parents=True)
        engine_log = strata_dir / "strata-ud-iq2_xs-ple_q8.log"
        engine_log.write_text(
            "INFO strata loaded\nWARN auth header Bearer leaked-engine-token and l3ms-leak99999\n",
            encoding="utf-8",
        )
        cfg = dict(SAMPLE_STRATA_JSON)
        cfg["log"] = str(engine_log)
        (strata_dir / "strata-ud-iq2_xs-ple_q8.json").write_text(json.dumps(cfg), encoding="utf-8")

        results_dir = self.repo_root / "logs" / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "inkling-refresh-ab.jsonl").write_text(
            json.dumps({"ts": "2026-11-26T10:00:00+00:00", "arm": "poly5", "tg_steady": 77.5, "edit_tg": 99.7, "pp2k": 1833, "pp8k": 2296, "pp20k": 2270, "vram_peak_mib": 11722}) + "\n",
            encoding="utf-8",
        )

        # Mock llama-swap state
        mock_state = {
            "running": [{"model": "qwen38-flash-next-plat", "state": "ready"}],
            "auth_headers": [],
            "smoke_payloads": [],
            "unload_calls": 0,
        }
        self.mock_state = mock_state
        expected_auth = f"Bearer {self.API_KEY}"

        class MockSwapHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def _send(self, code: int, body: object) -> None:
                raw = json.dumps(body).encode("utf-8") if not isinstance(body, str) else body.encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _check_auth(self) -> bool:
                auth = self.headers.get("Authorization", "")
                mock_state["auth_headers"].append(auth)
                if auth != expected_auth:
                    self._send(401, {"error": "unauthorized"})
                    return False
                return True

            def do_GET(self) -> None:
                if not self._check_auth():
                    return
                if self.path == "/running":
                    self._send(200, {"running": mock_state["running"]})
                elif self.path == "/v1/models":
                    self._send(200, {"data": [{"id": "qwen38-flash-next-plat"}, {"id": "inkling-2-mini"}]})
                elif self.path == "/unload":
                    mock_state["unload_calls"] += 1
                    mock_state["running"] = []
                    self._send(200, "OK")
                elif self.path.startswith("/upstream/"):
                    model_id = self.path.split("/")[2]
                    mock_state["running"] = [{"model": model_id, "state": "ready"}]
                    self._send(200, {"status": "ok"})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self) -> None:
                if not self._check_auth():
                    return
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8"))
                if self.path == "/v1/chat/completions":
                    mock_state["smoke_payloads"].append(payload)
                    self._send(
                        200,
                        {
                            "model": payload.get("model"),
                            "choices": [
                                {
                                    "message": {
                                        "content": "OK",
                                        "reasoning_content": "checking...",
                                    }
                                }
                            ],
                        },
                    )
                else:
                    self._send(404, {"error": "not found"})

        self.mock_swap = ThreadingHTTPServer(("127.0.0.1", 0), MockSwapHandler)
        self.mock_swap_port = self.mock_swap.server_address[1]
        self._swap_thread = threading.Thread(target=self.mock_swap.serve_forever, daemon=True)
        self._swap_thread.start()

        # Mock TelemetryCollector with deterministic fake system files
        fake_proc = self.repo_root / "fake_proc"
        fake_sys = self.repo_root / "fake_sys"
        (fake_proc / "sys" / "vm").mkdir(parents=True)
        (fake_proc / "meminfo").write_text(SAMPLE_MEMINFO, encoding="utf-8")
        (fake_proc / "stat").write_text(SAMPLE_STAT_2, encoding="utf-8")
        (fake_proc / "sys" / "vm" / "swappiness").write_text("10\n", encoding="utf-8")
        (fake_sys / "devices" / "system" / "cpu" / "cpu0" / "cpufreq").mkdir(parents=True)
        (fake_sys / "devices" / "system" / "cpu" / "cpu0" / "cpufreq" / "scaling_governor").write_text("performance\n", encoding="utf-8")
        (fake_sys / "kernel" / "mm" / "transparent_hugepage").mkdir(parents=True)
        (fake_sys / "kernel" / "mm" / "transparent_hugepage" / "enabled").write_text("[always] madvise never\n", encoding="utf-8")

        def fake_run_cmd(cmd: list[str], _timeout: float) -> tuple[int, str, str]:
            if cmd[0] == "nvidia-smi":
                return 0, SAMPLE_NVIDIA_CSV, ""
            if cmd[0] == "df":
                return 0, SAMPLE_DF_OUTPUT, ""
            if cmd[0] == "systemctl":
                return 0, "inactive\n", ""
            return 1, "", "unsupported"

        tel = telemetry.TelemetryCollector(proc_root=fake_proc, sys_root=fake_sys, run_cmd=fake_run_cmd)
        tel._prev_stat = telemetry.parse_proc_stat(SAMPLE_STAT_1)

        def fake_journal(lines: int) -> tuple[int, str, str]:
            sample = (
                f"2026-11-26T13:00:00 llama-swap started\n"
                f"2026-11-26T13:00:01 secret leak test {self.API_KEY} Bearer abcdef99999\n"
            )
            return 0, sample, ""

        self.dash = server.create_server(
            host="127.0.0.1",
            port=0,
            repo_root=self.repo_root,
            swap_url=f"http://127.0.0.1:{self.mock_swap_port}",
            api_key=self.API_KEY,
            telemetry=tel,
            sse_interval=0.15,
            journal_runner=fake_journal,
        )
        self.dash_port = self.dash.server_address[1]
        self._dash_thread = threading.Thread(target=self.dash.serve_forever, daemon=True)
        self._dash_thread.start()

    def tearDown(self) -> None:
        self.dash.shutdown()
        self.dash.server_close()
        self.mock_swap.shutdown()
        self.mock_swap.server_close()
        self._tmpdir.cleanup()

    def _get(self, path: str) -> tuple[int, str]:
        req = urllib.request.Request(f"http://127.0.0.1:{self.dash_port}{path}")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, exc.read().decode("utf-8")
            finally:
                exc.close()

    def _post(self, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode("utf-8") if body is not None else b""
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.dash_port}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_static_and_traversal_protection(self) -> None:
        code, html = self._get("/")
        self.assertEqual(code, 200)
        self.assertIn("L3MS", html)
        self.assertNotIn("L2M2", html)
        self.assertNotIn("L3DS", html)
        self.assertNotIn(self.API_KEY, html)

        code_js, js = self._get("/static/app.js")
        self.assertEqual(code_js, 200)
        self.assertIn("EventSource", js)

        code_bad, _ = self._get("/static/%2e%2e/server.py")
        self.assertEqual(code_bad, 403)

    def test_telemetry_endpoint(self) -> None:
        code, raw = self._get("/api/telemetry")
        self.assertEqual(code, 200)
        data = json.loads(raw)
        self.assertEqual(data["gpu"]["vram_used_mib"], 11562.0)
        self.assertEqual(data["cpu"]["p_core_pct"], 80.0)
        self.assertEqual(data["cpu"]["e_core_pct"], 20.0)
        self.assertTrue(data["tuning"]["governor_healthy"])
        self.assertEqual(data["tuning"]["vm_swappiness"], 10)
        self.assertEqual(len(data["disks"]), 3)

    def test_router_and_no_secret_leak(self) -> None:
        code, raw = self._get("/api/router")
        self.assertEqual(code, 200)
        self.assertNotIn(self.API_KEY, raw)
        data = json.loads(raw)
        self.assertTrue(data["reachable"])
        self.assertEqual(data["running"][0]["model"], "qwen38-flash-next-plat")
        self.assertEqual(data["active_engine_config"]["strata_env"]["STRATA_HC_Q8"], "1")
        ids = [m["id"] for m in data["catalog"]]
        self.assertIn("qwen38-flash-next-plat", ids)
        self.assertNotIn("legacy-unlisted-alias", ids)

        code2, raw2 = self._get("/api/router?include_unlisted=1")
        self.assertEqual(code2, 200)
        ids2 = [m["id"] for m in json.loads(raw2)["catalog"]]
        self.assertIn("legacy-unlisted-alias", ids2)

    def test_wake_smoke_and_unload(self) -> None:
        code, wake = self._post("/api/models/inkling-2-mini/wake")
        self.assertEqual(code, 200)
        self.assertTrue(wake["ok"])
        self.assertEqual(wake["evicted_candidates"], ["qwen38-flash-next-plat"])
        self.assertEqual(wake["running"][0]["model"], "inkling-2-mini")

        code_s, smoke = self._post("/api/models/inkling-2-mini/smoke")
        self.assertEqual(code_s, 200)
        self.assertTrue(smoke["ok"])
        self.assertTrue(smoke["has_reasoning"])
        self.assertEqual(smoke["snippet"], "OK")
        self.assertEqual(self.mock_state["smoke_payloads"][-1]["max_tokens"], 64)

        code_u, unload = self._post("/api/models/unload")
        self.assertEqual(code_u, 200)
        self.assertTrue(unload["ok"])
        self.assertEqual(self.mock_state["unload_calls"], 1)
        self.assertEqual(unload["running"], [])

        # Confirm every upstream request used the server-side Bearer key
        self.assertTrue(all(h == f"Bearer {self.API_KEY}" for h in self.mock_state["auth_headers"]))

    def test_logs_scrubbing_and_benches(self) -> None:
        code_j, raw_j = self._get("/api/logs?source=llama-swap&lines=50")
        self.assertEqual(code_j, 200)
        self.assertNotIn(self.API_KEY, raw_j)
        self.assertNotIn("abcdef99999", raw_j)
        self.assertIn("[REDACTED]", raw_j)

        code_e, raw_e = self._get("/api/logs?source=engine&lines=50")
        self.assertEqual(code_e, 200)
        self.assertNotIn("leaked-engine-token", raw_e)
        self.assertNotIn("l3ms-leak99999", raw_e)

        code_b, raw_b = self._get("/api/benches")
        self.assertEqual(code_b, 200)
        benches = json.loads(raw_b)
        self.assertEqual(benches["files"], ["inkling-refresh-ab.jsonl"])
        self.assertEqual(benches["runs"][0]["tg_steady"], 77.5)

    def test_sse_events_stream(self) -> None:
        req = urllib.request.Request(f"http://127.0.0.1:{self.dash_port}/api/events")
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/event-stream", resp.headers.get("Content-Type", ""))
            event_line = resp.readline().decode("utf-8").strip()
            data_line = resp.readline().decode("utf-8").strip()
            self.assertEqual(event_line, "event: snapshot")
            self.assertTrue(data_line.startswith("data: "))
            self.assertNotIn(self.API_KEY, data_line)
            payload = json.loads(data_line[len("data: ") :])
            self.assertEqual(payload["telemetry"]["gpu"]["vram_used_mib"], 11562.0)
            self.assertEqual(payload["router"]["running"][0]["model"], "qwen38-flash-next-plat")


if __name__ == "__main__":
    unittest.main()
