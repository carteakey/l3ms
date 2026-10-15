"""Exercise headless Blender argv without starting Blender."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("generate-blender-3d.sh")
FAKE_BLENDER = """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
Path(os.environ["BLENDER_ARGS_FILE"]).write_text(json.dumps(args))
print("FAKE_BLENDER_STDOUT_SENTINEL")
print("FAKE_BLENDER_STDERR_SENTINEL", file=sys.stderr)
if "--render-output" in args and os.environ.get("FAKE_BLENDER_WRITE_RENDER") == "1":
    pattern = args[args.index("--render-output") + 1]
    Path(pattern.replace("####", "0001") + ".png").write_bytes(b"fixture")
raise SystemExit(int(os.environ.get("FAKE_BLENDER_EXIT", "0")))
"""


class BlenderHeadlessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fake = self.root / "fake-blender"
        self.fake.write_text(FAKE_BLENDER)
        self.fake.chmod(0o700)
        self.capture = self.root / "argv.json"
        self.env = dict(os.environ, L3MS_BLENDER_BIN=str(self.fake),
                        L3MS_BLENDER_HEADLESS_ONLY="1",
                        BLENDER_ARGS_FILE=str(self.capture))

    def run_wrapper(self, *args, headless=True, **env):
        run_env = dict(self.env)
        run_env.update(env)
        if not headless:
            run_env.pop("L3MS_BLENDER_HEADLESS_ONLY", None)
        return subprocess.run(["/bin/bash", str(SCRIPT), *args], cwd=self.root,
                              env=run_env, capture_output=True, text=True)

    def recorded_argv(self):
        return json.loads(self.capture.read_text())

    def test_code_and_script_use_native_exception_exit_and_preserve_argument(self):
        code = "import bpy; print('exact ; code')"
        result = self.run_wrapper("--code", code, FAKE_BLENDER_EXIT="13")
        self.assertEqual(result.returncode, 13)
        self.assertEqual(self.recorded_argv(), ["--background", "--python-exit-code", "1",
                                                 "--python-expr", code])
        self.assertNotIn("FAKE_BLENDER", result.stdout + result.stderr)
        self.assertNotIn(code, result.stdout + result.stderr)

        self.capture.unlink()
        script = self.root / "script with spaces.py"
        script.touch()
        result = self.run_wrapper("--script", str(script))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.recorded_argv(), ["--background", "--python-exit-code", "1",
                                                 "--python", str(script)])

        self.capture.unlink()
        result = self.run_wrapper("--script", str(self.root / "missing.py"))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.capture.exists())

    def test_render_uses_fresh_fixed_png_target_and_refuses_collision(self):
        scene = self.root / "scene.blend"
        scene.touch()
        output = self.root / "private-output"
        output.mkdir()
        result = self.run_wrapper("--render", str(scene), "--output", str(output),
                                  FAKE_BLENDER_WRITE_RENDER="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        target = output / "result_0001.png"
        self.assertEqual(target.read_bytes(), b"fixture")
        self.assertEqual(self.recorded_argv(), ["--background", str(scene), "--render-output",
            str(output / "result_####"), "--render-format", "PNG", "--use-extension", "1",
            "--render-frame", "1"])

        self.capture.unlink()
        result = self.run_wrapper("--render", str(scene), "--output", str(output),
                                  FAKE_BLENDER_WRITE_RENDER="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.capture.exists())
        self.assertEqual(target.read_bytes(), b"fixture")

        missing = self.root / "missing.blend"
        result = self.run_wrapper("--render", str(missing), "--output", str(output))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.capture.exists())

    def test_only_one_headless_action_and_no_standalone_controls(self):
        script = self.root / "script.py"
        script.touch()
        cases = (
            (), ("--code", ""),
            ("--code", "x", "--code", "y"),
            ("--code", "x", "--script", str(script)),
            ("--prompt", "x", "--code", "y"),
            ("--launch",), ("--check",), ("--port", "9877", "--code", "x"),
            ("--help",), ("--code", "x", "--output", str(self.root)),
        )
        for args in cases:
            with self.subTest(args=args):
                self.capture.unlink(missing_ok=True)
                result = self.run_wrapper(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.capture.exists())

    def test_legacy_code_invocation_remains_unmodified(self):
        code = "import bpy; print('standalone')"
        result = self.run_wrapper("--code", code, headless=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.recorded_argv(), ["--background", "--python-expr", code])
        self.assertIn("FAKE_BLENDER_STDOUT_SENTINEL", result.stdout)


if __name__ == "__main__":
    unittest.main()
