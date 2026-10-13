"""Exercise native argv and prepared-only dispatch without loading a model."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("generate-ltx-2.5.sh")


class LtxPreparedTests(unittest.TestCase):
    def test_prepared_environment_and_native_image_triples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "checkout"
            checkout.mkdir()
            models = root / "models"
            for name in (
                "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
                "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
                "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
                "vae/ltx-2.5-audio-vae-bf16.safetensors",
                "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
            ):
                path = models / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            capture = root / "capture.json"
            env = dict(os.environ, HOME=str(root), L3MS_LTX_ROOT="checkout",
                       L3MS_LTX_MODEL_DIR=str(models), L3MS_LTX_PREPARED_ONLY="1",
                       L3MS_MEDIA_OUTPUT_DIR=str(root / "output"), CAPTURE=str(capture))
            command = ["/bin/bash", str(SCRIPT), "--prompt", "a boat; not a command"]
            missing = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(missing.returncode, 1)
            self.assertIn("Prepared LTX-2 Python environment not found", missing.stderr)
            self.assertFalse(capture.exists())
            interpreter = checkout / ".venv/bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text(
                "#!/usr/bin/python3\nimport json, os, sys\n"
                "with open(os.environ['CAPTURE'], 'w') as f:\n"
                " json.dump([sys.argv[1:], os.environ['HF_HUB_OFFLINE'], "
                "os.environ['TRANSFORMERS_OFFLINE']], f)\n"
            )
            interpreter.chmod(0o700)
            image = root / "image with spaces.png"
            image.touch()
            result = subprocess.run(command + ["--image", str(image), "0", "0.8",
                "--image", str(image), "120", "1.0", "--output", str(root / "result.mp4")],
                cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            argv, hub_offline, transformers_offline = json.loads(capture.read_text())
            self.assertEqual(argv[:2], ["-m", "ltx_pipelines.distilled"])
            self.assertEqual(argv[-8:], ["--image", str(image), "0", "0.8",
                                       "--image", str(image), "120", "1.0"])
            self.assertEqual(argv[argv.index("--prompt") + 1], "a boat; not a command")
            self.assertIn("--audio-vae-path", argv)
            self.assertIn("--video-vae-path", argv)
            self.assertEqual((hub_offline, transformers_offline), ("1", "1"))


if __name__ == "__main__":
    unittest.main()
