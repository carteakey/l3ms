"""Offline maintenance regressions; all script destinations live in private temp dirs."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="l2m2-maintenance-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.nas = self.root / "nas"
        self.archive = self.nas / "models/media"
        self.local = self.root / "local"
        self.archive.mkdir(parents=True)
        self.local.mkdir()
        self.env = {
            "PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(self.root / "home"),
            "L2M2_MEDIA_NAS_ROOT": str(self.archive),
            "L2M2_MEDIA_NAS_MOUNT": str(self.nas),
            "L2M2_MEDIA_LOCAL_ROOT": str(self.local),
        }
        self.executable("mountpoint", '#!/bin/bash\n[[ "$3" == "$L2M2_MEDIA_NAS_MOUNT" ]]\n')
        self.executable("findmnt", '#!/bin/bash\nif [[ "$*" == *"-M "* ]]; then echo nfs4; else echo /; fi\n')

    def executable(self, name, text):
        path = self.bin / name
        path.write_text(text)
        path.chmod(0o755)
        return path

    def run_script(self, name, *args):
        return subprocess.run(["bash", str(HERE / name), *args], env=self.env,
                              cwd=self.root, capture_output=True, text=True)

    def model(self, root):
        path = root / "HeartMuLa-GGUF"
        path.mkdir()
        (path / "weights.bin").write_bytes(b"original")
        return path

    def test_archive_rejects_same_size_corruption(self):
        source = self.model(self.local)
        rsync = shutil.which("rsync")
        self.assertIsNotNone(rsync)
        self.executable("rsync", f'''#!/bin/bash
{rsync} "$@" || exit $?
if [[ "$1" != *n* ]]; then
    printf corrupted > "$L2M2_MEDIA_NAS_ROOT/HeartMuLa-GGUF/weights.bin"
fi
''')
        result = self.run_script("archive-media-models.sh")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((source / "weights.bin").read_bytes(), b"original")
        self.assertIn("Archive differs", result.stderr)

    def test_archive_deletes_only_after_real_checksum_comparison(self):
        source = self.model(self.local)
        result = self.run_script("archive-media-models.sh")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(source.exists())
        self.assertEqual((self.archive / source.name / "weights.bin").read_bytes(), b"original")

    def test_archive_preserves_conflicting_existing_archive(self):
        source = self.model(self.local)
        archived = self.model(self.archive)
        (archived / "weights.bin").write_bytes(b"archived")
        result = self.run_script("archive-media-models.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((source / "weights.bin").read_bytes(), b"original")
        self.assertEqual((archived / "weights.bin").read_bytes(), b"archived")

    def test_archive_refuses_nested_mount(self):
        source = self.model(self.local)
        self.executable("findmnt", '#!/bin/bash\nif [[ "$*" == *"-M "* ]]; then echo nfs4; else printf "%s\\n" "$L2M2_MEDIA_LOCAL_ROOT/HeartMuLa-GGUF/nested"; fi\n')
        result = self.run_script("archive-media-models.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("containing a mount", result.stderr)
        self.assertTrue((source / "weights.bin").exists())

    def test_autofs_fails_before_any_destination_resolution(self):
        source = self.model(self.local)
        self.executable("findmnt", '#!/bin/bash\n[[ "$*" == *"-rnC -M "*" -t noautofs "* ]] || exit 99\nexit 1\n')
        marker = self.root / "unexpected-path-probe"
        self.env["PROBE_MARKER"] = str(marker)
        for command in ("realpath", "mountpoint", "rsync"):
            self.executable(command, '#!/bin/bash\ntouch "$PROBE_MARKER"\nexit 99\n')
        for script, args in (("archive-media-models.sh", ()),
                             ("restore-media-model.sh", (source.name,))):
            with self.subTest(script=script):
                result = self.run_script(script, *args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("autofs placeholder", result.stderr)
                self.assertFalse(marker.exists())
                self.assertTrue(source.exists())

    def test_archive_keeps_source_on_failed_verification(self):
        source = self.model(self.local)
        rsync = shutil.which("rsync")
        self.executable("rsync", f'#!/bin/bash\n[[ "$1" != *n* ]] || exit 23\nexec {rsync} "$@"\n')
        result = self.run_script("archive-media-models.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((source / "weights.bin").is_file())

    def test_archive_unmounted_or_symlink_destination_preserves_source(self):
        source = self.model(self.local)
        self.executable("mountpoint", "#!/bin/bash\nexit 1\n")
        self.assertNotEqual(self.run_script("archive-media-models.sh").returncode, 0)
        self.assertTrue(source.exists())
        self.executable("mountpoint", '#!/bin/bash\n[[ "$3" == "$L2M2_MEDIA_NAS_MOUNT" ]]\n')
        (self.archive / source.name).symlink_to(source, target_is_directory=True)
        self.assertNotEqual(self.run_script("archive-media-models.sh").returncode, 0)
        self.assertTrue(source.exists())

    def test_restore_refuses_existing_directory_and_symlink(self):
        archived = self.model(self.archive)
        target = self.model(self.local)
        (target / "weights.bin").write_bytes(b"local file")
        result = self.run_script("restore-media-model.sh", archived.name)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((target / "weights.bin").read_bytes(), b"local file")
        shutil.rmtree(target)
        target.symlink_to(archived, target_is_directory=True)
        self.assertNotEqual(self.run_script("restore-media-model.sh", archived.name).returncode, 0)
        self.assertTrue(target.is_symlink())

    def test_restore_stages_and_verifies(self):
        archived = self.model(self.archive)
        result = self.run_script("restore-media-model.sh", archived.name)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.local / archived.name / "weights.bin").read_bytes(), b"original")
        self.assertEqual(list(self.local.glob(".restore-*")), [])

    def test_restore_rejects_path_traversal(self):
        self.assertNotEqual(self.run_script("restore-media-model.sh", "../local").returncode, 0)

    def test_turbo_downloader_resolves_from_the_repository_root(self):
        revision = re.search(r"^comfy_ref=([0-9a-f]{40})$",
                             (HERE / "setup-comfyui.sh").read_text(), re.M).group(1)
        for layout in ("root",):
            with self.subTest(layout=layout):
                base = self.root / layout
                toolkit = base
                script = toolkit / "maintenance/setup-qwen-image-2.1-turbo-comfyui.sh"
                script.parent.mkdir(parents=True)
                shutil.copy2(HERE / script.name, script)
                downloader = base / "model_downloader/download_hf_model.py"
                downloader.parent.mkdir()
                downloader.touch()
                (toolkit / "vendor/ComfyUI/.git").mkdir(parents=True)
                venv = toolkit / ".venv-comfy/bin"
                venv.mkdir(parents=True)
                (venv / "hf").write_text("#!/bin/bash\nexit 1\n")
                (venv / "hf").chmod(0o755)
                fake_python = venv / "python"
                fake_python.write_text('''#!/usr/bin/python3
import os, pathlib, sys
assert pathlib.Path(sys.argv[1]).resolve() == pathlib.Path(os.environ['EXPECTED_DOWNLOADER'])
target = pathlib.Path(sys.argv[sys.argv.index('--local-dir') + 1])
(target / 'weights.bin').write_bytes(b'fixture')
''')
                fake_python.chmod(0o755)
                self.executable("git", f"#!/bin/bash\nprintf '%s' {revision}\n")
                (toolkit / "models").mkdir()
                (toolkit / "models/qwen-image-2.1-turbo-comfyui-files.txt").write_text(
                    f"fixture/repo revision weights.bin diffusion_models/weights.bin 7 {hashlib.sha256(b'fixture').hexdigest()}\n")
                env = dict(self.env, EXPECTED_DOWNLOADER=str(downloader),
                           L2M2_QWEN_IMAGE_21_TURBO_ROOT=str(base / "model-output"))
                result = subprocess.run(["bash", str(script)], env=env, cwd=self.root,
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue((toolkit / "vendor/ComfyUI/models/diffusion_models/weights.bin").is_symlink())

    def test_shared_comfy_setup_pins_are_consistent(self):
        revision = re.search(r"^comfy_ref=([0-9a-f]{40})$",
                             (HERE / "setup-comfyui.sh").read_text(), re.M).group(1)
        for model in ("flux-2-klein-9b", "flux-3-klein", "krea-2",
                      "qwen-image-2.1", "qwen-image-2.1-turbo", "z-image-turbo"):
            with self.subTest(model=model):
                script = (HERE / f"setup-{model}-comfyui.sh").read_text()
                self.assertIn(f'comfy_revision="{revision}"', script)

    def test_flux3_setup_preserves_unrelated_vae_alias(self):
        toolkit = self.root / "flux-toolkit"
        script = toolkit / "maintenance/setup-flux-3-klein-comfyui.sh"
        script.parent.mkdir(parents=True)
        shutil.copy2(HERE / script.name, script)
        (toolkit / "vendor/ComfyUI/.git").mkdir(parents=True)
        python = toolkit / ".venv-comfy/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("#!/bin/bash\nexit 99\n")
        python.chmod(0o755)
        revision = re.search(r'^comfy_revision="([0-9a-f]{40})"$',
                             script.read_text(), re.M).group(1)
        self.executable("git", f"#!/bin/bash\nprintf '%s' {revision}\n")
        model_root = self.root / "flux-weights"
        (model_root / "vae").mkdir(parents=True)
        (model_root / "vae/flux2-vae.safetensors").write_bytes(b"fixture")
        alias = model_root / "vae/full_encoder_small_decoder.safetensors"
        alias.write_bytes(b"unrelated user weights")
        (toolkit / "models").mkdir()
        (toolkit / "models/flux-3-klein-comfyui-files.txt").write_text(
            f"fixture/repo revision vae/flux2-vae.safetensors vae/flux2-vae.safetensors 7 {hashlib.sha256(b'fixture').hexdigest()}\n")
        result = subprocess.run(["bash", str(script)],
                                env=dict(self.env, L2M2_FLUX_3_KLEIN_ROOT=str(model_root)),
                                cwd=self.root, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(alias.is_symlink())
        self.assertEqual(alias.read_bytes(), b"unrelated user weights")

    def test_external_rubric_sent_without_network_or_credentials(self):
        spec = importlib.util.spec_from_file_location("grader", HERE / "grade-images-with-vision-tier.py")
        grader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(grader)
        rubric = self.root / "rubric.txt"
        rubric.write_text("Evaluate fixture colors only.")
        captured = []

        def response(request, **kwargs):
            captured.append(json.loads(request.data))
            return io.StringIO('{"choices": [{"message": {"content": "pass"}}]}')

        with patch.object(grader, "api_key", return_value="fixture-key"), \
             patch.object(grader, "thumbnail", return_value="data:fixture"), \
             patch.object(grader, "urlopen", side_effect=response), \
             patch("sys.argv", ["grader", "fixture.png", "--rubric-file", str(rubric)]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(grader.main(), 0)
        self.assertEqual(captured[0]["messages"][0]["content"][0]["text"], rubric.read_text())
        self.assertIn("composition", grader.DEFAULT_RUBRIC)


if __name__ == "__main__":
    unittest.main()
