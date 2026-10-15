#!/usr/bin/env python3
"""Exercise the staged Nagi install move without network or package installs."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


MAINTENANCE_DIR = Path(__file__).resolve().parent
TOOLKIT_ROOT = MAINTENANCE_DIR.parent
INSTALLER = MAINTENANCE_DIR / "prepare-nagi-smol-env.sh"
LOCK_FILE = TOOLKIT_ROOT / "requirements" / "nagi-smol.lock.txt"


FAKE_UV = r'''#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["UV_FIXTURE_CALL_LOG"]).open("a", encoding="utf-8") as calls:
    calls.write(json.dumps(args) + "\n")

if args[0] == "venv":
    environment = Path(args[-1])
    binary = environment / "bin"
    binary.mkdir(parents=True)
    python = binary / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    relocatable = "--relocatable" in args
    (environment / "fixture-relocatable").write_text(
        "yes" if relocatable else "no", encoding="utf-8"
    )
    if relocatable:
        activation = (
            'VIRTUAL_ENV="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." '
            '&& pwd)"\n'
        )
    else:
        activation = f"VIRTUAL_ENV={str(environment)!r}\n"
    (binary / "activate").write_text(activation, encoding="utf-8")
elif args[:2] == ["pip", "sync"]:
    python = Path(args[args.index("--python") + 1])
    environment = python.parent.parent
    relocatable = (environment / "fixture-relocatable").read_text(
        encoding="utf-8"
    ) == "yes"
    if relocatable:
        entrypoint = '#!/bin/sh\nexec "$(dirname -- "$0")/python" "$@"\n'
    else:
        entrypoint = f"#!{python}\n"
    launcher = environment / "bin" / "nagi-fixture"
    launcher.write_text(entrypoint, encoding="utf-8")
    launcher.chmod(0o755)
else:
    raise SystemExit(f"unexpected fixture uv command: {args!r}")
'''


class NagiEnvironmentRelocationTest(unittest.TestCase):
    def test_install_requests_relocatable_venv_before_staged_move(self):
        with tempfile.TemporaryDirectory(prefix="l3ds-nagi-relocation-") as temp:
            root = Path(temp)
            maintenance = root / "maintenance"
            requirements = root / "requirements"
            fake_bin = root / "fake-bin"
            maintenance.mkdir()
            requirements.mkdir()
            fake_bin.mkdir()
            installer = maintenance / INSTALLER.name
            shutil.copy2(INSTALLER, installer)
            shutil.copy2(LOCK_FILE, requirements / LOCK_FILE.name)

            fake_uv = fake_bin / "uv"
            fake_uv.write_text(FAKE_UV, encoding="utf-8")
            fake_uv.chmod(0o755)
            call_log = root / "uv-calls.jsonl"
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
            env["UV_FIXTURE_CALL_LOG"] = str(call_log)

            result = subprocess.run(
                ["bash", str(installer), "--install"],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in call_log.read_text().splitlines()]
            venv_args = next(args for args in calls if args[0] == "venv")
            sync_args = next(args for args in calls if args[:2] == ["pip", "sync"])
            self.assertIn("--relocatable", venv_args)
            self.assertIn("--strict", sync_args)
            self.assertIn("--torch-backend", sync_args)
            self.assertIn(str(requirements / LOCK_FILE.name), sync_args)

            staging = Path(venv_args[-1])
            self.assertEqual(staging.parent, root)
            self.assertTrue(staging.name.startswith(".nagi-smol-env."))
            self.assertFalse(staging.exists())
            final_environment = root / ".venv-decisions"
            self.assertTrue(final_environment.is_dir())
            self.assertTrue((final_environment / "bin" / "python").is_file())

            activation = (final_environment / "bin" / "activate").read_text(
                encoding="utf-8"
            )
            console_entrypoint = (
                final_environment / "bin" / "nagi-fixture"
            ).read_text(encoding="utf-8")
            self.assertNotIn(str(staging), activation)
            self.assertNotIn(str(staging), console_entrypoint)
            self.assertIn("BASH_SOURCE[0]", activation)
            self.assertIn('$(dirname -- "$0")/python', console_entrypoint)


if __name__ == "__main__":
    unittest.main()
