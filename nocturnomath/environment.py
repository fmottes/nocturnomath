"""The research environment is independent of the application's Python."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

MANAGED_RESEARCH_PACKAGES = ("ipykernel", "matplotlib", "numpy", "pandas")
PRIVATE_APP_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
    "SSH_AUTH_SOCK",
    "SSH_AGENT_PID",
    "SSH_ASKPASS",
    "GIT_ASKPASS",
)


class ResearchEnvironment:
    def __init__(self, workspace, python=None):
        self.workspace = Path(workspace).expanduser().resolve()
        config = self.workspace / ".nocturnomath/config.json"
        saved = json.loads(config.read_text()) if config.exists() else {}
        selected = python or saved.get("python") or ".nocturnomath/venv/bin/python"
        if selected == "managed":
            selected = ".nocturnomath/venv/bin/python"
        # Do not resolve the executable symlink: that would select base Python.
        self.python = Path(
            os.path.abspath(self.workspace / Path(selected).expanduser())
        )
        managed = self.workspace / ".nocturnomath/venv/bin/python"
        if self.python == managed:
            if not self.python.exists():
                self.command(
                    [
                        self.uv(),
                        "venv",
                        "--python",
                        sys.executable,
                        str(managed.parent.parent),
                    ]
                )
            try:
                self.command(
                    [
                        str(self.python),
                        "-c",
                        f"import {', '.join(MANAGED_RESEARCH_PACKAGES)}",
                    ]
                )
            except ValueError:
                # Repair an interrupted setup without asking uv to upgrade packages.
                self.command(
                    [
                        self.uv(),
                        "pip",
                        "install",
                        "--python",
                        str(self.python),
                        *MANAGED_RESEARCH_PACKAGES,
                    ]
                )
        self.command(
            [
                str(self.python),
                "-c",
                "import sys, pathlib; assert sys.prefix != sys.base_prefix or (pathlib.Path(sys.prefix)/'conda-meta').is_dir(), 'Select a virtual or conda environment'; import ipykernel, matplotlib",
            ]
        )
        self.version = self.command(
            [
                str(self.python),
                "-c",
                "import platform; print(platform.python_version())",
            ]
        ).strip()

    @staticmethod
    def uv():
        """Prefer the uv on PATH, then the binary bundled with the uv package."""
        executable = shutil.which("uv")
        if executable:
            return executable
        try:
            from uv import find_uv_bin

            return find_uv_bin()
        except (ImportError, FileNotFoundError):
            raise ValueError(
                "uv is required to prepare research environments. Install uv and restart the app."
            )

    def process_env(self):
        env = os.environ.copy()
        for key in (
            "VIRTUAL_ENV",
            "PYTHONHOME",
            "PYTHONPATH",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
            *PRIVATE_APP_ENV,
        ):
            env.pop(key, None)
        env["VIRTUAL_ENV"] = str(self.python.parent.parent)
        env["PATH"] = str(self.python.parent) + os.pathsep + env.get("PATH", "")
        env["PYTHONNOUSERSITE"] = "1"
        return env

    def command(self, argv):
        result = subprocess.run(
            argv,
            check=False,
            cwd=self.workspace,
            env=self.process_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
        if result.returncode:
            raise ValueError(result.stdout.strip() or "Environment command failed")
        return result.stdout

    def save(self):
        config = self.workspace / ".nocturnomath/config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(config.read_text()) if config.exists() else {}
        data["python"] = (
            os.path.relpath(self.python, self.workspace)
            if self.python.is_relative_to(self.workspace)
            else str(self.python)
        )
        config.write_text(json.dumps(data, indent=2) + "\n")

    def snapshot(self):
        packages = self.command(
            [
                str(self.python),
                "-c",
                "import importlib.metadata as m, json; print(json.dumps(sorted([(d.metadata['Name'], d.version) for d in m.distributions()])))",
            ]
        )
        return {
            "python": str(self.python),
            "version": self.version,
            "packages": json.loads(packages),
        }
