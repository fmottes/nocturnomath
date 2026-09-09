from pathlib import Path
from unittest.mock import patch

import pytest

from nocturnomath.session import ExplorationSession


class FakeKernel:
    def __init__(self, cwd=None, environment=None):
        self.on_start = None
        self.cwd = cwd
        self.busy = False
        self.errored = False
        self.alive = True
        self.restarts = 0
        self.interrupts = 0
        self.result = ("result", [], None)
        self.executed_codes = []

    def execute(self, code, timeout):
        self.executed_codes.append(code)
        return self.result

    def is_alive(self):
        return self.alive

    def set_cwd(self, cwd):
        self.cwd = cwd

    def restart(self, reason="restarted"):
        if self.on_start:
            self.on_start(reason)
        self.alive = True
        self.restarts += 1

    def interrupt(self):
        self.interrupts += 1

    def shutdown(self):
        self.alive = False


@pytest.fixture
def session(tmp_path):
    with (
        patch("nocturnomath.session.Kernel", FakeKernel),
        patch("nocturnomath.session.ResearchEnvironment", FakeEnvironment),
    ):
        return ExplorationSession(tmp_path)


class FakeEnvironment:
    def __init__(self, workspace, python=None):
        self.python = Path(workspace) / ".nocturnomath/venv/bin/python"
        self.version = "3.12"

    def save(self):
        pass

    def snapshot(self):
        return {"python": str(self.python), "version": self.version, "packages": []}
