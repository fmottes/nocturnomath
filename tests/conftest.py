from unittest.mock import patch

import pytest

from nocturnomath.session import ExplorationSession


class FakeKernel:
    def __init__(self, cwd=None):
        self.cwd = cwd
        self.busy = False
        self.errored = False
        self.alive = True
        self.restarts = 0
        self.interrupts = 0
        self.result = ("result", [], None)

    def execute(self, code, timeout):
        return self.result

    def is_alive(self):
        return self.alive

    def set_cwd(self, cwd):
        self.cwd = cwd

    def restart(self):
        self.alive = True
        self.restarts += 1

    def interrupt(self):
        self.interrupts += 1

    def shutdown(self):
        self.alive = False


@pytest.fixture
def session(tmp_path):
    with patch("nocturnomath.session.Kernel", FakeKernel):
        return ExplorationSession(tmp_path)
