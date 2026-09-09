import asyncio
import json
import sys
from unittest.mock import patch

import pytest

from nocturnomath.environment import ResearchEnvironment
from nocturnomath.web.runtime import WebRuntime


def test_default_ignores_neighbor_venv_and_preserves_python_symlink(tmp_path):
    neighbor = tmp_path / ".venv/bin"
    neighbor.mkdir(parents=True)
    (neighbor / "python").symlink_to(sys.executable)
    managed = tmp_path / ".nocturnomath/venv/bin"
    managed.mkdir(parents=True)
    (managed / "python").symlink_to(sys.executable)
    with patch.object(ResearchEnvironment, "command", return_value="3.12"):
        env = ResearchEnvironment(tmp_path)
        assert env.python == managed / "python"
        env.save()
        assert ResearchEnvironment(tmp_path).python == env.python
        env = ResearchEnvironment(tmp_path, str(neighbor / "python"))
        env.save()
        assert ResearchEnvironment(tmp_path).python == neighbor / "python"
        assert ResearchEnvironment(tmp_path, "managed").python == managed / "python"


def test_environment_does_not_inherit_app_python_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/app/packages")
    monkeypatch.setenv("VIRTUAL_ENV", "/app/venv")
    with patch.object(ResearchEnvironment, "command", return_value="3.12"):
        env = ResearchEnvironment(tmp_path, "/other/venv/bin/python")
    values = env.process_env()
    assert "PYTHONPATH" not in values
    assert values["VIRTUAL_ENV"] == "/other/venv"
    assert values["PATH"].split(":")[0] == "/other/venv/bin"


def test_resume_and_restart_record_fresh_kernel(session):
    original = session.kernel_id
    session.reset_client_session()
    second = session.kernel_id
    assert original != second
    session.resume_session("S001")
    assert session.kernel_id not in (original, second)
    assert "variables are gone" in session._kernel_notice
    current = session.kernel_id
    assert not session.resume_session("S001")["kernel_reset"]
    assert session.kernel_id == current
    assert json.loads(
        (session.workspace_path / session.environment_record).read_text()
    )["python"]


def test_bad_environment_switch_preserves_workspace_and_kernel(session, tmp_path):
    original = session.kernel
    with (
        patch(
            "nocturnomath.session.ResearchEnvironment",
            side_effect=ValueError("missing ipykernel"),
        ),
        pytest.raises(ValueError, match="ipykernel"),
    ):
        session.set_workspace(tmp_path / "other")
    assert session.kernel is original
    assert session.kernel.is_alive()
    assert session.workspace_path == tmp_path


@pytest.mark.asyncio
async def test_preparation_leaves_event_loop_free_and_rejects_queries(session):
    import threading

    entered = threading.Event()
    release = threading.Event()
    runtime = WebRuntime(session)

    def prepare():
        entered.set()
        release.wait(5)

    task = asyncio.create_task(runtime.transition(prepare))
    try:
        await asyncio.to_thread(entered.wait, 5)
        assert runtime.changing
        with pytest.raises(RuntimeError, match="prepared"):
            runtime.start_query("hello")
        with pytest.raises(RuntimeError, match="prepared"):
            await runtime.transition(prepare)
    finally:
        release.set()
        await task
    assert not runtime.changing


def test_installation_targets_selected_python_and_records_failure(session):
    from types import SimpleNamespace

    session.environment.uv = lambda: "/bin/uv"
    session.environment.process_env = dict

    def fail(command, **kwargs):
        assert command[:5] == [
            "/bin/uv",
            "pip",
            "install",
            "--python",
            str(session.environment.python),
        ]
        kwargs["stdout"].write("package could not be resolved")
        return SimpleNamespace(returncode=1)

    before = session.environment_record
    with patch("nocturnomath.session.subprocess.run", side_effect=fail):
        status, output = session.install_packages(["nonexistent-science-package"])
    assert status == "failed"
    assert "could not be resolved" in output
    changes = list(
        session.transcript_path.parent.glob("environment_changes/*/install.json")
    )
    record = json.loads(changes[0].read_text())
    assert record["before"] == before
    assert record["after"] == session.environment_record != before
    assert record["kernel_id"] == session.kernel_id
    assert not list(session.workspace.probes_path.iterdir())
