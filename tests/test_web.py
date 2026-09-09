from unittest.mock import AsyncMock, patch

import pytest
from conftest import FakeEnvironment, FakeKernel
from fastapi.testclient import TestClient

from nocturnomath.web import create_app


@pytest.fixture(autouse=True)
def sdk_model_catalog():
    with (
        patch("nocturnomath.session.ResearchEnvironment", FakeEnvironment),
        patch("nocturnomath.runtime.ClaudeSDKClient") as sdk,
    ):
        sdk.return_value.__aenter__.return_value.get_server_info = AsyncMock(
            return_value={
                "models": [
                    {"value": "default", "displayName": "Default (recommended)"},
                    {
                        "value": "sonnet",
                        "displayName": "Sonnet",
                        "resolvedModel": "claude-sonnet-5",
                    },
                ]
            }
        )
        yield sdk


def test_startup_discovers_models_without_querying(sdk_model_catalog):
    with TestClient(create_app()) as client:
        snapshot = client.get("/api/workspace").json()
        assert snapshot["models"] == ["sonnet"]
        assert snapshot["model_labels"]["sonnet"] == "claude-sonnet-5"
        sdk_model_catalog.return_value.__aenter__.return_value.query.assert_not_called()


def test_model_discovery_failure_has_no_fallback(sdk_model_catalog):
    sdk_model_catalog.return_value.__aenter__.side_effect = RuntimeError(
        "SDK unavailable"
    )
    with TestClient(create_app(model="custom-model")) as client:
        assert client.get("/api/workspace").json()["models"] == []


NoopKernel = FakeKernel


def test_landing_defers_workspace_creation_and_browses_folders(tmp_path):
    workspace = tmp_path / "chosen-workspace"
    workspace.mkdir()
    (tmp_path / ".hidden-folder").mkdir()
    (tmp_path / "not-a-folder.txt").write_text("keep")

    with patch("nocturnomath.session.Kernel", NoopKernel):
        app = create_app(navigator_root=tmp_path)
        with TestClient(app) as client:
            landing = client.get("/api/workspace").json()
            assert landing["is_open"] is False
            assert landing["path"] is None
            assert not (workspace / "notes.md").exists()
            assert not (workspace / "scratch").exists()
            assert not (workspace / ".nocturnomath").exists()
            assert client.get("/api/files").status_code == 409

            folders = client.get("/api/folders", params={"path": str(tmp_path)})
            assert folders.status_code == 200
            assert folders.json()["folders"] == [
                {"name": "chosen-workspace", "path": str(workspace)}
            ]

            missing = tmp_path / "does-not-exist"
            assert (
                client.post("/api/workspace", json={"path": str(missing)}).status_code
                == 400
            )
            assert not missing.exists()

            opened = client.post("/api/workspace", json={"path": str(workspace)})
            assert opened.status_code == 200
            assert opened.json()["workspace"]["is_open"] is True
            assert (
                opened.json()["workspace"]["evidence_path"]
                == ".nocturnomath/notes/evidence.md"
            )
            assert (workspace / ".nocturnomath" / "notes" / "evidence.md").is_file()
            assert not (workspace / ".nocturnomath" / "sessions" / "S001").exists()
            assert (workspace / ".nocturnomath" / "notes").is_dir()
            assert (workspace / ".nocturnomath" / "scratch").is_dir()


def test_landing_websocket_rejects_queries_without_a_workspace(tmp_path):
    app = create_app(navigator_root=tmp_path)
    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json()["workspace"]["is_open"] is False
        websocket.send_json({"action": "query", "text": "question"})
        assert websocket.receive_json() == {
            "type": "error",
            "message": "Open a workspace folder before using the agent.",
        }


def test_http_workspace_files_and_static_assets(session, tmp_path):
    (tmp_path / "report.md").write_text("# Report")
    report_figures = tmp_path / "report-figures"
    report_figures.mkdir()
    (report_figures / "embedded.png").write_bytes(b"embedded-figure")
    probe = session.workspace.start_probe("plot()", "a plot", "test")
    session.workspace.finish_probe(probe, "", [b"png-data"], "completed", None)
    app = create_app(session)
    with TestClient(app) as client:
        workspace = client.get("/api/workspace")
        assert workspace.status_code == 200
        assert workspace.json()["path"] == str(tmp_path)
        assert (
            client.get("/api/file", params={"path": "report.md"}).json()["content"]
            == "# Report"
        )
        assert client.get("/").status_code == 200
        static_script = client.get("/static/js/main.js")
        assert static_script.status_code == 200
        assert static_script.headers["cache-control"] == "no-store"
        assert client.get("/static/css/base.css").status_code == 200
        plots = client.get("/api/plots").json()["plots"]
        assert (
            plots[0]["url"]
            == "/api/asset?path=.nocturnomath/sessions/S001/probes/P001/plot-1.png"
        )
        assert client.get(plots[0]["url"]).content == b"png-data"
        assert (
            client.get(
                "/api/asset", params={"path": "report-figures/embedded.png"}
            ).content
            == b"embedded-figure"
        )
        assert (
            client.get("/api/asset", params={"path": "../outside.png"}).status_code
            == 403
        )


def test_current_session_downloads_as_notebook(session):
    app = create_app(session)
    with TestClient(app) as client:
        assert client.get("/api/session/notebook").status_code == 409
        session.log_transcript("user", text="question")
        session.log_transcript(
            "probe_started", probe_id="S001/P001", expected="3", code="print(3)"
        )
        session.log_transcript(
            "run", probe_id="S001/P001", output="3\n", images=[]
        )
        response = client.get("/api/session/notebook")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ipynb+json")
    assert response.headers["content-disposition"] == (
        f'attachment; filename="nocturnomath-{session.workspace_path.name}-S001.ipynb"'
    )
    assert response.json()["cells"][1]["source"] == "### Probe 1.1\n\n**Prediction:** 3"


def test_websocket_event_contract(session):
    session.query = AsyncMock()
    app = create_app(session)
    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        init = websocket.receive_json()
        assert init["type"] == "init"
        assert init["workspace"]["path"] == str(session.workspace_path)
        websocket.send_json({"action": "query", "text": "question"})
        assert websocket.receive_json() == {
            "type": "user_message",
            "text": "question",
        }


def test_new_session_restarts_kernel_and_broadcasts_reset(session):
    with (
        TestClient(create_app(session)) as client,
        client.websocket_connect("/ws") as websocket,
    ):
        websocket.receive_json()
        websocket.send_json({"action": "new_session"})
        assert websocket.receive_json()["type"] == "session_reset"
        assert session.kernel.restarts == 1
        websocket.send_json({"action": "query", "text": "/new"})
        assert websocket.receive_json()["type"] == "session_reset"
        assert session.kernel.restarts == 2
        assert client.post("/api/session/new").status_code == 200
        assert websocket.receive_json()["type"] == "session_reset"
        assert session.kernel.restarts == 3


def test_resume_session_can_restore_kernel(session):
    session.log_transcript("user", text="question")
    session.log_transcript("probe_started", probe_id="S001/P001", code="value = 3")
    session.reset_client_session()
    app = create_app(session)
    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        response = client.post(
            "/api/session/resume",
            json={"id": "S001", "restore_kernel": True},
        )
        event = websocket.receive_json()

    assert response.status_code == 200
    assert response.json()["probes_replayed"] == 1
    assert event["type"] == "session_resumed"
    assert event["kernel_restored"] is True
    assert session.kernel.executed_codes == ["value = 3"]


def test_model_selection_applies_to_next_query_without_resetting_context(session):
    seen_models = []

    async def query(text):
        seen_models.append(session.model)
        await session.emit("turn_complete")

    session.query = query
    session._sdk_session_id = "keep-context"
    with (
        TestClient(create_app(session)) as client,
        client.websocket_connect("/ws") as websocket,
    ):
        websocket.receive_json()
        websocket.send_json({"action": "query", "text": "question", "model": "sonnet"})
        assert websocket.receive_json()["type"] == "user_message"
        assert websocket.receive_json()["type"] == "turn_complete"
        assert seen_models == ["sonnet"]
        assert session._sdk_session_id == "keep-context"
        assert session.kernel.restarts == 0
        websocket.send_json({"action": "query", "text": "question", "model": "unknown"})
        assert websocket.receive_json()["type"] == "error"
        assert seen_models == ["sonnet"]


def test_exit_requests_launcher_shutdown_and_rejects_foreign_origin(session):
    from unittest.mock import Mock

    app = create_app(session)
    stop = Mock()
    app.state.request_shutdown = stop
    with TestClient(app) as client:
        assert (
            client.post(
                "/api/exit", headers={"Origin": "https://elsewhere.example"}
            ).status_code
            == 403
        )
        stop.assert_not_called()
        assert client.post("/api/exit").status_code == 200
        stop.assert_called_once()
        assert client.post("/api/exit").status_code == 200
        stop.assert_called_once()


def test_busy_http_mutations_return_conflict(session):
    session._is_busy = True
    app = create_app(session)
    with TestClient(app) as client:
        assert client.post("/api/kernel/restart").status_code == 409
        assert client.post("/api/session/new").status_code == 409
        assert (
            client.post(
                "/api/workspace", json={"path": str(session.workspace_path / "other")}
            ).status_code
            == 409
        )


def test_record_citations_resolve_to_saved_artifacts_and_entries(session):
    import re

    workspace = session.workspace
    probe = workspace.start_probe("print(3)", "3", "test")
    workspace.finish_probe(probe, "3\n", [b"png-data"], "completed", None)
    sources = workspace.evidence_sources(
        ["S001/P001/output.txt", "S001/P001/plot-1.png"]
    )
    workspace.notes.add_evidence("Value = 3.", sources)
    workspace.notes.add_thought("[E001] supports a positive value.", [])
    workspace.notes.add_thought(
        "[T001] is valid only under the conditions of [E001].", ["T001"]
    )
    workspace.notes.strike_evidence("E001")
    with TestClient(create_app(session)) as client:
        for document in (workspace.notes.evidence_path, workspace.notes.thoughts_path):
            for link in re.findall(r"\]\(([^)]+)\)", document.read_text()):
                target, _, anchor = link.partition("#")
                relative = (
                    (document.parent / target).resolve().relative_to(workspace.path)
                )
                response = client.get("/api/asset", params={"path": str(relative)})
                assert response.status_code == 200
                if anchor:
                    assert f'id="{anchor}"' in response.text
        with client.websocket_connect("/ws") as websocket:
            init = websocket.receive_json()
            assert (
                init["workspace"]["thoughts_path"] == ".nocturnomath/notes/thoughts.md"
            )
            websocket.send_json({"action": "query", "text": "/notes"})
            text = websocket.receive_json()["text"]
            assert "# Evidence" in text and "# Thoughts" in text and "<del>" in text
