from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from xprober.web import create_app


class NoopKernel:
    def __init__(self, cwd=None):
        self.cwd = cwd
        self.busy = False
        self.alive = True

    def is_alive(self):
        return self.alive

    def set_cwd(self, cwd):
        self.cwd = cwd

    def restart(self):
        self.alive = True

    def interrupt(self):
        pass

    def shutdown(self):
        self.alive = False


def test_landing_defers_workspace_creation_and_browses_folders(tmp_path):
    workspace = tmp_path / "chosen-workspace"
    workspace.mkdir()
    (tmp_path / "not-a-folder.txt").write_text("keep")

    with patch("xprober.session.Kernel", NoopKernel):
        app = create_app(navigator_root=tmp_path)
        with TestClient(app) as client:
            landing = client.get("/api/workspace").json()
            assert landing["is_open"] is False
            assert landing["path"] is None
            assert not (workspace / "notes.md").exists()
            assert not (workspace / "scratch").exists()
            assert not (workspace / "xprober").exists()
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
                == "xprober/notes/evidence.md"
            )
            assert (workspace / "xprober" / "notes" / "evidence.md").is_file()
            assert (workspace / "xprober" / "sessions" / "S001").is_dir()
            assert (workspace / "xprober" / "notes").is_dir()
            assert (workspace / "xprober" / "scratch").is_dir()


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
            == "/api/asset?path=xprober/sessions/S001/probes/P001/plot-1.png"
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
            assert init["workspace"]["thoughts_path"] == "xprober/notes/thoughts.md"
            websocket.send_json({"action": "query", "text": "/notes"})
            text = websocket.receive_json()["text"]
            assert "# Evidence" in text and "# Thoughts" in text and "<del>" in text
