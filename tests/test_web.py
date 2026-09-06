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
            assert (workspace / "notes.md").is_file()
            assert (workspace / "scratch").is_dir()


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
        assert client.get("/static/js/main.js").status_code == 200
        assert client.get("/static/css/base.css").status_code == 200


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
