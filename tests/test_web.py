import asyncio
import json
import re
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from conftest import FakeEnvironment, FakeKernel
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from nocturnomath.backends import ModelSpec
from nocturnomath.prompt import (
    DEFAULT_EXPLORATION_PROMPT,
    RUNTIME_PROMPT,
    build_system_prompt,
)
from nocturnomath.session import ExplorationSession
from nocturnomath.web import create_app


class FakeWebBackend:
    def __init__(self, provider, models=None, error=None):
        self.provider = provider
        self.uses_compact_hook = provider == "anthropic"
        self.models = list(models or [])
        self.error = error
        self.auth_seen = []
        self.turn_calls = 0

    async def discover_models(self, auth):
        self.auth_seen.append(auth)
        if self.error is not None:
            raise self.error
        return list(self.models)

    async def discover_models_staged(self, auth):
        return await self.discover_models(auth), lambda: None

    async def run_turn(self, *args, **kwargs):
        self.turn_calls += 1
        if False:
            yield None

    async def compact(self, *args, **kwargs):
        return None


@pytest.fixture(autouse=True)
def backend_model_catalog():
    backends = {
        "anthropic": FakeWebBackend(
            "anthropic",
            models=[
                ModelSpec(
                    id="claude-opus-5",
                    raw_id="claude-opus-5",
                    provider="anthropic",
                    label="claude-opus-5",
                    efforts=["low", "medium", "high", "xhigh", "max"],
                    is_default=True,
                ),
                ModelSpec(
                    id="claude-sonnet-5",
                    raw_id="claude-sonnet-5",
                    provider="anthropic",
                    label="claude-sonnet-5",
                    efforts=["low", "medium", "high", "xhigh"],
                ),
            ],
        ),
        "openai": FakeWebBackend(
            "openai",
            models=[
                ModelSpec(
                    id="gpt-6",
                    raw_id="gpt-6",
                    provider="openai",
                    label="gpt-6",
                    efforts=["low", "medium", "high", "ultra"],
                    is_default=True,
                )
            ],
        ),
        "google": FakeWebBackend("google", models=[]),
    }
    with (
        patch("nocturnomath.session.ResearchEnvironment", FakeEnvironment),
        patch("nocturnomath.session.default_backends", return_value=backends),
        patch("nocturnomath.runtime.default_backends", return_value=backends),
    ):
        yield backends


def test_startup_discovers_models_without_querying(backend_model_catalog):
    with TestClient(create_app()) as client:
        snapshot = client.get("/api/workspace").json()
        assert snapshot["models"] == ["claude-opus-5", "claude-sonnet-5", "gpt-6"]
        assert snapshot["model_labels"]["claude-sonnet-5"] == "claude-sonnet-5"
        assert snapshot["model_providers"] == {
            "claude-opus-5": "anthropic",
            "claude-sonnet-5": "anthropic",
            "gpt-6": "openai",
        }
        assert snapshot["model"] == "claude-opus-5"
        assert snapshot["effort"] == "high"
        assert snapshot["model_efforts"]["claude-opus-5"][-1] == "max"
        assert all(
            backend.turn_calls == 0 for backend in backend_model_catalog.values()
        )


def test_batch_monitor_publishes_the_original_probe_result_and_verdict(
    session, monkeypatch
):
    def scheduler(argv, workspace, env=None):
        output = (
            "42"
            if argv[0] == "sbatch"
            else "42|COMPLETED|0:0\n"
            if argv[0] == "sacct"
            else ""
        )
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr("nocturnomath.slurm._command", scheduler)
    session.log_transcript("user", text="Run a batch probe.")
    tools = {tool.name: tool for tool in session.tools}
    asyncio.run(
        tools["submit_slurm"].handler(
            {
                "code": "print(42)",
                "expected": "42",
                "time_minutes": 1,
                "cpus": 1,
                "memory_gb": 1,
                "result_files": ["plot-1.png"],
            }
        )
    )
    probe = session.workspace.probes_path / "P001"
    (probe / "stdout.txt").write_text("42\n")
    (probe / "stderr.txt").write_text("")
    (probe / "plot-1.png").write_bytes(b"saved plot")
    reviewed = threading.Event()

    async def review(probe_id):
        result = await tools["check_slurm_job"].handler({"probe_id": probe_id})
        assert result["content"][1]["type"] == "image"
        await tools["verdict"].handler({"text": "It printed 42 as predicted."})
        reviewed.set()
        return True

    monkeypatch.setattr(session, "review_slurm", review)
    with TestClient(create_app(session)) as client:
        assert reviewed.wait(timeout=3)
        snapshot = client.get("/api/workspace").json()
        records = snapshot["agents"][0]["records"]
        result = next(record for record in records if record["kind"] == "run")
        verdict = next(record for record in records if record["kind"] == "verdict")
        assert result["probe_id"] == verdict["probe_id"] == "S001/P001"
        assert result["backend"] == "slurm"
        assert (
            client.get("/api/asset", params={"path": result["images"][0]}).content
            == b"saved plot"
        )
        assert sum(record["kind"] == "user" for record in records) == 1


def test_startup_uses_first_model_when_backend_has_no_explicit_default(
    backend_model_catalog,
):
    backend_model_catalog["anthropic"].models = [
        ModelSpec(
            id="claude-sonnet-5",
            raw_id="claude-sonnet-5",
            provider="anthropic",
            label="claude-sonnet-5",
        ),
        ModelSpec(
            id="claude-opus-5",
            raw_id="claude-opus-5",
            provider="anthropic",
            label="claude-opus-5",
        ),
    ]
    backend_model_catalog["openai"].models = []
    backend_model_catalog["google"].models = []

    with TestClient(create_app()) as client:
        snapshot = client.get("/api/workspace").json()

    assert snapshot["model"] == "claude-sonnet-5"
    assert snapshot["model_labels"]["claude-opus-5"] == "claude-opus-5"


def test_startup_opens_the_dashboard_in_a_new_tab():
    with (
        patch("nocturnomath.web.app.asyncio.sleep", new=AsyncMock()),
        patch("nocturnomath.web.app.webbrowser.open") as open_browser,
        TestClient(create_app(browser_url="http://127.0.0.1:8000")),
    ):
        open_browser.assert_called_once_with("http://127.0.0.1:8000", new=2)


def test_authentication_can_be_changed_before_opening_a_workspace(
    backend_model_catalog,
):
    with (
        TestClient(create_app()) as client,
        client.websocket_connect("/ws") as websocket,
    ):
        websocket.receive_json()
        response = client.post(
            "/api/auth",
            json={
                "provider": "anthropic",
                "method": "subscription",
                "credential": "oauth-secret",
            },
        )
        event = websocket.receive_json()

        assert response.status_code == 200
        payload = response.json()
        assert payload["auth"]["providers"]["anthropic"]["method"] == "subscription"
        assert payload["models"] == [
            "claude-opus-5",
            "claude-sonnet-5",
            "gpt-6",
        ]
        assert payload["model"] == "claude-opus-5"
        assert "oauth-secret" not in response.text
        assert event["type"] == "auth_changed"
        assert event["provider"] == "anthropic"
        assert event["auth"]["providers"]["anthropic"]["method"] == "subscription"
        assert "oauth-secret" not in repr(event)

        seen_env = backend_model_catalog["anthropic"].auth_seen[-1].sdk_env()
        assert seen_env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-secret"
        assert seen_env["ANTHROPIC_API_KEY"] == ""

        openai_response = client.post(
            "/api/auth",
            json={
                "provider": "openai",
                "method": "api_key",
                "credential": "openai-secret",
            },
        )
        openai_event = websocket.receive_json()
        assert openai_response.status_code == 200
        assert (
            openai_response.json()["auth"]["providers"]["openai"]["method"] == "api_key"
        )
        assert "openai-secret" not in openai_response.text
        assert openai_event["provider"] == "openai"
        assert "openai-secret" not in repr(openai_event)
        assert (
            backend_model_catalog["openai"].auth_seen[-1].sdk_env()["OPENAI_API_KEY"]
            == "openai-secret"
        )


def test_authentication_rejects_missing_secrets_and_foreign_origins():
    with TestClient(create_app()) as client:
        missing = client.post(
            "/api/auth", json={"provider": "openai", "method": "api_key"}
        )
        assert missing.status_code == 400
        assert "API key" in missing.json()["detail"]

        disallowed = client.post(
            "/api/auth",
            json={
                "provider": "google",
                "method": "subscription",
                "credential": "secret",
            },
        )
        assert disallowed.status_code == 400
        assert "secret" not in disallowed.text

        foreign = client.post(
            "/api/auth",
            json={
                "provider": "anthropic",
                "method": "api_key",
                "credential": "secret",
            },
            headers={"Origin": "https://elsewhere.example"},
        )
        assert foreign.status_code == 403

        for body in (
            {
                "provider": "anthropic",
                "method": "secret-invalid-method",
                "credential": "secret",
            },
            {"provider": "invalid", "method": "cli", "credential": "secret"},
            {
                "provider": "openai",
                "method": "api_key",
                "credential": {"key": "secret"},
            },
        ):
            invalid = client.post("/api/auth", json=body)
            assert invalid.status_code == 422
            assert "secret" not in invalid.text
            assert invalid.json()["detail"][0]["loc"]


def test_authentication_ui_is_served_without_embedding_credentials():
    with TestClient(create_app()) as client:
        page = client.get("/").text
        assert 'id="btn-auth-landing"' in page
        assert 'id="auth-modal"' in page
        assert 'id="auth-provider"' in page
        assert 'value="openai"' in page
        assert 'value="google"' in page
        assert 'type="password"' in page
        external_scripts = re.findall(
            r'<script\b[^>]*src="https://[^>]+></script>', page
        )
        assert external_scripts
        assert all('integrity="sha384-' in tag for tag in external_scripts)
        assert all('crossorigin="anonymous"' in tag for tag in external_scripts)


def test_model_discovery_failure_has_no_fallback(backend_model_catalog):
    for backend in backend_model_catalog.values():
        backend.error = RuntimeError("CLI unavailable")
    with TestClient(create_app(model="custom-model")) as client:
        snapshot = client.get("/api/workspace").json()
        assert snapshot["models"] == []
        assert snapshot["model"] is None
        assert snapshot["model_providers"] == {}


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
            assert client.get("/api/documents").status_code == 409

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
                == ".nocturnomath/kb/evidence.md"
            )
            assert (workspace / ".nocturnomath" / "kb" / "evidence.md").is_file()
            assert not (workspace / ".nocturnomath" / "sessions" / "S001").exists()
            assert (workspace / ".nocturnomath" / "documents").is_dir()
            assert (workspace / ".nocturnomath" / "scratch").is_dir()


def test_browser_boundary_rejects_foreign_origins_and_navigation(tmp_path):
    app = create_app(navigator_root=tmp_path)
    with TestClient(app) as client:
        workspace = client.get("/api/workspace")
        assert "access-control-allow-origin" not in workspace.headers
        assert workspace.headers["x-content-type-options"] == "nosniff"
        assert (
            client.get(
                "/api/folders", params={"path": str(tmp_path.parent)}
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/workspace",
                json={"path": str(tmp_path)},
                headers={"Origin": "https://elsewhere.example"},
            ).status_code
            == 403
        )
        assert (
            client.get(
                "/api/workspace", headers={"Host": "elsewhere.example"}
            ).status_code
            == 400
        )
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect(
                "/ws", headers={"Origin": "https://elsewhere.example"}
            ),
        ):
            pass


def test_landing_websocket_rejects_queries_without_a_workspace(tmp_path):
    app = create_app(navigator_root=tmp_path)
    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json()["workspace"]["is_open"] is False
        assert (
            client.post(
                "/api/session/defaults",
                json={"model": "claude-opus-5", "effort": "high"},
            ).status_code
            == 409
        )
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
        assert workspace.json()["session_id"] == "S001"
        assert (
            client.get("/api/file", params={"path": "report.md"}).json()["content"]
            == "# Report"
        )
        assert client.get("/").status_code == 200
        page = client.get("/").text
        assert 'id="effort-select"' in page
        assert 'value="default">Effort:' not in page
        static_script = client.get("/static/js/main.js")
        assert static_script.status_code == 200
        assert static_script.headers["cache-control"] == "no-store"
        assert "appendProbeFinish(event.expected, event.code, event.output," in (
            static_script.text
        )
        assert client.get("/static/css/base.css").status_code == 200
        listing = client.get("/api/plots").json()
        assert listing["current_session"] == "S001"
        plots = listing["plots"]
        assert plots[0]["session"] == "S001"
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
        session.log_transcript("run", probe_id="S001/P001", output="3\n", images=[])
        response = client.get("/api/session/notebook")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ipynb+json")
    assert response.headers["content-disposition"] == (
        f'attachment; filename="nocturnomath-{session.workspace_path.name}-S001.ipynb"'
    )
    assert response.json()["cells"][1]["source"] == "### Probe 1.1\n\n**Prediction:** 3"


def test_websocket_event_contract(session):
    async def query(text):
        await session.emit("assistant_delta", text="streamed ")
        await session.emit("assistant_delta", text="reply")
        await session.emit("assistant_text", text="streamed reply")
        await session.emit("turn_complete", full_text="streamed reply")

    session.query = query
    session.context_usage = {
        "used_tokens": 13500,
        "window_tokens": 200000,
        "model": "test",
    }
    app = create_app(session)
    session.log_transcript("user", text="earlier question")
    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        init = websocket.receive_json()
        assert init["type"] == "init"
        assert init["workspace"]["path"] == str(session.workspace_path)
        assert [
            r["text"] for r in init["workspace"]["records"] if r["kind"] == "user"
        ] == ["earlier question"]
        agent_id = init["workspace"]["agents"][0]["agent_id"]
        assert init["workspace"]["agents"][0]["context_usage"] == session.context_usage
        websocket.send_json(
            {"action": "query", "agent_id": agent_id, "text": "question"}
        )
        user = websocket.receive_json()
        assert (user["type"], user["agent_id"], user["text"]) == (
            "user_message",
            agent_id,
            "question",
        )
        first_delta = websocket.receive_json()
        second_delta = websocket.receive_json()
        assert (first_delta["type"], first_delta["text"]) == (
            "assistant_delta",
            "streamed ",
        )
        assert first_delta["agent_id"] == agent_id
        assert (second_delta["type"], second_delta["text"]) == (
            "assistant_delta",
            "reply",
        )
        assistant = websocket.receive_json()
        assert assistant["type"] == "assistant_text"
        assert assistant["text"] == "streamed reply"
        assert isinstance(assistant["timestamp"], float)
        complete = websocket.receive_json()
        assert complete["type"] == "turn_complete"
        assert complete["full_text"] == "streamed reply"


def test_explorers_are_independent_and_idle_tabs_can_close(session):
    def factory(**kwargs):
        with patch("nocturnomath.session.Kernel", FakeKernel):
            return ExplorationSession(**kwargs)

    with TestClient(create_app(session, session_factory=factory)) as client:
        initial = client.get("/api/workspace").json()["agents"][0]
        created = client.post("/api/agents")
        assert created.status_code == 201
        second = created.json()
        assert second["agent_id"] != initial["agent_id"]
        assert (
            client.app.state.runtime.agents[initial["agent_id"]].kernel
            is not client.app.state.runtime.agents[second["agent_id"]].kernel
        )
        client.app.state.runtime.agents[second["agent_id"]]._is_busy = True
        assert client.delete(f"/api/agents/{second['agent_id']}").status_code == 409
        client.app.state.runtime.agents[second["agent_id"]]._is_busy = False
        assert client.delete(f"/api/agents/{second['agent_id']}").status_code == 200
        assert client.delete(f"/api/agents/{initial['agent_id']}").status_code == 409


def test_manual_compaction_route_is_scoped_and_refuses_busy_explorers(session):
    session.log_transcript("user", text="original question")
    session._sdk_session_id = "sdk-session"

    async def compact():
        await session.emit("status_change", status="compacting")
        await session.emit("context_compacted")
        await session.emit("status_change", status="idle")

    session.compact = compact
    with (
        TestClient(create_app(session)) as client,
        client.websocket_connect("/ws") as websocket,
    ):
        agent_id = websocket.receive_json()["workspace"]["agents"][0]["agent_id"]
        assert client.post("/api/agents/unknown/compact").status_code == 404
        session._is_busy = True
        assert client.post(f"/api/agents/{agent_id}/compact").status_code == 409
        session._is_busy = False

        response = client.post(f"/api/agents/{agent_id}/compact")
        events = [websocket.receive_json() for _ in range(3)]

    assert response.status_code == 200
    assert [event["type"] for event in events] == [
        "status_change",
        "context_compacted",
        "status_change",
    ]
    assert all(event["agent_id"] == agent_id for event in events)


def test_history_replaces_the_selected_idle_explorer(session):
    session.log_transcript("user", text="first exploration")

    def factory(**kwargs):
        with patch("nocturnomath.session.Kernel", FakeKernel):
            return ExplorationSession(**kwargs)

    with (
        TestClient(create_app(session, session_factory=factory)) as client,
        client.websocket_connect("/ws") as websocket,
    ):
        websocket.receive_json()
        agent_id = client.get("/api/workspace").json()["agents"][0]["agent_id"]
        response = client.post(
            f"/api/agents/{agent_id}/resume",
            json={"id": "S001", "restore_kernel": False},
        )
        resumed_evt = websocket.receive_json()
    assert response.status_code == 200
    body = response.json()
    assert body["agent_id"] == agent_id
    assert body["session_id"] == "S001"
    assert body["is_busy"] is False
    assert body["status"] == "idle"
    assert resumed_evt["type"] == "session_resumed"
    assert resumed_evt["is_busy"] is False
    assert resumed_evt["status"] == "idle"


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


def test_model_and_effort_apply_to_next_query_without_resetting_context(session):
    seen_settings = []

    async def query(text):
        seen_settings.append((session.model, session.effort))
        await session.emit("turn_complete")

    session.query = query
    session._sdk_session_id = "keep-context"
    with (
        TestClient(create_app(session)) as client,
        client.websocket_connect("/ws") as websocket,
    ):
        websocket.receive_json()
        websocket.send_json(
            {
                "action": "query",
                "text": "question",
                "model": "claude-sonnet-5",
                "effort": "xhigh",
            }
        )
        assert websocket.receive_json()["type"] == "user_message"
        assert websocket.receive_json()["type"] == "turn_complete"
        assert seen_settings == [("claude-sonnet-5", "xhigh")]
        assert session._sdk_session_id == "keep-context"
        assert session.kernel.restarts == 0
        websocket.send_json({"action": "query", "text": "question", "model": "unknown"})
        assert websocket.receive_json()["type"] == "error"
        websocket.send_json(
            {
                "action": "query",
                "text": "question",
                "model": "claude-sonnet-5",
                "effort": "extreme",
            }
        )
        assert websocket.receive_json()["type"] == "error"
        assert seen_settings == [("claude-sonnet-5", "xhigh")]


def test_scientific_prompt_is_saved_per_project_and_runtime_rules_stay_fixed(
    session, tmp_path
):
    original = session.workspace_path
    other = tmp_path / "other-project"
    other.mkdir()
    app = create_app(session, navigator_root=tmp_path)
    with TestClient(app) as client:
        initial = client.get("/api/workspace").json()
        assert initial["exploration_prompt"] == DEFAULT_EXPLORATION_PROMPT
        assert initial["exploration_prompt_is_custom"] is False

        assert client.put(
            "/api/settings/exploration-prompt", json={"prompt": "  Check the signal.  "}
        ).json() == {
            "exploration_prompt": "Check the signal.",
            "exploration_prompt_is_custom": True,
        }
        assert build_system_prompt(session.exploration_prompt) == (
            "Check the signal.\n\n" + RUNTIME_PROMPT.strip() + "\n"
        )
        assert (
            session.workspace.read_config()["exploration_prompt"] == "Check the signal."
        )
        assert (
            client.put(
                "/api/settings/exploration-prompt", json={"prompt": "   "}
            ).status_code
            == 400
        )

        with patch("nocturnomath.session.Kernel", FakeKernel):
            response = client.post("/api/workspace", json={"path": str(other)})
        assert response.status_code == 200
        assert client.get("/api/workspace").json()["exploration_prompt"] == (
            DEFAULT_EXPLORATION_PROMPT
        )
        with patch("nocturnomath.session.Kernel", FakeKernel):
            response = client.post("/api/workspace", json={"path": str(original)})
        assert response.status_code == 200
        assert client.get("/api/workspace").json()["exploration_prompt"] == (
            "Check the signal."
        )
        restored = client.put(
            "/api/settings/exploration-prompt", json={"prompt": None}
        ).json()
        assert restored["exploration_prompt"] == DEFAULT_EXPLORATION_PROMPT
        assert restored["exploration_prompt_is_custom"] is False


def test_workspace_defaults_apply_only_when_a_new_session_starts(session):
    app = create_app(session)
    with TestClient(app) as client:
        before = client.get("/api/workspace").json()
        assert before["model"] == "claude-opus-5"
        assert before["session_defaults"] == {
            "model": "claude-opus-5",
            "effort": "high",
        }

        changed = client.post(
            "/api/session/defaults",
            json={"model": "claude-sonnet-5", "effort": "xhigh"},
        )
        assert changed.status_code == 200
        assert changed.json()["session_defaults"] == {
            "model": "claude-sonnet-5",
            "effort": "xhigh",
        }
        assert session.model == "claude-opus-5"
        assert session.workspace.read_config()["default_model"] == "claude-sonnet-5"

        assert client.post("/api/session/new").status_code == 200
        after = client.get("/api/workspace").json()
        assert after["model"] == "claude-sonnet-5"
        assert after["effort"] == "xhigh"

        invalid = client.post(
            "/api/session/defaults",
            json={"model": "claude-sonnet-5", "effort": "max"},
        )
        assert invalid.status_code == 400


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
                "/api/auth",
                json={
                    "provider": "anthropic",
                    "method": "api_key",
                    "credential": "secret",
                },
            ).status_code
            == 409
        )
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
            assert init["workspace"]["thoughts_path"] == ".nocturnomath/kb/thoughts.md"
            websocket.send_json({"action": "query", "text": "/notes"})
            text = websocket.receive_json()["text"]
            assert "# Evidence" in text and "# Thoughts" in text and "<del>" in text


def test_documents_api_creates_edits_and_selects_documents(session):
    with TestClient(create_app(session)) as client:
        assert client.get("/api/documents").json() == {
            "documents": [],
            "folders": [],
            "documents_default": True,
        }

        created = client.post(
            "/api/documents", json={"title": "Inputs", "text": "# Inputs\n\nN = 10"}
        )
        assert created.status_code == 201
        assert created.json()["name"] == "Inputs.md"
        [document] = created.json()["documents"]
        assert document["name"] == "Inputs.md" and document["included"] is True
        assert (
            client.post("/api/documents", json={"title": "Inputs"}).status_code == 409
        )
        assert client.post("/api/documents", json={"title": " "}).status_code == 400

        read = client.get("/api/documents/Inputs.md")
        assert read.status_code == 200
        assert read.json()["content"] == "# Inputs\n\nN = 10\n"
        assert client.get("/api/documents/missing.md").status_code == 404
        assert client.get("/api/documents/..%2Fconfig.json").status_code in (400, 404)

        edited = client.put(
            "/api/documents/Inputs.md", json={"text": "# Inputs\n\nN = 20"}
        )
        assert edited.status_code == 200
        assert edited.json()["content"] == "# Inputs\n\nN = 20\n"

        toggled = client.post(
            "/api/documents/Inputs.md/include", json={"included": False}
        )
        assert toggled.json() == {"name": "Inputs.md", "included": False}
        assert client.get("/api/documents").json()["documents"][0]["included"] is False
        assert client.get("/api/workspace").json()["documents"][0]["included"] is False
        assert session.workspace.read_config()["document_choices"] == {
            "Inputs.md": False
        }

        assert client.delete("/api/documents/missing.md").status_code == 404
        session._sent_documents["Inputs.md"] = "# Inputs\n\nN = 20\n"
        rejected = client.delete(
            "/api/documents/Inputs.md", headers={"Origin": "https://example.com"}
        )
        assert rejected.status_code == 403
        assert (session.workspace.documents_path / "Inputs.md").is_file()
        deleted = client.delete("/api/documents/Inputs.md")
        assert deleted.status_code == 200
        assert deleted.json()["documents"] == []
        assert not (session.workspace.documents_path / "Inputs.md").exists()
        assert "Inputs.md" not in session.document_choices
        assert "Inputs.md" not in session._sent_documents

        client.post("/api/documents", json={"title": "Inputs", "text": "# Inputs"})
        changed = client.post("/api/documents/default", json={"enabled": False})
        assert changed.json()["documents_default"] is False
        assert changed.json()["documents"][0]["included"] is False
        config = json.loads(
            (session.workspace_path / ".nocturnomath" / "config.json").read_text()
        )
        assert config["documents_default"] is False


def test_documents_api_handles_nested_paths_and_empty_folders(session):
    with TestClient(create_app(session)) as client:
        created_folder = client.post(
            "/api/document-folders", json={"path": "reports/2026"}
        )
        assert created_folder.status_code == 201
        assert created_folder.json()["folders"] == ["reports", "reports/2026"]
        assert created_folder.json()["documents"] == []

        created = client.post(
            "/api/documents",
            json={"title": "First result", "text": "# First", "folder": "reports/2026"},
        )
        assert created.status_code == 201
        name = "reports/2026/First-result.md"
        assert created.json()["name"] == name
        assert created.json()["documents"][0]["name"] == name

        url = "/api/documents/reports%2F2026%2FFirst-result.md"
        assert client.get(url).json()["content"] == "# First\n"
        assert (
            client.put(url, json={"text": "# Revised"}).json()["content"]
            == "# Revised\n"
        )
        included = client.post(url + "/include", json={"included": False})
        assert included.status_code == 200
        assert included.json() == {"name": name, "included": False}
        assert client.get("/api/documents").json()["documents"][0]["included"] is False
        assert client.delete(url).status_code == 200
        assert client.get(url).status_code == 404
        assert client.get("/api/documents").json()["folders"] == [
            "reports",
            "reports/2026",
        ]

        for bad in ("../escape", ".hidden", "reports//bad"):
            assert (
                client.post("/api/document-folders", json={"path": bad}).status_code
                == 400
            )
        outside = "/api/documents/..%2Fkb%2Fevidence.md"
        assert client.get(outside).status_code in (400, 404)
        assert client.put(outside, json={"text": "unsafe"}).status_code in (400, 404)


def test_prompt_mode_setting_and_websocket_queue_steer_controls(session):
    entered_first = threading.Event()
    release_first = threading.Event()
    seen_queries = []

    async def fake_query(text):
        seen_queries.append(text)
        if len(seen_queries) == 1:
            entered_first.set()
            await asyncio.to_thread(release_first.wait, 5)
        await session.emit("assistant_text", text=f"reply:{text}")
        await session.emit("turn_complete", full_text=f"reply:{text}")

    session.query = fake_query

    def recv_until(ws, predicate):
        seen_events = []
        while True:
            event = ws.receive_json()
            seen_events.append(event)
            if predicate(event):
                return event, seen_events

    with (
        TestClient(create_app(session)) as client,
        client.websocket_connect("/ws") as websocket,
    ):
        init = websocket.receive_json()
        assert init["workspace"]["prompt_mode"] == "steer"
        assert client.get("/api/workspace").json()["prompt_mode"] == "steer"
        assert (
            client.put(
                "/api/settings/prompt-mode", json={"mode": "invalid"}
            ).status_code
            == 400
        )

        agent_id = init["workspace"]["agents"][0]["agent_id"]
        websocket.send_json({"action": "query", "agent_id": agent_id, "text": "first"})
        first_user = websocket.receive_json()
        assert (first_user["type"], first_user["text"]) == ("user_message", "first")
        assert entered_first.wait(timeout=5)

        websocket.send_json(
            {"action": "query", "agent_id": agent_id, "text": "steer-then-cancel"}
        )
        pending_evt, _ = recv_until(
            websocket, lambda e: e["type"] == "pending_prompts_changed"
        )
        assert pending_evt["pending_prompts"] == [
            {
                "id": pending_evt["pending_prompts"][0]["id"],
                "text": "steer-then-cancel",
                "mode": "steer",
            }
        ]
        first_id = pending_evt["pending_prompts"][0]["id"]

        updated = client.put("/api/settings/prompt-mode", json={"mode": "queue"})
        assert updated.status_code == 200
        assert updated.json() == {"prompt_mode": "queue"}
        assert session.workspace.read_config()["prompt_mode"] == "queue"

        observed = []
        while not (
            any(e["type"] == "prompt_mode_changed" for e in observed)
            and any(e["type"] == "pending_prompts_changed" for e in observed)
        ):
            observed.append(websocket.receive_json())
        mode_evt = next(e for e in observed if e["type"] == "prompt_mode_changed")
        assert mode_evt["prompt_mode"] == "queue"
        updated_pending = next(
            e for e in observed if e["type"] == "pending_prompts_changed"
        )
        assert updated_pending["pending_prompts"][0]["mode"] == "queue"

        websocket.send_json(
            {"action": "query", "agent_id": agent_id, "text": "keep-queued"}
        )
        two_pending, _ = recv_until(
            websocket, lambda e: e["type"] == "pending_prompts_changed"
        )
        assert [item["text"] for item in two_pending["pending_prompts"]] == [
            "steer-then-cancel",
            "keep-queued",
        ]

        websocket.send_json(
            {
                "action": "cancel_pending_prompt",
                "agent_id": agent_id,
                "prompt_id": first_id,
            }
        )
        after_cancel, _ = recv_until(
            websocket, lambda e: e["type"] == "pending_prompts_changed"
        )
        assert [item["text"] for item in after_cancel["pending_prompts"]] == [
            "keep-queued"
        ]

        release_first.set()
        _, tail = recv_until(
            websocket,
            lambda e: (
                e["type"] == "turn_complete"
                and e.get("full_text") == "reply:keep-queued"
            ),
        )
        user_messages = [e["text"] for e in tail if e["type"] == "user_message"]
        assert user_messages == ["keep-queued"]
        assert seen_queries == ["first", "keep-queued"]


def test_websocket_query_and_defaults_keep_provider_when_catalogue_collides(
    tmp_path, backend_model_catalog
):
    from nocturnomath.backends import SessionStarted, TextBlockDone, TurnResult

    workspace = tmp_path / "ws"
    workspace.mkdir()
    shared_openai = ModelSpec(
        id="shared",
        raw_id="shared",
        provider="openai",
        label="shared",
        efforts=["low", "high"],
        is_default=True,
    )
    shared_anthropic = ModelSpec(
        id="shared",
        raw_id="shared",
        provider="anthropic",
        label="shared",
        efforts=["low", "high"],
    )

    class RecordingBackend(FakeWebBackend):
        def __init__(self, provider, models=None):
            super().__init__(provider, models=models)
            self.calls = []

        async def run_turn(self, prompt, *, model, effort, **kwargs):
            self.turn_calls += 1
            self.calls.append((prompt, model, effort))
            yield SessionStarted(f"{self.provider}-1")
            yield TextBlockDone(f"{self.provider}:{model}")
            yield TurnResult()

    anthropic = RecordingBackend("anthropic", models=[])
    openai = RecordingBackend("openai", models=[shared_openai])
    backend_model_catalog["anthropic"] = anthropic
    backend_model_catalog["openai"] = openai
    backend_model_catalog["google"] = FakeWebBackend("google", models=[])

    with (
        patch("nocturnomath.session.Kernel", FakeKernel),
        TestClient(create_app(navigator_root=tmp_path)) as client,
        client.websocket_connect("/ws") as websocket,
    ):
        init = websocket.receive_json()
        assert init["workspace"]["is_open"] is False
        assert (
            client.post("/api/workspace", json={"path": str(workspace)}).status_code
            == 200
        )
        ws_updated = websocket.receive_json()
        assert ws_updated["type"] == "workspace_updated"
        assert ws_updated["workspace"]["models"] == ["shared"]
        assert ws_updated["workspace"]["model_providers"] == {"shared": "openai"}
        agent_id = ws_updated["workspace"]["agents"][0]["agent_id"]

        # Publish a colliding "shared" from Anthropic so OpenAI's moves to "openai/shared".
        anthropic.models = [shared_anthropic]
        assert (
            client.post(
                "/api/auth", json={"provider": "anthropic", "method": "cli"}
            ).status_code
            == 200
        )
        auth_evt = websocket.receive_json()
        assert auth_evt["type"] == "auth_changed"
        assert auth_evt["models"] == ["shared", "openai/shared"]
        assert auth_evt["model_providers"] == {
            "shared": "anthropic",
            "openai/shared": "openai",
        }

        # Submit a stale-catalogue payload targeting OpenAI's "shared": must go to OpenAI.
        websocket.send_json(
            {
                "action": "query",
                "agent_id": agent_id,
                "text": "hello openai",
                "model": "shared",
                "provider": "openai",
                "raw_model": "shared",
                "effort": "high",
            }
        )
        while True:
            event = websocket.receive_json()
            assert event["type"] != "error", event
            if event["type"] == "turn_complete":
                break
        assert anthropic.turn_calls == 0
        assert len(openai.calls) == 1
        assert openai.calls[0][1:] == ("shared", "high")
        assert client.app.state.runtime.get_agent(agent_id).model == "openai/shared"

        # Session defaults with the same stale OpenAI payload also resolve to "openai/shared".
        defaults_res = client.post(
            "/api/session/defaults",
            json={
                "model": "shared",
                "provider": "openai",
                "raw_model": "shared",
                "effort": "high",
            },
        )
        assert defaults_res.status_code == 200
        assert defaults_res.json()["session_defaults"] == {
            "model": "openai/shared",
            "effort": "high",
        }
        assert workspace.joinpath(".nocturnomath/config.json").exists()
        assert client.app.state.runtime.get_agent(agent_id).default_provider == "openai"

        # Drain idle status before testing removed provider rejection.
        while client.app.state.runtime.get_agent(agent_id).has_active_query():
            pass

        # If OpenAI drops "shared", the stale OpenAI payload must error instead of hitting Anthropic.
        openai.models = []
        assert (
            client.post(
                "/api/auth", json={"provider": "openai", "method": "cli"}
            ).status_code
            == 200
        )
        while websocket.receive_json()["type"] != "auth_changed":
            pass
        websocket.send_json(
            {
                "action": "query",
                "agent_id": agent_id,
                "text": "should be rejected",
                "model": "shared",
                "provider": "openai",
                "raw_model": "shared",
                "effort": "high",
            }
        )
        err = websocket.receive_json()
        while err["type"] != "error":
            err = websocket.receive_json()
        assert "Choose a model from the model selector." in err["message"]
        assert anthropic.turn_calls == 0


def test_prefixed_raw_model_ids_and_session_reset_resume_payloads(
    tmp_path, backend_model_catalog
):
    from nocturnomath.backends import SessionStarted, TextBlockDone, TurnResult

    workspace = tmp_path / "ws_raw"
    workspace.mkdir()
    oss_spec = ModelSpec(
        id="openai/gpt-oss-120b",
        raw_id="openai/gpt-oss-120b",
        provider="openai",
        label="GPT-OSS 120B",
        efforts=["low", "high"],
        is_default=True,
    )

    class PrefixedOpenAI(FakeWebBackend):
        async def run_turn(self, prompt, *, model, effort, **kwargs):
            self.turn_calls += 1
            assert model == "openai/gpt-oss-120b"
            yield SessionStarted("oai-oss-1")
            yield TextBlockDone("oss-ok")
            yield TurnResult()

    backend_model_catalog["anthropic"] = FakeWebBackend("anthropic", models=[])
    backend_model_catalog["openai"] = PrefixedOpenAI("openai", models=[oss_spec])
    backend_model_catalog["google"] = FakeWebBackend("google", models=[])

    with (
        patch("nocturnomath.session.Kernel", FakeKernel),
        TestClient(create_app(navigator_root=tmp_path)) as client,
        client.websocket_connect("/ws") as websocket,
    ):
        websocket.receive_json()
        ws_res = client.post("/api/workspace", json={"path": str(workspace)})
        assert ws_res.status_code == 200
        snap = ws_res.json()["workspace"]
        assert snap["session_id"] is None
        assert snap["model_raw_ids"] == {"openai/gpt-oss-120b": "openai/gpt-oss-120b"}
        assert client.get("/api/auth").json()["model_raw_ids"] == {
            "openai/gpt-oss-120b": "openai/gpt-oss-120b"
        }
        ws_evt = websocket.receive_json()
        agent_id = ws_evt["workspace"]["agents"][0]["agent_id"]

        websocket.send_json(
            {
                "action": "query",
                "agent_id": agent_id,
                "text": "create S001",
                "model": "openai/gpt-oss-120b",
                "provider": "openai",
                "raw_model": "openai/gpt-oss-120b",
                "effort": "high",
            }
        )
        while websocket.receive_json()["type"] != "turn_complete":
            pass
        while client.app.state.runtime.get_agent(agent_id).has_active_query():
            pass

        reset_res = client.post("/api/session/new")
        assert reset_res.status_code == 200
        reset_evt = websocket.receive_json()
        while reset_evt["type"] != "session_reset":
            reset_evt = websocket.receive_json()
        assert reset_evt["agent_id"] == agent_id
        assert reset_evt["session_id"] is None

        resume_res = client.post("/api/session/resume", json={"id": "S001"})
        assert resume_res.status_code == 200
        body = resume_res.json()
        assert body["session_id"] == "S001"
        assert body["can_compact"] is True
        assert body["agent_id"] == agent_id
        assert body["status"] == "idle"
        assert body["is_busy"] is False
        assert body["model"] == "openai/gpt-oss-120b"
        assert body["effort"] == "high"


def test_kernel_restart_conflicts_while_primary_agent_changing_and_user_message_carries_model(
    tmp_path, backend_model_catalog
):
    from nocturnomath.backends import SessionStarted, TextBlockDone, TurnResult

    workspace = tmp_path / "ws_state"
    workspace.mkdir()
    spec = ModelSpec(
        id="claude-test",
        raw_id="claude-test",
        provider="anthropic",
        label="Claude Test",
        efforts=["low", "medium"],
        is_default=True,
    )

    class SimpleBackend(FakeWebBackend):
        async def run_turn(self, prompt, **kwargs):
            yield SessionStarted("ant-1")
            yield TextBlockDone("ok")
            yield TurnResult()

    backend_model_catalog["anthropic"] = SimpleBackend("anthropic", models=[spec])
    backend_model_catalog["openai"] = FakeWebBackend("openai", models=[])
    backend_model_catalog["google"] = FakeWebBackend("google", models=[])

    with (
        patch("nocturnomath.session.Kernel", FakeKernel),
        TestClient(create_app(navigator_root=tmp_path)) as client,
        client.websocket_connect("/ws") as websocket,
    ):
        websocket.receive_json()
        assert (
            client.post("/api/workspace", json={"path": str(workspace)}).status_code
            == 200
        )
        websocket.receive_json()
        rt = client.app.state.runtime

        rt._changing_agents.add(rt.primary_agent_id)
        try:
            res = client.post("/api/kernel/restart")
            assert res.status_code == 409
        finally:
            rt._changing_agents.discard(rt.primary_agent_id)

        with patch.object(
            rt,
            "transition_agent",
            AsyncMock(side_effect=RuntimeError("already changing")),
        ):
            res2 = client.post("/api/kernel/restart")
            assert res2.status_code == 409

        websocket.send_json(
            {
                "action": "query",
                "agent_id": rt.primary_agent_id,
                "text": "hello",
                "model": "claude-test",
                "provider": "anthropic",
                "raw_model": "claude-test",
                "effort": "medium",
            }
        )
        seen = None
        for _ in range(20):
            evt = websocket.receive_json()
            if evt["type"] == "user_message":
                seen = evt
                break
        assert seen is not None
        assert seen["model"] == "claude-test"
        assert seen["effort"] == "medium"
        while websocket.receive_json()["type"] != "turn_complete":
            pass
        while rt.get_agent(rt.primary_agent_id).has_active_query():
            pass


@pytest.mark.asyncio
async def test_web_runtime_broadcast_handles_set_mutation(session):
    from nocturnomath.runtime import Runtime
    from nocturnomath.web.runtime import WebRuntime

    core = Runtime(session)
    web_rt = WebRuntime(core)
    delivered = []

    class MutatingSocket:
        def __init__(self, name, remove_other=None):
            self.name = name
            self.remove_other = remove_other

        async def send_json(self, message):
            delivered.append((self.name, message["type"]))
            if self.remove_other is not None:
                web_rt.websockets.discard(self.remove_other)

    other = MutatingSocket("other")
    first = MutatingSocket("first", remove_other=other)
    web_rt.websockets.update({first, other})

    await web_rt.broadcast("ping", {"value": 1})
    assert ("first", "ping") in delivered


def test_websocket_interrupt_is_nonblocking_and_second_stop_breaks_shield(
    tmp_path, backend_model_catalog
):
    from nocturnomath.backends import SessionStarted, TurnResult

    workspace = tmp_path / "ws_ws_int"
    workspace.mkdir()
    entered = threading.Event()
    unwinding = threading.Event()
    spec = ModelSpec(
        id="claude-test",
        raw_id="claude-test",
        provider="anthropic",
        label="Claude Test",
        is_default=True,
    )

    class ShieldedCancelBackend(FakeWebBackend):
        async def run_turn(self, prompt, **kwargs):
            entered.set()
            blocker = asyncio.create_task(asyncio.sleep(30))
            try:
                yield SessionStarted("ant-shield")
                await asyncio.Event().wait()
                yield TurnResult()
            except asyncio.CancelledError:
                unwinding.set()
                await asyncio.shield(blocker)
                raise
            finally:
                if not blocker.done():
                    blocker.cancel()
                    try:
                        await blocker
                    except asyncio.CancelledError:
                        pass

    backend_model_catalog["anthropic"] = ShieldedCancelBackend(
        "anthropic", models=[spec]
    )
    backend_model_catalog["openai"] = FakeWebBackend("openai", models=[])
    backend_model_catalog["google"] = FakeWebBackend("google", models=[])

    with (
        patch("nocturnomath.session.Kernel", FakeKernel),
        TestClient(create_app(navigator_root=tmp_path)) as client,
        client.websocket_connect("/ws") as websocket,
    ):
        websocket.receive_json()
        assert (
            client.post("/api/workspace", json={"path": str(workspace)}).status_code
            == 200
        )
        websocket.receive_json()
        rt = client.app.state.runtime
        agent_id = rt.primary_agent_id

        websocket.send_json(
            {
                "action": "query",
                "agent_id": agent_id,
                "text": "hang in shielded unwind",
                "model": "claude-test",
                "provider": "anthropic",
                "raw_model": "claude-test",
            }
        )
        assert entered.wait(timeout=5)

        # First Stop enters unwinding inside asyncio.shield(blocker); WS loop must stay responsive.
        websocket.send_json({"action": "interrupt", "agent_id": agent_id})
        assert unwinding.wait(timeout=5)

        websocket.send_json({"action": "query", "agent_id": agent_id, "text": "/notes"})
        notes_evt = None
        for _ in range(20):
            evt = websocket.receive_json()
            if evt["type"] == "system_message":
                notes_evt = evt
                break
        assert notes_evt is not None

        # Second Stop breaks out of the shielded unwind cleanly.
        websocket.send_json({"action": "interrupt", "agent_id": agent_id})
        cancelled_seen = False
        idle_seen = False
        for _ in range(20):
            evt = websocket.receive_json()
            if evt["type"] == "status_change" and evt["status"] == "cancelled":
                cancelled_seen = True
            if evt["type"] == "status_change" and evt["status"] == "idle":
                idle_seen = True
                break
        assert cancelled_seen and idle_seen
        assert not rt.get_agent(agent_id).has_active_query()


def test_agent_snapshot_is_idle_immediately_upon_idle_status_change(
    tmp_path, backend_model_catalog
):
    from nocturnomath.backends import SessionStarted, TextBlockDone, TurnResult

    workspace = tmp_path / "ws_turn_idle"
    workspace.mkdir()
    spec = ModelSpec(
        id="claude-test",
        raw_id="claude-test",
        provider="anthropic",
        label="Claude Test",
        is_default=True,
    )

    class SimpleTurnBackend(FakeWebBackend):
        async def run_turn(self, prompt, **kwargs):
            yield SessionStarted("ant-turn")
            yield TextBlockDone("Done.")
            yield TurnResult(
                context_usage={"used_tokens": 120, "window_tokens": 200000}
            )

    backend_model_catalog["anthropic"] = SimpleTurnBackend("anthropic", models=[spec])
    backend_model_catalog["openai"] = FakeWebBackend("openai", models=[])
    backend_model_catalog["google"] = FakeWebBackend("google", models=[])

    with (
        patch("nocturnomath.session.Kernel", FakeKernel),
        TestClient(create_app(navigator_root=tmp_path)) as client,
        client.websocket_connect("/ws") as websocket,
    ):
        websocket.receive_json()
        assert (
            client.post("/api/workspace", json={"path": str(workspace)}).status_code
            == 200
        )
        websocket.receive_json()
        agent_id = client.app.state.runtime.primary_agent_id

        websocket.send_json(
            {
                "action": "query",
                "agent_id": agent_id,
                "text": "check snapshot on idle status_change",
                "model": "claude-test",
                "provider": "anthropic",
                "raw_model": "claude-test",
            }
        )
        while True:
            evt = websocket.receive_json()
            if evt["type"] == "status_change" and evt.get("status") == "idle":
                break
        snap = client.get(f"/api/agents/{agent_id}").json()
        assert snap["is_busy"] is False
        assert snap["status"] == "idle"
        assert snap["can_compact"] is True


def test_api_auth_reports_provider_discovery_errors(backend_model_catalog):
    backend_model_catalog["google"].models = []
    backend_model_catalog["google"].last_discovery_error = "Invalid Gemini API key"
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/auth",
            json={
                "provider": "google",
                "method": "api_key",
                "credential": "AIza-bad-key",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert (
            payload["auth"]["providers"]["google"]["error"] == "Invalid Gemini API key"
        )
        assert "AIza-bad-key" not in response.text
        fetched = client.get("/api/auth").json()
        assert (
            fetched["auth"]["providers"]["google"]["error"] == "Invalid Gemini API key"
        )
