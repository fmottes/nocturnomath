import asyncio
import os
import threading
from unittest.mock import AsyncMock, patch

import pytest

from nocturnomath.runtime import Runtime
from nocturnomath.session import CLAUDE_CLI_NAME, claude_cli_candidates


@pytest.mark.asyncio
async def test_subscribers_receive_session_events(session):
    runtime = Runtime(session)
    runtime.start()
    events = []
    failures = []

    def broken(event_type, payload):
        failures.append(event_type)
        raise RuntimeError("subscriber is broken")

    async def collect(event_type, payload):
        events.append((event_type, payload))

    runtime.subscribe(broken)
    runtime.subscribe(collect)
    await session.emit("assistant_text", text="hello")

    assert failures == ["assistant_text"]
    assert events[0][0] == "assistant_text"
    assert events[0][1]["text"] == "hello"

    runtime.unsubscribe(collect)
    await session.emit("turn_complete")
    assert len(events) == 1


@pytest.mark.asyncio
async def test_start_query_validates_model_and_effort_and_reports_failures(session):
    runtime = Runtime(session)
    runtime.start()
    events = []
    runtime.subscribe(lambda event_type, payload: events.append((event_type, payload)))

    async def query(text):
        raise ValueError("query exploded")

    session.query = query
    runtime.models = ["sonnet"]
    runtime.model_efforts = {"sonnet": ["low", "medium", "high", "xhigh"]}

    with pytest.raises(ValueError, match="model selector"):
        runtime.start_query("question", "unknown")
    assert session.model != "unknown"

    with pytest.raises(ValueError, match="effort supported"):
        runtime.start_query("question", "sonnet", "extreme")
    assert session.model == "test"
    assert session.effort == "high"

    runtime.start_query("question", "sonnet", "low")
    assert session.model == "sonnet"
    assert session.effort == "low"
    assert runtime.model == "sonnet"
    assert runtime.effort == "low"
    await asyncio.gather(session._current_task, return_exceptions=True)

    assert ("error", {"message": "query exploded"}) in events
    assert session._current_task is None
    kinds = [
        record["kind"]
        for record in session.workspace.read_transcript(session.transcript_path)
    ]
    assert "meta" in kinds


def test_start_query_requires_a_discovered_model(session):
    session.model = None
    runtime = Runtime(session)

    with pytest.raises(ValueError, match="No model available"):
        runtime.start_query("question")


@pytest.mark.asyncio
async def test_transition_rejects_concurrent_changes(session):
    runtime = Runtime(session)
    entered = threading.Event()
    release = threading.Event()

    def prepare():
        entered.set()
        release.wait(5)

    task = asyncio.create_task(runtime.transition(prepare))
    try:
        await asyncio.to_thread(entered.wait, 5)
        with pytest.raises(RuntimeError, match="prepared"):
            await runtime.transition(prepare)
        with pytest.raises(RuntimeError, match="prepared"):
            runtime.start_query("question")
    finally:
        release.set()
        await task
    assert not runtime.changing

    session._is_busy = True
    with pytest.raises(RuntimeError, match="running a query"):
        await runtime.transition(prepare)


@pytest.mark.asyncio
async def test_agent_transition_blocks_queries_and_other_transitions(session):
    runtime = Runtime(session)
    entered = threading.Event()
    release = threading.Event()
    agent_id = runtime.primary_agent_id

    def prepare():
        entered.set()
        release.wait(5)

    task = asyncio.create_task(runtime.transition_agent(agent_id, prepare))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert runtime.agent_snapshot(agent_id)["is_busy"] is True
        with pytest.raises(RuntimeError, match="changing"):
            runtime.start_query("question", agent_id=agent_id)
        with pytest.raises(RuntimeError, match="already changing"):
            await runtime.transition_agent(agent_id, prepare)
        with pytest.raises(RuntimeError, match="explorer is changing"):
            await runtime.transition(prepare)
    finally:
        release.set()
        await task
    assert runtime.agent_snapshot(agent_id)["is_busy"] is False


def test_export_notebook_names_the_file_after_workspace_and_session(session):
    runtime = Runtime(session)

    with pytest.raises(ValueError, match="no messages"):
        runtime.export_notebook()

    session.log_transcript("user", text="question")
    filename, notebook = runtime.export_notebook()

    assert filename == f"nocturnomath-{session.workspace_path.name}-S001.ipynb"
    assert notebook["cells"]

    with pytest.raises(RuntimeError, match="Open a workspace"):
        Runtime().export_notebook()


@pytest.mark.asyncio
async def test_discovery_applies_startup_model_only_if_reported():
    runtime = Runtime(model="unavailable-model")
    with patch("nocturnomath.runtime.ClaudeSDKClient") as sdk:
        sdk.return_value.__aenter__.return_value.get_server_info = AsyncMock(
            return_value={
                "models": [{"value": "default", "resolvedModel": "claude-sonnet-5"}]
            }
        )
        await runtime.discover_models()

    assert sdk.call_args.args[0].model is None
    assert runtime.model == "claude-sonnet-5"


def test_cli_lookup_excludes_launch_directory_and_relative_path(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    relative_bin = workspace / "bin"
    installed_bin = tmp_path / "installed"
    for directory in (workspace, relative_bin, installed_bin):
        directory.mkdir()
        path = directory / CLAUDE_CLI_NAME
        path.touch(mode=0o700)

    monkeypatch.chdir(workspace)
    monkeypatch.setenv(
        "PATH", os.pathsep.join(("", str(workspace), "bin", str(installed_bin)))
    )

    assert claude_cli_candidates()[0] == installed_bin / CLAUDE_CLI_NAME


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_discovery",
    [
        pytest.param(RuntimeError("CLI unavailable"), id="failed-handshake"),
        pytest.param({"commands": []}, id="missing-catalogue"),
        pytest.param({"models": []}, id="empty-catalogue"),
        pytest.param({"models": [None, {"value": "default"}]}, id="unusable-models"),
    ],
)
async def test_cli_fallback_is_used_for_existing_and_new_chats(
    session, tmp_path, failed_discovery
):
    installed = tmp_path / "installed" / CLAUDE_CLI_NAME
    bundled = tmp_path / "bundled" / CLAUDE_CLI_NAME
    runtime = Runtime(session)

    def catalog(identifier):
        return {"models": [{"value": "default", "resolvedModel": identifier}]}

    with (
        patch(
            "nocturnomath.runtime.claude_cli_candidates",
            return_value=[installed, bundled],
        ),
        patch("nocturnomath.runtime.ClaudeSDKClient") as sdk,
    ):
        info = sdk.return_value.__aenter__.return_value.get_server_info
        info.side_effect = [
            catalog("local-model"),
            failed_discovery,
            catalog("bundled-model"),
        ]
        await runtime.discover_models()
        assert session._client_options().cli_path == installed
        assert session.model == "local-model"

        await runtime.authenticate("api_key", "unverified-key")

    assert [args.args[0].cli_path for args in sdk.call_args_list] == [
        installed,
        installed,
        bundled,
    ]
    assert runtime.models == ["bundled-model"]
    assert session.model == "bundled-model"
    assert session._client_options().cli_path == bundled
    new_session = runtime.get_agent(runtime.create_agent())
    assert new_session._client_options().cli_path == bundled


@pytest.mark.asyncio
async def test_catalogue_refresh_revalidates_chats_using_the_same_cli(
    session, tmp_path
):
    installed = tmp_path / "installed" / CLAUDE_CLI_NAME
    runtime = Runtime(session)
    common_model = {"value": "shared-model", "supportedEffortLevels": ["high"]}

    with (
        patch("nocturnomath.runtime.claude_cli_candidates", return_value=[installed]),
        patch("nocturnomath.runtime.ClaudeSDKClient") as sdk,
    ):
        info = sdk.return_value.__aenter__.return_value.get_server_info
        info.side_effect = [
            {
                "models": [
                    {"value": "default", "resolvedModel": "old-model"},
                    common_model,
                ]
            },
            {
                "models": [
                    {
                        "value": "default",
                        "resolvedModel": "new-model",
                        "supportedEffortLevels": ["medium"],
                    },
                    {**common_model, "supportedEffortLevels": ["low"]},
                ]
            },
        ]
        await runtime.discover_models()
        assert session.model == "old-model"
        other = runtime.get_agent(runtime.create_agent())
        other.model = "shared-model"
        other.effort = "high"
        session._sdk_session_id = "resumed-session"

        await runtime.authenticate("api_key", "unverified-key")

    options = session._client_options()
    assert (options.cli_path, options.model, options.effort, options.resume) == (
        installed,
        "new-model",
        "medium",
        "resumed-session",
    )
    assert (other.model, other.effort) == ("shared-model", "low")


@pytest.mark.asyncio
async def test_auth_selection_reaches_sdk_and_keeps_context(session):
    runtime = Runtime(session)
    session._sdk_session_id = "resumed-session"
    with patch("nocturnomath.runtime.ClaudeSDKClient") as sdk:
        sdk.return_value.__aenter__.return_value.get_server_info = AsyncMock(
            return_value={
                "models": [
                    {"value": "default"},
                    {
                        "value": "sonnet",
                        "resolvedModel": "claude-sonnet-5",
                        "supportedEffortLevels": ["low", "medium", "high"],
                    },
                ]
            }
        )
        result = await runtime.authenticate("subscription", "oauth-secret")

    options = sdk.call_args.args[0]
    assert options.env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-secret"
    assert options.env["ANTHROPIC_API_KEY"] == ""
    assert result["method"] == "subscription"
    assert "oauth-secret" not in repr(result)
    assert session.auth is runtime.auth
    assert session._sdk_session_id == "resumed-session"
    assert runtime.models == ["claude-sonnet-5"]


@pytest.mark.asyncio
async def test_auth_selection_survives_failed_model_discovery(session):
    runtime = Runtime(session)
    runtime.models = ["existing-model"]
    with patch("nocturnomath.runtime.ClaudeSDKClient") as sdk:
        sdk.return_value.__aenter__.side_effect = RuntimeError("CLI unavailable")
        result = await runtime.authenticate("api_key", "unverified-key")

    assert result["method"] == "api_key"
    assert session.auth.sdk_env()["ANTHROPIC_API_KEY"] == "unverified-key"
    assert runtime.models == []
    assert not runtime.changing


@pytest.mark.asyncio
async def test_auth_change_waits_for_idle_and_returns_to_automatic(session):
    runtime = Runtime(session)
    runtime.discover_models = AsyncMock()
    blocker = asyncio.Event()
    session._current_task = asyncio.create_task(blocker.wait())
    with pytest.raises(RuntimeError):
        await runtime.authenticate("api_key", "manual-key")
    assert session.auth.method == "claude_code"
    blocker.set()
    await session._current_task
    session._current_task = None

    await runtime.authenticate("api_key", "manual-key")
    assert session.auth.method == "api_key"
    result = await runtime.authenticate("claude_code")
    assert result["method"] == "claude_code"
    assert session.auth.sdk_env() == {}
    assert session.auth is runtime.auth


def test_runtime_and_session_share_one_auth(session, tmp_path):
    from nocturnomath.auth import ClaudeAuth

    manual = ClaudeAuth.interactive("api_key", "manual-key")
    runtime = Runtime(session, auth=manual)
    assert session.auth is manual
    assert runtime.auth is manual

    received = {}

    def factory(**kwargs):
        received.update(kwargs)
        return session

    fresh = Runtime(session_factory=factory, auth=manual)
    fresh.open_workspace(tmp_path)
    assert received["auth"] is manual
