import asyncio
import threading
from unittest.mock import AsyncMock, patch

import pytest

from nocturnomath.runtime import Runtime


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
async def test_start_query_validates_the_model_and_reports_failures(session):
    runtime = Runtime(session)
    runtime.start()
    events = []
    runtime.subscribe(lambda event_type, payload: events.append((event_type, payload)))

    async def query(text):
        raise ValueError("query exploded")

    session.query = query
    runtime.models = ["sonnet"]

    with pytest.raises(ValueError, match="model selector"):
        runtime.start_query("question", "unknown")
    assert session.model != "unknown"

    runtime.start_query("question", "sonnet")
    assert session.model == "sonnet"
    assert runtime.model == "sonnet"
    await asyncio.gather(session._current_task, return_exceptions=True)

    assert ("error", {"message": "query exploded"}) in events
    assert session._current_task is None
    kinds = [
        record["kind"]
        for record in session.workspace.read_transcript(session.transcript_path)
    ]
    assert "meta" in kinds


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
async def test_auth_selection_reaches_sdk_and_resets_resumed_context(session):
    runtime = Runtime(session)
    session._sdk_session_id = "previous-account-session"
    with patch("nocturnomath.runtime.ClaudeSDKClient") as sdk:
        sdk.return_value.__aenter__.return_value.get_server_info = AsyncMock(
            return_value={
                "models": [
                    {"value": "default"},
                    {"value": "sonnet", "resolvedModel": "claude-sonnet-5"},
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
    assert session._sdk_session_id is None
    assert runtime.models == ["sonnet"]


@pytest.mark.asyncio
async def test_cancelled_auth_selection_restores_credentials_catalog_and_context(
    session,
):
    runtime = Runtime(session)
    previous = runtime.auth
    session._sdk_session_id = "existing-session"
    runtime.models = ["existing-model"]

    async def discover_models():
        runtime.models = []
        raise asyncio.CancelledError()

    runtime.discover_models = discover_models
    with pytest.raises(asyncio.CancelledError):
        await runtime.authenticate("api_key", "bad-key")

    assert runtime.auth is previous
    assert session.auth is previous
    assert session._sdk_session_id == "existing-session"
    assert runtime.models == ["existing-model"]
    assert runtime.changing is False


@pytest.mark.asyncio
async def test_return_to_automatic_works_without_model_catalog(session):
    from nocturnomath.auth import ClaudeAuth

    session.set_auth(ClaudeAuth.interactive("api_key", "manual-key"))
    runtime = Runtime(session)
    runtime.discover_models = AsyncMock(return_value=False)
    result = await runtime.authenticate("claude_code")
    assert result["method"] == "claude_code"
    assert session.auth.sdk_env() == {}
    assert session.auth is runtime.auth


@pytest.mark.asyncio
async def test_reselecting_same_auth_preserves_context(session):
    runtime = Runtime(session)
    runtime.discover_models = AsyncMock(return_value=True)
    session._sdk_session_id = "existing-session"
    await runtime.authenticate("claude_code")
    assert session._sdk_session_id == "existing-session"
