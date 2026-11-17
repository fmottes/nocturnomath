import asyncio
import os
import threading
from unittest.mock import AsyncMock

import pytest

from nocturnomath.auth import AuthStore
from nocturnomath.backends import ModelSpec, find_cli_executables
from nocturnomath.backends.anthropic import CLAUDE_CLI_NAME, claude_cli_candidates
from nocturnomath.runtime import Runtime


class StubBackend:
    def __init__(self, provider, responses=None, error=None):
        self.provider = provider
        self.uses_compact_hook = provider == "anthropic"
        self.responses = list(responses or [[]])
        self.error = error
        self.auth_seen = []

    async def discover_models(self, auth):
        self.auth_seen.append(auth)
        if self.error is not None:
            raise self.error
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]

    async def run_turn(self, *args, **kwargs):
        if False:
            yield None

    async def compact(self, *args, **kwargs):
        return None


def stub_backends(anthropic=None, openai=None, google=None):
    return {
        "anthropic": anthropic or StubBackend("anthropic"),
        "openai": openai or StubBackend("openai"),
        "google": google or StubBackend("google"),
    }


@pytest.mark.asyncio
async def test_subscribers_receive_session_events(session):
    runtime = Runtime(session, backends=stub_backends())
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
    runtime = Runtime(session, backends=stub_backends())
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
    runtime = Runtime(session, backends=stub_backends())

    with pytest.raises(ValueError, match="No model available"):
        runtime.start_query("question")


@pytest.mark.asyncio
async def test_transition_rejects_concurrent_changes(session):
    runtime = Runtime(session, backends=stub_backends())
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
    runtime = Runtime(session, backends=stub_backends())
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
    runtime = Runtime(session, backends=stub_backends())

    with pytest.raises(ValueError, match="no messages"):
        runtime.export_notebook()

    session.log_transcript("user", text="question")
    filename, notebook = runtime.export_notebook()

    assert filename == f"nocturnomath-{session.workspace_path.name}-S001.ipynb"
    assert notebook["cells"]

    with pytest.raises(RuntimeError, match="Open a workspace"):
        Runtime(backends=stub_backends()).export_notebook()


@pytest.mark.asyncio
async def test_discovery_applies_startup_model_only_if_reported():
    runtime = Runtime(
        model="unavailable-model",
        backends=stub_backends(
            anthropic=StubBackend(
                "anthropic",
                [
                    [
                        ModelSpec(
                            id="claude-sonnet-5",
                            raw_id="claude-sonnet-5",
                            provider="anthropic",
                            label="claude-sonnet-5",
                            is_default=True,
                        )
                    ]
                ],
            )
        ),
    )
    await runtime.discover_models()
    assert runtime.model == "claude-sonnet-5"


@pytest.mark.asyncio
async def test_discovery_merges_all_providers_and_survives_partial_failure():
    runtime = Runtime(
        backends=stub_backends(
            anthropic=StubBackend(
                "anthropic",
                [
                    [
                        ModelSpec(
                            id="claude-opus-5",
                            raw_id="claude-opus-5",
                            provider="anthropic",
                            label="claude-opus-5",
                            efforts=["low", "high"],
                            is_default=True,
                        )
                    ]
                ],
            ),
            openai=StubBackend("openai", error=RuntimeError("codex missing")),
            google=StubBackend(
                "google",
                [
                    [
                        ModelSpec(
                            id="gemini-4-carbon",
                            raw_id="gemini-4-carbon",
                            provider="google",
                            label="Gemini 4 Carbon",
                            efforts=["medium", "high"],
                        )
                    ]
                ],
            ),
        )
    )
    await runtime.discover_models()
    assert runtime.models == ["claude-opus-5", "gemini-4-carbon"]
    assert runtime.model_providers == {
        "claude-opus-5": "anthropic",
        "gemini-4-carbon": "google",
    }
    assert runtime.model_labels["gemini-4-carbon"] == "Gemini 4 Carbon"
    assert runtime.model == "claude-opus-5"


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

    assert find_cli_executables(CLAUDE_CLI_NAME)[0] == installed_bin / CLAUDE_CLI_NAME
    assert claude_cli_candidates()[0] == installed_bin / CLAUDE_CLI_NAME


@pytest.mark.asyncio
async def test_catalogue_refresh_revalidates_chats_and_propagates_providers(session):
    anthropic = StubBackend(
        "anthropic",
        responses=[
            [
                ModelSpec(
                    id="old-model",
                    raw_id="old-model",
                    provider="anthropic",
                    label="old-model",
                    is_default=True,
                ),
                ModelSpec(
                    id="shared-model",
                    raw_id="shared-model",
                    provider="anthropic",
                    label="shared-model",
                    efforts=["high"],
                ),
            ],
            [
                ModelSpec(
                    id="new-model",
                    raw_id="new-model",
                    provider="anthropic",
                    label="new-model",
                    efforts=["medium"],
                    is_default=True,
                ),
                ModelSpec(
                    id="shared-model",
                    raw_id="shared-model",
                    provider="anthropic",
                    label="shared-model",
                    efforts=["low"],
                ),
            ],
        ],
    )
    runtime = Runtime(session, backends=stub_backends(anthropic=anthropic))
    await runtime.discover_models()
    assert session.model == "old-model"
    other = runtime.get_agent(runtime.create_agent())
    other.model = "shared-model"
    other.effort = "high"
    session._sdk_session_id = "resumed-session"

    await runtime.authenticate("anthropic", "api_key", "unverified-key")

    assert (session.model, session.effort, session._sdk_session_id) == (
        "new-model",
        "medium",
        "resumed-session",
    )
    assert (other.model, other.effort) == ("shared-model", "low")
    assert session.model_providers["new-model"] == "anthropic"


@pytest.mark.asyncio
async def test_auth_selection_reaches_backend_and_keeps_context(session):
    anthropic = StubBackend(
        "anthropic",
        responses=[
            [
                ModelSpec(
                    id="claude-sonnet-5",
                    raw_id="claude-sonnet-5",
                    provider="anthropic",
                    label="claude-sonnet-5",
                    efforts=["low", "medium", "high"],
                )
            ]
        ],
    )
    runtime = Runtime(session, backends=stub_backends(anthropic=anthropic))
    session._sdk_session_id = "resumed-session"

    result = await runtime.authenticate("anthropic", "subscription", "oauth-secret")

    seen_auth = anthropic.auth_seen[-1]
    assert seen_auth.sdk_env()["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-secret"
    assert seen_auth.sdk_env()["ANTHROPIC_API_KEY"] == ""
    assert result["providers"]["anthropic"]["method"] == "subscription"
    assert "oauth-secret" not in repr(result)
    assert session.auth is runtime.auth
    assert session._sdk_session_id == "resumed-session"
    assert runtime.models == ["claude-sonnet-5"]


@pytest.mark.asyncio
async def test_auth_selection_survives_failed_model_discovery(session):
    runtime = Runtime(
        session,
        backends=stub_backends(
            anthropic=StubBackend("anthropic", error=RuntimeError("CLI unavailable")),
            openai=StubBackend("openai", error=RuntimeError("CLI unavailable")),
            google=StubBackend("google", error=RuntimeError("CLI unavailable")),
        ),
    )
    runtime.models = ["existing-model"]

    result = await runtime.authenticate("anthropic", "api_key", "unverified-key")

    assert result["providers"]["anthropic"]["method"] == "api_key"
    assert (
        session.auth.get("anthropic").sdk_env()["ANTHROPIC_API_KEY"] == "unverified-key"
    )
    assert runtime.models == []
    assert not runtime.changing


@pytest.mark.asyncio
async def test_auth_change_waits_for_idle_and_returns_to_automatic(session):
    runtime = Runtime(session, backends=stub_backends())
    runtime.discover_models = AsyncMock()
    blocker = asyncio.Event()
    session._current_task = asyncio.create_task(blocker.wait())
    with pytest.raises(RuntimeError):
        await runtime.authenticate("openai", "api_key", "manual-key")
    assert session.auth.get("openai").method == "cli"
    blocker.set()
    await session._current_task
    session._current_task = None

    await runtime.authenticate("openai", "api_key", "manual-key")
    assert session.auth.get("openai").method == "api_key"
    result = await runtime.authenticate("openai", "cli")
    assert result["providers"]["openai"]["method"] == "cli"
    assert session.auth.get("openai").sdk_env() == {}
    assert session.auth is runtime.auth


def test_runtime_and_session_share_one_auth(session, tmp_path):
    manual = AuthStore()
    manual.set("anthropic", "api_key", "manual-key")
    runtime = Runtime(session, auth=manual, backends=stub_backends())
    assert session.auth is manual
    assert runtime.auth is manual

    received = {}

    def factory(**kwargs):
        received.update(kwargs)
        return session

    fresh = Runtime(session_factory=factory, auth=manual, backends=stub_backends())
    fresh.open_workspace(tmp_path)
    assert received["auth"] is manual
