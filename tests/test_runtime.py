import asyncio
import os
import threading
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from conftest import FakeEnvironment, FakeKernel

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

    async def discover_models_staged(self, auth):
        return await self.discover_models(auth), lambda: None

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


@pytest.mark.asyncio
async def test_empty_model_discovery_clears_active_sessions_and_blocks_queries(session):
    anthropic = StubBackend(
        "anthropic",
        responses=[
            [
                ModelSpec(
                    id="claude-opus-5",
                    raw_id="claude-opus-5",
                    provider="anthropic",
                    label="claude-opus-5",
                    efforts=["low", "high"],
                    is_default=True,
                )
            ],
            [],
        ],
    )
    runtime = Runtime(session, backends=stub_backends(anthropic=anthropic))
    await runtime.discover_models()
    other = runtime.get_agent(runtime.create_agent())
    assert session.model == "claude-opus-5"
    assert other.model == "claude-opus-5"

    await runtime.authenticate("anthropic", "cli")
    assert runtime.models == []
    assert runtime.model is None
    assert session.model is None
    assert session.effort is None
    assert session.default_model is None
    assert other.model is None
    assert other.effort is None
    with pytest.raises(ValueError, match="No model available"):
        runtime.start_query("question")


@pytest.mark.asyncio
async def test_colliding_model_ids_stay_bound_to_their_provider_across_refreshes(
    session,
):
    def shared_spec(provider, is_default=False):
        return ModelSpec(
            id="shared",
            raw_id="shared",
            provider=provider,
            label="shared",
            efforts=["low", "high"],
            is_default=is_default,
        )

    anthropic = StubBackend(
        "anthropic",
        responses=[
            [],
            [shared_spec("anthropic", is_default=True)],
            [],
            [shared_spec("anthropic", is_default=True)],
        ],
    )
    openai = StubBackend(
        "openai",
        responses=[
            [shared_spec("openai", is_default=True)],
            [shared_spec("openai")],
            [shared_spec("openai")],
            [shared_spec("openai")],
        ],
    )
    runtime = Runtime(
        session,
        backends=stub_backends(anthropic=anthropic, openai=openai),
    )

    # First refresh: only OpenAI reports "shared".
    await runtime.discover_models()
    assert runtime.models == ["shared"]
    assert runtime.model_providers["shared"] == "openai"
    runtime.set_session_defaults("shared", "low")
    assert session.workspace.read_config()["default_provider"] == "openai"

    # Second refresh: Anthropic now reports "shared" first, so OpenAI's moves to
    # "openai/shared". Active model and defaults stay on OpenAI rather than silently
    # switching to Anthropic.
    await runtime.discover_models()
    assert runtime.models == ["shared", "openai/shared"]
    assert runtime.model == "openai/shared"
    assert session.model == "openai/shared"
    assert session.default_model == "openai/shared"
    assert session.default_provider == "openai"

    # Third refresh: Anthropic drops "shared", so OpenAI returns to unqualified "shared".
    await runtime.discover_models()
    assert runtime.models == ["shared"]
    assert runtime.model == "shared"
    assert session.model == "shared"
    assert session.default_model == "shared"

    # Reopening a fresh runtime against saved workspace config ("default_model": "shared",
    # "default_provider": "openai") resolves "openai/shared" even when Anthropic reports first.
    reopened = Runtime(
        backends=stub_backends(anthropic=anthropic, openai=openai),
    )
    with (
        patch("nocturnomath.session.Kernel", FakeKernel),
        patch("nocturnomath.session.ResearchEnvironment", FakeEnvironment),
    ):
        reopened.open_workspace(session.workspace_path)
    await reopened.discover_models()
    assert reopened.models == ["shared", "openai/shared"]
    assert reopened.model == "openai/shared"
    assert reopened.session.model == "openai/shared"
    reopened.shutdown()


@pytest.mark.asyncio
async def test_out_of_order_auth_refresh_discards_stale_catalogue(session):
    api_entered = asyncio.Event()
    release_api = asyncio.Event()

    class BlockingOpenAI(StubBackend):
        async def discover_models(self, auth):
            self.auth_seen.append(auth)
            if auth.method == "api_key":
                api_entered.set()
                await release_api.wait()
                return [
                    ModelSpec(
                        id="stale-api-model",
                        raw_id="stale-api-model",
                        provider="openai",
                        label="stale-api-model",
                        efforts=["high"],
                        is_default=True,
                    )
                ]
            return [
                ModelSpec(
                    id="fresh-cli-model",
                    raw_id="fresh-cli-model",
                    provider="openai",
                    label="fresh-cli-model",
                    efforts=["high"],
                    is_default=True,
                )
            ]

    runtime = Runtime(
        session,
        backends=stub_backends(openai=BlockingOpenAI("openai")),
    )
    stale_task = asyncio.create_task(
        runtime.authenticate("openai", "api_key", "secret-key")
    )
    await api_entered.wait()

    # Newer auth selection completes while the older API-key discovery is paused.
    cli_auth = await runtime.authenticate("openai", "cli")
    assert cli_auth["providers"]["openai"]["method"] == "cli"
    assert runtime.models == ["fresh-cli-model"]

    # Releasing the older paused discovery must not overwrite the newer CLI catalogue.
    release_api.set()
    await stale_task
    assert runtime.auth.get("openai").method == "cli"
    assert runtime.models == ["fresh-cli-model"]
    assert runtime.model == "fresh-cli-model"


@pytest.mark.asyncio
async def test_steer_falls_back_to_immediate_followup_turn_when_no_probe_runs(session):
    runtime = Runtime(session, backends=stub_backends())
    runtime.start()
    events = []
    runtime.subscribe(lambda event_type, payload: events.append((event_type, payload)))

    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    queries = []

    async def query(text):
        await session.emit("status_change", status="thinking")
        try:
            queries.append(text)
            if len(queries) == 1:
                first_entered.set()
                await release_first.wait()
        finally:
            if not (session._draining_queries and session._pending_prompts):
                await session.emit("status_change", status="idle")

    session.query = query
    assert runtime.start_query("first") == "started"
    await first_entered.wait()

    assert runtime.start_query("steer 1") == "pending"
    assert runtime.start_query("steer 2") == "pending"
    release_first.set()
    await runtime.wait_for_query()

    assert queries == ["first", "steer 1\n\nsteer 2"]
    assert [payload["text"] for kind, payload in events if kind == "user_message"] == [
        "steer 1",
        "steer 2",
    ]
    assert [
        payload["status"] for kind, payload in events if kind == "status_change"
    ] == ["thinking", "thinking", "idle"]
    assert not session.has_active_query()


@pytest.mark.asyncio
async def test_queue_mode_runs_fifo_turns_applies_models_and_clears_on_interrupt(
    session,
):
    runtime = Runtime(session, backends=stub_backends())
    runtime.start()
    runtime.models = ["sonnet", "opus"]
    runtime.model_efforts = {"sonnet": ["low", "high"], "opus": ["high"]}
    await runtime.set_prompt_mode("queue")

    seen = []
    entered = asyncio.Event()
    release = asyncio.Event()

    async def query(text):
        seen.append((text, session.model, session.effort))
        if len(seen) == 1:
            entered.set()
            await release.wait()

    session.query = query
    assert runtime.start_query("first", "sonnet", "low") == "started"
    await entered.wait()

    assert runtime.start_query("second", "opus", "high") == "pending"
    assert runtime.start_query("third") == "pending"
    release.set()
    await runtime.wait_for_query()

    assert seen == [
        ("first", "sonnet", "low"),
        ("second", "opus", "high"),
        ("third", "opus", "high"),
    ]

    seen.clear()
    entered.clear()
    release.clear()
    assert runtime.start_query("fourth") == "started"
    await entered.wait()
    assert runtime.start_query("drop 1") == "pending"
    assert runtime.start_query("drop 2") == "pending"
    await session.interrupt()
    await runtime.wait_for_query()

    assert session._pending_prompts == []
    assert seen == [("fourth", "opus", "high")]


@pytest.mark.asyncio
async def test_steer_with_model_switch_queues_for_requested_provider(session):
    from nocturnomath.backends import SessionStarted, TextBlockDone, TurnResult

    class SteerProbeBackend(StubBackend):
        def __init__(self, provider, on_turn=None):
            super().__init__(provider)
            self.on_turn = on_turn
            self.calls = []

        async def run_turn(self, prompt, *, tools, model, effort, auth, **kwargs):
            self.calls.append((prompt, model, effort, auth))
            yield SessionStarted(f"{self.provider}-1")
            if self.on_turn is not None:
                await self.on_turn(tools)
            yield TextBlockDone(f"{self.provider} done")
            yield TurnResult()

    runtime_ref = {}
    snapshots = []

    async def on_anthropic_turn(tools):
        runtime = runtime_ref["runtime"]
        by_name = {spec.name: spec for spec in tools}
        first_run = await by_name["run"].handler({"code": "x = 1", "expected": "1"})
        assert not first_run.get("is_error")
        await by_name["verdict"].handler({"text": "x was set"})

        # Submitting with a different model in steer mode must queue instead of steering
        # into the active Anthropic turn.
        runtime.start_query("follow on openai 1", "gpt-5.6", "high")
        runtime.start_query("follow on openai 2", "gpt-5.6", "high")
        session.workspace.create_document("OpenAIHint", "Only for OpenAI")
        snapshots.append(session.pending_prompts_snapshot())

        second_run = await by_name["run"].handler(
            {"code": "print(x + 1)", "expected": "2"}
        )
        assert not second_run.get("is_error")
        assert "Probe S001/P002" in second_run["content"][0]["text"]
        assert "follow on openai" not in second_run["content"][0]["text"]
        assert "Only for OpenAI" not in second_run["content"][0]["text"]
        await by_name["verdict"].handler({"text": "x + 1 was printed"})

    anthropic = SteerProbeBackend("anthropic", on_turn=on_anthropic_turn)
    openai = SteerProbeBackend("openai")
    runtime = Runtime(
        session, backends=stub_backends(anthropic=anthropic, openai=openai)
    )
    runtime_ref["runtime"] = runtime
    runtime.models = ["claude-opus-5", "gpt-5.6"]
    runtime.model_providers = {"claude-opus-5": "anthropic", "gpt-5.6": "openai"}
    runtime.model_raw_ids = {"claude-opus-5": "claude-opus-5", "gpt-5.6": "gpt-5.6"}
    runtime.model_efforts = {"claude-opus-5": ["high"], "gpt-5.6": ["low", "high"]}
    session.model_providers = dict(runtime.model_providers)
    session.model_raw_ids = dict(runtime.model_raw_ids)
    session.model = "claude-opus-5"
    session.effort = "high"

    assert (
        runtime.start_query("start on anthropic", "claude-opus-5", "high") == "started"
    )
    await runtime.wait_for_query()

    assert [item["mode"] for item in snapshots[0]] == ["queue", "queue"]
    assert len(anthropic.calls) == 1
    assert len(openai.calls) == 1
    oai_prompt, oai_model, oai_effort, _ = openai.calls[0]
    assert oai_model == "gpt-5.6"
    assert oai_effort == "high"
    assert "Only for OpenAI" in oai_prompt
    assert oai_prompt.count("follow on openai 1") == 1
    assert oai_prompt.count("follow on openai 2") == 1
    assert "Only for OpenAI" not in anthropic.calls[0][0]


@pytest.mark.asyncio
async def test_steer_fallback_batch_not_duplicated_and_respects_provider_switch(
    session,
):
    from nocturnomath.backends import SessionStarted, TextBlockDone, TurnResult

    turn1_entered = asyncio.Event()
    release_turn1 = asyncio.Event()

    class BlockingAnthropic(StubBackend):
        async def run_turn(self, prompt, **kwargs):
            yield SessionStarted("ant-1")
            turn1_entered.set()
            await release_turn1.wait()
            yield TextBlockDone("anthropic reply")
            yield TurnResult()

    class RecordingOpenAI(StubBackend):
        def __init__(self):
            super().__init__("openai")
            self.calls = []

        async def run_turn(self, prompt, *, model, effort, **kwargs):
            self.calls.append((prompt, model, effort))
            yield SessionStarted("oai-1")
            yield TextBlockDone("openai reply")
            yield TurnResult()

    openai = RecordingOpenAI()
    runtime = Runtime(
        session,
        backends=stub_backends(anthropic=BlockingAnthropic("anthropic"), openai=openai),
    )
    runtime.models = ["claude-opus-5", "gpt-5.6"]
    runtime.model_providers = {"claude-opus-5": "anthropic", "gpt-5.6": "openai"}
    runtime.model_raw_ids = {"claude-opus-5": "claude-opus-5", "gpt-5.6": "gpt-5.6"}
    runtime.model_efforts = {"claude-opus-5": ["high"], "gpt-5.6": ["low", "high"]}
    session.model_providers = dict(runtime.model_providers)
    session.model_raw_ids = dict(runtime.model_raw_ids)
    session.model = "claude-opus-5"
    session.effort = "high"

    assert runtime.start_query("first turn", "claude-opus-5", "high") == "started"
    await turn1_entered.wait()
    assert runtime.start_query("batch msg one", "gpt-5.6", "low") == "pending"
    assert runtime.start_query("batch msg two", "gpt-5.6", "low") == "pending"
    assert runtime.start_query("third turn", "claude-opus-5", "high") == "pending"
    release_turn1.set()
    await runtime.wait_for_query()

    assert len(openai.calls) == 1
    oai_prompt, oai_model, oai_effort = openai.calls[0]
    assert oai_model == "gpt-5.6"
    assert oai_effort == "low"
    assert oai_prompt.count("batch msg one") == 1
    assert oai_prompt.count("batch msg two") == 1
    assert "third turn" not in oai_prompt
    user_records = [
        record["text"]
        for record in session.workspace.current_records()
        if record.get("kind") == "user"
    ]
    assert user_records == [
        "first turn",
        "batch msg one",
        "batch msg two",
        "third turn",
    ]


@pytest.mark.asyncio
async def test_prompt_queued_during_background_probe_task_drains_when_finished(
    session,
):
    from nocturnomath.backends import SessionStarted, TextBlockDone, TurnResult

    seen = []

    class RecordingBackend(StubBackend):
        async def run_turn(self, prompt, **kwargs):
            seen.append(prompt)
            yield SessionStarted("ant-1")
            yield TextBlockDone("ran queued prompt")
            yield TurnResult()

    runtime = Runtime(
        session, backends=stub_backends(anthropic=RecordingBackend("anthropic"))
    )
    release_bg = asyncio.Event()

    async def bg_probe():
        await release_bg.wait()

    bg_task = asyncio.create_task(bg_probe())
    session._probe_task = bg_task

    assert session._is_busy is False
    assert session._current_task is None
    assert runtime.start_query("wait for background task") == "pending"
    assert session.has_active_query() is True
    assert session.pending_prompts_snapshot()[0]["mode"] == "queue"

    release_bg.set()
    await bg_task
    await runtime.wait_for_query()

    assert len(seen) == 1
    assert seen[0].endswith("wait for background task")
    assert session._probe_task is None
    assert session._pending_prompts == []
    assert session.has_active_query() is False


@pytest.mark.asyncio
async def test_prompt_submitted_during_failed_turn_error_broadcast_is_cleared(
    session,
):
    from nocturnomath.backends import ModelSpec

    class FailingBackend(StubBackend):
        def __init__(self):
            super().__init__("anthropic")
            self.fail = True

        async def discover_models(self, auth):
            return [
                ModelSpec(
                    id="test",
                    raw_id="test",
                    provider="anthropic",
                    label="test",
                    is_default=True,
                )
            ]

        async def run_turn(self, prompt, **kwargs):
            if self.fail:
                raise RuntimeError("boom")
            from nocturnomath.backends import SessionStarted, TextBlockDone, TurnResult

            yield SessionStarted("ant-1")
            yield TextBlockDone(f"ok:{prompt}")
            yield TurnResult()

    backend = FailingBackend()
    runtime = Runtime(session, backends=stub_backends(anthropic=backend))
    runtime.start()

    async def on_event(event_type, payload):
        if event_type == "error" and payload.get("message") == "boom":
            runtime.start_query("raced prompt")

    runtime.subscribe(on_event)
    assert runtime.start_query("will fail") == "started"
    await runtime.wait_for_query()

    assert session._pending_prompts == []
    assert session.has_active_query() is False
    backend.fail = False
    assert runtime.start_query("after recovery") == "started"
    await runtime.wait_for_query()


@pytest.mark.asyncio
async def test_auth_refresh_during_queued_prompt_broadcast_preserves_bound_provider(
    session,
):
    from nocturnomath.backends import SessionStarted, TextBlockDone, TurnResult

    entered_turn1 = asyncio.Event()
    release_turn1 = asyncio.Event()

    class BlockingAnthropic(StubBackend):
        def __init__(self):
            super().__init__(
                "anthropic",
                responses=[
                    [
                        ModelSpec(
                            id="claude-opus-5",
                            raw_id="claude-opus-5",
                            provider="anthropic",
                            label="claude-opus-5",
                            efforts=["high"],
                            is_default=True,
                        )
                    ]
                ],
            )
            self.calls = []

        async def run_turn(self, prompt, *, model, effort, auth, **kwargs):
            self.calls.append((prompt, model, effort, auth.method))
            yield SessionStarted("ant-1")
            entered_turn1.set()
            await release_turn1.wait()
            yield TextBlockDone("anthropic done")
            yield TurnResult()

    class OpenAIBackendStub(StubBackend):
        def __init__(self):
            super().__init__(
                "openai",
                responses=[
                    [
                        ModelSpec(
                            id="gpt-5.6",
                            raw_id="gpt-5.6",
                            provider="openai",
                            label="gpt-5.6",
                            efforts=["low", "high"],
                        )
                    ],
                    [],
                ],
            )
            self.calls = []

        async def run_turn(self, prompt, *, model, effort, auth, **kwargs):
            self.calls.append((prompt, model, effort, auth.method, auth.credential))
            yield SessionStarted("oai-1")
            yield TextBlockDone("openai done")
            yield TurnResult()

    anthropic = BlockingAnthropic()
    openai = OpenAIBackendStub()
    runtime = Runtime(
        session, backends=stub_backends(anthropic=anthropic, openai=openai)
    )
    runtime.auth.set("openai", "api_key", "sk-oai-1")
    await runtime.discover_models()
    runtime.start()

    refresh_triggered = False

    async def on_event(event_type, payload):
        nonlocal refresh_triggered
        if (
            not refresh_triggered
            and event_type == "user_message"
            and payload.get("text") == "queued for openai"
        ):
            refresh_triggered = True
            runtime.auth.set("openai", "cli")
            await runtime.discover_models()

    runtime.subscribe(on_event)

    assert runtime.start_query("first turn", "claude-opus-5", "high") == "started"
    await entered_turn1.wait()
    assert runtime.start_query("queued for openai", "gpt-5.6", "high") == "pending"
    release_turn1.set()
    await runtime.wait_for_query()

    assert refresh_triggered is True
    assert runtime.models == ["claude-opus-5"]
    assert len(anthropic.calls) == 1
    assert len(openai.calls) == 1
    oai_prompt, oai_model, oai_effort, oai_method, oai_key = openai.calls[0]
    assert oai_prompt.endswith("queued for openai")
    assert (oai_model, oai_effort, oai_method, oai_key) == (
        "gpt-5.6",
        "high",
        "api_key",
        "sk-oai-1",
    )


@pytest.mark.asyncio
async def test_interrupting_queued_prompt_keeps_background_probe_worker_tracked(
    session,
):
    from nocturnomath.backends import SessionStarted

    worker_started = threading.Event()
    release_worker = threading.Event()

    def slow_install(packages):
        worker_started.set()
        release_worker.wait(timeout=5)
        return ("completed", f"installed {packages}", "done")

    session.install_packages = slow_install

    class InstallBackend(StubBackend):
        async def run_turn(self, prompt, *, tools, **kwargs):
            yield SessionStarted("ant-1")
            by_name = {spec.name: spec for spec in tools}
            await by_name["install_packages"].handler({"packages": ["numpy"]})

    runtime = Runtime(
        session, backends=stub_backends(anthropic=InstallBackend("anthropic"))
    )
    runtime.models = ["test"]
    runtime.model_providers = {"test": "anthropic"}
    runtime.model_raw_ids = {"test": "test"}

    assert runtime.start_query("install numpy") == "started"
    await asyncio.to_thread(worker_started.wait, 5)
    await session.interrupt()
    await runtime.wait_for_query()

    assert session._is_busy is False
    assert session._probe_task is not None and not session._probe_task.done()
    assert runtime.start_query("queued while worker runs") == "pending"

    await session.interrupt()
    await runtime.wait_for_query()

    assert session._pending_prompts == []
    assert session._probe_task is not None and not session._probe_task.done()
    assert session.has_active_query() is True
    with pytest.raises(RuntimeError, match="running a query"):
        session.require_idle("change workspace")

    probe_task = session._probe_task
    release_worker.set()
    await probe_task
    await asyncio.sleep(0)

    assert session._probe_task is None
    assert session.has_active_query() is False


def test_runtime_reuses_attached_session_backends_unless_overridden(session):
    attached = stub_backends()
    session.backends = attached

    runtime = Runtime(session)
    assert runtime.backends is attached
    assert runtime.backends["openai"] is attached["openai"]

    override = stub_backends()
    overridden = Runtime(session, backends=override)
    assert overridden.backends == override
    assert overridden.backends is not attached
    assert session.backends is overridden.backends


@pytest.mark.asyncio
async def test_saved_session_defaults_recover_after_empty_discovery_on_reset(session):
    session.set_session_defaults("gpt-5.4", "high", provider="openai")
    anthropic_spec = ModelSpec(
        id="claude-sonnet-5",
        raw_id="claude-sonnet-5",
        provider="anthropic",
        label="claude-sonnet-5",
        efforts=["medium"],
        is_default=True,
    )
    openai_spec = ModelSpec(
        id="gpt-5.4",
        raw_id="gpt-5.4",
        provider="openai",
        label="GPT-5.4",
        efforts=["medium", "high"],
    )
    runtime = Runtime(
        session,
        backends=stub_backends(
            anthropic=StubBackend("anthropic", responses=[[], [anthropic_spec]]),
            openai=StubBackend("openai", responses=[[], [openai_spec]]),
        ),
    )

    await runtime.discover_models()
    assert runtime.models == []
    assert session.default_model is None
    assert session.model is None

    # Recovering straight from empty discovery restores both saved defaults and
    # the unassigned active model.
    await runtime.discover_models()
    assert (session.model, session.effort) == ("gpt-5.4", "high")
    assert (
        session.default_model,
        session.default_provider,
        session.default_effort,
    ) == ("gpt-5.4", "openai", "high")

    # Partial discovery followed by full discovery keeps the active model
    # undisturbed while restoring saved workspace defaults for reset_session().
    runtime2 = Runtime(
        session,
        backends=stub_backends(
            anthropic=StubBackend(
                "anthropic", responses=[[anthropic_spec], [anthropic_spec]]
            ),
            openai=StubBackend("openai", responses=[[], [openai_spec]]),
        ),
    )
    await runtime2.discover_models()
    assert session.model == "claude-sonnet-5"
    assert session.default_model == "claude-sonnet-5"

    await runtime2.discover_models()
    assert session.model == "claude-sonnet-5"
    assert (
        session.default_model,
        session.default_provider,
        session.default_effort,
    ) == ("gpt-5.4", "openai", "high")

    runtime2.reset_session()
    assert (session.model, session.effort) == ("gpt-5.4", "high")
    assert (runtime2.model, runtime2.effort) == ("gpt-5.4", "high")


@pytest.mark.asyncio
async def test_superseded_discovery_does_not_overwrite_backend_cli_path(
    session, tmp_path
):
    import sys

    from nocturnomath.backends.anthropic import AnthropicBackend

    started_flag = tmp_path / "first_started"
    release_flag = tmp_path / "first_release"

    cli_first = tmp_path / "claude-first"
    cli_first.write_text(
        f"""#!{sys.executable}
import json, pathlib, sys, time
req = json.loads(sys.stdin.readline())
pathlib.Path({str(started_flag)!r}).write_text("started")
release = pathlib.Path({str(release_flag)!r})
while not release.exists():
    time.sleep(0.01)
print(json.dumps({{
    "type": "control_response",
    "response": {{
        "request_id": req["request_id"],
        "subtype": "success",
        "response": {{"models": [{{"value": "claude-old", "displayName": "Claude Old"}}]}},
    }},
}}), flush=True)
"""
    )
    cli_first.chmod(0o700)

    cli_second = tmp_path / "claude-second"
    cli_second.write_text(
        f"""#!{sys.executable}
import json, sys
req = json.loads(sys.stdin.readline())
print(json.dumps({{
    "type": "control_response",
    "response": {{
        "request_id": req["request_id"],
        "subtype": "success",
        "response": {{"models": [{{"value": "claude-new", "displayName": "Claude New"}}]}},
    }},
}}), flush=True)
"""
    )
    cli_second.chmod(0o700)

    backend = AnthropicBackend()
    runtime = Runtime(session, backends=stub_backends(anthropic=backend))
    calls = [[cli_first], [cli_second]]

    with patch(
        "nocturnomath.backends.anthropic.claude_cli_candidates",
        side_effect=lambda: calls.pop(0),
    ):
        task1 = asyncio.create_task(runtime.discover_models())
        try:
            async with asyncio.timeout(5):
                while not started_flag.exists():
                    await asyncio.sleep(0.01)

            await runtime.authenticate("anthropic", "api_key", "sk-new")
            assert runtime.models == ["claude-new"]
            assert backend.cli_path == cli_second

            release_flag.write_text("go")
            await task1
        finally:
            release_flag.write_text("go")
            if not task1.done():
                await task1

    assert runtime.models == ["claude-new"]
    assert backend.cli_path == cli_second


@pytest.mark.asyncio
async def test_discovery_commit_failure_is_isolated_per_provider(session):
    anthropic_spec = ModelSpec(
        id="claude-sonnet-5",
        raw_id="claude-sonnet-5",
        provider="anthropic",
        label="claude-sonnet-5",
        efforts=["medium"],
        is_default=True,
    )
    openai_spec = ModelSpec(
        id="gpt-5.4",
        raw_id="gpt-5.4",
        provider="openai",
        label="GPT-5.4",
        efforts=["high"],
    )

    class FailingCommitOpenAI(StubBackend):
        async def discover_models_staged(self, auth):
            def failing_commit() -> None:
                raise OSError("disk full")

            return [openai_spec], failing_commit

    runtime = Runtime(
        session,
        backends=stub_backends(
            anthropic=StubBackend("anthropic", responses=[[anthropic_spec]]),
            openai=FailingCommitOpenAI("openai"),
        ),
    )

    await runtime.discover_models()
    assert runtime.models == ["claude-sonnet-5"]
    assert runtime.model == "claude-sonnet-5"
    assert session.model == "claude-sonnet-5"


@pytest.mark.asyncio
async def test_active_chat_falls_back_to_workspace_default_after_empty_then_partial_discovery(
    session,
):
    from nocturnomath.backends import SessionStarted, TextBlockDone, TurnResult

    anthropic_spec = ModelSpec(
        id="claude-sonnet-5",
        raw_id="claude-sonnet-5",
        provider="anthropic",
        label="claude-sonnet-5",
        efforts=["medium"],
        is_default=True,
    )
    openai_spec = ModelSpec(
        id="gpt-5.4",
        raw_id="gpt-5.4",
        provider="openai",
        label="GPT-5.4",
        efforts=["high"],
    )

    class ActiveOpenAI(StubBackend):
        async def run_turn(self, prompt, **kwargs):
            yield SessionStarted("oai-1")
            yield TextBlockDone("ok")
            yield TurnResult()

    runtime = Runtime(
        session,
        backends=stub_backends(
            anthropic=StubBackend(
                "anthropic",
                responses=[[anthropic_spec], [], [anthropic_spec]],
            ),
            openai=ActiveOpenAI("openai", responses=[[openai_spec], [], []]),
        ),
    )

    await runtime.discover_models()
    assert (
        runtime.start_query("run on openai", "gpt-5.4", "high", provider="openai")
        == "started"
    )
    await runtime.wait_for_query()
    assert session._active_provider == "openai"

    await runtime.discover_models()
    assert runtime.models == []
    assert session.model is None
    assert runtime.model is None

    await runtime.discover_models()
    assert runtime.models == ["claude-sonnet-5"]
    assert session.model == "claude-sonnet-5"
    assert runtime.model == "claude-sonnet-5"


@pytest.mark.asyncio
async def test_manual_resume_and_replace_agent_restore_saved_chat_model_and_same_provider_fallback(
    session,
):
    from nocturnomath.backends import SessionStarted, TextBlockDone, TurnResult
    from nocturnomath.session import ExplorationSession

    claude_spec = ModelSpec(
        id="claude-sonnet-5",
        raw_id="claude-sonnet-5",
        provider="anthropic",
        label="claude-sonnet-5",
        efforts=["medium"],
        is_default=True,
    )
    oai_preview = ModelSpec(
        id="gpt-5.4-preview",
        raw_id="gpt-5.4-preview",
        provider="openai",
        label="GPT-5.4 Preview",
        efforts=["medium", "high"],
    )
    oai_stable = ModelSpec(
        id="gpt-5.4",
        raw_id="gpt-5.4",
        provider="openai",
        label="GPT-5.4",
        efforts=["medium", "high"],
    )

    class ActiveOpenAI(StubBackend):
        async def run_turn(self, prompt, **kwargs):
            yield SessionStarted("oai-1")
            yield TextBlockDone("ok")
            yield TurnResult()

    def make_session(**kwargs):
        with (
            patch("nocturnomath.session.Kernel", FakeKernel),
            patch("nocturnomath.session.ResearchEnvironment", FakeEnvironment),
        ):
            return ExplorationSession(**kwargs)

    runtime = Runtime(
        session,
        session_factory=make_session,
        backends=stub_backends(
            anthropic=StubBackend(
                "anthropic",
                responses=[[claude_spec], [claude_spec]],
            ),
            openai=ActiveOpenAI(
                "openai",
                responses=[[oai_preview], [oai_stable]],
            ),
        ),
    )

    await runtime.discover_models()
    assert (
        runtime.start_query(
            "run on openai preview", "gpt-5.4-preview", "high", provider="openai"
        )
        == "started"
    )
    await runtime.wait_for_query()

    # Reset onto workspace default (Claude), then verify manual resume and
    # replace_agent both restore the chat's OpenAI model and effort.
    runtime.reset_session()
    assert session.model == "claude-sonnet-5"

    resumed = runtime.resume_session("S001")
    assert (session.model, session.effort) == ("gpt-5.4-preview", "high")
    assert (runtime.model, runtime.effort) == ("gpt-5.4-preview", "high")
    assert (resumed["model"], resumed["effort"]) == ("gpt-5.4-preview", "high")

    runtime.reset_session()
    assert runtime.session.model == "claude-sonnet-5"

    _, replaced_data, _ = runtime.replace_agent(runtime.primary_agent_id, "S001")
    replaced = runtime.session
    assert (replaced.model, replaced.effort) == ("gpt-5.4-preview", "high")
    assert (runtime.model, runtime.effort) == ("gpt-5.4-preview", "high")
    assert (replaced_data["model"], replaced_data["effort"]) == (
        "gpt-5.4-preview",
        "high",
    )

    # Refresh catalogue without gpt-5.4-preview: active model falls back within
    # OpenAI (gpt-5.4), not across providers to claude-sonnet-5.
    await runtime.discover_models()
    assert replaced.model == "gpt-5.4"
    assert replaced.effort == "high"
    assert runtime.model == "gpt-5.4"


@pytest.mark.asyncio
async def test_partial_refresh_and_unavailable_resume_use_owner_then_workspace_default(
    session,
):
    from nocturnomath.backends import SessionStarted, TextBlockDone, TurnResult
    from nocturnomath.session import ExplorationSession

    claude_spec = ModelSpec(
        id="claude-sonnet-5",
        raw_id="claude-sonnet-5",
        provider="anthropic",
        label="claude-sonnet-5",
        efforts=["medium"],
        is_default=True,
    )
    openai_spec = ModelSpec(
        id="gpt-5.4",
        raw_id="gpt-5.4",
        provider="openai",
        label="GPT-5.4",
        efforts=["medium", "high"],
    )
    gemini_spec = ModelSpec(
        id="gemini-4-carbon",
        raw_id="gemini-4-carbon",
        provider="google",
        label="Gemini 4 Carbon",
        efforts=["medium", "high"],
    )

    class ActiveOpenAI(StubBackend):
        async def run_turn(self, prompt, **kwargs):
            yield SessionStarted("oai-1")
            yield TextBlockDone("ok")
            yield TurnResult()

    def make_session(**kwargs):
        with (
            patch("nocturnomath.session.Kernel", FakeKernel),
            patch("nocturnomath.session.ResearchEnvironment", FakeEnvironment),
        ):
            return ExplorationSession(**kwargs)

    runtime = Runtime(
        session,
        session_factory=make_session,
        backends=stub_backends(
            anthropic=StubBackend(
                "anthropic",
                responses=[[claude_spec], [claude_spec], [claude_spec]],
            ),
            openai=ActiveOpenAI(
                "openai",
                responses=[[openai_spec], [openai_spec], []],
            ),
            google=StubBackend(
                "google",
                responses=[[gemini_spec], [], [gemini_spec]],
            ),
        ),
    )

    # 1. Initial discovery + save Gemini as workspace default + run a turn on OpenAI.
    await runtime.discover_models()
    runtime.set_session_defaults("gemini-4-carbon", "medium", provider="google")
    assert (
        runtime.start_query("turn on openai", "gpt-5.4", "high", provider="openai")
        == "started"
    )
    await runtime.wait_for_query()
    assert session._active_provider == "openai"
    assert (session.model, session.effort) == ("gpt-5.4", "high")

    # 2. Unsent Google selection disappears while OpenAI owner + Claude remain:
    # falls back to chat owner (OpenAI) first.
    session.model = "gemini-4-carbon"
    session.effort = "medium"
    await runtime.discover_models()
    assert (session.model, session.effort) == ("gpt-5.4", "high")
    assert (runtime.model, runtime.effort) == ("gpt-5.4", "high")

    # 3. OpenAI disappears while Claude (sdk_default_model) and configured workspace
    # default Gemini remain: falls back to workspace default (Gemini), not Claude.
    await runtime.discover_models()
    assert (session.model, session.effort) == ("gemini-4-carbon", "medium")
    assert (runtime.model, runtime.effort) == ("gemini-4-carbon", "medium")

    # 4. Manual resume of S001 (OpenAI unavailable) from a tab on Claude falls back
    # to the configured workspace default rather than keeping Claude.
    runtime.reset_session()
    session.model = "claude-sonnet-5"
    session.effort = "medium"
    runtime.model = "claude-sonnet-5"
    runtime.effort = "medium"

    resumed = runtime.resume_session("S001")
    assert (session.model, session.effort) == ("gemini-4-carbon", "medium")
    assert (runtime.model, runtime.effort) == ("gemini-4-carbon", "medium")
    assert (resumed["model"], resumed["effort"]) == ("gemini-4-carbon", "medium")

    runtime.reset_session()
    runtime.session.model = "claude-sonnet-5"
    runtime.session.effort = "medium"
    runtime.model = "claude-sonnet-5"
    runtime.effort = "medium"

    _, replaced_data, _ = runtime.replace_agent(runtime.primary_agent_id, "S001")
    replaced = runtime.session
    assert (replaced.model, replaced.effort) == ("gemini-4-carbon", "medium")
    assert (runtime.model, runtime.effort) == ("gemini-4-carbon", "medium")
    assert (replaced_data["model"], replaced_data["effort"]) == (
        "gemini-4-carbon",
        "medium",
    )


@pytest.mark.asyncio
async def test_post_interrupt_prompt_submitted_during_unwind_runs_instead_of_being_cleared(
    session,
):
    from nocturnomath.backends import SessionStarted, TextBlockDone, TurnResult

    started_turn = asyncio.Event()
    finish_unwind = asyncio.Event()
    seen_prompts: list[str] = []

    class SlowUnwindBackend(StubBackend):
        async def run_turn(self, prompt, **kwargs):
            seen_prompts.append(prompt)
            if len(seen_prompts) == 1:
                try:
                    yield SessionStarted("ant-1")
                    started_turn.set()
                    await asyncio.Event().wait()
                finally:
                    await finish_unwind.wait()
                return
            yield SessionStarted("ant-1")
            yield TextBlockDone("second-done")
            yield TurnResult()

    runtime = Runtime(
        session, backends=stub_backends(anthropic=SlowUnwindBackend("anthropic"))
    )
    runtime.start()
    try:
        assert runtime.start_query("first") == "started"
        await asyncio.wait_for(started_turn.wait(), timeout=2)
        assert runtime.start_query("queued-before-stop") == "pending"

        interrupt_task = asyncio.create_task(session.interrupt())
        await asyncio.sleep(0.02)
        # Pre-interrupt prompt was cleared immediately; submit a new prompt while unwinding.
        assert session._pending_prompts == []
        assert runtime.start_query("submitted-after-stop") == "pending"
        finish_unwind.set()
        await interrupt_task
        await runtime.wait_for_query()

        assert len(seen_prompts) == 2
        assert seen_prompts[1].endswith("submitted-after-stop")
    finally:
        await runtime.drain()
        runtime.shutdown()


@pytest.mark.asyncio
async def test_failed_compaction_clears_queued_prompts_and_blocks_duplicate_resume(
    session,
):
    class FailingCompactBackend(StubBackend):
        async def compact(self, **kwargs):
            await asyncio.sleep(0.03)
            raise RuntimeError("compact exploded")

    runtime = Runtime(
        session, backends=stub_backends(anthropic=FailingCompactBackend("anthropic"))
    )
    runtime.start()
    try:
        session._sdk_session_id = "ant-1"
        session._active_provider = "anthropic"
        session.activate_session()
        session.log_transcript("user", text="seed")

        compact_task = asyncio.create_task(
            runtime.compact_agent(runtime.primary_agent_id)
        )
        await asyncio.sleep(0.005)
        assert runtime.start_query("queued-during-failing-compact") == "pending"
        with pytest.raises(RuntimeError, match="compact exploded"):
            await compact_task
        assert session._pending_prompts == []
        assert session.status == "idle"

        other_id = await runtime.create_agent_async()
        runtime.reset_session()
        runtime.replace_agent(other_id, "S001")
        with pytest.raises(RuntimeError, match="already open in another explorer"):
            runtime.resume_session("S001")
    finally:
        await runtime.drain()
        runtime.shutdown()


@pytest.mark.asyncio
async def test_interrupting_compaction_cleans_up_state_and_handles_same_tick_prompt(
    session,
):
    from nocturnomath.backends import CompactResult, SessionStarted, TurnResult

    entered = asyncio.Event()
    release = asyncio.Event()
    seen_prompts: list[str] = []

    class InteractiveCompactBackend(StubBackend):
        async def run_turn(self, prompt, **kwargs):
            seen_prompts.append(prompt)
            yield SessionStarted("ant-1")
            yield TurnResult()

        async def compact(self, **kwargs):
            entered.set()
            await release.wait()
            return CompactResult(session_id="ant-1", context_usage=None)

    runtime = Runtime(
        session,
        backends=stub_backends(anthropic=InteractiveCompactBackend("anthropic")),
    )
    runtime.start()
    try:
        session._sdk_session_id = "ant-1"
        session._active_provider = "anthropic"
        session.activate_session()
        session.log_transcript("user", text="seed")

        compact_task = asyncio.create_task(
            runtime.compact_agent(runtime.primary_agent_id)
        )
        await entered.wait()
        assert runtime.start_query("queued-during-compact") == "pending"

        await session.interrupt()
        with pytest.raises(RuntimeError, match="Compaction was cancelled."):
            await compact_task
        assert session.status == "idle"
        assert session._pending_prompts == []

        # Same-tick prompt before compact() starts should queue and drain after compaction.
        entered.clear()
        release.clear()
        compact_task = asyncio.create_task(
            runtime.compact_agent(runtime.primary_agent_id)
        )
        await asyncio.sleep(0)
        assert not entered.is_set()
        assert runtime.start_query("same-tick-before-compact-starts") == "pending"
        release.set()
        await compact_task
        await runtime.wait_for_query()
        assert len(seen_prompts) == 1
        assert seen_prompts[0].endswith("same-tick-before-compact-starts")
    finally:
        release.set()
        await runtime.drain()
        runtime.shutdown()


@pytest.mark.asyncio
async def test_outer_disconnect_preserves_compaction_while_server_cancel_clears_queue(
    session,
):
    from nocturnomath.backends import CompactResult, SessionStarted, TurnResult

    entered = asyncio.Event()
    release = asyncio.Event()
    seen_prompts: list[str] = []

    class DisconnectCompactBackend(StubBackend):
        async def run_turn(self, prompt, **kwargs):
            seen_prompts.append(prompt)
            yield SessionStarted("ant-1")
            yield TurnResult()

        async def compact(self, **kwargs):
            entered.set()
            await release.wait()
            return CompactResult(
                session_id="ant-2",
                context_usage={"used_tokens": 500, "window_tokens": 200000},
            )

    runtime = Runtime(
        session, backends=stub_backends(anthropic=DisconnectCompactBackend("anthropic"))
    )
    runtime.start()
    try:
        session._sdk_session_id = "ant-1"
        session._active_provider = "anthropic"
        session.activate_session()
        session.log_transcript("user", text="seed")

        # 1. Outer caller (e.g. HTTP client) disconnects: shielded compaction finishes and drains queued prompts.
        outer_task = asyncio.create_task(
            runtime.compact_agent(runtime.primary_agent_id)
        )
        await entered.wait()
        assert runtime.start_query("after-outer-disconnect") == "pending"
        outer_task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await outer_task
        await runtime.wait_for_query()
        assert len(seen_prompts) == 1
        assert seen_prompts[0].endswith("after-outer-disconnect")

        # 2. Server-initiated cancellation of session._current_task without session.interrupt()
        # clears pre-queued prompts from that turn.
        turn_entered = asyncio.Event()

        class BlockingTurnBackend(StubBackend):
            async def run_turn(self, prompt, **kwargs):
                turn_entered.set()
                await asyncio.Event().wait()
                yield TurnResult()

        session.backends["anthropic"] = BlockingTurnBackend("anthropic")
        assert runtime.start_query("first-turn") == "started"
        await turn_entered.wait()
        assert runtime.start_query("queued-before-server-cancel") == "pending"
        session._current_task.cancel()
        await runtime.wait_for_query()
        assert session._pending_prompts == []
        assert session.status == "idle"
    finally:
        release.set()
        await runtime.drain()
        runtime.shutdown()


@pytest.mark.asyncio
async def test_open_workspace_validation_failure_preserves_explorers_and_concurrent_close_keeps_one(
    session, tmp_path
):
    runtime = Runtime(session, backends=stub_backends())
    runtime.start()
    try:
        second_id = await runtime.create_agent_async()
        assert len(runtime.agents) == 2
        bad_key = (session.workspace_path, "S001/P001")
        runtime._slurm_review_attempts.add(bad_key)

        bad_ws = tmp_path / "bad_ws"
        bad_ws.mkdir()

        def failing_factory(**kwargs):
            if Path(kwargs["workspace_path"]).resolve() == bad_ws.resolve():
                raise ValueError("Invalid Python interpreter")
            return session

        runtime.session_factory = failing_factory
        with pytest.raises(ValueError, match="Invalid Python interpreter"):
            await runtime.transition(runtime.open_workspace, bad_ws)

        assert len(runtime.agents) == 2
        assert runtime.session is session
        assert bad_key in runtime._slurm_review_attempts

        results = await asyncio.gather(
            runtime.transition_agent(
                runtime.primary_agent_id, runtime.close_agent, runtime.primary_agent_id
            ),
            runtime.transition_agent(second_id, runtime.close_agent, second_id),
            return_exceptions=True,
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert "Keep one explorer open" in str(errors[0])
        assert len(runtime.agents) == 1
        assert runtime.session is runtime.agents[runtime.primary_agent_id]
    finally:
        await runtime.drain()
        runtime.shutdown()


@pytest.mark.asyncio
async def test_turn_complete_and_idle_handoff_do_not_clobber_next_turn_or_pending_prompts(
    session,
):
    from nocturnomath.backends import SessionStarted, TextBlockDone, TurnResult

    observed_calls = []

    class TrackingBackend(StubBackend):
        def __init__(self, provider: str, models: list[ModelSpec]):
            super().__init__(provider)
            self._specs = models

        async def discover_models(self, auth):
            return list(self._specs)

        async def run_turn(self, prompt, *, model, effort, **kwargs):
            await asyncio.sleep(0.01)
            observed_calls.append(
                (
                    self.provider,
                    model,
                    effort,
                    prompt.splitlines()[-1],
                    session._query_task is not None,
                    session.active_model,
                )
            )
            yield SessionStarted(f"{self.provider}-1")
            yield TextBlockDone("ok")
            yield TurnResult()

    ant = TrackingBackend(
        "anthropic",
        [
            ModelSpec(
                id="claude-sonnet-5",
                raw_id="claude-sonnet-5",
                provider="anthropic",
                label="Claude",
                efforts=["high"],
                is_default=True,
            )
        ],
    )
    oai = TrackingBackend(
        "openai",
        [
            ModelSpec(
                id="gpt-5.4",
                raw_id="gpt-5.4",
                provider="openai",
                label="GPT",
                efforts=["low", "high"],
                is_default=True,
            )
        ],
    )
    runtime = Runtime(
        session,
        backends={"anthropic": ant, "openai": oai, "google": StubBackend("google")},
    )
    runtime.start()
    try:
        await runtime.discover_models()
        triggered_on_complete = False
        triggered_on_idle = False

        async def listener(event_type, payload):
            nonlocal triggered_on_complete, triggered_on_idle
            await asyncio.sleep(0)
            if event_type == "turn_complete" and not triggered_on_complete:
                triggered_on_complete = True
                assert (
                    runtime.start_query(
                        "from-turn-complete",
                        model="gpt-5.4",
                        provider="openai",
                        raw_model="gpt-5.4",
                        effort="low",
                    )
                    == "pending"
                )
            elif (
                event_type == "status_change"
                and payload.get("status") == "idle"
                and not triggered_on_idle
            ):
                triggered_on_idle = True
                assert (
                    runtime.start_query(
                        "from-idle-status",
                        model="claude-sonnet-5",
                        provider="anthropic",
                        raw_model="claude-sonnet-5",
                        effort="high",
                    )
                    == "started"
                )
                await asyncio.sleep(0)
                assert runtime.start_query("queued-during-turn-3") == "pending"

        runtime.subscribe(listener)
        assert runtime.start_query("initial-turn") == "started"
        for _ in range(5):
            await runtime.wait_for_query()
            await asyncio.sleep(0.02)
        assert [item[:4] for item in observed_calls] == [
            ("anthropic", "claude-sonnet-5", "high", "initial-turn"),
            ("openai", "gpt-5.4", "low", "from-turn-complete"),
            ("anthropic", "claude-sonnet-5", "high", "from-idle-status"),
            ("anthropic", "claude-sonnet-5", "high", "queued-during-turn-3"),
        ]
        assert all(item[4] is True for item in observed_calls)
        assert [item[5] for item in observed_calls] == [
            "claude-sonnet-5",
            "gpt-5.4",
            "claude-sonnet-5",
            "claude-sonnet-5",
        ]
    finally:
        await runtime.drain()
        runtime.shutdown()


@pytest.mark.asyncio
async def test_discover_models_handles_concurrent_agent_creation_and_closure(session):
    from nocturnomath.session import ExplorationSession

    spec = ModelSpec(
        id="claude-sonnet-5",
        raw_id="claude-sonnet-5",
        provider="anthropic",
        label="Claude Sonnet 5",
        efforts=["high"],
        is_default=True,
    )

    def factory(**kwargs):
        with (
            patch("nocturnomath.session.Kernel", FakeKernel),
            patch("nocturnomath.session.ResearchEnvironment", FakeEnvironment),
        ):
            return ExplorationSession(**kwargs)

    runtime = Runtime(
        session,
        session_factory=factory,
        backends=stub_backends(anthropic=StubBackend("anthropic", [[spec]])),
    )
    runtime.start()
    try:
        await runtime.discover_models()
        stop_churn = threading.Event()
        churn_errors = []

        def churn():
            try:
                while not stop_churn.is_set():
                    aid = runtime.create_agent(session.workspace_path)
                    runtime.close_agent(aid)
            except Exception as exc:
                churn_errors.append(exc)

        worker = threading.Thread(target=churn)
        worker.start()
        try:
            for _ in range(15):
                await runtime.authenticate("anthropic", "cli")
        finally:
            stop_churn.set()
            worker.join(timeout=5)

        assert churn_errors == []
        assert runtime.models == ["claude-sonnet-5"]
    finally:
        await runtime.drain()
        runtime.shutdown()
