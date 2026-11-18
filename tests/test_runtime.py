import asyncio
import os
import threading
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
