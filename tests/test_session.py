import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
from conftest import FakeEnvironment, FakeKernel

from nocturnomath.auth import AuthStore
from nocturnomath.backends import (
    CompactionNotice,
    CompactResult,
    MessageUsage,
    ModelSpec,
    SessionResumeError,
    SessionStarted,
    TextBlockDone,
    TextDelta,
    TurnResult,
    TurnStarted,
)
from nocturnomath.session import ExplorationSession


@dataclass
class TurnCall:
    prompt: str
    system_prompt: str
    model: str
    effort: str | None
    auth: Any
    resume_session_id: str | None
    workspace_path: Any


class FakeBackend:
    def __init__(
        self,
        provider="anthropic",
        *,
        uses_compact_hook=True,
        session_id="sdk-session",
        events=None,
        compact_result=None,
        fail_turn=None,
        fail_compact=None,
    ):
        self.provider = provider
        self.uses_compact_hook = uses_compact_hook
        self.session_id = session_id
        self.custom_events = events
        self.compact_result = compact_result
        self.fail_turn = fail_turn
        self.fail_compact = fail_compact
        self.calls: list[TurnCall] = []
        self.compact_calls: list[dict[str, Any]] = []

    async def discover_models(self, auth):
        return [
            ModelSpec(id="test", raw_id="test", provider=self.provider, label="test")
        ]

    async def run_turn(
        self,
        prompt,
        *,
        system_prompt,
        tools,
        workspace_path,
        model,
        effort,
        auth,
        resume_session_id=None,
        tool_timeout_s=660,
    ):
        self.tool_timeout_s = tool_timeout_s
        self.calls.append(
            TurnCall(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                effort=effort,
                auth=auth,
                resume_session_id=resume_session_id,
                workspace_path=workspace_path,
            )
        )
        if self.fail_turn is not None:
            raise self.fail_turn
        if self.custom_events is not None:
            for item in self.custom_events:
                yield item
            return
        yield SessionStarted(self.session_id)
        yield TextDelta("ans")
        yield TextDelta("wer")
        yield TextBlockDone("answer")
        yield TurnResult()

    async def compact(
        self,
        *,
        system_prompt,
        tools,
        workspace_path,
        model,
        effort,
        auth,
        resume_session_id,
    ):
        self.compact_calls.append(
            {
                "system_prompt": system_prompt,
                "model": model,
                "effort": effort,
                "auth": auth,
                "resume_session_id": resume_session_id,
            }
        )
        if self.fail_compact is not None:
            raise self.fail_compact
        return self.compact_result or CompactResult(
            session_id=resume_session_id,
            context_usage={
                "used_tokens": 4200,
                "window_tokens": 200000,
                "model": "test",
            },
        )


def attach_backend(session, backend: FakeBackend, provider="anthropic"):
    session.backends[provider] = backend
    return backend


@pytest.mark.asyncio
async def test_compaction_is_recorded_and_reloads_scientific_record_when_needed(
    session,
):
    events = []
    session.subscribe(lambda event_type, payload: events.append(event_type))
    anthropic = attach_backend(
        session,
        FakeBackend(
            uses_compact_hook=True,
            events=[
                SessionStarted("sdk-session"),
                CompactionNotice(),
                TextBlockDone("continued"),
                TurnResult(),
            ],
        ),
        "anthropic",
    )
    await session.query("question")

    assert "context_compacted" in events
    assert any(
        record["kind"] == "compaction" for record in session.workspace.current_records()
    )
    assert not session._record_dirty
    assert anthropic.calls[-1].workspace_path == session.workspace_path

    # Non-hook providers (OpenAI / Google) mark the record dirty so the next turn
    # reloads evidence.md and thoughts.md explicitly.
    session.model = "gpt-5.6-sol"
    attach_backend(
        session,
        FakeBackend(
            provider="openai",
            uses_compact_hook=False,
            session_id="thr-1",
            events=[
                SessionStarted("thr-1"),
                CompactionNotice(),
                TextBlockDone("ok"),
                TurnResult(),
            ],
        ),
        "openai",
    )
    await session.query("next")
    assert session._record_dirty is True


@pytest.mark.asyncio
async def test_manual_compaction_keeps_transcript_and_kernel_and_refreshes_usage(
    session,
):
    events = []
    session.subscribe(lambda event_type, payload: events.append((event_type, payload)))
    backend = attach_backend(session, FakeBackend())
    await session.query("original question")
    kernel = session.kernel
    await session.compact()

    assert backend.compact_calls[-1]["resume_session_id"] == "sdk-session"
    assert session.kernel is kernel
    assert session._sdk_session_id == "sdk-session"
    assert session.context_usage == {
        "used_tokens": 4200,
        "window_tokens": 200000,
        "model": "test",
    }
    records = session.workspace.current_records()
    assert [record["text"] for record in records if record["kind"] == "user"] == [
        "original question"
    ]
    assert any(record["kind"] == "compaction" for record in records)
    assert any(kind == "context_compacted" for kind, _ in events)
    assert events[-1][1]["status"] == "idle"


@pytest.mark.asyncio
async def test_manual_compaction_requires_context_and_confirmation(session):
    with pytest.raises(RuntimeError, match="no conversation context"):
        await session.compact()
    attach_backend(
        session,
        FakeBackend(fail_compact=RuntimeError("Claude did not confirm")),
    )
    await session.query("original question")
    with pytest.raises(RuntimeError, match="did not confirm"):
        await session.compact()
    assert not session.has_active_query()
    assert not any(
        record["kind"] == "compaction" for record in session.workspace.current_records()
    )


@pytest.mark.asyncio
async def test_manual_compaction_on_non_hook_backend_marks_record_dirty(session):
    session.model = "gpt-5.6-sol"
    openai = attach_backend(
        session,
        FakeBackend(
            provider="openai",
            uses_compact_hook=False,
            session_id="thr-1",
            compact_result=CompactResult(session_id="thr-1", context_usage=None),
        ),
        "openai",
    )
    await session.query("original question")
    await session.compact()

    assert session.context_usage is None
    assert session._record_dirty is True
    assert not session.has_active_query()
    assert any(
        record["kind"] == "compaction" for record in session.workspace.current_records()
    )
    session.workspace.notes.add_evidence("Temp = 4 K.", ["source"])
    await session.query("follow up")
    assert "Temp = 4 K." in openai.calls[-1].prompt


@pytest.mark.asyncio
async def test_context_usage_is_reported_and_restored_from_history(session):
    events = []
    session.subscribe(lambda event_type, payload: events.append((event_type, payload)))
    expected = {"used_tokens": 13500, "window_tokens": 200000, "model": "test"}
    attach_backend(
        session,
        FakeBackend(
            events=[
                SessionStarted("sdk-session"),
                TextBlockDone("answer"),
                TurnResult(context_usage=expected),
            ]
        ),
    )
    await session.query("question")

    assert session.context_usage == expected
    assert (
        next(
            payload["context_usage"]
            for event_type, payload in events
            if event_type == "context_usage_changed"
        )
        == expected
    )
    records = session.workspace.current_records()
    assert records[-1]["context_usage"] == expected

    session.start_new_transcript()
    assert session.context_usage is None
    resumed = session.resume_session("S001")
    assert resumed["context_usage"] == expected
    assert session.context_usage == expected


@pytest.mark.asyncio
async def test_query_streams_text_tracks_context_and_logs(session):
    events = []
    session.subscribe(lambda event_type, payload: events.append((event_type, payload)))
    backend = attach_backend(session, FakeBackend())
    await session.query("question")

    assert session._sdk_session_id == "sdk-session"
    call = backend.calls[-1]
    assert call.effort == "high"
    assert call.workspace_path == session.workspace_path
    assert [event_type for event_type, _ in events] == [
        "status_change",
        "assistant_delta",
        "assistant_delta",
        "assistant_text",
        "turn_complete",
        "status_change",
    ]
    payloads = dict(events)
    deltas = [
        payload["text"]
        for event_type, payload in events
        if event_type == "assistant_delta"
    ]
    assert deltas == ["ans", "wer"]
    assert payloads["assistant_text"]["text"] == "answer"
    assert payloads["turn_complete"]["full_text"] == "answer"
    records = session.workspace.read_transcript(session.transcript_path)
    assert [record["kind"] for record in records] == [
        "kernel_started",
        "user",
        "meta",
        "agent",
    ]
    assert [record["text"] for record in records if record["kind"] == "agent"] == [
        "answer"
    ]


@pytest.mark.asyncio
async def test_pending_probe_verdict_survives_next_message_and_resume(session):
    tools = {tool.name: tool for tool in session.tools}
    await tools["run"].handler({"code": "print(42)", "expected": "prints 42"})
    assert session._pending_verdict_probe_id == "S001/P001"

    backend = attach_backend(session, FakeBackend())
    await session.query("Continue")
    assert "S001/P001 still needs a verdict" in backend.calls[-1].prompt
    assert session._pending_verdict_probe_id == "S001/P001"
    blocked = await tools["run"].handler({"code": "print(43)", "expected": "prints 43"})
    assert blocked["is_error"]

    session.start_new_transcript()
    session.resume_session("S001")
    assert session._pending_verdict_probe_id == "S001/P001"
    await tools["verdict"].handler({"text": "The result matched."})
    session.start_new_transcript()
    session.resume_session("S001")
    assert session._pending_verdict is None


@pytest.mark.asyncio
async def test_batch_review_is_an_internal_turn_and_preserves_next_message_documents(
    session,
):
    session.log_transcript("user", text="Original question")
    session._track_session_id("original-sdk-session", "anthropic")
    session.workspace.save_document("input.md", "Selected for the next message", False)
    session.refresh_document_settings()
    session.set_document_included("input.md", True)
    events = []
    session.subscribe(lambda kind, payload: events.append((kind, payload)))

    backend = attach_backend(session, FakeBackend())
    await session.review_slurm("S001/P001")

    assert backend.calls[-1].resume_session_id == "original-sdk-session"
    assert "S001/P001" in backend.calls[-1].prompt
    assert "Selected for the next message" not in backend.calls[-1].prompt
    assert session._sent_documents == {}
    records = session.workspace.current_records()
    assert [record["text"] for record in records if record["kind"] == "user"] == [
        "Original question"
    ]
    assert any(
        record["kind"] == "slurm_review" and record["probe_id"] == "S001/P001"
        for record in records
    )
    assert any(kind == "slurm_review" for kind, _ in events)

    await session.query("Next user message")
    assert "Selected for the next message" in backend.calls[-1].prompt


@pytest.mark.asyncio
async def test_query_passes_selected_effort_to_backend(session):
    session.effort = "xhigh"
    backend = attach_backend(session, FakeBackend())
    await session.query("question")

    assert backend.calls[-1].effort == "xhigh"


@pytest.mark.asyncio
async def test_query_passes_selected_authentication_only_to_backend(session):
    session.auth = AuthStore()
    session.auth.set("anthropic", "api_key", "top-secret-key")
    backend = attach_backend(session, FakeBackend())
    await session.query("question")

    env = backend.calls[-1].auth.sdk_env()
    assert env["ANTHROPIC_API_KEY"] == "top-secret-key"
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == ""
    assert env["ANTHROPIC_AUTH_TOKEN"] == ""
    transcript = session.transcript_path.read_text()
    assert "top-secret-key" not in transcript


@pytest.mark.asyncio
async def test_switching_providers_mid_chat_keeps_kernel_and_recaps_history(session):
    anthropic = attach_backend(
        session, FakeBackend(provider="anthropic", session_id="ant-1"), "anthropic"
    )
    openai = attach_backend(
        session,
        FakeBackend(provider="openai", uses_compact_hook=False, session_id="oai-1"),
        "openai",
    )
    (session.workspace.documents_path / "notes.md").write_text("# Notes\nalpha = 1\n")

    restarts_before = session.kernel.restarts
    await session.query("first question")
    assert anthropic.calls[-1].resume_session_id is None
    assert "### notes.md" in anthropic.calls[-1].prompt

    # Same provider switches model while preserving the native session ID.
    session.model = "claude-sonnet-5-5"
    await session.query("same provider follow up")
    assert anthropic.calls[-1].model == "claude-sonnet-5-5"
    assert anthropic.calls[-1].resume_session_id == "ant-1"
    assert "first question" not in anthropic.calls[-1].prompt

    # Switching to OpenAI mid-session keeps the live kernel, resets native session,
    # prepends a recap of earlier turns without duplicating the new message, and
    # re-sends active documents to the new provider.
    session.model = "gpt-5.6-sol"
    await session.query("cross provider question")
    assert session.kernel.restarts == restarts_before
    oai_call = openai.calls[-1]
    assert oai_call.resume_session_id is None
    assert (
        "live Jupyter kernel and its in-memory variables are still intact"
        in oai_call.prompt
    )
    assert "Me: first question" in oai_call.prompt
    assert "Me: same provider follow up" in oai_call.prompt
    assert oai_call.prompt.count("cross provider question") == 1
    assert "### notes.md" in oai_call.prompt

    # Subsequent turn on OpenAI resumes the OpenAI thread natively.
    await session.query("openai follow up")
    assert openai.calls[-1].resume_session_id == "oai-1"

    # Resuming S001 while an Anthropic model is selected still retains the saved
    # native OpenAI thread until the next message actually picks a backend.
    session.start_new_transcript()
    session.model = "claude-opus-5-5"
    resumed = session.resume_session("S001")
    assert resumed["context_restored"] is True
    assert session._sdk_session_id == "oai-1"
    assert session._resume_prefix is None

    # Switching back to OpenAI before sending uses that preserved native thread.
    session.model = "gpt-5.6-sol"
    await session.query("still native on openai")
    assert openai.calls[-1].resume_session_id == "oai-1"

    # Switching to Anthropic after resuming converts the preserved thread into a recap.
    session.start_new_transcript()
    session.model = "claude-opus-5-5"
    resumed_then_switch = session.resume_session("S001")
    assert resumed_then_switch["context_restored"] is True
    await session.query("switch after resume")
    assert anthropic.calls[-1].resume_session_id is None
    assert "Me: still native on openai" in anthropic.calls[-1].prompt


@pytest.mark.asyncio
async def test_active_query_guards_state_and_interrupts(session):
    blocker = asyncio.Event()
    task = asyncio.create_task(blocker.wait())
    session._current_task = task

    with pytest.raises(RuntimeError):
        session.reset_client_session()
    with pytest.raises(RuntimeError):
        session.set_workspace(session.workspace_path / "other")

    await session.interrupt()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_resume_reads_current_records_and_reuses_session_probe_folder(session):
    session.log_transcript("user", text="original investigation")
    session.log_transcript("meta", provider="anthropic", sdk_session_id="sdk-session")
    session.workspace.start_probe("print(3)", "3", "test")
    session.workspace.notes.add_evidence("Value = 3.", ["source"])
    session.reset_client_session()
    session.workspace.notes.strike_evidence("E001")
    resumed = session.resume_session("S001")
    assert resumed["id"] == "S001"
    assert session.workspace.start_probe("print(4)", "4", "test").name == "P002"
    backend = attach_backend(session, FakeBackend())
    await session.query("continue")
    assert "<del>" in backend.calls[-1].prompt
    assert "Value = 3." in backend.calls[-1].prompt


def test_resume_with_kernel_replays_recorded_probe_code_in_order(session):
    session.log_transcript("user", text="original investigation")
    session.log_transcript("probe_started", probe_id="S001/P001", code="x = 1")
    session.log_transcript("run", probe_id="S001/P001", code="x = 1")
    session.log_transcript("probe_started", probe_id="S001/P002", code="y = x + 1")
    session.reset_client_session()

    resumed = session.resume_session("S001", restore_kernel=True)

    assert session.kernel.executed_codes == ["x = 1", "y = x + 1"]
    assert resumed["kernel_restored"] is True
    assert resumed["probes_replayed"] == 2
    assert "replaying 2 stored probes" in session._kernel_notice
    records = session.workspace.read_transcript(session.transcript_path)
    assert sum(record["kind"] == "run" for record in records) == 1


@pytest.mark.asyncio
async def test_included_documents_are_sent_when_new_or_changed(session):
    documents = session.workspace.documents_path
    (documents / "inputs.md").write_text("# Inputs\n\nN = 10\n")
    (documents / "draft.md").write_text("# Draft\n")
    session.set_document_included("draft.md", False)

    backend = attach_backend(session, FakeBackend())
    await session.query("first")
    first = backend.calls[-1].prompt
    assert "### inputs.md\n\n# Inputs\n\nN = 10\n" in first
    assert "draft.md" not in first
    assert (
        first.index("# Evidence") < first.index("### inputs.md") < first.index("first")
    )

    await session.query("second")
    assert "inputs.md" not in backend.calls[-1].prompt
    (documents / "inputs.md").write_text("# Inputs\n\nN = 20\n")
    await session.query("third")
    third = backend.calls[-1].prompt
    assert "N = 20" in third and "N = 10" not in third

    session.set_documents_default(False)
    await session.query("fourth")
    assert "inputs.md" not in backend.calls[-1].prompt
    assert session.list_documents() == [
        {**item, "included": False} for item in session.workspace.list_documents()
    ]

    session.set_document_included("inputs.md", True)
    (documents / "inputs.md").write_text("# Inputs\n\nN = 30\n")
    await session.query("fifth")
    assert "N = 30" in backend.calls[-1].prompt
    await session.query("sixth")
    assert "inputs.md" not in backend.calls[-1].prompt


def test_document_selections_persist_per_workspace(session, tmp_path):
    from unittest.mock import patch

    session.workspace.save_document("inputs.md", "# Inputs")
    session.workspace.save_document("draft.md", "# Draft")
    session.set_document_included("draft.md", False)

    with (
        patch("nocturnomath.session.Kernel", FakeKernel),
        patch("nocturnomath.session.ResearchEnvironment", FakeEnvironment),
    ):
        another_session = ExplorationSession(session.workspace_path, model="test")
    assert {
        item["name"]: item["included"] for item in another_session.list_documents()
    } == {
        "draft.md": False,
        "inputs.md": True,
    }

    another_session.set_document_included("inputs.md", False)
    assert all(not item["included"] for item in session.list_documents())

    other_workspace = tmp_path / "other"
    session.set_workspace(other_workspace)
    session.workspace.save_document("draft.md", "# Other draft")
    assert session.list_documents()[0]["included"] is True
    session.set_workspace(another_session.workspace_path)
    assert all(not item["included"] for item in session.list_documents())

    session.delete_document("draft.md")
    assert "draft.md" not in session.workspace.read_config()["document_choices"]
    session.set_documents_default(True)
    assert session.workspace.read_config()["document_choices"] == {}
    assert session.list_documents()[0]["included"] is True


@pytest.mark.asyncio
async def test_nested_document_saved_by_tool_enters_next_message_context(session):
    save = next(tool for tool in session.tools if tool.name == "save_document")
    await save.handler(
        {"path": "reports/summary.md", "text": "# Summary", "replace_existing": False}
    )
    backend = attach_backend(session, FakeBackend())
    await session.query("Read the report")
    assert "### reports/summary.md\n\n# Summary\n" in backend.calls[-1].prompt
    await session.query("Continue")
    assert "### reports/summary.md" not in backend.calls[-1].prompt

    await save.handler(
        {
            "path": "reports/summary.md",
            "text": "# Updated",
            "replace_existing": True,
        }
    )
    await session.query("Continue again")
    assert "### reports/summary.md\n\n# Updated\n" in backend.calls[-1].prompt


@pytest.mark.asyncio
async def test_project_prompt_applies_to_next_query_without_resetting_context(session):
    backend = attach_backend(session, FakeBackend())
    await session.query("first")
    session_id = session._sdk_session_id
    session.set_exploration_prompt("Explore this project's data carefully.")
    await session.query("second")

    call = backend.calls[-1]
    assert call.resume_session_id == session_id
    assert call.system_prompt.startswith("Explore this project's data carefully.\n\n")
    assert "# Nocturnomath runtime contract" in call.system_prompt
    assert session.workspace.read_config()["exploration_prompt"] == (
        "Explore this project's data carefully."
    )


@pytest.mark.asyncio
async def test_document_context_is_retried_when_submission_fails(session):
    (session.workspace.documents_path / "inputs.md").write_text("# Inputs\n")

    attach_backend(session, FakeBackend(fail_turn=RuntimeError("send failed")))
    await session.query("first")
    assert session._sent_documents == {}

    backend = attach_backend(session, FakeBackend())
    await session.query("second")
    assert "### inputs.md\n\n# Inputs\n" in backend.calls[-1].prompt
    assert "# Evidence" in backend.calls[-1].prompt


@pytest.mark.asyncio
async def test_query_logs_usage_records(session):
    attach_backend(
        session,
        FakeBackend(
            events=[
                SessionStarted("sdk-session"),
                MessageUsage(
                    message_id="msg_1",
                    model="m",
                    usage={"input_tokens": 10, "output_tokens": 5},
                ),
                TextBlockDone("hi"),
                TurnResult(
                    total_cost_usd=0.01,
                    usage={"input_tokens": 10, "output_tokens": 5},
                    num_turns=1,
                ),
            ]
        ),
    )
    await session.query("hello")
    records = [
        json.loads(line) for line in session.transcript_path.read_text().splitlines()
    ]
    msg = [r for r in records if r["kind"] == "message_usage"]
    total = [r for r in records if r["kind"] == "usage"]
    assert len(msg) == 1 and msg[0]["message_id"] == "msg_1"
    assert msg[0]["usage"]["output_tokens"] == 5
    assert len(total) == 1 and total[0]["total_cost_usd"] == 0.01
    assert total[0]["num_turns"] == 1


@pytest.mark.asyncio
async def test_failed_turn_after_session_started_does_not_commit_prompt_state(session):
    (session.workspace.documents_path / "inputs.md").write_text("# Inputs\n")

    class EmitStartThenFailBackend(FakeBackend):
        async def run_turn(self, *args, **kwargs):
            yield SessionStarted("ephemeral-thread")
            raise RuntimeError("invalid model")

    attach_backend(session, EmitStartThenFailBackend())
    assert await session.query("first") is False
    assert session._sdk_session_id is None
    assert session._active_provider is None
    assert session._session_initialized is False
    assert session._sent_documents == {}
    assert session._kernel_notice != ""

    backend = attach_backend(session, FakeBackend(session_id="real-thread"))
    assert await session.query("second") is True
    assert session._sdk_session_id == "real-thread"
    assert "# Evidence" in backend.calls[-1].prompt
    assert "### inputs.md\n\n# Inputs\n" in backend.calls[-1].prompt


@pytest.mark.asyncio
async def test_stale_native_session_falls_back_to_recap_and_recovers(session):
    (session.workspace.documents_path / "inputs.md").write_text("# Inputs\n")
    session.model = "gpt-5.6-sol"

    class RecoveringBackend(FakeBackend):
        def __init__(self):
            super().__init__(provider="openai", uses_compact_hook=False)
            self.turn_num = 0

        async def run_turn(self, prompt, **kwargs):
            self.calls.append(
                TurnCall(
                    prompt=prompt,
                    system_prompt=kwargs["system_prompt"],
                    model=kwargs["model"],
                    effort=kwargs["effort"],
                    auth=kwargs["auth"],
                    resume_session_id=kwargs.get("resume_session_id"),
                    workspace_path=kwargs["workspace_path"],
                )
            )
            self.turn_num += 1
            if self.turn_num == 1:
                yield SessionStarted("stale-thread")
                yield TextBlockDone("first answer")
                yield TurnResult()
                return
            if kwargs.get("resume_session_id") == "stale-thread":
                raise SessionResumeError("Thread not found")
            yield SessionStarted("fresh-thread")
            yield TextBlockDone("recovered answer")
            yield TurnResult()

    backend = attach_backend(session, RecoveringBackend(), "openai")
    events: list[tuple[str, dict[str, Any]]] = []
    session.subscribe(lambda et, payload: events.append((et, payload)))
    assert await session.query("first question") is True
    assert session._sdk_session_id == "stale-thread"

    assert await session.query("follow up") is True
    assert len(backend.calls) == 3
    assert backend.calls[1].resume_session_id == "stale-thread"
    assert backend.calls[2].resume_session_id is None
    assert (
        "live Jupyter kernel and its in-memory variables are still intact"
        in backend.calls[2].prompt
    )
    assert "Me: first question" in backend.calls[2].prompt
    assert backend.calls[2].prompt.count("follow up") == 1
    assert "### inputs.md\n\n# Inputs\n" in backend.calls[2].prompt
    assert session._sdk_session_id == "fresh-thread"
    assert any(
        et == "system_message"
        and "rebuilt context from the saved transcript" in payload["text"]
        for et, payload in events
    )
    records = [
        json.loads(line) for line in session.transcript_path.read_text().splitlines()
    ]
    assert any(
        r.get("kind") == "meta"
        and r.get("context_rebuilt") is True
        and r.get("stale_sdk_session_id") == "stale-thread"
        for r in records
    )


@pytest.mark.asyncio
async def test_compact_omits_model_when_selected_model_belongs_to_another_provider(
    session,
):
    anthropic = attach_backend(
        session, FakeBackend(provider="anthropic", session_id="ant-1"), "anthropic"
    )
    await session.query("first")
    session.model = "gpt-5.6-sol"

    await session.compact()
    assert anthropic.compact_calls[-1]["model"] is None
    assert anthropic.compact_calls[-1]["effort"] is None


@pytest.mark.asyncio
async def test_tool_run_or_turn_started_before_failure_commits_thread_and_docs(
    session,
):
    (session.workspace.documents_path / "inputs.md").write_text("# Inputs\n")

    class ToolThenFailBackend(FakeBackend):
        async def run_turn(self, prompt, *, tools, **kwargs):
            self.calls.append(
                TurnCall(
                    prompt=prompt,
                    system_prompt=kwargs["system_prompt"],
                    model=kwargs["model"],
                    effort=kwargs["effort"],
                    auth=kwargs["auth"],
                    resume_session_id=kwargs.get("resume_session_id"),
                    workspace_path=kwargs["workspace_path"],
                )
            )
            yield SessionStarted("thread-via-tool")
            by_name = {spec.name: spec for spec in tools}
            await by_name["list_documents"].handler({})
            raise RuntimeError("interrupted mid-probe")

    backend = attach_backend(session, ToolThenFailBackend(provider="openai"), "openai")
    session.model = "gpt-5.6-sol"
    assert await session.query("run a tool first") is False
    assert session._sdk_session_id == "thread-via-tool"
    assert session._active_provider == "openai"
    assert "inputs.md" in session._sent_documents

    # Next turn resumes the committed thread without resending unchanged inputs.md.
    class TurnStartedThenFailBackend(FakeBackend):
        async def run_turn(self, prompt, **kwargs):
            self.calls.append(
                TurnCall(
                    prompt=prompt,
                    system_prompt=kwargs["system_prompt"],
                    model=kwargs["model"],
                    effort=kwargs["effort"],
                    auth=kwargs["auth"],
                    resume_session_id=kwargs.get("resume_session_id"),
                    workspace_path=kwargs["workspace_path"],
                )
            )
            yield TurnStarted()
            raise RuntimeError("aborted during reasoning")

    turn_started_backend = attach_backend(
        session, TurnStartedThenFailBackend(provider="openai"), "openai"
    )
    session._session_initialized = False
    assert await session.query("follow up") is False
    assert session._session_initialized is True
    assert turn_started_backend.calls[-1].resume_session_id == "thread-via-tool"
    assert "### inputs.md" not in turn_started_backend.calls[-1].prompt
    assert backend.calls[-1].resume_session_id is None


@pytest.mark.asyncio
async def test_concurrent_record_dirty_and_non_hook_compaction_reload_labels(session):
    started = asyncio.Event()
    release = asyncio.Event()

    class PausedBackend(FakeBackend):
        async def run_turn(self, prompt, **kwargs):
            self.calls.append(
                TurnCall(
                    prompt=prompt,
                    system_prompt=kwargs["system_prompt"],
                    model=kwargs["model"],
                    effort=kwargs["effort"],
                    auth=kwargs["auth"],
                    resume_session_id=kwargs.get("resume_session_id"),
                    workspace_path=kwargs["workspace_path"],
                )
            )
            started.set()
            await release.wait()
            yield SessionStarted("oai-1")
            yield TextBlockDone("ok")
            yield TurnResult()

    attach_backend(
        session, PausedBackend(provider="openai", uses_compact_hook=False), "openai"
    )
    session.model = "gpt-5.6-sol"
    task = asyncio.create_task(session.query("first turn"))
    await started.wait()
    # Another explorer modifies the shared record after first turn's prompt was built
    # but before the turn makes progress.
    session.workspace.notes.add_evidence("Observed later.", ["source"])
    session.mark_record_dirty()
    release.set()
    assert await task is True
    assert session._record_dirty is True

    backend2 = attach_backend(
        session, FakeBackend(provider="openai", uses_compact_hook=False), "openai"
    )
    session.timeout_s = 900
    await session.query("second turn")
    assert backend2.tool_timeout_s == 960
    assert (
        "The shared scientific record changed in another exploration."
        in backend2.calls[-1].prompt
    )
    assert "Observed later." in backend2.calls[-1].prompt

    events: list[tuple[str, dict[str, Any]]] = []
    session.subscribe(lambda et, payload: events.append((et, payload)))
    compacting_backend = attach_backend(
        session,
        FakeBackend(
            provider="openai",
            uses_compact_hook=False,
            events=[
                SessionStarted("oai-1"),
                CompactionNotice(),
                TextBlockDone("after compaction"),
                TurnResult(),
            ],
        ),
        "openai",
    )
    await session.query("trigger auto compact")
    compacted = [payload for et, payload in events if et == "context_compacted"]
    assert compacted and compacted[-1]["record_reloaded"] is False
    assert session._record_dirty is True

    await session.query("post compaction turn")
    assert "Conversation history was compacted." in compacting_backend.calls[-1].prompt
    assert (
        "The shared scientific record changed in another exploration."
        not in compacting_backend.calls[-1].prompt
    )


@pytest.mark.asyncio
async def test_steer_during_active_turn_delivers_before_second_probe_without_resending_initial_docs(
    session,
):
    (session.workspace.documents_path / "initial.md").write_text("# Initial\n")
    steered_texts = []

    class SteeredTurnBackend(FakeBackend):
        async def run_turn(self, prompt, *, tools, **kwargs):
            self.calls.append(
                TurnCall(
                    prompt=prompt,
                    system_prompt=kwargs["system_prompt"],
                    model=kwargs["model"],
                    effort=kwargs["effort"],
                    auth=kwargs["auth"],
                    resume_session_id=kwargs.get("resume_session_id"),
                    workspace_path=kwargs["workspace_path"],
                )
            )
            yield SessionStarted("sdk-session")
            by_name = {spec.name: spec for spec in tools}
            first_run = await by_name["run"].handler(
                {"code": "x = 1", "expected": "sets x"}
            )
            assert not first_run.get("is_error")
            verdict = await by_name["verdict"].handler({"text": "x was set"})
            assert not verdict.get("is_error")

            session.enqueue_prompt("Now multiply x by 2", mode="steer")
            session.workspace.create_document("Added", "Factor is 2")

            steered = await by_name["run"].handler(
                {"code": "print(x)", "expected": "1"}
            )
            assert not steered.get("is_error")
            steered_texts.append(steered["content"][0]["text"])

            third_run = await by_name["run"].handler(
                {"code": "print(x * 2)", "expected": "2"}
            )
            assert not third_run.get("is_error")
            assert "Probe S001/P002" in third_run["content"][0]["text"]
            await by_name["verdict"].handler({"text": "x * 2 was computed"})

            yield TextBlockDone("Done.")
            yield TurnResult()

    attach_backend(session, SteeredTurnBackend())
    assert await session.query("Start") is True
    assert len(steered_texts) == 1
    assert "Now multiply x by 2" in steered_texts[0]
    assert "Factor is 2" in steered_texts[0]
    assert "# Initial" not in steered_texts[0]
    assert session.kernel.executed_codes == ["x = 1", "print(x * 2)"]


@pytest.mark.asyncio
async def test_session_status_tracks_kernel_probing(session):
    session._is_busy = True
    await session.emit("status_change", status="thinking")
    assert session.status == "thinking"

    await session.emit("probe_start", probe_id="S001/P001", kernel_alive=True)
    assert session.status == "probing"

    await session.emit("probe_finish", probe_id="S001/P001", status="completed")
    assert session.status == "thinking"

    await session.emit("probe_start", probe_id="S001/P002", backend="slurm")
    assert session.status == "thinking"

    session._is_busy = False
    await session.emit("status_change", status="idle")
    assert session.status == "idle"


@pytest.mark.asyncio
async def test_query_snapshots_turn_model_effort_and_auth_across_concurrent_refresh(
    session,
):
    openai = attach_backend(session, FakeBackend(provider="openai"), "openai")
    anthropic = attach_backend(session, FakeBackend(provider="anthropic"), "anthropic")
    session.model_providers = {"gpt-5.6-sol": "openai", "claude-sonnet-5": "anthropic"}
    session.model_raw_ids = {
        "gpt-5.6-sol": "gpt-5.6-sol",
        "claude-sonnet-5": "claude-sonnet-5",
    }
    session.model = "gpt-5.6-sol"
    session.effort = "high"

    async def mutate_on_thinking(event_type, payload):
        if event_type == "status_change" and payload.get("status") == "thinking":
            session.auth.set("openai", "api_key", "sk-later")
            session.model = "claude-sonnet-5"
            session.effort = "low"
            session.model_providers = {"claude-sonnet-5": "anthropic"}
            session.model_raw_ids = {"claude-sonnet-5": "claude-sonnet-5"}

    session.subscribe(mutate_on_thinking)
    assert await session.query("run with snapshotted turn config") is True
    assert anthropic.calls == []
    assert len(openai.calls) == 1
    call = openai.calls[0]
    assert call.model == "gpt-5.6-sol"
    assert call.effort == "high"
    assert call.auth.method == "cli"


@pytest.mark.asyncio
async def test_recap_kernel_preserved_reflects_kernel_notice_and_provider_switch(
    session,
):
    anthropic = attach_backend(
        session,
        FakeBackend(provider="anthropic", session_id="ant-1"),
        "anthropic",
    )
    openai = attach_backend(
        session,
        FakeBackend(provider="openai", session_id="oai-1"),
        "openai",
    )
    session.model_providers = {"claude-sonnet-5": "anthropic", "gpt-5.6-sol": "openai"}
    session.model_raw_ids = {
        "claude-sonnet-5": "claude-sonnet-5",
        "gpt-5.6-sol": "gpt-5.6-sol",
    }
    session.model = "claude-sonnet-5"
    assert await session.query("first question") is True
    assert session._kernel_notice == ""

    # Switching providers while the kernel stayed live marks the kernel preserved.
    session.model = "gpt-5.6-sol"
    assert await session.query("switch while kernel live") is True
    assert (
        "The live Jupyter kernel and its in-memory variables are still intact"
        in openai.calls[-1].prompt
    )

    # After restarting the kernel, switching providers must NOT claim the kernel is intact.
    session.kernel.restart("manual restart")
    assert session._kernel_notice
    session.model = "claude-sonnet-5"
    assert await session.query("switch after restart") is True
    rebuild_prompt = anthropic.calls[-1].prompt
    assert (
        "The live Jupyter kernel and its in-memory variables are still intact"
        not in rebuild_prompt
    )
    assert "The kernel may no longer hold the state it had then" in rebuild_prompt
