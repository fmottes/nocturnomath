import asyncio
import json
import os
import shlex
import subprocess
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
)

from nocturnomath.auth import ClaudeAuth


def stream_event(delta, parent_tool_use_id=None):
    return StreamEvent(
        uuid="stream-event",
        session_id="sdk-session",
        event={"type": "content_block_delta", "index": 0, "delta": delta},
        parent_tool_use_id=parent_tool_use_id,
    )


class FakeClient:
    prompts: ClassVar[list[str]] = []
    options_used: ClassVar[list[Any]] = []

    def __init__(self, options):
        self.options = options
        FakeClient.options_used.append(options)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def query(self, prompt):
        self.prompts.append(prompt)

    async def receive_response(self):
        yield stream_event({"type": "text_delta", "text": "ans"})
        yield stream_event({"type": "text_delta", "text": "wer"})
        yield stream_event({"type": "thinking_delta", "thinking": "musing"})
        yield stream_event({"type": "text_delta", "text": "sub"}, "tool-use-1")
        yield AssistantMessage(
            content=[TextBlock("answer")], model="test", session_id="sdk-session"
        )


class FailingClient(FakeClient):
    async def query(self, prompt):
        raise RuntimeError("send failed")


class UsageClient(FakeClient):
    async def receive_response(self):
        yield AssistantMessage(
            content=[TextBlock("answer")],
            model="test",
            session_id="sdk-session",
            usage={
                "input_tokens": 1000,
                "cache_read_input_tokens": 12000,
                "cache_creation_input_tokens": 500,
                "output_tokens": 300,
            },
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="sdk-session",
            model_usage={"test": {"contextWindow": 200000}},
        )


class CompactingClient(FakeClient):
    async def receive_response(self):
        yield SystemMessage(subtype="compact_boundary", data={"trigger": "auto"})
        yield AssistantMessage(
            content=[TextBlock("continued")], model="test", session_id="sdk-session"
        )

    async def get_context_usage(self):
        return {"totalTokens": 4200, "rawMaxTokens": 200000, "model": "test"}


class NoBoundaryClient(FakeClient):
    async def receive_response(self):
        yield AssistantMessage(content=[], model="test", session_id="sdk-session")


class SlowUsageClient(CompactingClient):
    async def get_context_usage(self):
        await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_compaction_is_recorded_and_reloads_current_scientific_record(session):
    events = []
    session.subscribe(lambda event_type, payload: events.append(event_type))
    with patch("nocturnomath.session.ClaudeSDKClient", CompactingClient):
        await session.query("question")

    assert "context_compacted" in events
    assert any(
        record["kind"] == "compaction" for record in session.workspace.current_records()
    )
    options = FakeClient.options_used[-1]
    hook = json.loads(options.settings)["hooks"]["SessionStart"][0]
    assert hook["matcher"] == "compact"
    assert "nocturnomath.compaction" in hook["hooks"][0]["command"]
    assert options.env["NOCTURNOMATH_WORKSPACE_PATH"] == str(session.workspace_path)

    session.workspace.notes.evidence_path.write_text(
        "# Evidence\n\n## E001\n\nNew result\n"
    )
    result = await asyncio.to_thread(
        subprocess.run,
        shlex.split(hook["hooks"][0]["command"]),
        input='{"source":"compact"}',
        text=True,
        capture_output=True,
        cwd=session.workspace_path,
        env={**os.environ, **options.env},
        check=True,
    )
    context = json.loads(result.stdout)["hookSpecificOutput"]
    assert context["hookEventName"] == "SessionStart"
    assert "New result" in context["additionalContext"]
    assert "# Thoughts" in context["additionalContext"]


@pytest.mark.asyncio
async def test_manual_compaction_keeps_transcript_and_kernel_and_refreshes_usage(
    session,
):
    events = []
    session.subscribe(lambda event_type, payload: events.append((event_type, payload)))
    with patch("nocturnomath.session.ClaudeSDKClient", FakeClient):
        await session.query("original question")
    kernel = session.kernel
    with patch("nocturnomath.session.ClaudeSDKClient", CompactingClient):
        await session.compact()

    assert FakeClient.prompts[-1] == "/compact"
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
    with patch("nocturnomath.session.ClaudeSDKClient", FakeClient):
        await session.query("original question")
    with (
        patch("nocturnomath.session.ClaudeSDKClient", NoBoundaryClient),
        pytest.raises(RuntimeError, match="did not confirm"),
    ):
        await session.compact()
    assert not session.has_active_query()
    assert not any(
        record["kind"] == "compaction" for record in session.workspace.current_records()
    )


@pytest.mark.asyncio
async def test_manual_compaction_does_not_wait_for_slow_usage_refresh(session):
    with patch("nocturnomath.session.ClaudeSDKClient", FakeClient):
        await session.query("original question")
    with (
        patch("nocturnomath.session.ClaudeSDKClient", SlowUsageClient),
        patch("nocturnomath.session.CONTEXT_USAGE_TIMEOUT_S", 0.01),
    ):
        await session.compact()

    assert session.context_usage is None
    assert not session.has_active_query()
    assert any(
        record["kind"] == "compaction" for record in session.workspace.current_records()
    )


@pytest.mark.asyncio
async def test_context_usage_is_reported_and_restored_from_history(session):
    events = []
    session.subscribe(lambda event_type, payload: events.append((event_type, payload)))
    with patch("nocturnomath.session.ClaudeSDKClient", UsageClient):
        await session.query("question")

    expected = {
        "used_tokens": 13500,
        "window_tokens": 200000,
        "model": "test",
    }
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
    with patch("nocturnomath.session.ClaudeSDKClient", FakeClient):
        await session.query("question")

    assert session._sdk_session_id == "sdk-session"
    options = FakeClient.options_used[-1]
    assert options.include_partial_messages is True
    assert options.effort == "high"
    assert options.tools == []
    assert options.strict_mcp_config is True
    assert options.setting_sources == ["user"]
    assert options.skills == []
    assert options.cwd == session.workspace_path
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
async def test_query_passes_selected_effort_to_the_sdk(session):
    session.effort = "xhigh"

    with patch("nocturnomath.session.ClaudeSDKClient", FakeClient):
        await session.query("question")

    assert FakeClient.options_used[-1].effort == "xhigh"


@pytest.mark.asyncio
async def test_query_passes_selected_authentication_only_to_the_sdk(session):
    session.auth = ClaudeAuth.interactive("api_key", "top-secret-key")
    with patch("nocturnomath.session.ClaudeSDKClient", FakeClient):
        await session.query("question")

    options = FakeClient.options_used[-1]
    assert options.env["ANTHROPIC_API_KEY"] == "top-secret-key"
    assert options.env["CLAUDE_CODE_OAUTH_TOKEN"] == ""
    assert options.env["ANTHROPIC_AUTH_TOKEN"] == ""
    transcript = session.transcript_path.read_text()
    assert "top-secret-key" not in transcript


@pytest.mark.asyncio
async def test_active_query_guards_state_and_interrupts(session):
    blocker = asyncio.Event()
    task = asyncio.create_task(blocker.wait())
    session._current_task = task

    with pytest.raises(RuntimeError):
        session.reset_client_session()
    with pytest.raises(RuntimeError):
        session.set_workspace(session.workspace_path / "other")
    with pytest.raises(RuntimeError):
        session.set_carry_chat_context(False)

    await session.interrupt()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_resume_reads_current_records_and_reuses_session_probe_folder(session):
    session.log_transcript("user", text="original investigation")
    session.log_transcript("meta", sdk_session_id="sdk-session")
    session.workspace.start_probe("print(3)", "3", "test")
    session.workspace.notes.add_evidence("Value = 3.", ["source"])
    session.reset_client_session()
    session.workspace.notes.strike_evidence("E001")
    resumed = session.resume_session("S001")
    assert resumed["id"] == "S001"
    assert session.workspace.start_probe("print(4)", "4", "test").name == "P002"
    with patch("nocturnomath.session.ClaudeSDKClient", FakeClient):
        await session.query("continue")
    assert "<del>" in FakeClient.prompts[-1]
    assert "Value = 3." in FakeClient.prompts[-1]


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

    with patch("nocturnomath.session.ClaudeSDKClient", FakeClient):
        await session.query("first")
        first = FakeClient.prompts[-1]
        assert "### inputs.md\n\n# Inputs\n\nN = 10\n" in first
        assert "draft.md" not in first
        assert (
            first.index("# Evidence")
            < first.index("### inputs.md")
            < first.index("first")
        )

        await session.query("second")
        assert "inputs.md" not in FakeClient.prompts[-1]

        (documents / "inputs.md").write_text("# Inputs\n\nN = 20\n")
        await session.query("third")
        third = FakeClient.prompts[-1]
        assert "N = 20" in third and "N = 10" not in third

        session.set_documents_default(False)
        await session.query("fourth")
        assert "inputs.md" not in FakeClient.prompts[-1]
        assert session.list_documents() == [
            {**item, "included": False} for item in session.workspace.list_documents()
        ]

        session.set_document_included("inputs.md", True)
        session.set_carry_chat_context(False)
        await session.query("fifth")
        await session.query("sixth")
        assert "N = 20" in FakeClient.prompts[-1]


@pytest.mark.asyncio
async def test_document_context_is_retried_when_submission_fails(session):
    (session.workspace.documents_path / "inputs.md").write_text("# Inputs\n")

    with patch("nocturnomath.session.ClaudeSDKClient", FailingClient):
        await session.query("first")
    assert session._sent_documents == {}

    with patch("nocturnomath.session.ClaudeSDKClient", FakeClient):
        await session.query("second")
    assert "### inputs.md\n\n# Inputs\n" in FakeClient.prompts[-1]
    assert "# Evidence" in FakeClient.prompts[-1]
