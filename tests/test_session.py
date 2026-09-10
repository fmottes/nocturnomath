import asyncio
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from claude_agent_sdk import AssistantMessage, StreamEvent, TextBlock

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


@pytest.mark.asyncio
async def test_query_streams_text_tracks_context_and_logs(session):
    events = []
    session.subscribe(lambda event_type, payload: events.append((event_type, payload)))
    with patch("nocturnomath.session.ClaudeSDKClient", FakeClient):
        await session.query("question")

    assert session._sdk_session_id == "sdk-session"
    assert FakeClient.options_used[-1].include_partial_messages is True
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
