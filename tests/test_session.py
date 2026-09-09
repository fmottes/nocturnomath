import asyncio
from typing import ClassVar
from unittest.mock import patch

import pytest
from claude_agent_sdk import AssistantMessage, TextBlock


class FakeClient:
    prompts: ClassVar[list[str]] = []

    def __init__(self, options):
        self.options = options

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def query(self, prompt):
        self.prompts.append(prompt)

    async def receive_response(self):
        yield AssistantMessage(
            content=[TextBlock("answer")], model="test", session_id="sdk-session"
        )


@pytest.mark.asyncio
async def test_query_emits_text_tracks_context_and_logs(session):
    events = []
    session.subscribe(lambda event_type, payload: events.append(event_type))
    with patch("nocturnomath.session.ClaudeSDKClient", FakeClient):
        await session.query("question")

    assert session._sdk_session_id == "sdk-session"
    assert "assistant_text" in events
    assert events[0] == "status_change"
    assert events[-1] == "status_change"
    records = session.workspace.read_transcript(session.transcript_path)
    assert [record["kind"] for record in records] == [
        "kernel_started",
        "user",
        "meta",
        "agent",
    ]


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
