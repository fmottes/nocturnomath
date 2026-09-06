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
    with patch("xprober.session.ClaudeSDKClient", FakeClient):
        await session.query("question")

    assert session._sdk_session_id == "sdk-session"
    assert "assistant_text" in events
    assert events[0] == "status_change"
    assert events[-1] == "status_change"
    records = session.workspace.read_transcript(session.transcript_path)
    assert [record["kind"] for record in records] == ["user", "meta", "agent"]


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
