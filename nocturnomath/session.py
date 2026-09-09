"""Conversation orchestration for a scientific exploration session."""

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    TextBlock,
    create_sdk_mcp_server,
)

from .kernel import Kernel
from .prompt import SYSTEM_PROMPT
from .tools import build_tools
from .workspace import Workspace

logger = logging.getLogger("nocturnomath")

CARRY_CHAT_CONTEXT = True


class ExplorationSession:
    """Own one kernel, one workspace, and one Claude conversation at a time."""

    def __init__(
        self,
        workspace_path: Path | str = ".",
        model: str = "claude-opus-5",
        timeout_s: int = 600,
        image_cap: int = 2,
        carry_chat_context: bool = CARRY_CHAT_CONTEXT,
    ):
        self.model = model
        self.timeout_s = timeout_s
        self.image_cap = image_cap
        self.carry_chat_context = carry_chat_context
        self.event_subscribers: list[Callable[[str, dict[str, Any]], Any]] = []
        self._is_busy = False
        self._current_client: ClaudeSDKClient | None = None
        self._current_task: asyncio.Task | None = None
        self._probe_task: asyncio.Task | None = None
        self._session_initialized = False
        self._results_since_note = 0
        self._pending_verdict: str | None = None
        self._sdk_session_id: str | None = None
        self._resume_prefix: str | None = None

        self.workspace = Workspace(workspace_path)
        self.kernel = Kernel(cwd=self.workspace.path)
        self.tools = build_tools(self)

    @property
    def workspace_path(self) -> Path:
        return self.workspace.path

    @property
    def transcript_path(self) -> Path:
        return self.workspace.transcript_path

    def has_active_query(self) -> bool:
        return (
            self._is_busy
            or self.kernel.busy
            or (self._probe_task is not None and not self._probe_task.done())
            or (self._current_task is not None and not self._current_task.done())
        )

    def require_idle(self, action: str):
        if self.has_active_query():
            raise RuntimeError(f"Cannot {action} while the agent is running a query.")

    def subscribe(self, callback: Callable[[str, dict[str, Any]], Any]):
        self.event_subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[str, dict[str, Any]], Any]):
        if callback in self.event_subscribers:
            self.event_subscribers.remove(callback)

    async def emit(self, event_type: str, **data):
        payload = {"type": event_type, "timestamp": time.time(), **data}
        for subscriber in list(self.event_subscribers):
            try:
                result = subscriber(event_type, payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.error(f"Error in event subscriber: {exc}")

    def log_transcript(self, kind: str, **fields):
        self.workspace.log_transcript(kind, **fields)

    def opening_notes(self) -> str:
        return self.workspace.opening_notes()

    def start_new_transcript(self):
        self.workspace.start_new_transcript()

    def set_workspace(self, new_dir: Path | str):
        self.require_idle("change workspace")
        new_path = Path(new_dir).resolve()
        self.workspace.set_path(new_path)
        self._session_initialized = False
        self._sdk_session_id = None
        self._resume_prefix = None
        self._results_since_note = 0
        self._pending_verdict = None
        self.kernel.set_cwd(new_path)
        logger.info(f"Switched workspace to: {new_path}")

    def list_markdown_files(self) -> list[dict[str, Any]]:
        return self.workspace.list_markdown_files()

    def read_file(self, relative_path: str) -> dict[str, Any]:
        return self.workspace.read_file(relative_path)

    def list_plots(self) -> list[dict[str, Any]]:
        return self.workspace.list_plots()

    def list_sessions(self) -> list[dict[str, Any]]:
        return self.workspace.list_sessions()

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        return self.workspace.load_session(session_id)

    def resume_session(self, session_id: str) -> dict[str, Any]:
        if self.has_active_query():
            raise RuntimeError("Cannot resume while the agent is running a query.")

        path = self.workspace.transcript_file(session_id)
        records = self.workspace.read_transcript(path)
        sdk_id = self.workspace.sdk_id(records)
        self.workspace.transcript_path = path
        self._sdk_session_id = sdk_id
        self._results_since_note = 0
        self._pending_verdict = None
        if self.carry_chat_context:
            self._session_initialized = False
            self._resume_prefix = None if sdk_id else self.workspace.recap(records)
        else:
            self._session_initialized = False
            self._resume_prefix = None

        self.log_transcript("resumed")
        logger.info(f"Resumed chat {path.name}")
        return {
            "id": path.parent.name,
            "records": records,
            "carry_chat_context": self.carry_chat_context,
            "context_restored": self.carry_chat_context and sdk_id is not None,
        }

    def set_carry_chat_context(self, enabled: bool) -> bool:
        self.require_idle("change context settings")
        enabled = bool(enabled)
        if enabled == self.carry_chat_context:
            return enabled
        self.carry_chat_context = enabled
        self._sdk_session_id = None
        self._resume_prefix = None
        self._session_initialized = False
        self.log_transcript("meta", carry_chat_context=enabled)
        logger.info(f"carry_chat_context set to {enabled}")
        return enabled

    async def interrupt(self):
        if self.kernel.busy:
            self.kernel.interrupt()
            await self.emit("kernel_interrupted")
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            await self.emit("query_cancelled")

    def shutdown(self):
        self.kernel.shutdown()

    async def query(self, user_text: str):
        if self._is_busy:
            raise RuntimeError("Agent is already running a query.")

        self._is_busy = True
        self._pending_verdict = None
        await self.emit("status_change", status="thinking")
        self.log_transcript("user", text=user_text)
        options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            mcp_servers={"explore": create_sdk_mcp_server("explore", tools=self.tools)},
            disallowed_tools=[
                "Bash",
                "BashOutput",
                "KillShell",
                "Task",
                "Agent",
                "Workflow",
                "Monitor",
            ],
            permission_mode="bypassPermissions",
            model=self.model,
            max_buffer_size=20 * 1024 * 1024,
            # Deliberately preserve the existing SDK working-directory behavior.
            resume=self._sdk_session_id if self.carry_chat_context else None,
        )

        try:
            async with ClaudeSDKClient(options) as client:
                self._current_client = client
                prefix = ""
                resume_prefix = self._resume_prefix
                self._resume_prefix = None
                if self.carry_chat_context and resume_prefix:
                    prefix = self.opening_notes() + resume_prefix
                    self._session_initialized = True
                elif not self.carry_chat_context or not self._session_initialized:
                    prefix = self.opening_notes()
                    self._session_initialized = True

                prompt = prefix + user_text if prefix else user_text
                await client.query(prompt)
                accumulated_text = []
                async for message in client.receive_response():
                    self._track_session_id(getattr(message, "session_id", None))
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                accumulated_text.append(block.text)
                                self.log_transcript("agent", text=block.text)
                                await self.emit("assistant_text", text=block.text)
                await self.emit(
                    "turn_complete", full_text="\n\n".join(accumulated_text)
                )
        except asyncio.CancelledError:
            await self.emit("status_change", status="cancelled")
            raise
        except Exception as exc:
            logger.exception("Error during query execution")
            await self.emit("error", message=str(exc))
        finally:
            self._is_busy = False
            self._current_client = None
            await self.emit("status_change", status="idle")

    def _track_session_id(self, session_id: str | None):
        if not session_id or session_id == self._sdk_session_id:
            return
        self._sdk_session_id = session_id
        self.log_transcript("meta", sdk_session_id=session_id)

    def reset_client_session(self):
        self.require_idle("start a new session")
        self.kernel.restart()
        self._session_initialized = False
        self._sdk_session_id = None
        self._resume_prefix = None
        self._results_since_note = 0
        self._pending_verdict = None
        self.start_new_transcript()
