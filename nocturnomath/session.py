"""Conversation orchestration for a scientific exploration session."""

import asyncio
import json
import logging
import subprocess
import time
import uuid
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

from .environment import ResearchEnvironment
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
        python: str | None = None,
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

        self.environment = ResearchEnvironment(workspace_path, python)
        self.kernel = Kernel(cwd=workspace_path, environment=self.environment)
        try:
            self.workspace = Workspace(workspace_path)
            self.environment.save()
            self.kernel.on_start = self.record_kernel_start
            self.record_kernel_start("opened")
        except Exception:
            self.kernel.shutdown()
            raise
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

    def set_workspace(self, new_dir: Path | str, python=None):
        self.require_idle("change workspace")
        new_path = Path(new_dir).resolve()
        environment = ResearchEnvironment(new_path, python)
        kernel = Kernel(cwd=new_path, environment=environment)
        try:
            workspace = Workspace(new_path)
            environment.save()
        except Exception:
            kernel.shutdown()
            raise
        self.kernel.shutdown()
        self.workspace = workspace
        self.environment = environment
        self.kernel = kernel
        self.kernel.on_start = self.record_kernel_start
        self.record_kernel_start("workspace changed")
        self._session_initialized = False
        self._sdk_session_id = None
        self._resume_prefix = None
        self._results_since_note = 0
        self._pending_verdict = None
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
        changed = path != self.workspace.transcript_path
        previous_path = self.workspace.transcript_path
        self.workspace.transcript_path = path
        if changed:
            try:
                self.kernel.restart("historical session resumed")
            except Exception:
                self.workspace.transcript_path = previous_path
                raise
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
            "kernel_reset": changed,
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

                prefix += self._kernel_notice
                self._kernel_notice = ""
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
        previous_path = self.workspace.transcript_path
        self.start_new_transcript()
        try:
            self.kernel.restart("new session")
        except Exception:
            self.workspace.transcript_path = previous_path
            raise
        self._session_initialized = False
        self._sdk_session_id = None
        self._resume_prefix = None
        self._results_since_note = 0
        self._pending_verdict = None

    def record_environment(self):
        directory = self.transcript_path.parent / "environments"
        directory.mkdir(exist_ok=True)
        path = directory / f"{uuid.uuid4().hex}.json"
        path.write_text(json.dumps(self.environment.snapshot(), indent=2) + "\n")
        self.environment_record = str(path.relative_to(self.workspace.path))

    def record_kernel_start(self, reason):
        self.record_environment()
        self.kernel_id = uuid.uuid4().hex
        self.log_transcript(
            "kernel_started",
            kernel_id=self.kernel_id,
            reason=reason,
            environment=self.environment_record,
        )
        self._kernel_notice = (
            f"\nKernel {self.kernel_id} started ({reason}). All prior in-memory variables are gone. "
            f"Python: {self.environment.python}. Workspace: {self.workspace.path}. "
            "Reload only what the next probe needs; do not automatically replay old probes.\n"
        )

    def install_packages(self, packages):
        if not packages or any(not p or p.startswith("-") for p in packages):
            raise ValueError("Provide package requirements, not installer options.")
        directory = (
            self.transcript_path.parent / "environment_changes" / uuid.uuid4().hex
        )
        directory.mkdir(parents=True)
        command = [
            self.environment.uv(),
            "pip",
            "install",
            "--python",
            str(self.environment.python),
            *packages,
        ]
        metadata = {
            "command": command,
            "kernel_id": self.kernel_id,
            "before": self.environment_record,
            "started": time.time(),
            "status": "started",
        }
        path = directory / "install.json"
        path.write_text(json.dumps(metadata, indent=2) + "\n")
        try:
            with (directory / "output.txt").open("w") as output:
                result = subprocess.run(
                    command,
                    check=False,
                    cwd=self.workspace.path,
                    env=self.environment.process_env(),
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                )
            metadata["status"] = "completed" if result.returncode == 0 else "failed"
        except Exception as exc:
            metadata["status"] = "failed"
            with (directory / "output.txt").open("a") as output:
                output.write(str(exc))
        finally:
            metadata["finished"] = time.time()
            try:
                self.record_environment()
                metadata["after"] = self.environment_record
            finally:
                path.write_text(json.dumps(metadata, indent=2) + "\n")
        self.log_transcript(
            "packages_installed",
            **metadata,
            output=str((directory / "output.txt").relative_to(self.workspace.path)),
        )
        return metadata["status"], (directory / "output.txt").read_text()
