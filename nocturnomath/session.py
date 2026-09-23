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
    StreamEvent,
    TextBlock,
    create_sdk_mcp_server,
)

from .auth import ClaudeAuth
from .environment import ResearchEnvironment
from .kernel import Kernel
from .prompt import build_system_prompt
from .tools import build_tools
from .workspace import Workspace

logger = logging.getLogger("nocturnomath")

CARRY_CHAT_CONTEXT = True
DEFAULT_EFFORT = "high"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def _stream_delta_text(event: StreamEvent) -> str | None:
    """Return the incremental assistant text a stream event carries, if any."""
    if event.parent_tool_use_id:
        return None
    raw = event.event or {}
    if raw.get("type") != "content_block_delta":
        return None
    delta = raw.get("delta") or {}
    if delta.get("type") != "text_delta":
        return None
    return delta.get("text") or None


class ExplorationSession:
    """Own one kernel, one workspace, and one Claude conversation at a time."""

    def __init__(
        self,
        workspace_path: Path | str = ".",
        model: str | None = None,
        effort: str | None = None,
        timeout_s: int = 600,
        image_cap: int = 2,
        carry_chat_context: bool = CARRY_CHAT_CONTEXT,
        python: str | None = None,
        auth: ClaudeAuth | None = None,
    ):
        if effort is not None and effort not in EFFORT_LEVELS:
            raise ValueError(f"Unknown effort level: {effort}")
        self.model = model
        self.effort = effort or DEFAULT_EFFORT
        self.default_model = model
        self.default_effort: str | None = self.effort
        self.timeout_s = timeout_s
        self.image_cap = image_cap
        self.carry_chat_context = carry_chat_context
        self.auth = auth or ClaudeAuth()
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
        self._pending_kernel_start: tuple[str, str] | None = None
        self._record_dirty = False
        # Documents chosen for the next message, and what this conversation already saw.
        self.document_choices: dict[str, bool] = {}
        self._sent_documents: dict[str, str] = {}

        self.environment = ResearchEnvironment(workspace_path, python)
        self.kernel = Kernel(cwd=workspace_path, environment=self.environment)
        try:
            self.workspace = Workspace(workspace_path)
            self.environment.save()
            self.load_document_settings()
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
        self.activate_session()
        self.workspace.log_transcript(kind, **fields)

    def activate_session(self):
        if not self.workspace.session_pending:
            return
        self.workspace.ensure_session()
        if self._pending_kernel_start:
            kernel_id, reason = self._pending_kernel_start
            self._pending_kernel_start = None
            self._write_kernel_start(kernel_id, reason)

    def opening_notes(self) -> str:
        return self.workspace.opening_notes()

    def mark_record_dirty(self):
        """Ensure a concurrent explorer sees the shared record on its next turn."""
        self._record_dirty = True

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
        self.load_document_settings()
        logger.info(f"Switched workspace to: {new_path}")

    def load_document_settings(self):
        self.documents_default = bool(
            self.workspace.read_config().get("documents_default", True)
        )
        self.document_choices = {}
        self._sent_documents = {}

    def list_documents(self) -> list[dict[str, Any]]:
        """Documents with their inclusion flag; unseen files take the default."""
        documents = self.workspace.list_documents()
        names = {document["name"] for document in documents}
        self.document_choices = {
            name: self.document_choices.get(name, self.documents_default)
            for name in names
        }
        for document in documents:
            document["included"] = self.document_choices[document["name"]]
        return documents

    def set_document_included(self, name: str, included: bool) -> bool:
        self.workspace.read_document(name)
        self.document_choices[name] = bool(included)
        return self.document_choices[name]

    def delete_document(self, name: str):
        self.workspace.delete_document(name)
        self.document_choices.pop(name, None)
        self._sent_documents.pop(name, None)

    def set_documents_default(self, enabled: bool) -> bool:
        """Persist the default and reset the current selection to match it."""
        self.documents_default = bool(enabled)
        self.workspace.update_config(documents_default=self.documents_default)
        self.document_choices = {}
        return self.documents_default

    def set_session_defaults(self, model: str, effort: str | None):
        """Persist the model and effort applied when the next session starts."""
        self.default_model = model
        self.default_effort = effort
        self.workspace.update_config(default_model=model, default_effort=effort)

    def document_context(self) -> tuple[str, dict[str, str]]:
        """Build document context and the contents to mark sent after submission."""
        sections = []
        sent_documents = {}
        for document in self.list_documents():
            if not document["included"]:
                continue
            content = self.workspace.read_document(document["name"])["content"]
            if (
                self.carry_chat_context
                and self._sent_documents.get(document["name"]) == content
            ):
                continue
            sent_documents[document["name"]] = content
            sections.append(f"### {document['name']}\n\n{content.rstrip()}\n")
        if not sections:
            return "", sent_documents
        context = (
            "Documents (extra context selected by the user, stored in "
            ".nocturnomath/documents/):\n\n" + "\n".join(sections) + "\n---\n\n"
        )
        return context, sent_documents

    def read_file(self, relative_path: str) -> dict[str, Any]:
        return self.workspace.read_file(relative_path)

    def list_plots(self) -> list[dict[str, Any]]:
        return self.workspace.list_plots()

    def list_sessions(self) -> list[dict[str, Any]]:
        return self.workspace.list_sessions()

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        return self.workspace.load_session(session_id)

    def resume_session(
        self, session_id: str, restore_kernel: bool = False
    ) -> dict[str, Any]:
        if self.has_active_query():
            raise RuntimeError("Cannot resume while the agent is running a query.")

        path = self.workspace.transcript_file(session_id)
        records = self.workspace.read_transcript(path)
        sdk_id = self.workspace.sdk_id(records)
        changed = path != self.workspace.transcript_path
        previous_state = self.workspace.session_state()
        self.workspace.use_transcript(path)
        if changed or restore_kernel:
            try:
                reason = (
                    "session kernel restored"
                    if restore_kernel
                    else "historical session resumed"
                )
                self.kernel.restart(reason)
                replayed = self._replay_probe_code(records) if restore_kernel else 0
            except Exception:
                self.workspace.restore_session_state(previous_state)
                raise
        else:
            replayed = 0
        self._sdk_session_id = sdk_id
        self._sent_documents = {}
        self._results_since_note = 0
        self._pending_verdict = None
        if self.carry_chat_context:
            self._session_initialized = False
            self._resume_prefix = None if sdk_id else self.workspace.recap(records)
        else:
            self._session_initialized = False
            self._resume_prefix = None

        if restore_kernel:
            probe_word = "probe" if replayed == 1 else "probes"
            self._kernel_notice = (
                f"\nKernel {self.kernel_id} was reconstructed by replaying {replayed} stored "
                f"{probe_word} in order. Treat its in-memory state as an approximation of the end "
                "of the resumed session.\n"
            )
        self.log_transcript(
            "resumed", restore_kernel=restore_kernel, probes_replayed=replayed
        )
        logger.info(f"Resumed chat {path.name}")
        return {
            "id": path.parent.name,
            "kernel_reset": changed or restore_kernel,
            "kernel_restored": restore_kernel,
            "probes_replayed": replayed,
            "records": records,
            "carry_chat_context": self.carry_chat_context,
            "context_restored": self.carry_chat_context and sdk_id is not None,
        }

    def _replay_probe_code(self, records: list[dict[str, Any]]) -> int:
        """Rebuild kernel state from recorded probes without creating new artifacts."""
        replayed = 0
        for record in records:
            if record.get("kind") != "probe_started":
                continue
            code = record.get("code")
            if not isinstance(code, str) or not code.strip():
                continue
            self.kernel.execute(code, self.timeout_s)
            replayed += 1
        return replayed

    def set_carry_chat_context(self, enabled: bool) -> bool:
        self.require_idle("change context settings")
        enabled = bool(enabled)
        if enabled == self.carry_chat_context:
            return enabled
        self.carry_chat_context = enabled
        self._sdk_session_id = None
        self._resume_prefix = None
        self._sent_documents = {}
        self._session_initialized = False
        if not self.workspace.session_pending:
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
        if self.model is None:
            raise RuntimeError("No model available.")

        self._is_busy = True
        self._pending_verdict = None
        try:
            # Snapshot the selection before yielding so changes made while this turn runs
            # apply to the next message.
            document_prefix, sent_documents = self.document_context()
            await self.emit("status_change", status="thinking")
            self.log_transcript("user", text=user_text)
            options = ClaudeAgentOptions(
                # The research MCP tools are the complete execution surface. Do not
                # inherit Claude Code's built-ins or unrelated MCP configuration.
                tools=[],
                system_prompt=build_system_prompt(),
                mcp_servers={
                    "explore": create_sdk_mcp_server("explore", tools=self.tools)
                },
                strict_mcp_config=True,
                # Keep the user's authentication/provider settings, but do not load
                # instructions or hooks from the selected research workspace.
                setting_sources=["user"],
                skills=[],
                permission_mode="bypassPermissions",
                model=self.model,
                effort=self.effort,
                max_buffer_size=20 * 1024 * 1024,
                include_partial_messages=True,
                env=self.auth.sdk_env(),
                cwd=self.workspace_path,
                resume=self._sdk_session_id if self.carry_chat_context else None,
            )
            async with ClaudeSDKClient(options) as client:
                self._current_client = client
                prefix = ""
                resume_prefix = self._resume_prefix
                initialize_session = False
                if self.carry_chat_context and resume_prefix:
                    prefix = self.opening_notes() + resume_prefix
                    initialize_session = True
                elif not self.carry_chat_context or not self._session_initialized:
                    prefix = self.opening_notes()
                    initialize_session = True
                elif self._record_dirty:
                    prefix = (
                        "The shared scientific record changed in another exploration. "
                        "Treat this current copy as authoritative:\n\n"
                        + self.opening_notes()
                    )

                prefix += document_prefix
                kernel_notice = self._kernel_notice
                prefix += kernel_notice
                prompt = prefix + user_text if prefix else user_text
                await client.query(prompt)
                self._resume_prefix = None
                if initialize_session:
                    self._session_initialized = True
                if self._kernel_notice == kernel_notice:
                    self._kernel_notice = ""
                self._record_dirty = False
                self._sent_documents.update(sent_documents)
                accumulated_text = []
                async for message in client.receive_response():
                    self._track_session_id(getattr(message, "session_id", None))
                    if isinstance(message, StreamEvent):
                        delta = _stream_delta_text(message)
                        if delta:
                            await self.emit("assistant_delta", text=delta)
                    elif isinstance(message, AssistantMessage):
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
        previous_state = self.workspace.session_state()
        self.start_new_transcript()
        try:
            self.kernel.restart("new session")
        except Exception:
            self.workspace.restore_session_state(previous_state)
            raise
        self._session_initialized = False
        self._sdk_session_id = None
        self._resume_prefix = None
        self._sent_documents = {}
        self._results_since_note = 0
        self._pending_verdict = None
        self.model = self.default_model
        self.effort = self.default_effort

    def record_environment(self):
        directory = self.transcript_path.parent / "environments"
        directory.mkdir(exist_ok=True)
        path = directory / f"{uuid.uuid4().hex}.json"
        path.write_text(json.dumps(self.environment.snapshot(), indent=2) + "\n")
        self.environment_record = str(path.relative_to(self.workspace.path))

    def record_kernel_start(self, reason):
        self.kernel_id = uuid.uuid4().hex
        self._kernel_notice = (
            f"\nKernel {self.kernel_id} started ({reason}). All prior in-memory variables are gone. "
            f"Python: {self.environment.python}. Workspace: {self.workspace.path}. "
            "Reload only what the next probe needs; do not automatically replay old probes.\n"
        )
        if self.workspace.session_pending:
            self._pending_kernel_start = (self.kernel_id, reason)
            return
        self._pending_kernel_start = None
        self._write_kernel_start(self.kernel_id, reason)

    def _write_kernel_start(self, kernel_id, reason):
        self.record_environment()
        self.workspace.log_transcript(
            "kernel_started",
            kernel_id=kernel_id,
            reason=reason,
            environment=self.environment_record,
        )

    def install_packages(self, packages):
        self.activate_session()
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
