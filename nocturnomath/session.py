"""Conversation orchestration for a scientific exploration session."""

import asyncio
import json
import logging
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .auth import AuthStore, ProviderAuth, ProviderId
from .backends import (
    DEFAULT_EFFORT,
    EFFORT_LEVELS,
    AgentBackend,
    CompactionNotice,
    MessageUsage,
    SessionResumeError,
    SessionStarted,
    TextBlockDone,
    TextDelta,
    TurnResult,
    TurnStarted,
    default_backends,
)
from .environment import ResearchEnvironment
from .kernel import Kernel
from .prompt import DEFAULT_EXPLORATION_PROMPT, build_system_prompt
from .tools import ToolSpec, build_tools
from .workspace import Workspace

logger = logging.getLogger("nocturnomath")

PROMPT_MODES = ("steer", "queue")
DEFAULT_PROMPT_MODE = "steer"


@dataclass
class PendingPrompt:
    """A prompt submitted while the explorer is already working."""

    id: str
    text: str
    mode: str
    model: str | None = None
    effort: str | None = None
    provider: ProviderId | None = None
    raw_model: str | None = None


@dataclass(frozen=True)
class BoundTurnConfig:
    """Model, effort, provider, and credential bound for one accepted turn."""

    catalogue_model: str
    provider: ProviderId
    raw_model: str
    effort: str | None
    auth: ProviderAuth


class ExplorationSession:
    """Own one kernel, one workspace, and one multi-provider conversation at a time."""

    def __init__(
        self,
        workspace_path: Path | str = ".",
        model: str | None = None,
        effort: str | None = None,
        timeout_s: int = 600,
        image_cap: int = 2,
        python: str | None = None,
        auth: AuthStore | None = None,
        backends: Mapping[ProviderId, AgentBackend] | None = None,
        model_providers: Mapping[str, ProviderId] | None = None,
        model_raw_ids: Mapping[str, str] | None = None,
    ):
        if effort is not None and effort not in EFFORT_LEVELS:
            raise ValueError(f"Unknown effort level: {effort}")
        self.model = model
        self.effort = effort or DEFAULT_EFFORT
        self.default_model = model
        self.default_effort: str | None = self.effort
        self.default_provider: ProviderId | None = None
        self.timeout_s = timeout_s
        self.image_cap = image_cap
        self.auth = auth or AuthStore()
        self.backends: dict[ProviderId, AgentBackend] = (
            dict(backends) if backends is not None else default_backends()
        )
        self.model_providers: dict[str, ProviderId] = (
            dict(model_providers) if model_providers is not None else {}
        )
        self.model_raw_ids: dict[str, str] = (
            dict(model_raw_ids) if model_raw_ids is not None else {}
        )
        self.event_subscribers: list[Callable[[str, dict[str, Any]], Any]] = []
        self._is_busy = False
        self.status = "idle"
        self._current_task: asyncio.Task | None = None
        self._probe_task: asyncio.Task | None = None
        self._probe_operation_lock = asyncio.Lock()
        self._session_initialized = False
        self._results_since_note = 0
        self._pending_verdict: str | None = None
        self._pending_verdict_probe_id: str | None = None
        self._slurm_review_probe_id: str | None = None
        self._active_provider: ProviderId | None = None
        self._owner_provider: ProviderId | None = None
        self._active_model: str | None = None
        self._active_raw_model: str | None = None
        self._active_effort: str | None = None
        self._sdk_session_id: str | None = None
        self._resume_prefix: str | None = None
        self._needs_recap = False
        self._recap_allow_preserved_kernel = False
        self._kernel_interrupt_requested = False
        self._kernel_notice = ""
        self._pending_kernel_start: tuple[str, str] | None = None
        self._record_dirty_seq = 0
        self._record_acked_seq = 0
        self._record_reload_reason = "concurrent"
        self._pending_prompts: list[PendingPrompt] = []
        self._turn_model: str | None = None
        self._turn_selection: tuple[ProviderId, str, str | None] | None = None
        self._bound_turn: BoundTurnConfig | None = None
        self._last_pending_task: asyncio.Task[Any] | None = None
        self._commit_turn_accepted: Callable[[], None] | None = None
        self._draining_queries = False
        self._pending_user_logs: list[str] | None = None
        self.context_usage: dict[str, Any] | None = None
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
    def active_model(self) -> str | None:
        return self._turn_model if self._turn_model is not None else self.model

    @property
    def workspace_path(self) -> Path:
        return self.workspace.path

    @property
    def transcript_path(self) -> Path:
        return self.workspace.transcript_path

    @property
    def can_compact(self) -> bool:
        return self._sdk_session_id is not None and self._active_provider is not None

    def provider_for_model(self, model: str | None) -> ProviderId:
        if model and model in self.model_providers:
            return self.model_providers[model]
        if model:
            if model.startswith(("openai/", "gpt-")):
                return "openai"
            if model.startswith(("google/", "gemini-")):
                return "google"
        return "anthropic"

    def raw_model_id(self, model: str) -> str:
        if model in self.model_raw_ids:
            return self.model_raw_ids[model]
        provider = self.provider_for_model(model)
        prefix = f"{provider}/"
        return model.removeprefix(prefix)

    def has_active_query(self) -> bool:
        return (
            self._is_busy
            or self.kernel.busy
            or self._probe_operation_lock.locked()
            or bool(self._pending_prompts)
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
        if event_type == "status_change":
            self.status = data["status"]
        elif (
            event_type == "probe_start"
            and self._is_busy
            and "kernel_alive" in data
            and data.get("backend") != "slurm"
        ):
            self.status = "probing"
        elif (
            event_type == "probe_finish"
            and self._is_busy
            and self.status == "probing"
            and data.get("backend") != "slurm"
            and data.get("status") != "slurm-error"
        ):
            self.status = "thinking"
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

    @property
    def _record_dirty(self) -> bool:
        return self._record_dirty_seq > self._record_acked_seq

    @_record_dirty.setter
    def _record_dirty(self, dirty: bool):
        if dirty:
            if not self._record_dirty:
                self._record_dirty_seq += 1
        else:
            self._record_acked_seq = self._record_dirty_seq

    def mark_record_dirty(self, reason: str = "concurrent"):
        """Ensure an explorer reloads the shared record on its next turn."""
        self._record_dirty_seq += 1
        self._record_reload_reason = reason

    def _record_reload_heading(self) -> str:
        if self._record_reload_reason == "compaction":
            return (
                "Conversation history was compacted. "
                "Treat this current scientific record as authoritative:\n\n"
            )
        return (
            "The shared scientific record changed in another exploration. "
            "Treat this current copy as authoritative:\n\n"
        )

    def start_new_transcript(self):
        self.workspace.start_new_transcript()
        self._owner_provider = None
        self._active_model = None
        self._active_raw_model = None
        self._active_effort = None
        self.context_usage = None

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
        self._active_provider = None
        self._owner_provider = None
        self._active_model = None
        self._active_raw_model = None
        self._active_effort = None
        self._sdk_session_id = None
        self._resume_prefix = None
        self._needs_recap = False
        self._recap_allow_preserved_kernel = False
        self._kernel_interrupt_requested = False
        self._results_since_note = 0
        self._pending_verdict = None
        self._pending_verdict_probe_id = None
        self._record_dirty_seq = 0
        self._record_acked_seq = 0
        self._record_reload_reason = "concurrent"
        self._pending_prompts = []
        self._turn_model = None
        self._turn_selection = None
        self._bound_turn = None
        self._pending_user_logs = None
        self.context_usage = None
        self.load_document_settings()
        logger.info(f"Switched workspace to: {new_path}")

    def load_document_settings(self):
        self.refresh_document_settings()
        self._sent_documents = {}

    def refresh_document_settings(self):
        """Read the document selections shared by this workspace's sessions."""
        config = self.workspace.read_config()
        self.documents_default = bool(config.get("documents_default", True))
        choices = config.get("document_choices", {})
        self.document_choices = (
            {
                name: included
                for name, included in choices.items()
                if isinstance(name, str) and isinstance(included, bool)
            }
            if isinstance(choices, dict)
            else {}
        )

    def list_documents(self) -> list[dict[str, Any]]:
        """Documents with their inclusion flag; unseen files take the default."""
        self.refresh_document_settings()
        documents = self.workspace.list_documents()
        for document in documents:
            document["included"] = self.document_choices.get(
                document["name"], self.documents_default
            )
        return documents

    def set_document_included(self, name: str, included: bool) -> bool:
        self.workspace.read_document(name)
        self.refresh_document_settings()
        self.document_choices[name] = bool(included)
        self.workspace.update_config(document_choices=self.document_choices)
        return self.document_choices[name]

    def delete_document(self, name: str):
        self.workspace.delete_document(name)
        self.refresh_document_settings()
        if self.document_choices.pop(name, None) is not None:
            self.workspace.update_config(document_choices=self.document_choices)
        self._sent_documents.pop(name, None)

    def set_documents_default(self, enabled: bool) -> bool:
        """Persist the default and reset the current selection to match it."""
        self.documents_default = bool(enabled)
        self.document_choices = {}
        self.workspace.update_config(
            documents_default=self.documents_default, document_choices={}
        )
        return self.documents_default

    def set_session_defaults(
        self,
        model: str,
        effort: str | None,
        provider: ProviderId | None = None,
    ):
        """Persist the model and effort applied when the next session starts."""
        self.default_model = model
        self.default_effort = effort
        self.default_provider = provider
        self.workspace.update_config(
            default_model=model,
            default_effort=effort,
            default_provider=provider,
        )

    @property
    def exploration_prompt_override(self) -> str | None:
        prompt = self.workspace.read_config().get("exploration_prompt")
        return prompt if isinstance(prompt, str) and prompt.strip() else None

    @property
    def exploration_prompt(self) -> str:
        return self.exploration_prompt_override or DEFAULT_EXPLORATION_PROMPT

    def set_exploration_prompt(self, prompt: str | None) -> str:
        if prompt is not None:
            prompt = prompt.strip()
            if not prompt:
                raise ValueError("Enter a scientific prompt, or restore the default.")
        self.workspace.update_config(exploration_prompt=prompt)
        return self.exploration_prompt

    @property
    def prompt_mode(self) -> str:
        mode = self.workspace.read_config().get("prompt_mode")
        return mode if mode in PROMPT_MODES else DEFAULT_PROMPT_MODE

    def set_prompt_mode(self, mode: str) -> str:
        if mode not in PROMPT_MODES:
            raise ValueError("Choose 'steer' or 'queue' for prompt delivery.")
        self.workspace.update_config(prompt_mode=mode)
        return mode

    def _baseline_selection(
        self,
    ) -> tuple[ProviderId | None, str | None, str | None]:
        if self._turn_selection is not None:
            return self._turn_selection
        if self.model is None:
            return (None, None, self.effort)
        return (
            self.provider_for_model(self.model),
            self.raw_model_id(self.model),
            self.effort,
        )

    def _prompt_selection(
        self,
        item: PendingPrompt,
        base: tuple[ProviderId | None, str | None, str | None],
    ) -> tuple[ProviderId | None, str | None, str | None]:
        if item.model is not None:
            provider = item.provider or self.provider_for_model(item.model)
            raw_model = item.raw_model or self.raw_model_id(item.model)
            effort = (
                (item.effort if item.effort is not None else base[2])
                if (provider, raw_model) == (base[0], base[1])
                else item.effort
            )
            return (provider, raw_model, effort)
        return (
            base[0],
            base[1],
            item.effort if item.effort is not None else base[2],
        )

    def _steerable_prefix_count(self) -> int:
        if self._slurm_review_probe_id is not None:
            return 0
        base = self._baseline_selection()
        count = 0
        for item in self._pending_prompts:
            if item.mode != "steer" or self._prompt_selection(item, base) != base:
                break
            count += 1
        return count

    def pending_prompts_snapshot(self) -> list[dict[str, str]]:
        """Serialize pending prompts without exposing backend model overrides."""
        if (
            self.status == "compacting"
            or self._slurm_review_probe_id is not None
            or (
                self._turn_selection is None
                and self._probe_task is not None
                and not self._probe_task.done()
            )
        ):
            steerable = 0
        else:
            steerable = self._steerable_prefix_count()
        return [
            {
                "id": item.id,
                "text": item.text,
                "mode": "steer"
                if item.mode == "steer" and index < steerable
                else "queue",
            }
            for index, item in enumerate(self._pending_prompts)
        ]

    def enqueue_prompt(
        self,
        text: str,
        *,
        mode: str | None = None,
        model: str | None = None,
        effort: str | None = None,
    ) -> PendingPrompt:
        """Schedule a prompt while this explorer is already working."""
        resolved_mode = mode if mode in PROMPT_MODES else self.prompt_mode
        prompt = PendingPrompt(
            id=uuid.uuid4().hex,
            text=text,
            mode=resolved_mode,
            model=model,
            effort=effort,
            provider=self.provider_for_model(model) if model is not None else None,
            raw_model=self.raw_model_id(model) if model is not None else None,
        )
        self._pending_prompts.append(prompt)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            self._last_pending_task = loop.create_task(
                self.emit(
                    "pending_prompts_changed",
                    pending_prompts=self.pending_prompts_snapshot(),
                )
            )
        return prompt

    async def remove_pending_prompt(self, prompt_id: str) -> bool:
        """Remove one pending prompt if it has not been delivered yet."""
        before = len(self._pending_prompts)
        self._pending_prompts = [
            item for item in self._pending_prompts if item.id != prompt_id
        ]
        if len(self._pending_prompts) == before:
            return False
        await self.emit(
            "pending_prompts_changed",
            pending_prompts=self.pending_prompts_snapshot(),
        )
        return True

    async def clear_pending_prompts(self) -> None:
        """Drop all pending prompts and notify listeners if any were present."""
        had_pending = bool(self._pending_prompts)
        self._pending_prompts.clear()
        if had_pending:
            await self.emit("pending_prompts_changed", pending_prompts=[])

    def pop_next_turn_prompts(self) -> list[PendingPrompt]:
        """Pop the next queued turn, batching matching leading steer prompts together."""
        if not self._pending_prompts:
            return []
        first = self._pending_prompts.pop(0)
        if first.mode != "steer":
            return [first]
        target = self._prompt_selection(first, self._baseline_selection())
        batch = [first]
        while (
            self._pending_prompts
            and self._pending_prompts[0].mode == "steer"
            and self._prompt_selection(self._pending_prompts[0], target) == target
        ):
            batch.append(self._pending_prompts.pop(0))
        return batch

    async def consume_steer_message(self) -> dict[str, Any] | None:
        """Consume matching leading steer messages before starting the next probe."""
        if self._steerable_prefix_count() == 0:
            return None
        # Yield briefly so ready assistant text arrives before logging user input.
        await asyncio.sleep(0)
        count = self._steerable_prefix_count()
        if count == 0:
            return None
        steered = self._pending_prompts[:count]
        self._pending_prompts = self._pending_prompts[count:]
        if self._commit_turn_accepted is not None:
            self._commit_turn_accepted()

        prefix = ""
        if self._record_dirty:
            prefix = self._record_reload_heading() + self.opening_notes()
            self._record_dirty = False
        document_prefix, sent_documents = self.document_context()
        if sent_documents:
            self._sent_documents.update(sent_documents)
            prefix += document_prefix

        for item in steered:
            self.log_transcript("user", text=item.text)
            await self.emit(
                "user_message",
                session_id=self.workspace.session_id,
                text=item.text,
            )
        await self.emit(
            "pending_prompts_changed",
            pending_prompts=self.pending_prompts_snapshot(),
        )
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Nothing ran: the user sent a new message while you were working. "
                        "Do not call `verdict` for this unstarted probe. "
                        "Read and address the user's message before starting the next probe "
                        "(or reply without a probe if none is needed):\n\n"
                        + prefix
                        + "\n\n".join(item.text for item in steered)
                    ),
                }
            ]
        }

    def document_context(
        self, *, sent: Mapping[str, str] | None = None
    ) -> tuple[str, dict[str, str]]:
        """Build document context and the contents to mark sent after submission."""
        baseline = self._sent_documents if sent is None else sent
        sections = []
        sent_documents = {}
        for document in self.list_documents():
            if not document["included"]:
                continue
            content = self.workspace.read_document(document["name"])["content"]
            if baseline.get(document["name"]) == content:
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

    def _restore_pending_verdict(
        self, records: list[dict[str, Any]], *, include_slurm: bool = True
    ):
        self._pending_verdict = None
        self._pending_verdict_probe_id = None
        reviewed = {
            record.get("probe_id")
            for record in records
            if record.get("kind") == "verdict"
        }
        for record in reversed(records):
            if record.get("kind") != "run":
                continue
            if not include_slurm and record.get("backend") == "slurm":
                continue
            status = record.get("status", "completed")
            if status in ("failed", "error") and record.get("backend") != "slurm":
                continue
            probe_id = record.get("probe_id")
            if probe_id and probe_id not in reviewed and record.get("expected"):
                self._pending_verdict = record["expected"]
                self._pending_verdict_probe_id = probe_id
                break

    def resume_session(
        self, session_id: str, restore_kernel: bool = False
    ) -> dict[str, Any]:
        if self.has_active_query():
            raise RuntimeError("Cannot resume while the agent is running a query.")

        path = self.workspace.transcript_file(session_id)
        records = self.workspace.read_transcript(path)
        saved_provider, sdk_id = self.workspace.sdk_session_info(records)
        owner_provider, saved_model, saved_effort = (
            self.workspace.session_provider_and_model(records)
        )
        changed = (
            self.workspace.session_pending or path != self.workspace.transcript_path
        )
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

        resolved_owner = (
            owner_provider
            if owner_provider in self.backends
            else saved_provider
            if saved_provider in self.backends
            else None
        )
        can_restore_native = (
            sdk_id is not None
            and saved_provider in self.backends
            and (resolved_owner is None or saved_provider == resolved_owner)
        )
        if can_restore_native:
            self._active_provider = saved_provider
            self._sdk_session_id = sdk_id
            self._resume_prefix = None
            self._needs_recap = False
            self._recap_allow_preserved_kernel = False
        else:
            self._active_provider = None
            self._sdk_session_id = None
            self._resume_prefix = self.workspace.recap(records)
            self._needs_recap = True
            self._recap_allow_preserved_kernel = False
        self._owner_provider = resolved_owner
        self._active_model = saved_model
        self._active_raw_model = (
            self.raw_model_id(saved_model) if saved_model is not None else None
        )
        self._active_effort = saved_effort

        self._sent_documents = {}
        self._record_dirty_seq = 0
        self._record_acked_seq = 0
        self._record_reload_reason = "concurrent"
        self._results_since_note = 0
        self._pending_prompts = []
        self._turn_model = None
        self._turn_selection = None
        self._bound_turn = None
        self._pending_user_logs = None
        self._restore_pending_verdict(records)
        latest_ctx_record = (
            next(
                (
                    record["context_usage"]
                    for record in reversed(records)
                    if record.get("kind") == "meta" and "context_usage" in record
                ),
                None,
            )
            if can_restore_native
            else None
        )
        self.context_usage = (
            latest_ctx_record if isinstance(latest_ctx_record, dict) else None
        )
        self._session_initialized = False

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
            "context_restored": can_restore_native,
            "context_usage": self.context_usage,
        }

    def _replay_probe_code(self, records: list[dict[str, Any]]) -> int:
        """Rebuild kernel state from recorded probes without creating new artifacts."""
        replayed = 0
        for record in records:
            if (
                record.get("kind") != "probe_started"
                or record.get("backend") == "slurm"
            ):
                continue
            code = record.get("code")
            if not isinstance(code, str) or not code.strip():
                continue
            self.kernel.execute(code, self.timeout_s)
            replayed += 1
        return replayed

    async def interrupt(self):
        await self.clear_pending_prompts()
        if self.kernel.busy:
            self._kernel_interrupt_requested = True
            self.kernel.interrupt()
            await self.emit("kernel_interrupted")
        task = self._current_task
        if task and not task.done():
            task.cancel()
            await self.emit("query_cancelled")
        if task is not None and task is not asyncio.current_task():
            await asyncio.gather(task, return_exceptions=True)

    def shutdown(self):
        self.kernel.shutdown()

    async def _record_compaction(self, *, record_reloaded: bool = True):
        self.log_transcript("compaction", record_reloaded=record_reloaded)
        await self.emit("context_compacted", record_reloaded=record_reloaded)

    async def compact(self):
        current = asyncio.current_task()
        if (
            self._is_busy
            or self.kernel.busy
            or self._probe_operation_lock.locked()
            or bool(self._pending_prompts)
            or (self._probe_task is not None and not self._probe_task.done())
            or (
                self._current_task is not None
                and not self._current_task.done()
                and self._current_task is not current
            )
        ):
            raise RuntimeError(
                "Cannot compact this explorer while the agent is running a query."
            )
        if not self._sdk_session_id or self._active_provider is None:
            raise RuntimeError("This explorer has no conversation context to compact.")
        provider = self._active_provider
        backend = self.backends[provider]
        compact_model = (
            self.raw_model_id(self.model)
            if self.model and self.provider_for_model(self.model) == provider
            else None
        )
        compact_effort = self.effort if compact_model is not None else None
        compact_auth = self.auth.get(provider)
        self._is_busy = True
        self.status = "compacting"
        started_at = time.monotonic()
        try:
            await self.emit("status_change", status="compacting")
            try:
                result = await backend.compact(
                    system_prompt=build_system_prompt(self.exploration_prompt),
                    tools=self.tools,
                    workspace_path=self.workspace_path,
                    model=compact_model,
                    effort=compact_effort,
                    auth=compact_auth,
                    resume_session_id=self._sdk_session_id,
                )
            except SessionResumeError as exc:
                prior_records = self.workspace.current_records()
                self._sdk_session_id = None
                self._active_provider = None
                self._sent_documents = {}
                self.context_usage = None
                self._session_initialized = False
                self._needs_recap = True
                self._recap_allow_preserved_kernel = True
                self._resume_prefix = (
                    self.workspace.recap(
                        prior_records, kernel_preserved=not self._kernel_notice
                    )
                    if prior_records
                    else None
                )
                self.log_transcript("meta", context_usage=None)
                await self.emit("context_usage_changed", context_usage=None)
                raise RuntimeError(
                    "Saved provider session is no longer available; the next message will rebuild context from the transcript."
                ) from exc
            self._track_session_id(result.session_id, provider)
            await self._record_compaction(record_reloaded=backend.uses_compact_hook)
            if not backend.uses_compact_hook:
                self.mark_record_dirty(reason="compaction")
            logger.info(
                "%s compaction completed in %.1fs",
                provider,
                time.monotonic() - started_at,
            )
            self.context_usage = result.context_usage
            self.log_transcript("meta", context_usage=self.context_usage)
            await self.emit("context_usage_changed", context_usage=self.context_usage)
        finally:
            self._is_busy = False
            await self.emit("status_change", status="idle")

    def _owner_turn_model(
        self, owner_provider: ProviderId
    ) -> tuple[str | None, str | None]:
        if self.model and self.provider_for_model(self.model) == owner_provider:
            return self.model, self.effort
        if self._active_model:
            for candidate, candidate_provider in self.model_providers.items():
                if candidate_provider == owner_provider and (
                    candidate == self._active_model
                    or self.model_raw_ids.get(candidate) == self._active_model
                ):
                    return candidate, self._active_effort
            if not self.model_providers and (
                self._active_provider == owner_provider
                or self._owner_provider == owner_provider
                or self.provider_for_model(self._active_model) == owner_provider
            ):
                return self._active_model, self._active_effort
        if (
            self.default_model
            and self.provider_for_model(self.default_model) == owner_provider
            and (not self.model_providers or self.default_model in self.model_providers)
        ):
            return self.default_model, self.default_effort
        for candidate, candidate_provider in self.model_providers.items():
            if candidate_provider == owner_provider:
                return candidate, None
        return None, None

    def bind_next_turn(
        self,
        model: str | None = None,
        effort: str | None = None,
    ) -> BoundTurnConfig:
        """Bind the model, provider, effort, and auth for the next turn."""
        catalogue_model = model if model is not None else self.model
        if catalogue_model is None:
            raise RuntimeError("No model available.")
        provider = self.provider_for_model(catalogue_model)
        chosen_effort = effort if model is not None else self.effort
        bound = BoundTurnConfig(
            catalogue_model=catalogue_model,
            provider=provider,
            raw_model=self.raw_model_id(catalogue_model),
            effort=chosen_effort,
            auth=self.auth.get(provider),
        )
        self._bound_turn = bound
        return bound

    async def review_slurm(self, probe_id: str):
        return await self.query(
            f"Runtime event: Slurm probe {probe_id} has finished and its output is saved. "
            f"Call check_slurm_job for {probe_id} to inspect its output and plots, then "
            "record its verdict against the original prediction if it has none, including "
            "if the job failed or was cancelled. Preserve an already recorded verdict and "
            "finish any interrupted correction. This continues the original probe. "
            "If the probe failed (including missing required outputs) and the saved "
            "output supports a concrete correction, "
            "explain the cause and the correction, then use submit_slurm to submit a "
            "corrected script or resource request as a new probe. Keep the original "
            "scientific question and prediction, and preserve the failed output. "
            "After resubmitting, stop this turn and let the monitor wait for the new job. "
            "If the probe completed successfully, stop after its verdict. Do not retry a null or "
            "unexpected scientific result as though it were an execution failure. "
            "If no justified correction remains or user input is needed, explain why "
            "and stop. Do not blindly resubmit an unchanged job, restart a job the "
            "user deliberately cancelled, or launch an unrelated investigation.",
            slurm_probe_id=probe_id,
        )

    async def query(self, user_text: str, *, slurm_probe_id: str | None = None):
        if self._is_busy:
            raise RuntimeError("Agent is already running a query.")
        owner_provider = (
            self._active_provider or self._owner_provider
            if slurm_probe_id is not None
            else None
        )
        bound = self._bound_turn
        self._bound_turn = None
        if owner_provider is not None and (
            bound is None or bound.provider != owner_provider
        ):
            chosen_model, chosen_effort = self._owner_turn_model(owner_provider)
            if chosen_model is None:
                raise RuntimeError(
                    f"Provider {owner_provider} is not available to review {slurm_probe_id}."
                )
            bound = self.bind_next_turn(chosen_model, chosen_effort)
            self._bound_turn = None
        elif bound is None:
            bound = self.bind_next_turn()
            self._bound_turn = None

        turn_catalogue_model = bound.catalogue_model
        provider = bound.provider
        turn_raw_model = bound.raw_model
        turn_effort = bound.effort
        turn_auth = bound.auth
        backend = self.backends[provider]
        self._is_busy = True
        self._turn_model = turn_catalogue_model
        self._turn_selection = (provider, turn_raw_model, turn_effort)
        try:
            prior_records = self.workspace.current_records()
            switching_provider = (
                self._active_provider is not None and provider != self._active_provider
            )
            rebuilding_threadless = (
                not switching_provider
                and self._sdk_session_id is None
                and self._session_initialized
            )
            resetting_context = switching_provider or rebuilding_threadless
            turn_resume_prefix = (
                (
                    self.workspace.recap(
                        prior_records, kernel_preserved=not self._kernel_notice
                    )
                    if prior_records
                    else None
                )
                if resetting_context
                else (
                    self.workspace.recap(
                        prior_records,
                        kernel_preserved=self._recap_allow_preserved_kernel
                        and not self._kernel_notice,
                    )
                    if prior_records
                    else None
                )
                if self._needs_recap
                else self._resume_prefix
            )
            base_sent_documents: Mapping[str, str] = (
                {} if resetting_context else self._sent_documents
            )
            initial_resume_id = None if resetting_context else self._sdk_session_id

            # Snapshot the selection before yielding so changes made while this turn runs
            # apply to the next message.
            sent_documents: dict[str, str] = {}
            initialize_session = False
            kernel_notice = ""
            prompt_record_seq = 0
            prompt = user_text

            def prepare_prompt():
                nonlocal sent_documents, initialize_session, kernel_notice
                nonlocal prompt_record_seq, prompt
                prompt_record_seq = self._record_dirty_seq
                document_prefix, sent_documents = (
                    self.document_context(sent=base_sent_documents)
                    if slurm_probe_id is None
                    else ("", {})
                )
                prefix = ""
                resume_prefix = turn_resume_prefix
                initialize_session = False
                if resume_prefix:
                    prefix = self.opening_notes() + resume_prefix
                    initialize_session = True
                elif not self._session_initialized or resetting_context:
                    prefix = self.opening_notes()
                    initialize_session = True
                elif self._record_dirty:
                    prefix = self._record_reload_heading() + self.opening_notes()

                prefix += document_prefix
                if self._pending_verdict is not None:
                    prefix += (
                        f"Probe {self._pending_verdict_probe_id} still needs a verdict "
                        f"against its prediction: {self._pending_verdict}. "
                        "Call verdict before starting another probe.\n\n"
                    )
                kernel_notice = self._kernel_notice
                prefix += kernel_notice
                prompt = prefix + user_text if prefix else user_text

            prepare_prompt()
            await self.emit("status_change", status="thinking")
            if slurm_probe_id is None:
                user_logs = self._pending_user_logs or [user_text]
                self._pending_user_logs = None
                for message_text in user_logs:
                    self.log_transcript("user", text=message_text)
            else:
                self.log_transcript("slurm_review", probe_id=slurm_probe_id)
                await self.emit("slurm_review", probe_id=slurm_probe_id)

            accumulated_text: list[str] = []
            for attempt in range(2):
                accepted = False
                pending_session_id: str | None = None
                resume_id = initial_resume_id if attempt == 0 else self._sdk_session_id

                def commit_prompt_state() -> bool:
                    nonlocal accepted
                    if accepted:
                        return True
                    accepted = True
                    self._resume_prefix = None
                    self._needs_recap = False
                    self._recap_allow_preserved_kernel = False
                    if initialize_session:
                        self._session_initialized = True
                    if self._kernel_notice == kernel_notice:
                        self._kernel_notice = ""
                    self._record_acked_seq = max(
                        self._record_acked_seq, prompt_record_seq
                    )
                    if resetting_context:
                        self._sdk_session_id = None
                        self._sent_documents = dict(sent_documents)
                        self.context_usage = None
                    else:
                        self._sent_documents.update(sent_documents)
                    self._active_provider = provider
                    self._owner_provider = provider
                    self._active_model = turn_catalogue_model
                    self._active_raw_model = turn_raw_model
                    self._active_effort = turn_effort
                    return False

                def commit_progress():
                    nonlocal pending_session_id
                    prev_selection = (
                        self._active_provider,
                        self._active_model,
                        self._active_raw_model,
                        self._active_effort,
                    )
                    new_selection = (
                        provider,
                        turn_catalogue_model,
                        turn_raw_model,
                        turn_effort,
                    )
                    was_accepted = commit_prompt_state()
                    selection_changed = (
                        not was_accepted and prev_selection != new_selection
                    )
                    if pending_session_id is not None:
                        self._track_session_id(
                            pending_session_id,
                            provider,
                            force_meta=selection_changed,
                        )
                        pending_session_id = None
                    elif selection_changed:
                        meta_fields: dict[str, Any] = {
                            "provider": provider,
                            "model": turn_catalogue_model,
                            "raw_model": turn_raw_model,
                        }
                        if self._sdk_session_id is not None:
                            meta_fields["sdk_session_id"] = self._sdk_session_id
                        if turn_effort is not None:
                            meta_fields["effort"] = turn_effort
                        self.log_transcript("meta", **meta_fields)

                def wrap_tool(spec: ToolSpec) -> ToolSpec:
                    async def wrapped_handler(args: dict[str, Any]) -> dict[str, Any]:
                        commit_progress()
                        return await spec.handler(args)

                    return ToolSpec(
                        name=spec.name,
                        description=spec.description,
                        input_schema=spec.input_schema,
                        handler=wrapped_handler,
                    )

                turn_tools = [wrap_tool(spec) for spec in self.tools]
                tool_timeout_s = max(self.timeout_s, 600) + 60

                try:
                    self._commit_turn_accepted = commit_progress
                    try:
                        async with aclosing(
                            backend.run_turn(
                                prompt,
                                system_prompt=build_system_prompt(
                                    self.exploration_prompt
                                ),
                                tools=turn_tools,
                                workspace_path=self.workspace_path,
                                model=turn_raw_model,
                                effort=turn_effort,
                                auth=turn_auth,
                                resume_session_id=resume_id,
                                tool_timeout_s=tool_timeout_s,
                            )
                        ) as turn_events:
                            async for event in turn_events:
                                if isinstance(event, SessionStarted):
                                    if accepted:
                                        self._track_session_id(
                                            event.session_id, provider
                                        )
                                    else:
                                        pending_session_id = event.session_id
                                    continue
                                if isinstance(event, TurnStarted):
                                    commit_progress()
                                    continue
                                commit_progress()
                                if isinstance(event, TextDelta):
                                    await self.emit("assistant_delta", text=event.text)
                                elif isinstance(event, CompactionNotice):
                                    await self._record_compaction(
                                        record_reloaded=backend.uses_compact_hook
                                    )
                                    if not backend.uses_compact_hook:
                                        self.mark_record_dirty(reason="compaction")
                                elif isinstance(event, MessageUsage):
                                    self.log_transcript(
                                        "message_usage",
                                        message_id=event.message_id,
                                        model=event.model,
                                        usage=event.usage,
                                    )
                                elif isinstance(event, TextBlockDone):
                                    accumulated_text.append(event.text)
                                    self.log_transcript("agent", text=event.text)
                                    await self.emit("assistant_text", text=event.text)
                                elif isinstance(event, TurnResult):
                                    if any(
                                        value is not None
                                        for value in (
                                            event.usage,
                                            event.model_usage,
                                            event.total_cost_usd,
                                            event.duration_ms,
                                            event.duration_api_ms,
                                            event.num_turns,
                                        )
                                    ):
                                        self.log_transcript(
                                            "usage",
                                            usage=event.usage,
                                            model_usage=event.model_usage,
                                            total_cost_usd=event.total_cost_usd,
                                            duration_ms=event.duration_ms,
                                            duration_api_ms=event.duration_api_ms,
                                            num_turns=event.num_turns,
                                        )
                                    if event.context_usage is not None:
                                        self.context_usage = event.context_usage
                                        self.log_transcript(
                                            "meta", context_usage=self.context_usage
                                        )
                                        await self.emit(
                                            "context_usage_changed",
                                            context_usage=self.context_usage,
                                        )
                        commit_progress()
                        break
                    finally:
                        self._commit_turn_accepted = None
                except SessionResumeError as exc:
                    if attempt == 0 and resume_id is not None and not accepted:
                        logger.warning(
                            "Could not resume %s session %s (%s); rebuilding context from transcript.",
                            provider,
                            resume_id,
                            exc,
                        )
                        self._sdk_session_id = None
                        self._active_provider = None
                        self._owner_provider = provider
                        self._sent_documents = {}
                        self.context_usage = None
                        self._session_initialized = False
                        self.log_transcript(
                            "meta",
                            context_rebuilt=True,
                            provider=provider,
                            stale_sdk_session_id=resume_id,
                        )
                        await self.emit(
                            "system_message",
                            text="Original provider session was unavailable; rebuilt context from the saved transcript.",
                        )
                        self._needs_recap = True
                        self._recap_allow_preserved_kernel = True
                        self._resume_prefix = (
                            self.workspace.recap(
                                prior_records, kernel_preserved=not self._kernel_notice
                            )
                            if prior_records
                            else None
                        )
                        turn_resume_prefix = self._resume_prefix
                        base_sent_documents = {}
                        prepare_prompt()
                        continue
                    raise RuntimeError(str(exc)) from exc

            await self.emit("turn_complete", full_text="\n\n".join(accumulated_text))
            return True
        except asyncio.CancelledError:
            await self.emit("status_change", status="cancelled")
            raise
        except Exception as exc:
            await self.clear_pending_prompts()
            logger.exception("Error during query execution")
            await self.emit("error", message=str(exc))
            return False
        finally:
            self._pending_user_logs = None
            self._commit_turn_accepted = None
            self._turn_model = None
            self._turn_selection = None
            self._is_busy = False
            if not (self._draining_queries and self._pending_prompts):
                await self.emit("status_change", status="idle")

    def _track_session_id(
        self,
        session_id: str | None,
        provider: ProviderId | None = None,
        *,
        force_meta: bool = False,
    ):
        resolved_provider = (
            provider or self._active_provider or self.provider_for_model(self.model)
        )
        if not session_id or (
            not force_meta
            and session_id == self._sdk_session_id
            and resolved_provider == self._active_provider
        ):
            return
        self._active_provider = resolved_provider
        self._owner_provider = resolved_provider
        self._sdk_session_id = session_id
        meta_fields: dict[str, Any] = {
            "provider": resolved_provider,
            "sdk_session_id": session_id,
        }
        if (
            self._turn_selection is not None
            and self._turn_selection[0] == resolved_provider
        ):
            meta_fields["model"] = self._turn_model or self._turn_selection[1]
            meta_fields["raw_model"] = self._turn_selection[1]
            if self._turn_selection[2] is not None:
                meta_fields["effort"] = self._turn_selection[2]
        else:
            active_model = self._active_model or self.model
            if active_model and (
                (
                    active_model == self._active_model
                    and self._owner_provider == resolved_provider
                )
                or self.provider_for_model(active_model) == resolved_provider
            ):
                meta_fields["model"] = active_model
                meta_fields["raw_model"] = (
                    self._active_raw_model
                    if active_model == self._active_model and self._active_raw_model
                    else self.raw_model_id(active_model)
                )
                if (
                    self._active_effort is not None
                    and active_model == self._active_model
                ):
                    meta_fields["effort"] = self._active_effort
                elif active_model == self.model and self.effort is not None:
                    meta_fields["effort"] = self.effort
        self.log_transcript("meta", **meta_fields)

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
        self._active_provider = None
        self._owner_provider = None
        self._active_model = None
        self._active_raw_model = None
        self._active_effort = None
        self._sdk_session_id = None
        self._resume_prefix = None
        self._needs_recap = False
        self._recap_allow_preserved_kernel = False
        self._kernel_interrupt_requested = False
        self._sent_documents = {}
        self._results_since_note = 0
        self._pending_verdict = None
        self._pending_verdict_probe_id = None
        self._record_dirty_seq = 0
        self._record_acked_seq = 0
        self._record_reload_reason = "concurrent"
        self._pending_prompts = []
        self._turn_model = None
        self._turn_selection = None
        self._bound_turn = None
        self._pending_user_logs = None
        self.context_usage = None
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
        if (
            isinstance(packages, (str, bytes))
            or not isinstance(packages, Sequence)
            or not packages
            or any(
                not isinstance(p, str) or not p or p.startswith("-") for p in packages
            )
        ):
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
            "packages": packages,
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
            metadata["message"] = (
                f"Package installation {metadata['status']}: {', '.join(packages)}. "
                "Installer output saved in this session's environment_changes folder."
            )
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
        return (
            metadata["status"],
            (directory / "output.txt").read_text(),
            metadata["message"],
        )
