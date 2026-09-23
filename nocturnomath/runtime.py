"""Transport-agnostic runtime shared by the web app and the terminal client."""

import asyncio
import logging
import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from .auth import AuthMethod, ClaudeAuth
from .session import DEFAULT_EFFORT, EFFORT_LEVELS, ExplorationSession

logger = logging.getLogger("nocturnomath.runtime")

EVIDENCE_PATH = ".nocturnomath/kb/evidence.md"
THOUGHTS_PATH = ".nocturnomath/kb/thoughts.md"


class Runtime:
    """Own the optional workspace and publish its events to any interface."""

    def __init__(
        self,
        session: ExplorationSession | None = None,
        *,
        model: str | None = None,
        effort: str | None = None,
        timeout_s: int = 600,
        image_cap: int = 2,
        navigator_root: Path | str = ".",
        session_factory: Callable[..., ExplorationSession] = ExplorationSession,
        auth: ClaudeAuth | None = None,
    ):
        # ``session`` remains the terminal client's single-explorer facade.
        # The web client uses the ordered registry below and always supplies an id.
        self.session = session
        self.agents: dict[str, ExplorationSession] = {}
        self.primary_agent_id: str | None = None
        self._auth = auth or (session.auth if session else ClaudeAuth())
        if session:
            session.auth = self._auth
        self.startup_default_model = model
        self.startup_default_effort = effort
        self.model = session.model if session else None
        self.effort = session.effort if session else (effort or DEFAULT_EFFORT)
        self.models = []
        self.model_labels = {}
        self.model_efforts: dict[str, list[str]] = {}
        self.sdk_default_model: str | None = None
        self.timeout_s = session.timeout_s if session else timeout_s
        self.image_cap = session.image_cap if session else image_cap
        self.navigator_root = Path(navigator_root).expanduser().resolve()
        self.session_factory = session_factory
        self.subscribers: list[Callable[[str, dict[str, Any]], Any]] = []
        self._started = False
        self.changing = False
        self._changing_agents: set[str] = set()
        self._agent_subscribers: dict[str, Callable[[str, dict[str, Any]], Any]] = {}
        self._workspace_defaults_pending = session is not None
        if session is not None:
            self.primary_agent_id = self._new_agent_id()
            self.agents[self.primary_agent_id] = session

    @property
    def has_workspace(self) -> bool:
        return bool(self.agents)

    @property
    def auth(self) -> ClaudeAuth:
        """The credential the next SDK client uses; the session owns it once open."""
        return self.session.auth if self.session else self._auth

    @auth.setter
    def auth(self, auth: ClaudeAuth):
        self._auth = auth
        for session in self.agents.values():
            session.auth = auth

    def start(self):
        self._started = True
        for agent_id, session in self.agents.items():
            self._subscribe_to_agent(agent_id, session)

    async def discover_models(self):
        """Read the CLI's model catalog without sending an inference request."""
        models: list[str] = []
        labels: dict[str, str] = {}
        model_efforts: dict[str, list[str]] = {}
        sdk_default_model = None
        try:
            async with asyncio.timeout(15):
                async with ClaudeSDKClient(
                    ClaudeAgentOptions(
                        tools=[],
                        mcp_servers={},
                        strict_mcp_config=True,
                        setting_sources=["user"],
                        skills=[],
                        env=self.auth.sdk_env(),
                    )
                ) as client:
                    info = await client.get_server_info()
            for model in (info or {}).get("models", []):
                value = model.get("value")
                resolved = model.get("resolvedModel")
                if not isinstance(value, str) or not value:
                    continue
                identifier = (
                    resolved if isinstance(resolved, str) and resolved else None
                )
                if value == "default":
                    sdk_default_model = identifier
                if identifier is None and value != "default":
                    identifier = value
                if identifier is None:
                    continue
                if identifier not in models:
                    models.append(identifier)
                labels[identifier] = identifier
                levels = model.get("supportedEffortLevels")
                supported = (
                    [level for level in levels if level in EFFORT_LEVELS]
                    if isinstance(levels, list)
                    else list(EFFORT_LEVELS)
                    if model.get("supportsEffort")
                    else []
                )
                current = model_efforts.setdefault(identifier, [])
                current.extend(level for level in supported if level not in current)
        except Exception as exc:
            logger.warning("Could not discover Claude models: %s", exc)
        self.models = models
        self.model_labels = labels
        self.model_efforts = model_efforts
        self.sdk_default_model = (
            sdk_default_model
            if sdk_default_model in models
            else models[0]
            if models
            else None
        )
        self.model = self._select_model(self.model or self.startup_default_model)
        self.effort = self._select_effort(self.model, self.effort)
        if self.agents:
            if self._workspace_defaults_pending:
                self._apply_workspace_defaults()
            else:
                for session in self.agents.values():
                    session.default_model = self._select_model(session.default_model)
                    session.default_effort = self._select_effort(
                        session.default_model, session.default_effort
                    )

    def _select_model(self, preferred: str | None) -> str | None:
        if preferred in self.models:
            return preferred
        return self.sdk_default_model

    def _select_effort(self, model: str | None, preferred: str | None) -> str | None:
        levels = self.model_efforts.get(model or "", [])
        if preferred in levels:
            return preferred
        if DEFAULT_EFFORT in levels:
            return DEFAULT_EFFORT
        return levels[0] if levels else None

    def _apply_workspace_defaults(self):
        session = self.require_session()
        config = session.workspace.read_config()
        configured_model = config.get("default_model")
        model_preference = (
            configured_model
            if isinstance(configured_model, str)
            else self.startup_default_model
        )
        default_model = self._select_model(model_preference)
        configured_effort = config.get("default_effort")
        effort_preference = (
            configured_effort
            if isinstance(configured_effort, str)
            else self.startup_default_effort
        )
        default_effort = self._select_effort(default_model, effort_preference)
        for session in self.agents.values():
            self._apply_defaults_to(session, default_model, default_effort)
        self.model = default_model
        self.effort = default_effort
        self._workspace_defaults_pending = False

    @staticmethod
    def _apply_defaults_to(
        session: ExplorationSession, model: str | None, effort: str | None
    ):
        session.default_model = model
        session.default_effort = effort
        session.model = model
        session.effort = effort

    def _workspace_defaults(
        self, session: ExplorationSession
    ) -> tuple[str | None, str | None]:
        config = session.workspace.read_config()
        model = self._select_model(
            config.get("default_model")
            if isinstance(config.get("default_model"), str)
            else self.startup_default_model
        )
        effort = self._select_effort(
            model,
            config.get("default_effort")
            if isinstance(config.get("default_effort"), str)
            else self.startup_default_effort,
        )
        return model, effort

    def set_session_defaults(self, model: str, effort: str | None):
        session = self.require_session()
        if model not in self.models:
            raise ValueError("Choose a model from the model selector.")
        supported = self.model_efforts.get(model, [])
        if effort is not None and effort not in supported:
            raise ValueError("Choose an effort supported by the selected model.")
        if supported and effort is None:
            raise ValueError("Choose an effort supported by the selected model.")
        session.set_session_defaults(model, effort)
        for other in self.agents.values():
            other.default_model = model
            other.default_effort = effort

    def reset_session(self):
        """Start a session with the workspace's configured defaults."""
        session = self.require_session()
        session.reset_client_session()
        self.model = session.model
        self.effort = session.effort

    async def authenticate(
        self, method: AuthMethod, credential: str | None = None
    ) -> dict[str, str | None]:
        """Select the credential for the next SDK client.

        Nothing is verified here: the CLI reports a bad credential on the next
        message. The model catalogue is refreshed as a courtesy and may be empty.
        """
        self.require_all_idle("change Claude authentication")
        self.auth = ClaudeAuth.interactive(method, credential)
        await self.discover_models()
        return self.auth.public()

    def shutdown(self):
        self._started = False
        self.changing = False
        for agent_id in list(self.agents):
            self._unsubscribe_agent(agent_id)
            self.agents[agent_id].shutdown()

    def open_workspace(self, path: Path | str, python=None) -> ExplorationSession:
        """Open an existing directory, creating runtime state only after validation."""
        if self.agents:
            self.require_all_idle("change workspace")
        target = Path(path).expanduser().resolve()
        if not target.is_dir():
            raise ValueError("Choose an existing folder for the workspace.")

        # Keep the terminal's long-standing in-place session contract when it
        # is the only explorer. Multi-tab workspace changes intentionally tear
        # down every independent kernel below.
        if len(self.agents) == 1:
            session = self.require_session()
            session.set_workspace(target, python)
            self._workspace_defaults_pending = True
            if self.models:
                self._apply_workspace_defaults()
            return session

        for agent_id in list(self.agents):
            self._unsubscribe_agent(agent_id)
            self.agents[agent_id].shutdown()
        self.agents.clear()
        self.primary_agent_id = None
        self.session = None
        self._workspace_defaults_pending = True
        self.create_agent(target, python=python)
        self._workspace_defaults_pending = True
        if self.models:
            self._apply_workspace_defaults()
        return self.require_session()

    def _new_agent_id(self) -> str:
        return uuid.uuid4().hex

    def _subscribe_to_agent(self, agent_id: str, session: ExplorationSession):
        if not self._started or agent_id in self._agent_subscribers:
            return

        async def forward(event_type: str, payload: dict[str, Any]):
            if event_type == "record_changed":
                for other_id, other in self.agents.items():
                    if other_id != agent_id:
                        other.mark_record_dirty()
            await self.emit(event_type, {"agent_id": agent_id, **payload})

        session.subscribe(forward)
        self._agent_subscribers[agent_id] = forward

    def _unsubscribe_agent(self, agent_id: str):
        session = self.agents.get(agent_id)
        callback = self._agent_subscribers.pop(agent_id, None)
        if session and callback:
            session.unsubscribe(callback)

    def subscribe(self, callback: Callable[[str, dict[str, Any]], Any]):
        self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[str, dict[str, Any]], Any]):
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    async def emit(self, event_type: str, data: dict[str, Any] | None = None):
        payload = data or {}
        for subscriber in list(self.subscribers):
            try:
                result = subscriber(event_type, payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.error(f"Error in runtime subscriber: {exc}")

    def create_agent(
        self, workspace_path: Path | str | None = None, *, python: str | None = None
    ) -> str:
        """Create an independent exploration in the current workspace."""
        if workspace_path is None:
            workspace_path = self.require_session().workspace_path
        session = self.session_factory(
            workspace_path=workspace_path,
            python=python,
            model=self.model,
            effort=self.effort,
            timeout_s=self.timeout_s,
            image_cap=self.image_cap,
            auth=self._auth,
        )
        agent_id = self._new_agent_id()
        self.agents[agent_id] = session
        if self.primary_agent_id is None:
            self.primary_agent_id = agent_id
            self.session = session
        if self.models:
            default_model, default_effort = self._workspace_defaults(session)
            self._apply_defaults_to(session, default_model, default_effort)
        self._subscribe_to_agent(agent_id, session)
        return agent_id

    def get_agent(self, agent_id: str | None = None) -> ExplorationSession:
        if agent_id is None:
            return self.require_session()
        try:
            return self.agents[agent_id]
        except KeyError as exc:
            raise ValueError("Explorer not found.") from exc

    def agent_snapshot(self, agent_id: str) -> dict[str, Any]:
        session = self.get_agent(agent_id)
        return {
            "agent_id": agent_id,
            "session_id": None
            if session.workspace.session_pending
            else session.workspace.session_id,
            "records": session.workspace.current_records(),
            "model": session.model,
            "effort": session.effort,
            "carry_chat_context": session.carry_chat_context,
            "kernel_alive": session.kernel.is_alive(),
            "kernel_busy": session.kernel.busy,
            "is_busy": session.has_active_query() or agent_id in self._changing_agents,
            "documents": session.list_documents(),
            "documents_default": session.documents_default,
        }

    def close_agent(self, agent_id: str):
        if len(self.agents) == 1:
            raise RuntimeError("Keep one explorer open.")
        session = self.get_agent(agent_id)
        session.require_idle("close this explorer")
        self._unsubscribe_agent(agent_id)
        session.shutdown()
        del self.agents[agent_id]
        if self.primary_agent_id == agent_id:
            self.primary_agent_id = next(iter(self.agents))
            self.session = self.agents[self.primary_agent_id]

    def find_agent_for_session(self, session_id: str) -> str | None:
        for agent_id, session in self.agents.items():
            if (
                not session.workspace.session_pending
                and session.workspace.session_id == session_id
            ):
                return agent_id
        return None

    def replace_agent(
        self, agent_id: str, session_id: str, restore_kernel: bool = False
    ) -> tuple[str, dict[str, Any], bool]:
        """Load history into an idle tab, unless it is already open elsewhere."""
        existing = self.find_agent_for_session(session_id)
        if existing is not None and existing != agent_id:
            return existing, self.agent_snapshot(existing), True
        old = self.get_agent(agent_id)
        old.require_idle("resume a past session")
        replacement = self.session_factory(
            workspace_path=old.workspace_path,
            model=old.model,
            effort=old.effort,
            timeout_s=old.timeout_s,
            image_cap=old.image_cap,
            auth=self._auth,
        )
        try:
            data = replacement.resume_session(session_id, restore_kernel)
        except Exception:
            replacement.shutdown()
            raise
        self._unsubscribe_agent(agent_id)
        old.shutdown()
        self.agents[agent_id] = replacement
        if self.primary_agent_id == agent_id:
            self.session = replacement
        self._subscribe_to_agent(agent_id, replacement)
        return agent_id, {**self.agent_snapshot(agent_id), **data}, False

    def workspace_snapshot(self) -> dict[str, Any]:
        """Describe the current workspace for a client that just connected."""
        session = self.session
        if session is None:
            return {
                "is_open": False,
                "path": None,
                "model": self.model,
                "effort": self.effort,
                "models": self.models,
                "model_labels": self.model_labels,
                "model_efforts": self.model_efforts,
                "session_defaults": None,
                "evidence_path": EVIDENCE_PATH,
                "thoughts_path": THOUGHTS_PATH,
                "plots": [],
                "agents": [],
                "navigator_root": str(self.navigator_root),
                "auth": self.auth.public(),
            }
        return {
            "is_open": True,
            "path": str(session.workspace_path),
            # Legacy single-explorer fields keep the terminal-adjacent HTTP
            # surface useful while web clients consume ``agents`` below.
            "session_id": session.workspace.session_id,
            "records": session.workspace.current_records(),
            "environment": {
                "python": str(session.environment.python),
                "version": session.environment.version,
            },
            "model": session.model,
            "effort": session.effort,
            "models": self.models,
            "model_labels": self.model_labels,
            "model_efforts": self.model_efforts,
            "session_defaults": {
                "model": session.default_model,
                "effort": session.default_effort,
            },
            "evidence_path": EVIDENCE_PATH,
            "thoughts_path": THOUGHTS_PATH,
            "kernel_alive": session.kernel.is_alive(),
            "kernel_busy": session.kernel.busy,
            "is_busy": session.has_active_query(),
            "carry_chat_context": session.carry_chat_context,
            "documents": session.list_documents(),
            "documents_default": session.documents_default,
            "plots": session.list_plots(),
            "agents": [self.agent_snapshot(agent_id) for agent_id in self.agents],
            "auth": self.auth.public(),
        }

    def export_notebook(
        self, agent_id: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Render the active session as a notebook and the filename to save it under."""
        session = self.get_agent(agent_id)
        if session.workspace.session_pending:
            raise ValueError("The current session has no messages to download.")
        notebook = session.workspace.export_notebook(
            session.workspace.session_id, session.environment.version
        )
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", session.workspace_path.name).strip(".-")
        filename = (
            f"nocturnomath-{name or 'workspace'}-{session.workspace.session_id}.ipynb"
        )
        return filename, notebook

    def require_session(self) -> ExplorationSession:
        if self.session is None:
            raise RuntimeError("Open a workspace folder before using the agent.")
        return self.session

    def require_all_idle(self, action: str):
        if self._changing_agents:
            raise RuntimeError(f"Cannot {action} while an explorer is changing.")
        for session in self.agents.values():
            session.require_idle(action)

    async def transition(self, operation, *args):
        if self.changing:
            raise RuntimeError(
                "The research environment is being prepared. Please wait."
            )
        self.require_all_idle("change kernel or workspace")
        self.changing = True
        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
        finally:
            self.changing = False

    async def transition_agent(self, agent_id: str, operation, *args):
        if self.changing:
            raise RuntimeError(
                "The research environment is being prepared. Please wait."
            )
        if agent_id in self._changing_agents:
            raise RuntimeError("This explorer is already changing. Please wait.")
        self.get_agent(agent_id).require_idle("change this explorer")
        self._changing_agents.add(agent_id)
        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
        finally:
            self._changing_agents.discard(agent_id)

    def start_query(
        self,
        text: str,
        model: str | None = None,
        effort: str | None = None,
        agent_id: str | None = None,
    ):
        if self.changing:
            raise RuntimeError(
                "The research environment is being prepared. Please wait."
            )
        if agent_id in self._changing_agents:
            raise RuntimeError("This explorer is changing. Please wait.")
        session = self.get_agent(agent_id)
        if session.has_active_query():
            raise RuntimeError("The agent is already running a query.")
        if model is None and session.model is None:
            raise ValueError("No model available.")
        if model is not None and model not in self.models:
            raise ValueError("Choose a model from the model selector.")
        target_model = model or session.model
        supported_efforts = self.model_efforts.get(target_model or "", [])
        if effort is not None and effort not in supported_efforts:
            raise ValueError("Choose an effort supported by the selected model.")

        metadata = {}
        if model is not None:
            session.model = model
            if agent_id is None or agent_id == self.primary_agent_id:
                self.model = model
            metadata["model"] = model
            selected_effort = self._select_effort(model, effort or session.effort)
            session.effort = selected_effort
            if agent_id is None or agent_id == self.primary_agent_id:
                self.effort = selected_effort
            metadata["effort"] = selected_effort
        elif effort is not None:
            session.effort = effort
            if agent_id is None or agent_id == self.primary_agent_id:
                self.effort = effort
            metadata["effort"] = effort
        if metadata:
            session.log_transcript("meta", **metadata)
        # An accepted first message allocates its session before the UI sees it.
        session.activate_session()
        task = asyncio.create_task(self._run_query(session, text, agent_id))
        session._current_task = task

    @property
    def current_task(self) -> asyncio.Task | None:
        return self.session._current_task if self.session else None

    async def wait_for_query(self):
        """Wait for the running query without re-raising its cancellation."""
        task = self.current_task
        if task:
            await asyncio.gather(task, return_exceptions=True)

    async def drain(self):
        """Interrupt every explorer and preserve any partial probe artifacts."""
        sessions = list(self.agents.values())
        await asyncio.gather(*(session.interrupt() for session in sessions))
        await asyncio.gather(
            *(
                asyncio.gather(
                    *(
                        task
                        for task in (session._current_task, session._probe_task)
                        if task is not None
                    ),
                    return_exceptions=True,
                )
                for session in sessions
            )
        )

    async def _run_query(
        self, session: ExplorationSession, text: str, agent_id: str | None
    ):
        try:
            await session.query(text)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Query task failed")
            payload = {"message": str(exc)}
            if agent_id is not None:
                payload["agent_id"] = agent_id
            await self.emit("error", payload)
        finally:
            if session._current_task is asyncio.current_task():
                session._current_task = None
