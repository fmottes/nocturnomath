"""Transport-agnostic runtime shared by the web app and the terminal client."""

import asyncio
import logging
import re
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
        self.session = session
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
        self._subscribed_session: ExplorationSession | None = None
        self._workspace_defaults_pending = session is not None

    @property
    def has_workspace(self) -> bool:
        return self.session is not None

    @property
    def auth(self) -> ClaudeAuth:
        """The credential the next SDK client uses; the session owns it once open."""
        return self.session.auth if self.session else self._auth

    @auth.setter
    def auth(self, auth: ClaudeAuth):
        self._auth = auth
        if self.session:
            self.session.auth = auth

    def start(self):
        self._started = True
        self._subscribe_to_session()

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
                        tools=[], mcp_servers={}, env=self.auth.sdk_env()
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
        if self.session:
            self.session.model = self.model
            self.session.effort = self.effort
            if self._workspace_defaults_pending:
                self._apply_workspace_defaults()
            else:
                self.session.default_model = self._select_model(
                    self.session.default_model
                )
                self.session.default_effort = self._select_effort(
                    self.session.default_model, self.session.default_effort
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
        session.default_model = default_model
        session.default_effort = default_effort
        session.model = default_model
        session.effort = default_effort
        self.model = default_model
        self.effort = default_effort
        self._workspace_defaults_pending = False

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
        if self.session:
            self.session.require_idle("change Claude authentication")
        self.auth = ClaudeAuth.interactive(method, credential)
        await self.discover_models()
        return self.auth.public()

    def shutdown(self):
        self._started = False
        self.changing = False
        if self._subscribed_session:
            self._subscribed_session.unsubscribe(self.forward_session_event)
            self._subscribed_session = None
        if self.session:
            self.session.shutdown()

    def open_workspace(self, path: Path | str, python=None) -> ExplorationSession:
        """Open an existing directory, creating runtime state only after validation."""
        if self.session is not None:
            self.session.require_idle("change workspace")
        target = Path(path).expanduser().resolve()
        if not target.is_dir():
            raise ValueError("Choose an existing folder for the workspace.")

        if self.session is None:
            self.session = self.session_factory(
                workspace_path=target,
                python=python,
                model=self.model,
                effort=self.effort,
                timeout_s=self.timeout_s,
                image_cap=self.image_cap,
                auth=self.auth,
            )
            self._subscribe_to_session()
        else:
            self.session.set_workspace(target, python)
        self._workspace_defaults_pending = True
        if self.models:
            self._apply_workspace_defaults()
        return self.session

    def _subscribe_to_session(self):
        if not self._started or not self.session:
            return
        if self._subscribed_session is self.session:
            return
        if self._subscribed_session:
            self._subscribed_session.unsubscribe(self.forward_session_event)
        self.session.subscribe(self.forward_session_event)
        self._subscribed_session = self.session

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

    async def forward_session_event(self, event_type: str, payload: dict[str, Any]):
        await self.emit(event_type, payload)

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
                "kernel_alive": False,
                "kernel_busy": False,
                "is_busy": False,
                "carry_chat_context": True,
                "evidence_path": EVIDENCE_PATH,
                "thoughts_path": THOUGHTS_PATH,
                "documents": [],
                "documents_default": True,
                "plots": [],
                "records": [],
                "navigator_root": str(self.navigator_root),
                "auth": self.auth.public(),
            }
        return {
            "is_open": True,
            "path": str(session.workspace_path),
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
            "kernel_alive": session.kernel.is_alive(),
            "kernel_busy": session.kernel.busy,
            "is_busy": session._is_busy,
            "carry_chat_context": session.carry_chat_context,
            "evidence_path": EVIDENCE_PATH,
            "thoughts_path": THOUGHTS_PATH,
            "documents": session.list_documents(),
            "documents_default": session.documents_default,
            "plots": session.list_plots(),
            "auth": self.auth.public(),
        }

    def export_notebook(self) -> tuple[str, dict[str, Any]]:
        """Render the active session as a notebook and the filename to save it under."""
        session = self.require_session()
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

    async def transition(self, operation, *args):
        if self.changing:
            raise RuntimeError(
                "The research environment is being prepared. Please wait."
            )
        if self.session:
            self.session.require_idle("change kernel or workspace")
        self.changing = True
        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
        finally:
            self.changing = False

    def start_query(
        self, text: str, model: str | None = None, effort: str | None = None
    ):
        if self.changing:
            raise RuntimeError(
                "The research environment is being prepared. Please wait."
            )
        if self.session is None:
            raise RuntimeError("Open a workspace folder before starting a query.")
        if self.session.has_active_query():
            raise RuntimeError("The agent is already running a query.")
        if model is None and self.session.model is None:
            raise ValueError("No model available.")
        if model is not None and model not in self.models:
            raise ValueError("Choose a model from the model selector.")
        target_model = model or self.session.model
        supported_efforts = self.model_efforts.get(target_model or "", [])
        if effort is not None and effort not in supported_efforts:
            raise ValueError("Choose an effort supported by the selected model.")

        metadata = {}
        if model is not None:
            self.session.model = model
            self.model = model
            metadata["model"] = model
            selected_effort = self._select_effort(model, effort or self.session.effort)
            self.session.effort = selected_effort
            self.effort = selected_effort
            metadata["effort"] = selected_effort
        elif effort is not None:
            self.session.effort = effort
            self.effort = effort
            metadata["effort"] = effort
        if metadata:
            self.session.log_transcript("meta", **metadata)
        task = asyncio.create_task(self._run_query(text))
        self.session._current_task = task

    @property
    def current_task(self) -> asyncio.Task | None:
        return self.session._current_task if self.session else None

    async def wait_for_query(self):
        """Wait for the running query without re-raising its cancellation."""
        task = self.current_task
        if task:
            await asyncio.gather(task, return_exceptions=True)

    async def drain(self):
        """Interrupt the session and let its query and probe finish before shutdown."""
        session = self.session
        if session is None:
            return
        task = session._current_task
        await session.interrupt()
        if task:
            await asyncio.gather(task, return_exceptions=True)
        if session._probe_task:
            await asyncio.gather(session._probe_task, return_exceptions=True)

    async def _run_query(self, text: str):
        session = self.session
        if session is None:
            return
        try:
            await session.query(text)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Query task failed")
            await self.emit("error", {"message": str(exc)})
        finally:
            if session._current_task is asyncio.current_task():
                session._current_task = None
