"""Mutable runtime state shared by the web routes and WebSocket handler."""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from fastapi import WebSocket

from ..session import ExplorationSession

logger = logging.getLogger("exploration-web-app")


class WebRuntime:
    """Own the optional browser workspace and its live WebSocket clients."""

    def __init__(
        self,
        session: ExplorationSession | None = None,
        *,
        model: str = "claude-opus-5",
        timeout_s: int = 600,
        image_cap: int = 2,
        navigator_root: Path | str = ".",
        session_factory: Callable[..., ExplorationSession] = ExplorationSession,
    ):
        self.session = session
        self.model = session.model if session else model
        self.models = [self.model]
        self.model_labels = {self.model: self.model}
        self.timeout_s = session.timeout_s if session else timeout_s
        self.image_cap = session.image_cap if session else image_cap
        self.navigator_root = Path(navigator_root).expanduser().resolve()
        self.session_factory = session_factory
        self.websockets: set[WebSocket] = set()
        self._started = False
        self._subscribed_session: ExplorationSession | None = None

    @property
    def has_workspace(self) -> bool:
        return self.session is not None

    def start(self):
        self._started = True
        self._subscribe_to_session()

    async def discover_models(self):
        """Read the CLI's model catalog without sending an inference request."""
        try:
            async with asyncio.timeout(15):
                async with ClaudeSDKClient(
                    ClaudeAgentOptions(tools=[], mcp_servers={})
                ) as client:
                    info = await client.get_server_info()
            for model in (info or {}).get("models", []):
                value = model.get("value")
                if not isinstance(value, str) or not value:
                    continue
                if value not in self.models:
                    self.models.append(value)
                self.model_labels[value] = model.get("displayName") or value
        except Exception as exc:
            logger.warning(
                "Could not discover Claude models; retaining configured model: %s", exc
            )

    def shutdown(self):
        self._started = False
        if self._subscribed_session:
            self._subscribed_session.unsubscribe(self.forward_session_event)
            self._subscribed_session = None
        if self.session:
            self.session.shutdown()

    def open_workspace(self, path: Path | str) -> ExplorationSession:
        """Open an existing directory, creating runtime state only after validation."""
        if self.session is not None:
            self.session.require_idle("change workspace")
        target = Path(path).expanduser().resolve()
        if not target.is_dir():
            raise ValueError("Choose an existing folder for the workspace.")

        if self.session is None:
            self.session = self.session_factory(
                workspace_path=target,
                model=self.model,
                timeout_s=self.timeout_s,
                image_cap=self.image_cap,
            )
            self._subscribe_to_session()
        else:
            self.session.set_workspace(target)
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

    async def broadcast(self, event_type: str, data: dict[str, Any]):
        if not self.websockets:
            return
        message = {"type": event_type, **data}
        dead = []
        for websocket in self.websockets:
            try:
                await websocket.send_json(message)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self.websockets.discard(websocket)

    async def forward_session_event(self, event_type: str, payload: dict[str, Any]):
        await self.broadcast(event_type, payload)

    def start_query(self, text: str, model: str | None = None):
        if self.session is None:
            raise RuntimeError("Open a workspace folder before starting a query.")
        if self.session.has_active_query():
            raise RuntimeError("The agent is already running a query.")
        if model is not None:
            if model not in self.models:
                raise ValueError("Choose a model from the model selector.")
            self.session.model = model
            self.model = model
            self.session.log_transcript("meta", model=model)
        task = asyncio.create_task(self._run_query(text))
        self.session._current_task = task

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
            await self.broadcast("error", {"message": str(exc)})
        finally:
            if session._current_task is asyncio.current_task():
                session._current_task = None
