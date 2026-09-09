"""Runtime state shared by the web routes and the WebSocket handler."""

from typing import Any

from fastapi import WebSocket

from ..runtime import Runtime


class WebRuntime(Runtime):
    """Fan runtime events out to every connected browser."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.websockets: set[WebSocket] = set()
        self.subscribe(self.broadcast)

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
