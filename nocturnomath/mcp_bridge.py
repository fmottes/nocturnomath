"""Local authenticated stdio-to-loopback MCP bridge for session tools."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import socket
import sys
import threading
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from typing import Any, Self

from .tools import ToolSpec

logger = logging.getLogger("nocturnomath.mcp_bridge")

_PROTOCOL_VERSION_RE = re.compile(r"^20\d\d-\d\d-\d\d$")
_DEFAULT_PROTOCOL_VERSION = "2024-11-05"


def _jsonrpc_result(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _jsonrpc_error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": code, "message": message},
    }


class McpToolBridge:
    """Serve ``ToolSpec`` handlers over an authenticated loopback TCP socket."""

    def __init__(self, tools: Sequence[ToolSpec], *, server_name: str = "explore"):
        self._tools = list(tools)
        self._by_name = {spec.name: spec for spec in self._tools}
        self._server_name = server_name
        self.token = secrets.token_urlsafe(32)
        self._server: asyncio.AbstractServer | None = None
        self.port: int | None = None
        self._client_tasks: set[asyncio.Task[None]] = set()

    async def __aenter__(self) -> Self:
        self._server = await asyncio.start_server(
            self._handle_connection, host="127.0.0.1", port=0
        )
        sockets = self._server.sockets
        if not sockets:
            raise RuntimeError("MCP bridge failed to bind a loopback socket.")
        self.port = sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *args: object) -> None:
        for task in list(self._client_tasks):
            task.cancel()
        if self._client_tasks:
            await asyncio.gather(*self._client_tasks, return_exceptions=True)
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self.port = None

    def stdio_server_config(self) -> dict[str, Any]:
        """Return standard MCP stdio server configuration for CLI subprocesses."""
        if self.port is None:
            raise RuntimeError("MCP bridge must be entered before requesting config.")
        return {
            "command": sys.executable,
            "args": ["-m", "nocturnomath.mcp_bridge"],
            "env": {
                "NOCTURNOMATH_MCP_PORT": str(self.port),
                "NOCTURNOMATH_MCP_TOKEN": self.token,
            },
        }

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._client_tasks.add(task)
        try:
            lines = self._iter_lines(reader)
            auth_line = await anext(lines, None)
            if not auth_line:
                return
            try:
                auth_msg = json.loads(auth_line)
            except json.JSONDecodeError:
                return
            presented = auth_msg.get("auth") if isinstance(auth_msg, dict) else None
            if not isinstance(presented, str) or not secrets.compare_digest(
                presented, self.token
            ):
                return

            write_lock = asyncio.Lock()
            in_flight: set[asyncio.Task[None]] = set()

            async def handle_line(raw: str) -> None:
                try:
                    request = json.loads(raw)
                except json.JSONDecodeError:
                    response = _jsonrpc_error(None, -32700, "Parse error")
                else:
                    response = await self._dispatch(request)
                if response is not None:
                    async with write_lock:
                        writer.write((json.dumps(response) + "\n").encode("utf-8"))
                        await writer.drain()

            try:
                async for line in lines:
                    if not line:
                        continue
                    req_task = asyncio.create_task(handle_line(line))
                    in_flight.add(req_task)
                    req_task.add_done_callback(in_flight.discard)
                if in_flight:
                    await asyncio.gather(*in_flight, return_exceptions=True)
            finally:
                for req_task in list(in_flight):
                    req_task.cancel()
                if in_flight:
                    await asyncio.gather(*in_flight, return_exceptions=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("MCP bridge client closed with error: %s", exc)
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()
            if task is not None:
                self._client_tasks.discard(task)

    @staticmethod
    async def _iter_lines(reader: asyncio.StreamReader) -> AsyncIterator[str]:
        """Yield complete lines of any size without a 64 KiB StreamReader limit."""
        buffer = bytearray()
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                if buffer:
                    yield buffer.decode("utf-8", errors="replace").strip()
                break
            buffer.extend(chunk)
            while True:
                newline = buffer.find(b"\n")
                if newline == -1:
                    break
                line = bytes(buffer[:newline])
                del buffer[: newline + 1]
                yield line.decode("utf-8", errors="replace").strip()

    async def _dispatch(self, request: Any) -> dict[str, Any] | None:
        if not isinstance(request, dict):
            return _jsonrpc_error(None, -32600, "Invalid Request")
        msg_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        if not isinstance(method, str):
            return _jsonrpc_error(msg_id, -32600, "Invalid Request")
        # JSON-RPC notifications carry no `id` and expect no reply.
        if "id" not in request or msg_id is None:
            return None

        if method == "initialize":
            requested = (
                params.get("protocolVersion") if isinstance(params, dict) else None
            )
            version = (
                requested
                if isinstance(requested, str)
                and _PROTOCOL_VERSION_RE.fullmatch(requested)
                else _DEFAULT_PROTOCOL_VERSION
            )
            return _jsonrpc_result(
                msg_id,
                {
                    "protocolVersion": version,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": self._server_name, "version": "0.3.1"},
                },
            )
        if method == "ping":
            return _jsonrpc_result(msg_id, {})
        if method == "tools/list":
            return _jsonrpc_result(
                msg_id,
                {
                    "tools": [
                        {
                            "name": spec.name,
                            "description": spec.description,
                            "inputSchema": spec.input_schema,
                        }
                        for spec in self._tools
                    ]
                },
            )
        if method == "tools/call":
            if not isinstance(params, dict) or not isinstance(params.get("name"), str):
                return _jsonrpc_error(msg_id, -32602, "Tool name is required")
            name = params["name"]
            spec = self._by_name.get(name)
            if spec is None:
                return _jsonrpc_error(msg_id, -32602, f"Unknown tool: {name}")
            arguments = params.get("arguments")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                return _jsonrpc_error(
                    msg_id, -32602, "Tool arguments must be an object"
                )
            try:
                result = await spec.handler(arguments)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return _jsonrpc_result(
                    msg_id,
                    {
                        "content": [{"type": "text", "text": str(exc)}],
                        "isError": True,
                    },
                )
            return _jsonrpc_result(
                msg_id,
                {
                    "content": result.get("content", []),
                    "isError": bool(result.get("is_error", False)),
                },
            )
        return _jsonrpc_error(msg_id, -32601, f"Method not found: {method}")


def main() -> None:
    """Forward newline-delimited JSON-RPC between stdio and the parent bridge."""
    port = int(os.environ["NOCTURNOMATH_MCP_PORT"])
    token = os.environ["NOCTURNOMATH_MCP_TOKEN"]

    with socket.create_connection(("127.0.0.1", port)) as sock:
        sock.sendall((json.dumps({"auth": token}) + "\n").encode("utf-8"))
        reader = sock.makefile("rb")

        def pump_stdin() -> None:
            try:
                while True:
                    line = sys.stdin.buffer.readline()
                    if not line:
                        break
                    sock.sendall(line)
            except OSError:
                pass
            finally:
                try:
                    sock.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

        stdin_thread = threading.Thread(target=pump_stdin, daemon=True)
        stdin_thread.start()
        try:
            while True:
                line = reader.readline()
                if not line:
                    break
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
        finally:
            stdin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
