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
import tempfile
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Self

from .tools import ToolSpec

logger = logging.getLogger("nocturnomath.mcp_bridge")

_PROTOCOL_VERSION_RE = re.compile(r"^20\d\d-\d\d-\d\d$")
_DEFAULT_PROTOCOL_VERSION = "2024-11-05"
MCP_PROGRESS_INTERVAL_S = 15.0


def _package_version() -> str:
    try:
        return version("nocturnomath")
    except PackageNotFoundError:
        return "0.0.0"


def _jsonrpc_result(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _jsonrpc_error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": code, "message": message},
    }


def _check_json_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return True


def _validate_property_value(name: str, value: Any, prop: dict[str, Any]) -> None:
    expected = prop.get("type")
    if isinstance(expected, str):
        if not _check_json_type(value, expected):
            raise ValueError(f"Argument '{name}' must be of type {expected}.")
    elif (
        isinstance(expected, list)
        and expected
        and not any(isinstance(t, str) and _check_json_type(value, t) for t in expected)
    ):
        joined = " | ".join(str(t) for t in expected)
        raise ValueError(f"Argument '{name}' must be of type {joined}.")
    if "enum" in prop and isinstance(prop["enum"], list) and value not in prop["enum"]:
        raise ValueError(f"Argument '{name}' must be one of {prop['enum']}.")
    if isinstance(value, list):
        min_items = prop.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            raise ValueError(
                f"Argument '{name}' must contain at least {min_items} item(s)."
            )
        items_schema = prop.get("items")
        if isinstance(items_schema, dict):
            for index, item in enumerate(value):
                _validate_property_value(f"{name}[{index}]", item, items_schema)


def _is_rpc_id(value: Any) -> bool:
    return isinstance(value, (str, int)) and not isinstance(value, bool)


def _validate_arguments(
    schema: dict[str, Any], arguments: dict[str, Any]
) -> dict[str, Any]:
    properties = (
        schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    )
    required_list = (
        schema.get("required") if isinstance(schema.get("required"), list) else []
    )
    disallow_extra = schema.get("additionalProperties") is False
    required = {item for item in required_list if isinstance(item, str)}
    cleaned: dict[str, Any] = {}
    for key, value in arguments.items():
        if disallow_extra and key not in properties:
            raise ValueError(f"Unexpected argument: '{key}'.")
        prop = properties.get(key)
        if value is None:
            prop_type = prop.get("type") if isinstance(prop, dict) else None
            allows_null = prop_type == "null" or (
                isinstance(prop_type, list) and "null" in prop_type
            )
            if allows_null:
                cleaned[key] = None
                continue
            if key not in required:
                continue
            raise ValueError(f"Argument '{key}' is required and cannot be null.")
        if isinstance(prop, dict):
            _validate_property_value(key, value, prop)
        cleaned[key] = value
    for key in required_list:
        if isinstance(key, str) and key not in cleaned:
            raise ValueError(f"Missing required argument: '{key}'.")
    return cleaned


class McpToolBridge:
    """Serve ``ToolSpec`` handlers over an authenticated loopback TCP socket."""

    def __init__(self, tools: Sequence[ToolSpec], *, server_name: str = "explore"):
        self._tools = list(tools)
        self._by_name = {spec.name: spec for spec in self._tools}
        self._server_name = server_name
        self.token = secrets.token_urlsafe(32)
        self._server: asyncio.AbstractServer | None = None
        self.port: int | None = None
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._auth_file: Path | None = None
        self._client_tasks: set[asyncio.Task[None]] = set()

    async def __aenter__(self) -> Self:
        self._server = await asyncio.start_server(
            self._handle_connection, host="127.0.0.1", port=0
        )
        try:
            sockets = self._server.sockets
            if not sockets:
                raise RuntimeError("MCP bridge failed to bind a loopback socket.")
            self.port = sockets[0].getsockname()[1]
            self._temp_dir = tempfile.TemporaryDirectory(prefix="nocturnomath-mcp-")
            auth_file = Path(self._temp_dir.name) / "bridge.json"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            fd = os.open(auth_file, flags, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"port": self.port, "token": self.token}, handle)
            self._auth_file = auth_file
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def cancel_in_flight(self) -> None:
        """Cancel active client connections and running tool handlers immediately."""
        tasks = [task for task in self._client_tasks if not task.done()]
        for task in tasks:
            task.cancel()
        if not tasks:
            return
        try:
            done, _ = await asyncio.wait(tasks)
            for finished in done:
                if not finished.cancelled():
                    with suppress(Exception):
                        finished.exception()
        except asyncio.CancelledError:
            for client_task in list(self._client_tasks):
                if not client_task.done():
                    client_task.cancel()
            remaining = [
                client_task
                for client_task in self._client_tasks
                if not client_task.done()
            ]
            if remaining:
                done, _ = await asyncio.wait(remaining, timeout=0.5)
                for finished in done:
                    if not finished.cancelled():
                        with suppress(Exception):
                            finished.exception()
            raise

    async def __aexit__(self, *args: object) -> None:
        try:
            await self.cancel_in_flight()
        finally:
            try:
                if self._server is not None:
                    self._server.close()
                    with suppress(asyncio.CancelledError, Exception):
                        await self._server.wait_closed()
                    self._server = None
            finally:
                self.port = None
                if self._temp_dir is not None:
                    with suppress(Exception):
                        self._temp_dir.cleanup()
                    self._temp_dir = None
                self._auth_file = None

    def stdio_server_config(self) -> dict[str, Any]:
        """Return standard MCP stdio server configuration for CLI subprocesses."""
        if self.port is None or self._auth_file is None:
            raise RuntimeError("MCP bridge must be entered before requesting config.")
        return {
            "command": sys.executable,
            "args": ["-m", "nocturnomath.mcp_bridge", str(self._auth_file)],
            "env": {},
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
            in_flight_by_id: dict[Any, asyncio.Task[None]] = {}
            already_cancelled: set[asyncio.Task[None]] = set()

            async def send_json(payload: dict[str, Any]) -> None:
                async with write_lock:
                    writer.write((json.dumps(payload) + "\n").encode("utf-8"))
                    await writer.drain()

            async def handle_parsed(request: Any, msg_id: Any) -> None:
                current = asyncio.current_task()
                try:
                    response = await self._dispatch(request, notify=send_json)
                    if response is not None:
                        await send_json(response)
                finally:
                    if _is_rpc_id(msg_id) and in_flight_by_id.get(msg_id) is current:
                        in_flight_by_id.pop(msg_id, None)

            def _drain_done(done_tasks: set[asyncio.Task[None]]) -> None:
                for finished in done_tasks:
                    if not finished.cancelled():
                        with suppress(Exception):
                            finished.exception()

            try:
                async for line in lines:
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        await send_json(_jsonrpc_error(None, -32700, "Parse error"))
                        continue
                    if (
                        isinstance(parsed, dict)
                        and parsed.get("method") == "notifications/cancelled"
                    ):
                        raw_params = parsed.get("params")
                        cancel_id = (
                            raw_params.get("requestId")
                            if isinstance(raw_params, dict)
                            else None
                        )
                        if _is_rpc_id(cancel_id):
                            target = in_flight_by_id.get(cancel_id)
                            if (
                                target is not None
                                and not target.done()
                                and target not in already_cancelled
                            ):
                                already_cancelled.add(target)
                                target.cancel()
                        continue
                    msg_id = parsed.get("id") if isinstance(parsed, dict) else None
                    valid_id = msg_id if _is_rpc_id(msg_id) else None
                    req_task = asyncio.create_task(handle_parsed(parsed, valid_id))
                    in_flight.add(req_task)
                    if valid_id is not None:
                        in_flight_by_id[valid_id] = req_task
                    req_task.add_done_callback(in_flight.discard)
                if in_flight:
                    done, _ = await asyncio.wait(list(in_flight))
                    _drain_done(done)
            finally:
                for req_task in list(in_flight):
                    if not req_task.done() and req_task not in already_cancelled:
                        already_cancelled.add(req_task)
                        req_task.cancel()
                if in_flight:
                    try:
                        done, _ = await asyncio.wait(list(in_flight))
                        _drain_done(done)
                    except asyncio.CancelledError:
                        for req_task in list(in_flight):
                            if not req_task.done():
                                req_task.cancel()
                        if in_flight:
                            done, _ = await asyncio.wait(list(in_flight), timeout=0.5)
                            _drain_done(done)
                        raise
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

    async def _dispatch(
        self,
        request: Any,
        *,
        notify: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any] | None:
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
                    "serverInfo": {
                        "name": self._server_name,
                        "version": _package_version(),
                    },
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
                arguments = _validate_arguments(spec.input_schema, arguments)
            except ValueError as exc:
                return _jsonrpc_result(
                    msg_id,
                    {
                        "content": [{"type": "text", "text": str(exc)}],
                        "isError": True,
                    },
                )
            meta = params.get("_meta")
            progress_token = (
                meta.get("progressToken") if isinstance(meta, dict) else None
            )

            async def send_progress() -> None:
                step = 0
                while True:
                    await asyncio.sleep(MCP_PROGRESS_INTERVAL_S)
                    step += 1
                    with suppress(Exception):
                        await notify(
                            {
                                "jsonrpc": "2.0",
                                "method": "notifications/progress",
                                "params": {
                                    "progressToken": progress_token,
                                    "progress": step,
                                },
                            }
                        )

            progress_task = (
                asyncio.create_task(send_progress())
                if progress_token is not None and notify is not None
                else None
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
            finally:
                if progress_task is not None:
                    progress_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await progress_task
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
    if len(sys.argv) > 1:
        info = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        port = int(info["port"])
        token = str(info["token"])
    else:
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
