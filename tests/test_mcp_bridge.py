import asyncio
import base64
import json
import os
import stat
from pathlib import Path

import pytest

from nocturnomath.mcp_bridge import McpToolBridge
from nocturnomath.tools import ToolSpec, tool


@pytest.mark.asyncio
async def test_session_tools_expose_normalized_json_schemas(session):
    assert len(session.tools) == 12
    by_name = {spec.name: spec for spec in session.tools}
    for spec in by_name.values():
        assert isinstance(spec, ToolSpec)
        assert spec.input_schema["type"] == "object"
        assert isinstance(spec.input_schema["properties"], dict)

    assert by_name["verdict"].input_schema == {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    assert by_name["save_document"].input_schema == {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "text": {"type": "string"},
            "replace_existing": {"type": "boolean"},
        },
        "required": ["path", "text", "replace_existing"],
    }
    assert by_name["restart_kernel"].input_schema["properties"] == {}
    assert "probe_id" in by_name["check_slurm_job"].input_schema["properties"]


@pytest.mark.asyncio
async def test_mcp_bridge_serves_stdio_subprocess_roundtrip():
    png_bytes = b"\x89PNG\r\n\x1a\nfake"
    started_slow = asyncio.Event()
    release_slow = asyncio.Event()

    @tool("echo", "Echo text and optional png.", {"text": str, "with_image": bool})
    async def echo(args):
        if args.get("text") == "fail":
            return {
                "content": [{"type": "text", "text": "bad input"}],
                "is_error": True,
            }
        content = [{"type": "text", "text": f"echo: {args['text']}"}]
        if args.get("with_image"):
            content.append(
                {
                    "type": "image",
                    "data": base64.b64encode(png_bytes).decode("ascii"),
                    "mimeType": "image/png",
                }
            )
        return {"content": content}

    @tool("slow", "Wait for release.", {})
    async def slow(_args):
        started_slow.set()
        await release_slow.wait()
        return {"content": [{"type": "text", "text": "done"}]}

    async with McpToolBridge([echo, slow]) as bridge:
        cfg = bridge.stdio_server_config()
        assert bridge.token not in json.dumps(cfg)
        assert cfg["env"] == {}
        auth_file = Path(cfg["args"][-1])
        assert auth_file.is_file()
        if os.name == "posix":
            assert stat.S_IMODE(auth_file.stat().st_mode) == 0o600
            assert stat.S_IMODE(auth_file.parent.stat().st_mode) == 0o700
        proc = await asyncio.create_subprocess_exec(
            cfg["command"],
            *cfg["args"],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **cfg["env"]},
        )

        async def send(msg):
            proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
            await proc.stdin.drain()

        async def recv():
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=5)
            return json.loads(line)

        try:
            await send(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2025-03-26"},
                }
            )
            init = await recv()
            assert init["result"]["protocolVersion"] == "2025-03-26"
            assert init["result"]["serverInfo"]["name"] == "explore"

            # Coalesced notification + tools/list must both be handled properly.
            proc.stdin.write(
                (
                    json.dumps(
                        {"jsonrpc": "2.0", "method": "notifications/initialized"}
                    )
                    + "\n"
                    + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
                    + "\n"
                ).encode("utf-8")
            )
            await proc.stdin.drain()
            listed = await recv()
            assert [item["name"] for item in listed["result"]["tools"]] == [
                "echo",
                "slow",
            ]

            await send(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "echo",
                        "arguments": {"text": "hi", "with_image": True},
                    },
                }
            )
            called = await recv()
            assert called["result"]["isError"] is False
            assert called["result"]["content"][0] == {
                "type": "text",
                "text": "echo: hi",
            }
            assert called["result"]["content"][1]["type"] == "image"
            assert base64.b64decode(called["result"]["content"][1]["data"]) == png_bytes

            await send(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {"name": "echo", "arguments": {"text": "fail"}},
                }
            )
            failed = await recv()
            assert failed["result"]["isError"] is True
            assert failed["result"]["content"][0]["text"] == "bad input"

            # Ping must still answer while a slow tool call is running.
            await send(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {"name": "slow", "arguments": {}},
                }
            )
            await asyncio.wait_for(started_slow.wait(), timeout=5)
            await send({"jsonrpc": "2.0", "id": 6, "method": "ping"})
            ping_resp = await recv()
            assert ping_resp == {"jsonrpc": "2.0", "id": 6, "result": {}}
            release_slow.set()
            slow_resp = await recv()
            assert slow_resp["id"] == 5
            assert slow_resp["result"]["content"][0]["text"] == "done"

            # Direct TCP connection with wrong token is rejected immediately.
            reader, writer = await asyncio.open_connection("127.0.0.1", bridge.port)
            writer.write(
                b'{"auth":"wrong-token"}\n{"jsonrpc":"2.0","id":1,"method":"ping"}\n'
            )
            await writer.drain()
            assert await asyncio.wait_for(reader.read(), timeout=5) == b""
            writer.close()
            await writer.wait_closed()
        finally:
            proc.stdin.close()
            await asyncio.wait_for(proc.wait(), timeout=5)
    assert not auth_file.exists()
