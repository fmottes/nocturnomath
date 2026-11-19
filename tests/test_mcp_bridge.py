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

    @tool(
        "echo",
        "Echo text and optional png.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "with_image": {"type": "boolean"},
            },
            "required": ["text"],
        },
    )
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


@pytest.mark.asyncio
async def test_mcp_bridge_emits_progress_notifications(monkeypatch):
    from nocturnomath import mcp_bridge

    monkeypatch.setattr(mcp_bridge, "MCP_PROGRESS_INTERVAL_S", 0.03)
    started = asyncio.Event()
    release = asyncio.Event()

    @tool("slow", "Wait for release.", {})
    async def slow(_args):
        started.set()
        await release.wait()
        return {"content": [{"type": "text", "text": "done"}]}

    async with McpToolBridge([slow]) as bridge:
        cfg = bridge.stdio_server_config()
        proc = await asyncio.create_subprocess_exec(
            cfg["command"],
            *cfg["args"],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **cfg["env"]},
        )
        try:
            proc.stdin.write(
                (
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {
                                "name": "slow",
                                "arguments": {},
                                "_meta": {"progressToken": "tok-1"},
                            },
                        }
                    )
                    + "\n"
                ).encode("utf-8")
            )
            await proc.stdin.drain()
            await asyncio.wait_for(started.wait(), timeout=5)
            progress_msg = json.loads(
                await asyncio.wait_for(proc.stdout.readline(), timeout=5)
            )
            assert progress_msg["method"] == "notifications/progress"
            assert progress_msg["params"]["progressToken"] == "tok-1"
            assert progress_msg["params"]["progress"] >= 1
            release.set()
            result_msg = json.loads(
                await asyncio.wait_for(proc.stdout.readline(), timeout=5)
            )
            assert result_msg["id"] == 1
            assert result_msg["result"]["content"][0]["text"] == "done"
        finally:
            proc.stdin.close()
            await asyncio.wait_for(proc.wait(), timeout=5)


@pytest.mark.asyncio
async def test_mcp_bridge_validates_schema_types_and_strips_optional_nulls():
    seen: list[dict] = []

    @tool(
        "probe_like",
        "Check schema validation.",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "expected": {"type": "string"},
                "gpus": {"type": "integer"},
                "replace_existing": {"type": "boolean"},
                "packages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["code"],
        },
    )
    async def probe_like(args):
        seen.append(dict(args))
        return {"content": [{"type": "text", "text": "ok"}]}

    bridge = McpToolBridge([probe_like])

    async def call(arguments):
        resp = await bridge._dispatch(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "probe_like", "arguments": arguments},
            }
        )
        return resp["result"]

    ok = await call({"code": "print(1)", "expected": None, "gpus": None})
    assert ok["isError"] is False
    assert seen == [{"code": "print(1)"}]

    for bad in (
        {},
        {"code": None},
        {"code": 123},
        {"code": "x", "gpus": True},
        {"code": "x", "replace_existing": "false"},
        {"code": "x", "packages": "scipy"},
        {"code": "x", "packages": []},
        {"code": "x", "packages": [1]},
    ):
        res = await call(bad)
        assert res["isError"] is True, bad
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_mcp_bridge_enforces_additional_properties_false():
    @tool("strict", "Strict no-args tool.", {})
    async def strict(_args):
        return {"content": [{"type": "text", "text": "ok"}]}

    bridge = McpToolBridge([strict])
    resp = await bridge._dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "strict", "arguments": {"extra": 1}},
        }
    )
    assert resp["result"]["isError"] is True
    assert "Unexpected argument" in resp["result"]["content"][0]["text"]


@pytest.mark.asyncio
async def test_mcp_bridge_single_cancel_shield_and_malformed_ids():
    started = asyncio.Event()
    shield_release = asyncio.Event()
    cancel_count = 0
    shield_completed = False

    @tool("shielded_run", "Hold lock inside shield on first cancel.", {})
    async def shielded_run(_args):
        nonlocal cancel_count, shield_completed
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancel_count += 1
            await asyncio.shield(shield_release.wait())
            shield_completed = True
            raise
        return {"content": [{"type": "text", "text": "done"}]}

    async with McpToolBridge([shielded_run]) as bridge:
        cfg = bridge.stdio_server_config()
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

        try:
            await send(
                {
                    "jsonrpc": "2.0",
                    "id": {"bad": "id"},
                    "method": "ping",
                }
            )
            bad_id_resp = json.loads(
                await asyncio.wait_for(proc.stdout.readline(), timeout=5)
            )
            assert bad_id_resp["result"] == {}

            await send(
                {
                    "jsonrpc": "2.0",
                    "id": 10,
                    "method": "tools/call",
                    "params": {"name": "shielded_run", "arguments": {}},
                }
            )
            await asyncio.wait_for(started.wait(), timeout=5)

            # Malformed notifications/cancelled must be ignored without closing connection.
            await send(
                {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": []}
            )
            await send(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/cancelled",
                    "params": {"requestId": [1]},
                }
            )
            await send({"jsonrpc": "2.0", "id": 11, "method": "ping"})
            ping_resp = json.loads(
                await asyncio.wait_for(proc.stdout.readline(), timeout=5)
            )
            assert ping_resp == {"jsonrpc": "2.0", "id": 11, "result": {}}

            # CLI notifications/cancelled followed by bridge.cancel_in_flight() must only
            # cancel `req_task` once so the shield finishes before shutdown.
            await send(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/cancelled",
                    "params": {"requestId": 10},
                }
            )
            await asyncio.sleep(0.05)
            cancel_task = asyncio.create_task(bridge.cancel_in_flight())
            await asyncio.sleep(0.05)
            assert cancel_count == 1
            assert not cancel_task.done()
            shield_release.set()
            await asyncio.wait_for(cancel_task, timeout=5)
            assert shield_completed is True
            assert cancel_count == 1
        finally:
            proc.stdin.close()
            await asyncio.wait_for(proc.wait(), timeout=5)
