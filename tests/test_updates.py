"""Tests for the startup GitHub update check and CLI/web wiring."""

from __future__ import annotations

import asyncio
import io
from unittest.mock import patch

import httpx
import pytest
from rich.console import Console

from nocturnomath.cli.terminal import TerminalApp
from nocturnomath.runtime import Runtime
from nocturnomath.updates import PYPROJECT_URL, UPGRADE_COMMAND, check_for_update


def _mock_client(handler):
    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    return lambda *a, **kw: real_client(*a, transport=transport, **kw)


@pytest.mark.asyncio
async def test_check_for_update_compares_versions_and_fails_silently(monkeypatch):
    monkeypatch.delenv("NOCTURNOMATH_NO_UPDATE_CHECK", raising=False)

    def ok(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == PYPROJECT_URL
        return httpx.Response(200, text='[project]\nversion = "0.10.0"\n')

    with patch("nocturnomath.updates.httpx.AsyncClient", _mock_client(ok)):
        info = await check_for_update(current="0.3.1")
        assert info == {
            "current": "0.3.1",
            "latest": "0.10.0",
            "url": "https://github.com/fmottes/nocturnomath",
            "command": UPGRADE_COMMAND,
        }
        assert await check_for_update(current="0.10.0") is None
        assert await check_for_update(current="1.0.0") is None

    def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    with patch("nocturnomath.updates.httpx.AsyncClient", _mock_client(offline)):
        assert await check_for_update(current="0.3.1") is None

    monkeypatch.setenv("NOCTURNOMATH_NO_UPDATE_CHECK", "1")
    with patch("nocturnomath.updates.httpx.AsyncClient", _mock_client(ok)):
        assert await check_for_update(current="0.3.1") is None


@pytest.mark.asyncio
async def test_runtime_emits_update_for_web_snapshot_and_cli(tmp_path):
    update_info = {
        "current": "0.3.1",
        "latest": "0.4.0",
        "url": "https://github.com/fmottes/nocturnomath",
        "command": UPGRADE_COMMAND,
    }
    events: list[tuple[str, dict]] = []

    async def fake_check():
        return dict(update_info)

    with patch("nocturnomath.runtime.check_for_update", side_effect=fake_check):
        runtime = Runtime(navigator_root=tmp_path)
        runtime.subscribe(lambda kind, payload: events.append((kind, payload)))
        buf = io.StringIO()
        app = TerminalApp(runtime, console=Console(file=buf, force_terminal=False))
        runtime.subscribe(app.render_event)
        runtime.start()
        assert runtime._update_task is not None
        await asyncio.wait_for(runtime._update_task, timeout=2.0)

        assert ("update_available", update_info) in events
        assert runtime.workspace_snapshot()["update_available"] == update_info
        app.banner()
        assert buf.getvalue().count("update available: 0.3.1 → 0.4.0") == 1
        await runtime.drain()
        runtime.shutdown()
