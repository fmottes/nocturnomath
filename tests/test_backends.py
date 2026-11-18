import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from nocturnomath.auth import ProviderAuth
from nocturnomath.backends import (
    AnthropicBackend,
    CompactionNotice,
    GoogleBackend,
    MessageUsage,
    OpenAIBackend,
    SessionResumeError,
    SessionStarted,
    TextBlockDone,
    TextDelta,
    TurnResult,
)
from nocturnomath.tools import tool


def _write_fake_cli(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(0o700)
    return path


@pytest.mark.asyncio
async def test_anthropic_backend_discovery_turn_and_compact(tmp_path):
    log_path = tmp_path / "anthropic_log.json"
    fake = _write_fake_cli(
        tmp_path / "claude",
        f"""
import json, os, sys
log_path = {str(log_path)!r}
argv = sys.argv[1:]
logs = json.loads(open(log_path).read()) if os.path.exists(log_path) else []
logs.append({{"argv": argv, "oauth": os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"), "api_key": os.environ.get("ANTHROPIC_API_KEY")}})
open(log_path, "w").write(json.dumps(logs))

for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("type") == "control_request":
        req = msg["request"]
        subtype = req["subtype"]
        if subtype == "initialize":
            resp = {{"models": [
                {{"value": "default", "resolvedModel": "claude-opus-5-5", "supportedEffortLevels": ["low", "medium", "high", "xhigh", "max"]}},
                {{"value": "sonnet", "resolvedModel": "claude-sonnet-5-5", "supportedEffortLevels": ["low", "medium", "high"]}},
            ]}}
        elif subtype == "get_context_usage":
            resp = {{"totalTokens": 4200, "rawMaxTokens": 200000, "model": "claude-opus-5-5"}}
        else:
            resp = {{}}
        print(json.dumps({{"type": "control_response", "response": {{"subtype": "success", "request_id": msg["request_id"], "response": resp}}}}), flush=True)
    elif msg.get("type") == "user":
        text = msg["message"]["content"]
        if text == "/compact":
            print(json.dumps({{"type": "system", "subtype": "compact_boundary", "session_id": "ant-1"}}), flush=True)
            print(json.dumps({{"type": "result", "subtype": "success", "is_error": False, "session_id": "ant-1"}}), flush=True)
        else:
            print(json.dumps({{"type": "stream_event", "session_id": "ant-1", "parent_tool_use_id": None, "event": {{"type": "content_block_delta", "delta": {{"type": "text_delta", "text": "hel"}}}}}}), flush=True)
            print(json.dumps({{"type": "stream_event", "session_id": "ant-1", "parent_tool_use_id": "sub", "event": {{"type": "content_block_delta", "delta": {{"type": "text_delta", "text": "ignored"}}}}}}), flush=True)
            print(json.dumps({{"type": "assistant", "session_id": "ant-1", "parent_tool_use_id": None, "message": {{"id": "m1", "model": "claude-opus-5-5", "usage": {{"input_tokens": 100, "cache_read_input_tokens": 50, "cache_creation_input_tokens": 0}}, "content": [{{"type": "text", "text": "hello"}}]}}}}), flush=True)
            print(json.dumps({{"type": "result", "subtype": "success", "is_error": False, "session_id": "ant-1", "total_cost_usd": 0.02, "modelUsage": {{"claude-opus-5-5": {{"contextWindow": 200000}}}}}}), flush=True)
""",
    )
    backend = AnthropicBackend()
    auth = ProviderAuth.interactive("anthropic", "subscription", "oauth-secret")
    with patch(
        "nocturnomath.backends.anthropic.claude_cli_candidates", return_value=[fake]
    ):
        models = await backend.discover_models(auth)

    assert [m.id for m in models] == ["claude-opus-5-5", "claude-sonnet-5-5"]
    assert models[0].is_default is True
    assert models[0].efforts == ["low", "medium", "high", "xhigh", "max"]

    @tool("ping", "Ping.", {})
    async def ping(_args):
        return {"content": [{"type": "text", "text": "ok"}]}

    events = [
        event
        async for event in backend.run_turn(
            "question",
            system_prompt="System",
            tools=[ping],
            workspace_path=tmp_path,
            model="claude-opus-5-5",
            effort="high",
            auth=auth,
        )
    ]
    assert SessionStarted("ant-1") in events
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["hel"]
    assert [e.text for e in events if isinstance(e, TextBlockDone)] == ["hello"]
    usage_msgs = [e for e in events if isinstance(e, MessageUsage)]
    assert usage_msgs[0].message_id == "m1"
    result = next(e for e in events if isinstance(e, TurnResult))
    assert result.context_usage == {
        "used_tokens": 150,
        "window_tokens": 200000,
        "model": "claude-opus-5-5",
    }

    compact = await backend.compact(
        system_prompt="System",
        tools=[ping],
        workspace_path=tmp_path,
        model="claude-opus-5-5",
        effort="high",
        auth=auth,
        resume_session_id="ant-1",
    )
    assert compact.session_id == "ant-1"
    assert compact.context_usage == {
        "used_tokens": 4200,
        "window_tokens": 200000,
        "model": "claude-opus-5-5",
    }

    logs = json.loads(log_path.read_text())
    assert logs[1]["oauth"] == "oauth-secret"
    assert logs[1]["api_key"] == ""
    assert "--tools" in logs[1]["argv"]
    assert "--strict-mcp-config" in logs[1]["argv"]
    assert "NOCTURNOMATH_MCP_TOKEN" not in json.dumps(logs[1]["argv"])
    mcp_idx = logs[1]["argv"].index("--mcp-config") + 1
    mcp_payload = json.loads(logs[1]["argv"][mcp_idx])["mcpServers"]["explore"]
    assert mcp_payload["env"] == {}
    assert mcp_payload["args"][-1].endswith("bridge.json")
    assert "--resume=ant-1" in logs[2]["argv"]


@pytest.mark.asyncio
async def test_openai_backend_discovery_turn_compact_and_api_key_isolation(
    tmp_path, monkeypatch
):
    real_home = tmp_path / "real_codex_home"
    real_home.mkdir()
    (real_home / "config.toml").write_text(
        '[mcp_servers.user_extra]\ncommand = "echo"\n', encoding="utf-8"
    )
    (real_home / "auth.json").write_text('{"keep":"untouched"}\n', encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(real_home))

    log_path = tmp_path / "openai_log.json"
    fake = _write_fake_cli(
        tmp_path / "codex",
        f"""
import json, os, sys
log_path = {str(log_path)!r}
logs = json.loads(open(log_path).read()) if os.path.exists(log_path) else []
logs.append({{"argv": sys.argv[1:], "codex_home": os.environ.get("CODEX_HOME"), "cwd": os.getcwd()}})
open(log_path, "w").write(json.dumps(logs))

for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"userAgent": "codex"}}}}), flush=True)
    elif method == "account/login/start":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{}}}}), flush=True)
    elif method == "account/read":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"requiresOpenaiAuth": True, "account": {{"type": "apiKey"}}}}}}), flush=True)
    elif method == "model/list":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"data": [
            {{"id": "gpt-5.6-sol", "model": "gpt-5.6-sol", "isDefault": True, "supportedReasoningEfforts": [{{"reasoningEffort": "low"}}, {{"reasoningEffort": "high"}}, {{"reasoningEffort": "ultra"}}]}},
        ], "nextCursor": None}}}}), flush=True)
    elif method in ("thread/start", "thread/resume"):
        logs[-1]["thread_params"] = msg.get("params")
        open(log_path, "w").write(json.dumps(logs))
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"thread": {{"id": "thr-1"}}}}}}), flush=True)
    elif method == "turn/start":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"turn": {{"id": "turn-1"}}}}}}), flush=True)
        print(json.dumps({{"jsonrpc": "2.0", "method": "item/agentMessage/delta", "params": {{"delta": "chunk"}}}}), flush=True)
        print(json.dumps({{"jsonrpc": "2.0", "method": "item/completed", "params": {{"item": {{"type": "agentMessage", "text": "chunk done"}}}}}}), flush=True)
        print(json.dumps({{"jsonrpc": "2.0", "method": "thread/tokenUsage/updated", "params": {{"tokenUsage": {{"modelContextWindow": 272000, "last": {{"totalTokens": 900}}, "total": {{"totalTokens": 900}}}}}}}}), flush=True)
        print(json.dumps({{"jsonrpc": "2.0", "method": "turn/completed", "params": {{"turn": {{"id": "turn-1", "status": "completed"}}}}}}), flush=True)
    elif method == "thread/compact/start":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{}}}}), flush=True)
        print(json.dumps({{"jsonrpc": "2.0", "method": "thread/tokenUsage/updated", "params": {{"tokenUsage": {{"modelContextWindow": 272000, "last": {{"totalTokens": 300}}}}}}}}), flush=True)
        print(json.dumps({{"jsonrpc": "2.0", "method": "thread/compacted", "params": {{}}}}), flush=True)
""",
    )
    backend = OpenAIBackend()
    api_auth = ProviderAuth.interactive("openai", "api_key", "sk-openai-secret")
    with patch(
        "nocturnomath.backends.openai.codex_cli_candidates", return_value=[fake]
    ):
        models = await backend.discover_models(api_auth)
        assert len(models) == 1
        assert models[0].id == "gpt-5.6-sol"
        assert models[0].efforts == ["low", "high", "ultra"]

        @tool("ping", "Ping.", {})
        async def ping(_args):
            return {"content": [{"type": "text", "text": "ok"}]}

        events = [
            event
            async for event in backend.run_turn(
                "hi",
                system_prompt="System",
                tools=[ping],
                workspace_path=tmp_path,
                model="gpt-5.6-sol",
                effort="ultra",
                auth=api_auth,
            )
        ]
        assert SessionStarted("thr-1") in events
        assert [e.text for e in events if isinstance(e, TextDelta)] == ["chunk"]
        assert [e.text for e in events if isinstance(e, TextBlockDone)] == [
            "chunk done"
        ]
        result = next(e for e in events if isinstance(e, TurnResult))
        assert result.context_usage == {
            "used_tokens": 900,
            "window_tokens": 272000,
            "model": "gpt-5.6-sol",
        }

        compact = await backend.compact(
            system_prompt="System",
            tools=[ping],
            workspace_path=tmp_path,
            model="gpt-5.6-sol",
            effort="ultra",
            auth=api_auth,
            resume_session_id="thr-1",
        )
        assert compact.session_id == "thr-1"
        assert compact.context_usage["used_tokens"] == 300

        # In CLI mode, user-configured MCP servers from CODEX_HOME/config.toml are disabled,
        # and the temporary API-key CODEX_HOME is removed immediately.
        api_home_path = Path(backend._api_key_home.name)
        assert api_home_path.is_dir()
        cli_models = await backend.discover_models(ProviderAuth("openai", "cli"))
        assert len(cli_models) == 1
        assert backend._api_key_home is None
        assert not api_home_path.exists()

    logs = json.loads(log_path.read_text())
    assert logs[0]["codex_home"] != str(real_home)
    assert logs[1]["cwd"] != str(tmp_path)
    assert 'cli_auth_credentials_store="ephemeral"' in logs[0]["argv"]
    assert 'cli_auth_credentials_store="ephemeral"' in logs[1]["argv"]
    assert 'cli_auth_credentials_store="ephemeral"' not in logs[3]["argv"]
    assert "developerInstructions" not in logs[1]["thread_params"]
    assert "baseInstructions" in logs[1]["thread_params"]
    assert "NOCTURNOMATH_MCP_TOKEN" not in json.dumps(logs[1]["argv"])
    assert 'default_tools_approval_mode="approve"' in " ".join(logs[1]["argv"])
    assert "mcp_servers.node_repl.enabled=false" in logs[3]["argv"]
    assert "mcp_servers.user_extra.enabled=false" in logs[3]["argv"]
    assert (real_home / "auth.json").read_text() == '{"keep":"untouched"}\n'
    backend._cleanup_api_home()


@pytest.mark.asyncio
async def test_google_backend_discovery_permissions_turn_and_compact(tmp_path):
    log_path = tmp_path / "google_log.json"
    fake = _write_fake_cli(
        tmp_path / "agy",
        f"""
import json, os, sys
log_path = {str(log_path)!r}
logs = json.loads(open(log_path).read()) if os.path.exists(log_path) else []
logs.append({{"argv": sys.argv[1:], "gemini_key": os.environ.get("GEMINI_API_KEY"), "cwd": os.getcwd()}})
open(log_path, "w").write(json.dumps(logs))

for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"protocolVersion": 1}}}}), flush=True)
    elif method == "authenticate":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{}}}}), flush=True)
    elif method in ("session/new", "session/resume"):
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"sessionId": "agy-1", "models": {{"availableModels": [
            {{"modelId": "gemini-4-carbon-high", "name": "Gemini 4 Carbon (High)"}},
            {{"modelId": "gemini-4-carbon-low", "name": "Gemini 4 Carbon (Low)"}},
            {{"modelId": "gemini-pro-orchestrator", "name": "Orchestrator (Carbon)"}},
            {{"modelId": "claude-opus-5-5-high", "name": "Claude Opus 5.5 (High)"}},
        ]}}}}}}), flush=True)
    elif method == "session/set_model":
        logs[-1]["set_model"] = msg["params"]["modelId"]
        open(log_path, "w").write(json.dumps(logs))
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{}}}}), flush=True)
    elif method == "session/prompt":
        prompt_text = msg["params"]["prompt"][0]["text"]
        if prompt_text == "/compact":
            print(json.dumps({{"jsonrpc": "2.0", "method": "session/update", "params": {{"sessionId": "agy-1", "update": {{"sessionUpdate": "usage_update", "used": 200, "size": 1000000}}}}}}), flush=True)
            print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"stopReason": "end_turn"}}}}), flush=True)
            continue
        # First ask permission for disallowed built-in run_command, then allowed mcp_explore_run.
        print(json.dumps({{"jsonrpc": "2.0", "id": "perm_1", "method": "session/request_permission", "params": {{"sessionId": "agy-1", "toolCall": {{"name": "run_command"}}, "options": [{{"optionId": "allow", "kind": "allow_once"}}, {{"optionId": "deny", "kind": "reject_once"}}]}}}}), flush=True)
        resp1 = json.loads(sys.stdin.readline())
        print(json.dumps({{"jsonrpc": "2.0", "id": "perm_2", "method": "session/request_permission", "params": {{"sessionId": "agy-1", "toolCall": {{"name": "mcp_explore_run"}}, "options": [{{"optionId": "allow", "kind": "allow_once"}}, {{"optionId": "deny", "kind": "reject_once"}}]}}}}), flush=True)
        resp2 = json.loads(sys.stdin.readline())
        logs[-1]["perms"] = [resp1["result"]["outcome"]["optionId"], resp2["result"]["outcome"]["optionId"]]
        open(log_path, "w").write(json.dumps(logs))
        print(json.dumps({{"jsonrpc": "2.0", "method": "session/update", "params": {{"sessionId": "agy-1", "update": {{"sessionUpdate": "agent_message_chunk", "content": {{"type": "text", "text": "part1"}}}}}}}}), flush=True)
        print(json.dumps({{"jsonrpc": "2.0", "method": "session/update", "params": {{"sessionId": "agy-1", "update": {{"sessionUpdate": "tool_call", "name": "mcp_explore_run"}}}}}}), flush=True)
        print(json.dumps({{"jsonrpc": "2.0", "method": "session/update", "params": {{"sessionId": "agy-1", "update": {{"sessionUpdate": "agent_message_chunk", "content": {{"type": "text", "text": "part2"}}}}}}}}), flush=True)
        print(json.dumps({{"jsonrpc": "2.0", "method": "session/update", "params": {{"sessionId": "agy-1", "update": {{"sessionUpdate": "usage_update", "used": 1234, "size": 1048576}}}}}}), flush=True)
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"stopReason": "end_turn", "usage": {{"inputTokens": 1200, "outputTokens": 34}}}}}}), flush=True)
""",
    )
    backend = GoogleBackend()
    auth = ProviderAuth.interactive("google", "api_key", "AIza-secret")

    @tool("run", "Run.", {"code": str})
    async def run(_args):
        return {"content": [{"type": "text", "text": "ok"}]}

    with patch("nocturnomath.backends.google.agy_cli_candidates", return_value=[fake]):
        models = await backend.discover_models(auth)
        assert [m.id for m in models] == ["gemini-4-carbon", "gemini-pro-orchestrator"]
        assert models[0].efforts == ["low", "high"]
        assert models[0].label == "Gemini 4 Carbon"
        assert models[1].efforts == []

        events = [
            event
            async for event in backend.run_turn(
                "explore",
                system_prompt="System",
                tools=[run],
                workspace_path=tmp_path,
                model="gemini-4-carbon",
                effort="high",
                auth=auth,
            )
        ]
        assert SessionStarted("agy-1") in events
        assert [e.text for e in events if isinstance(e, TextDelta)] == [
            "part1",
            "part2",
        ]
        assert [e.text for e in events if isinstance(e, TextBlockDone)] == [
            "part1",
            "part2",
        ]
        result = next(e for e in events if isinstance(e, TurnResult))
        assert result.context_usage == {
            "used_tokens": 1234,
            "window_tokens": 1048576,
            "model": "gemini-4-carbon",
        }

        compact = await backend.compact(
            system_prompt="System",
            tools=[run],
            workspace_path=tmp_path,
            model="gemini-4-carbon",
            effort="high",
            auth=auth,
            resume_session_id="agy-1",
        )
        assert compact.session_id == "agy-1"
        assert compact.context_usage == {
            "used_tokens": 200,
            "window_tokens": 1000000,
            "model": "gemini-4-carbon",
        }

    logs = json.loads(log_path.read_text())
    assert "--gemini-dir" in logs[1]["argv"]
    assert "--include-default-customizations=false" in logs[1]["argv"]
    assert logs[1]["set_model"] == "gemini-4-carbon-high"
    assert logs[1]["perms"] == ["deny", "allow"]
    assert logs[1]["cwd"] != str(tmp_path)
    backend._cleanup_api_dir()


@pytest.mark.asyncio
async def test_anthropic_drains_large_stderr_and_surfaces_early_exits(tmp_path):
    noisy = _write_fake_cli(
        tmp_path / "claude_noisy",
        """
import json, sys
sys.stderr.write("W" * (160 * 1024) + "\\n")
sys.stderr.flush()
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("type") == "control_request":
        print(json.dumps({"type": "control_response", "response": {"subtype": "success", "request_id": msg["request_id"], "response": {}}}), flush=True)
    elif msg.get("type") == "user":
        print(json.dumps({"type": "assistant", "session_id": "ant-noisy", "parent_tool_use_id": None, "message": {"id": "m1", "model": "claude", "content": [{"type": "text", "text": "ok"}]}}), flush=True)
        print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "session_id": "ant-noisy"}), flush=True)
""",
    )
    backend = AnthropicBackend(cli_path=noisy)
    events = [
        e
        async for e in backend.run_turn(
            "hi",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="claude",
            effort=None,
            auth=ProviderAuth("anthropic", "cli"),
        )
    ]
    assert [e.text for e in events if isinstance(e, TextBlockDone)] == ["ok"]

    silent_exit = _write_fake_cli(
        tmp_path / "claude_silent_exit",
        """
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("type") == "control_request":
        print(json.dumps({"type": "control_response", "response": {"subtype": "success", "request_id": msg["request_id"], "response": {}}}), flush=True)
    elif msg.get("type") == "user":
        sys.stderr.write("unexpected eof detail\\n")
        sys.stderr.flush()
        sys.exit(0)
""",
    )
    with pytest.raises(RuntimeError, match="unexpected eof detail"):
        async for _ in AnthropicBackend(cli_path=silent_exit).run_turn(
            "hi",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="claude",
            effort=None,
            auth=ProviderAuth("anthropic", "cli"),
        ):
            pass

    missing_resume = _write_fake_cli(
        tmp_path / "claude_missing_resume",
        """
import json, sys
sys.stderr.write("No conversation found with session ID: stale-id\\n")
sys.stderr.flush()
print(json.dumps({"type": "result", "is_error": True, "errors": ["No conversation found with session ID: stale-id"]}), flush=True)
sys.exit(1)
""",
    )
    with pytest.raises(SessionResumeError, match="No conversation found"):
        async for _ in AnthropicBackend(cli_path=missing_resume).run_turn(
            "hi",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="claude",
            effort=None,
            auth=ProviderAuth("anthropic", "cli"),
            resume_session_id="stale-id",
        ):
            pass


@pytest.mark.asyncio
async def test_openai_buffered_resume_usage_delta_and_context_compaction_item(tmp_path):
    fake = _write_fake_cli(
        tmp_path / "codex_resume",
        """
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {}}), flush=True)
    elif method == "thread/resume":
        if msg["params"]["threadId"] == "missing":
            print(json.dumps({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": "Thread not found"}}), flush=True)
            continue
        # Emit prior-turn token usage notification before thread/resume response!
        print(json.dumps({"jsonrpc": "2.0", "method": "thread/tokenUsage/updated", "params": {"turnId": "turn-prev", "tokenUsage": {"modelContextWindow": 200000, "last": {"totalTokens": 100, "inputTokens": 80, "outputTokens": 20}, "total": {"totalTokens": 100, "inputTokens": 80, "outputTokens": 20}}}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"thread": {"id": "thr-1"}}}), flush=True)
    elif method == "turn/start":
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"turn": {"id": "turn-now"}}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "method": "item/completed", "params": {"item": {"type": "contextCompaction"}}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "method": "thread/compacted", "params": {}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "resumed"}}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "method": "thread/tokenUsage/updated", "params": {"turnId": "turn-now", "tokenUsage": {"modelContextWindow": 200000, "last": {"totalTokens": 250, "inputTokens": 200, "outputTokens": 50}, "total": {"totalTokens": 350, "inputTokens": 280, "outputTokens": 70}}}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "method": "turn/completed", "params": {"turn": {"id": "turn-now", "status": "completed"}}}), flush=True)
    elif method == "thread/compact/start":
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "method": "item/completed", "params": {"item": {"type": "contextCompaction"}}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "method": "turn/completed", "params": {"turn": {"id": "compact-1", "status": "completed"}}}), flush=True)
""",
    )
    backend = OpenAIBackend(cli_path=fake)
    auth = ProviderAuth("openai", "cli")

    with pytest.raises(SessionResumeError, match="Thread not found"):
        async for _ in backend.run_turn(
            "hi",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gpt-5.6-sol",
            effort=None,
            auth=auth,
            resume_session_id="missing",
        ):
            pass

    events = [
        e
        async for e in backend.run_turn(
            "hi",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gpt-5.6-sol",
            effort=None,
            auth=auth,
            resume_session_id="thr-1",
        )
    ]
    assert sum(isinstance(e, CompactionNotice) for e in events) == 1
    result = next(e for e in events if isinstance(e, TurnResult))
    assert result.usage == {"totalTokens": 250, "inputTokens": 200, "outputTokens": 50}
    assert result.context_usage["used_tokens"] == 250

    compact = await backend.compact(
        system_prompt="Sys",
        tools=[],
        workspace_path=tmp_path,
        model="gpt-5.6-sol",
        effort=None,
        auth=auth,
        resume_session_id="thr-1",
    )
    assert compact.session_id == "thr-1"


@pytest.mark.asyncio
async def test_google_executor_errors_cancelled_permissions_and_compact(tmp_path):
    log_path = tmp_path / "agy_edge_log.json"
    fake = _write_fake_cli(
        tmp_path / "agy_edge",
        f"""
import json, os, sys
log_path = {str(log_path)!r}
logs = json.loads(open(log_path).read()) if os.path.exists(log_path) else []
logs.append({{"argv": sys.argv[1:]}})
open(log_path, "w").write(json.dumps(logs))

for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"protocolVersion": 1}}}}), flush=True)
    elif method == "authenticate":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{}}}}), flush=True)
    elif method == "session/new":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"sessionId": "agy-1", "models": {{"currentModelId": "gemini-pro-orchestrator", "availableModels": [
            {{"modelId": "gemini-4-carbon-high", "name": "Gemini 4 Carbon (High)"}},
            {{"modelId": "gemini-pro-orchestrator", "name": "Orchestrator (Carbon)"}},
        ]}}}}}}), flush=True)
    elif method == "session/resume":
        if msg["params"]["sessionId"] == "missing":
            print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "error": {{"code": -32000, "message": "Session not found"}}}}), flush=True)
            continue
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"sessionId": msg["params"]["sessionId"]}}}}), flush=True)
    elif method == "session/set_model":
        logs[-1]["set_model"] = msg["params"]["modelId"]
        open(log_path, "w").write(json.dumps(logs))
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{}}}}), flush=True)
    elif method == "session/prompt":
        text = msg["params"]["prompt"][0]["text"]
        if text == "trigger_perm_and_error":
            print(json.dumps({{"jsonrpc": "2.0", "id": "p1", "method": "session/request_permission", "params": {{"sessionId": "agy-1", "toolCall": {{"name": "unknown_tool"}}, "options": []}}}}), flush=True)
            resp = json.loads(sys.stdin.readline())
            logs[-1]["perm_outcome"] = resp["result"]["outcome"]
            open(log_path, "w").write(json.dumps(logs))
            print(json.dumps({{"jsonrpc": "2.0", "method": "session/update", "params": {{"sessionId": "agy-1", "update": {{"sessionUpdate": "agent_message_chunk", "content": {{"type": "text", "text": "\\n\\u26a0 agent executor error: Error 400, API key not valid"}}}}}}}}), flush=True)
            print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"stopReason": "end_turn"}}}}), flush=True)
        elif text == "/compact":
            sid = msg["params"]["sessionId"]
            if sid == "no-compact":
                print(json.dumps({{"jsonrpc": "2.0", "method": "session/update", "params": {{"sessionId": sid, "update": {{"sessionUpdate": "available_commands_update", "availableCommands": [{{"name": "plan"}}]}}}}}}), flush=True)
            elif sid == "stray-reply":
                print(json.dumps({{"jsonrpc": "2.0", "method": "session/update", "params": {{"sessionId": sid, "update": {{"sessionUpdate": "agent_message_chunk", "content": {{"type": "text", "text": "Nothing to compact yet."}}}}}}}}), flush=True)
                print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"stopReason": "end_turn"}}}}), flush=True)
            else:
                print(json.dumps({{"jsonrpc": "2.0", "method": "session/update", "params": {{"sessionId": sid, "update": {{"sessionUpdate": "available_commands_update", "availableCommands": [{{"name": "compact"}}]}}}}}}), flush=True)
                print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"stopReason": "end_turn", "usage": {{"inputTokens": 450, "cachedReadTokens": 50}}}}}}), flush=True)
""",
    )
    backend = GoogleBackend(cli_path=fake)
    api_auth = ProviderAuth.interactive("google", "api_key", "AIza-secret")
    with patch("nocturnomath.backends.google.agy_cli_candidates", return_value=[fake]):
        models = await backend.discover_models(api_auth)
        assert [m.is_default for m in models] == [False, True]
        api_dir_path = Path(backend._api_key_dir.name)
        assert api_dir_path.is_dir()

        cli_auth = ProviderAuth("google", "cli")
        await backend.discover_models(cli_auth)
        assert backend._api_key_dir is None
        assert not api_dir_path.exists()

    with pytest.raises(SessionResumeError, match="Session not found"):
        async for _ in backend.run_turn(
            "hi",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gemini-4-carbon",
            effort="high",
            auth=cli_auth,
            resume_session_id="missing",
        ):
            pass

    with pytest.raises(RuntimeError, match="API key not valid"):
        async for _ in backend.run_turn(
            "trigger_perm_and_error",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gemini-4-carbon",
            effort="high",
            auth=cli_auth,
        ):
            pass

    with pytest.raises(RuntimeError, match="does not support /compact"):
        await backend.compact(
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gemini-4-carbon",
            effort="high",
            auth=cli_auth,
            resume_session_id="no-compact",
        )

    with pytest.raises(RuntimeError, match="Nothing to compact yet"):
        await backend.compact(
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gemini-4-carbon",
            effort="high",
            auth=cli_auth,
            resume_session_id="stray-reply",
        )

    compact = await backend.compact(
        system_prompt="Sys",
        tools=[],
        workspace_path=tmp_path,
        model="gemini-4-carbon",
        effort="high",
        auth=cli_auth,
        resume_session_id="ok-compact",
    )
    assert compact.context_usage == {
        "used_tokens": 500,
        "window_tokens": None,
        "model": "gemini-4-carbon",
    }
    logs = json.loads(log_path.read_text())
    assert logs[3]["perm_outcome"] == {"outcome": "cancelled"}
    assert logs[-1]["set_model"] == "gemini-4-carbon-high"


@pytest.mark.asyncio
async def test_anthropic_discovery_rejects_unauthenticated_cli_without_env(
    tmp_path, monkeypatch
):
    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
    ):
        monkeypatch.delenv(var, raising=False)

    fake = _write_fake_cli(
        tmp_path / "claude_unauth",
        """
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("type") == "control_request":
        print(json.dumps({
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": msg["request_id"],
                "response": {
                    "account": {"tokenSource": "none", "apiProvider": "firstParty"},
                    "models": [{"value": "sonnet", "resolvedModel": "claude-sonnet-5-5"}],
                },
            },
        }), flush=True)
""",
    )
    fake_api_key_login = _write_fake_cli(
        tmp_path / "claude_api_key_login",
        """
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("type") == "control_request":
        print(json.dumps({
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": msg["request_id"],
                "response": {
                    "account": {
                        "tokenSource": "none",
                        "apiKeySource": "/login managed key",
                        "apiProvider": "firstParty",
                    },
                    "models": [{"value": "sonnet", "resolvedModel": "claude-sonnet-5-5"}],
                },
            },
        }), flush=True)
""",
    )
    backend = AnthropicBackend()
    with patch(
        "nocturnomath.backends.anthropic.claude_cli_candidates", return_value=[fake]
    ):
        assert await backend.discover_models(ProviderAuth("anthropic", "cli")) == []
        api_models = await backend.discover_models(
            ProviderAuth.interactive("anthropic", "api_key", "sk-ant-test")
        )
        assert [m.id for m in api_models] == ["claude-sonnet-5-5"]

    with patch(
        "nocturnomath.backends.anthropic.claude_cli_candidates",
        return_value=[fake_api_key_login],
    ):
        managed_models = await backend.discover_models(ProviderAuth("anthropic", "cli"))
        assert [m.id for m in managed_models] == ["claude-sonnet-5-5"]


@pytest.mark.asyncio
async def test_openai_tool_timeout_early_compaction_and_transient_resume_error(
    tmp_path,
):
    log_path = tmp_path / "codex_opts.json"
    fake = _write_fake_cli(
        tmp_path / "codex_timeout_test",
        f"""
import json, sys
open({str(log_path)!r}, "w").write(json.dumps(sys.argv[1:]))
for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{}}}}), flush=True)
    elif method == "thread/start":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"thread": {{"id": "thr-1"}}}}}}), flush=True)
    elif method == "turn/start":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"turn": {{"id": "turn-1"}}}}}}), flush=True)
        print(json.dumps({{"jsonrpc": "2.0", "method": "turn/completed", "params": {{"turn": {{"id": "turn-1", "status": "completed"}}}}}}), flush=True)
    elif method == "thread/resume":
        if msg["params"]["threadId"] == "crash":
            sys.stderr.write("fatal transport error\\n")
            sys.stderr.flush()
            sys.exit(1)
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"thread": {{"id": "thr-1"}}}}}}), flush=True)
    elif method == "thread/compact/start":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{}}}}), flush=True)
        # Emit contextCompaction without turn/completed or thread/compacted and stay open.
        print(json.dumps({{"jsonrpc": "2.0", "method": "item/completed", "params": {{"item": {{"type": "contextCompaction"}}}}}}), flush=True)
""",
    )
    backend = OpenAIBackend(cli_path=fake)
    auth = ProviderAuth("openai", "cli")

    events = [
        e
        async for e in backend.run_turn(
            "hi",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gpt-5.6-sol",
            effort=None,
            auth=auth,
            tool_timeout_s=720,
        )
    ]
    assert isinstance(events[-1], TurnResult)
    argv = json.loads(log_path.read_text())
    assert any("tool_timeout_sec=720" in arg for arg in argv)

    # A crash during thread/resume must NOT be converted into SessionResumeError.
    with pytest.raises(RuntimeError, match="fatal transport error") as exc_info:
        async for _ in backend.run_turn(
            "hi",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gpt-5.6-sol",
            effort=None,
            auth=auth,
            resume_session_id="crash",
        ):
            pass
    assert not isinstance(exc_info.value, SessionResumeError)

    # Compaction with contextCompaction item finishes after the short trailing wait.
    result = await asyncio.wait_for(
        backend.compact(
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gpt-5.6-sol",
            effort=None,
            auth=auth,
            resume_session_id="thr-1",
        ),
        timeout=6.0,
    )
    assert result.session_id == "thr-1"


@pytest.mark.asyncio
async def test_google_transient_resume_error_and_compact_timeout_message(tmp_path):
    fake = _write_fake_cli(
        tmp_path / "agy_timeout",
        """
import json, sys, time
for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"protocolVersion": 1}}), flush=True)
    elif method == "session/resume":
        if msg["params"]["sessionId"] == "crash":
            sys.stderr.write("unexpected rpc crash\\n")
            sys.stderr.flush()
            sys.exit(1)
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": "agy-1"}}), flush=True)
    elif method == "session/set_model":
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {}}), flush=True)
    elif method == "session/prompt":
        time.sleep(1.0)
""",
    )
    backend = GoogleBackend(cli_path=fake)
    auth = ProviderAuth("google", "cli")

    with pytest.raises(RuntimeError, match="unexpected rpc crash") as exc_info:
        async for _ in backend.run_turn(
            "hi",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gemini-4-carbon",
            effort=None,
            auth=auth,
            resume_session_id="crash",
        ):
            pass
    assert not isinstance(exc_info.value, SessionResumeError)

    with (
        patch("nocturnomath.backends.google.COMPACT_TIMEOUT_S", 0.05),
        pytest.raises(RuntimeError, match="timed out"),
    ):
        await backend.compact(
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gemini-4-carbon",
            effort=None,
            auth=auth,
            resume_session_id="agy-1",
        )


@pytest.mark.asyncio
async def test_terminate_process_drains_paused_stdout_after_sigkill():
    import signal

    from nocturnomath.backends.base import StderrCollector, terminate_process

    script = (
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "sys.stdout.write('READY\\n')\n"
        "sys.stdout.flush()\n"
        "sys.stdout.write('x' * 512 * 1024)\n"
        "sys.stdout.flush()\n"
        "time.sleep(30)\n"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stderr = StderrCollector(proc.stderr)
    try:
        assert await asyncio.wait_for(proc.stdout.readline(), timeout=5.0) == b"READY\n"
        await asyncio.sleep(0.1)
        assert proc.returncode is None
        await asyncio.wait_for(terminate_process(proc, stderr), timeout=5.0)
        assert proc.returncode == -signal.SIGKILL
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        await stderr.close()


@pytest.mark.asyncio
async def test_google_backend_surfaces_rejected_effort_in_fallback_config(tmp_path):
    log_path = tmp_path / "google_config_log.json"
    fake = _write_fake_cli(
        tmp_path / "agy",
        f"""
import json, os, sys
log_path = {str(log_path)!r}
methods = json.loads(open(log_path).read()) if os.path.exists(log_path) else []

for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    methods.append(method)
    open(log_path, "w").write(json.dumps(methods))
    if method == "initialize":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"protocolVersion": 1}}}}), flush=True)
    elif method == "session/new":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"sessionId": "agy-cfg"}}}}), flush=True)
    elif method == "session/set_model":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "error": {{"code": -32601, "message": "set_model unsupported"}}}}), flush=True)
    elif method == "session/set_config_option":
        if msg["params"]["configId"] == "model":
            print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{}}}}), flush=True)
        elif msg["params"]["configId"] == "thought_level":
            print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "error": {{"code": -32602, "message": "Unsupported thought_level: high"}}}}), flush=True)
""",
    )
    backend = GoogleBackend(cli_path=fake)
    auth = ProviderAuth("google", "cli")

    with pytest.raises(RuntimeError, match="Unsupported thought_level: high"):
        async for _ in backend.run_turn(
            "hi",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gemini-4-carbon",
            effort="high",
            auth=auth,
        ):
            pass

    seen_methods = json.loads(log_path.read_text())
    assert "session/set_model" in seen_methods
    assert seen_methods.count("session/set_config_option") == 2
    assert "session/prompt" not in seen_methods
