import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from nocturnomath.auth import ProviderAuth
from nocturnomath.backends import (
    AnthropicBackend,
    GoogleBackend,
    MessageUsage,
    OpenAIBackend,
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

        # In CLI mode, user-configured MCP servers from CODEX_HOME/config.toml are disabled.
        cli_models = await backend.discover_models(ProviderAuth("openai", "cli"))
        assert len(cli_models) == 1

    logs = json.loads(log_path.read_text())
    assert logs[0]["codex_home"] != str(real_home)
    assert logs[1]["cwd"] != str(tmp_path)
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
