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

_ENV_VARS_TO_CLEAR = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "CODEX_API_KEY",
    "CODEX_ACCESS_TOKEN",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_BASE_URL",
    "GOOGLE_GENAI_USE_VERTEXAI",
)


@pytest.fixture(autouse=True)
def clean_provider_env(monkeypatch):
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)


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
async def test_openai_backend_discovery_turn_compact_and_cli_isolation(
    tmp_path, monkeypatch
):
    real_home = tmp_path / "real_codex_home"
    real_home.mkdir()
    (real_home / "config.toml").write_text(
        '[mcp_servers.user_extra]\ncommand = "echo"\n', encoding="utf-8"
    )
    (real_home / "auth.json").write_text('{"keep":"untouched"}\n', encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(real_home))
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "oauth-access-token")

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
    elif method == "account/read":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"requiresOpenaiAuth": True, "account": {{"type": "chatgpt"}}}}}}), flush=True)
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
    cli_auth = ProviderAuth("openai", "cli")
    with patch(
        "nocturnomath.backends.openai.codex_cli_candidates", return_value=[fake]
    ):
        models = await backend.discover_models(cli_auth)
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
                auth=cli_auth,
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
            auth=cli_auth,
            resume_session_id="thr-1",
        )
        assert compact.session_id == "thr-1"
        assert compact.context_usage["used_tokens"] == 300

    logs = json.loads(log_path.read_text())
    assert logs[0]["codex_home"] == str(real_home)
    assert logs[1]["cwd"] != str(tmp_path)
    assert "developerInstructions" not in logs[1]["thread_params"]
    assert "baseInstructions" in logs[1]["thread_params"]
    assert "NOCTURNOMATH_MCP_TOKEN" not in json.dumps(logs[1]["argv"])
    assert 'default_tools_approval_mode="approve"' in " ".join(logs[1]["argv"])
    assert "mcp_servers.node_repl.enabled=false" not in logs[1]["argv"]
    assert "mcp_servers.user_extra.enabled=false" in logs[0]["argv"]
    assert "mcp_servers.user_extra.enabled=false" in logs[1]["argv"]
    assert (real_home / "auth.json").read_text() == '{"keep":"untouched"}\n'


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
    auth = ProviderAuth("google", "cli")

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
    assert "--include-default-customizations=false" in logs[1]["argv"]
    assert logs[1]["set_model"] == "gemini-4-carbon-high"
    assert logs[1]["perms"] == ["deny", "allow"]
    assert logs[1]["cwd"] != str(tmp_path)


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
    cli_auth = ProviderAuth("google", "cli")
    with patch("nocturnomath.backends.google.agy_cli_candidates", return_value=[fake]):
        models = await backend.discover_models(cli_auth)
        assert [m.is_default for m in models] == [False, True]

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
    assert logs[2]["perm_outcome"] == {"outcome": "cancelled"}
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
        sub_models = await backend.discover_models(
            ProviderAuth.interactive("anthropic", "subscription", "oauth-tok-test")
        )
        assert [m.id for m in sub_models] == ["claude-sonnet-5-5"]

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


def test_missing_thread_and_session_helpers_ignore_unrelated_does_not_exist():
    from nocturnomath.backends.google import _is_missing_session_error
    from nocturnomath.backends.openai import _is_missing_thread_error

    assert not _is_missing_thread_error("Working directory does not exist")
    assert not _is_missing_thread_error("File /tmp/foo.py does not exist")
    assert _is_missing_thread_error("Thread thr-123 does not exist")
    assert _is_missing_thread_error("No rollout found for id")
    assert _is_missing_thread_error("Thread not found")

    assert not _is_missing_session_error("Working directory does not exist")
    assert not _is_missing_session_error(
        "No session cookie provided? Wait no directory"
    )
    assert _is_missing_session_error("Session agy-1 does not exist")
    assert _is_missing_session_error("Session not found")
    assert _is_missing_session_error("No session found")


@pytest.mark.asyncio
async def test_anthropic_backend_enforces_mcp_tool_timeout_floor(tmp_path, monkeypatch):
    seen_timeouts: list[str] = []
    fake = _write_fake_cli(
        tmp_path / "claude",
        f"""
import json, os, sys
req = json.loads(sys.stdin.readline())
print(json.dumps({{
    "type": "control_response",
    "response": {{"request_id": req["request_id"], "subtype": "success", "response": {{}}}},
}}), flush=True)
with open({str(tmp_path / "env.log")!r}, "a") as fh:
    fh.write(os.environ.get("MCP_TOOL_TIMEOUT", "") + "\\n")
print(json.dumps({{"type": "result", "subtype": "success"}}), flush=True)
""",
    )
    backend = AnthropicBackend(cli_path=fake)
    auth = ProviderAuth("anthropic", "cli")

    monkeypatch.setenv("MCP_TOOL_TIMEOUT", "1000")
    async for _ in backend.run_turn(
        "hi",
        system_prompt="Sys",
        tools=[],
        workspace_path=tmp_path,
        model="claude-sonnet-5",
        effort=None,
        auth=auth,
        tool_timeout_s=660,
    ):
        pass

    monkeypatch.setenv("MCP_TOOL_TIMEOUT", "900000")
    async for _ in backend.run_turn(
        "hi",
        system_prompt="Sys",
        tools=[],
        workspace_path=tmp_path,
        model="claude-sonnet-5",
        effort=None,
        auth=auth,
        tool_timeout_s=660,
    ):
        pass

    seen_timeouts = (tmp_path / "env.log").read_text().splitlines()
    assert seen_timeouts == ["660000", "900000"]


@pytest.mark.asyncio
async def test_google_cli_mode_disables_builtin_tools_and_keeps_vertex_on_cli(
    tmp_path, monkeypatch
):
    log_path = tmp_path / "agy_cli_env_log.json"
    fake = _write_fake_cli(
        tmp_path / "agy",
        f"""
import json, os, sys
log_path = {str(log_path)!r}
entry = {{"argv": sys.argv[1:], "methods": [], "meta": None}}
for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    entry["methods"].append(method)
    if method in ("session/new", "session/resume"):
        entry["meta"] = (msg.get("params") or {{}}).get("_meta")
        open(log_path, "w").write(json.dumps(entry))
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"sessionId": "agy-env-1", "models": {{"availableModels": [{{"modelId": "gemini-4-carbon-ultra", "name": "Gemini 4 Carbon (Ultra)"}}]}}}}}}), flush=True)
    elif method in ("initialize", "session/set_model"):
        open(log_path, "w").write(json.dumps(entry))
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{}}}}), flush=True)
    elif method == "session/prompt":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"stopReason": "end_turn"}}}}), flush=True)
""",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "other-provider-secret")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    backend = GoogleBackend(cli_path=fake)
    cli_auth = ProviderAuth("google", "cli")
    assert cli_auth.direct_api_key() is None
    assert "OPENAI_API_KEY" not in cli_auth.process_env()

    with patch("nocturnomath.backends.google.agy_cli_candidates", return_value=[fake]):
        models = await backend.discover_models(cli_auth)
        assert len(models) == 1
        assert models[0].efforts == ["ultra"]
        async for _ in backend.run_turn(
            "hi",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gemini-4-carbon",
            effort="ultra",
            auth=cli_auth,
        ):
            pass

    entry = json.loads(log_path.read_text())
    assert "authenticate" not in entry["methods"]
    assert entry["meta"]["disableBuiltInTools"] is True
    assert "disableBuiltInTools" not in entry["meta"].get("antigravity", {})


@pytest.mark.asyncio
async def test_anthropic_filters_subagent_text_denies_unknown_tools_and_caps_efforts(
    tmp_path,
):
    from nocturnomath.backends.anthropic import _parse_models

    parsed = _parse_models(
        [
            {
                "value": "claude-next",
                "resolvedModel": "claude-next",
                "supportsEffort": True,
            }
        ]
    )
    assert "ultra" not in parsed[0].efforts

    log_path = tmp_path / "claude_perm.json"
    fake = _write_fake_cli(
        tmp_path / "claude",
        f"""
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("type") == "control_request":
        print(json.dumps({{"type": "control_response", "response": {{"subtype": "success", "request_id": msg["request_id"], "response": {{}}}}}}), flush=True)
    elif msg.get("type") == "user":
        print(json.dumps({{"type": "control_request", "request_id": "c1", "request": {{"subtype": "can_use_tool", "tool_name": "Bash", "input": {{}}}}}}), flush=True)
        r1 = json.loads(sys.stdin.readline())["response"]["response"]["behavior"]
        print(json.dumps({{"type": "control_request", "request_id": "c2", "request": {{"subtype": "can_use_tool", "tool_name": "mcp__explore__ping", "input": {{}}}}}}), flush=True)
        r2 = json.loads(sys.stdin.readline())["response"]["response"]["behavior"]
        open({str(log_path)!r}, "w").write(json.dumps([r1, r2]))
        print(json.dumps({{"type": "assistant", "session_id": "ant-sub", "parent_tool_use_id": "sub-1", "message": {{"content": [{{"type": "text", "text": "subagent internal"}}]}} }}), flush=True)
        print(json.dumps({{"type": "assistant", "session_id": "ant-sub", "parent_tool_use_id": None, "message": {{"content": [{{"type": "text", "text": "main reply"}}]}} }}), flush=True)
        print(json.dumps({{"type": "result", "subtype": "success", "is_error": False, "session_id": "ant-sub"}}), flush=True)
""",
    )

    @tool("ping", "Ping.", {})
    async def ping(_args):
        return {"content": [{"type": "text", "text": "ok"}]}

    events = [
        e
        async for e in AnthropicBackend(cli_path=fake).run_turn(
            "hi",
            system_prompt="Sys",
            tools=[ping],
            workspace_path=tmp_path,
            model="claude-next",
            effort="high",
            auth=ProviderAuth("anthropic", "cli"),
        )
    ]
    assert json.loads(log_path.read_text()) == ["deny", "allow"]
    assert [e.text for e in events if isinstance(e, TextBlockDone)] == ["main reply"]


def test_openai_toml_mcp_override_handles_non_bmp_unicode_and_enables_explore():
    import tomllib

    from nocturnomath.backends.openai import _toml_mcp_override

    override = _toml_mcp_override(
        {
            "command": "/opt/🚀/python",
            "args": ["-m", "nocturnomath.mcp_bridge", "/tmp/🧪/bridge.json"],
            "env": {"FOO": "bar\nbaz"},
        },
        tool_timeout_s=660,
    )
    parsed = tomllib.loads(override)["mcp_servers"]["explore"]
    assert parsed["enabled"] is True
    assert parsed["command"] == "/opt/🚀/python"
    assert parsed["args"][-1] == "/tmp/🧪/bridge.json"
    assert parsed["env"] == {"FOO": "bar\nbaz"}


@pytest.mark.asyncio
async def test_compact_discards_buffered_precompaction_usage_and_handles_agy_cancel(
    tmp_path,
):
    fake_codex = _write_fake_cli(
        tmp_path / "codex_compact_buf",
        """
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {}}), flush=True)
    elif method == "thread/resume":
        print(json.dumps({"jsonrpc": "2.0", "method": "thread/tokenUsage/updated", "params": {"turnId": "old", "tokenUsage": {"modelContextWindow": 200000, "last": {"totalTokens": 190000}}}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"thread": {"id": "thr-1"}}}), flush=True)
    elif method == "thread/compact/start":
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "method": "thread/compacted", "params": {}}), flush=True)
""",
    )
    compact_codex = await OpenAIBackend(cli_path=fake_codex).compact(
        system_prompt="Sys",
        tools=[],
        workspace_path=tmp_path,
        model="gpt-5.6-sol",
        effort=None,
        auth=ProviderAuth("openai", "cli"),
        resume_session_id="thr-1",
    )
    assert compact_codex.context_usage is None

    fake_agy = _write_fake_cli(
        tmp_path / "agy_compact_buf",
        """
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"protocolVersion": 1}}), flush=True)
    elif method == "session/resume":
        sid = msg["params"]["sessionId"]
        print(json.dumps({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": sid, "update": {"sessionUpdate": "usage_update", "used": 180000, "size": 1000000}}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": sid}}), flush=True)
    elif method == "session/set_model":
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {}}), flush=True)
    elif method == "session/prompt":
        prompt_text = msg["params"]["prompt"][0]["text"]
        if msg["params"]["sessionId"] == "agy-cancel":
            print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "cancelled"}}), flush=True)
            continue
        assert prompt_text == "/compact"
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn", "usage": {"inputTokens": 400}}}), flush=True)
""",
    )
    agy_backend = GoogleBackend(cli_path=fake_agy)
    compact_agy = await agy_backend.compact(
        system_prompt="Sys",
        tools=[],
        workspace_path=tmp_path,
        model="gemini-4-carbon",
        effort="high",
        auth=ProviderAuth("google", "cli"),
        resume_session_id="agy-1",
    )
    assert compact_agy.context_usage == {
        "used_tokens": 400,
        "window_tokens": None,
        "model": "gemini-4-carbon",
    }
    with pytest.raises(asyncio.CancelledError):
        async for _ in agy_backend.run_turn(
            "hi",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gemini-4-carbon",
            effort="high",
            auth=ProviderAuth("google", "cli"),
            resume_session_id="agy-cancel",
        ):
            pass


@pytest.mark.asyncio
async def test_terminate_process_closes_transport_for_already_exited_process_with_stdout():
    from nocturnomath.backends.base import StderrCollector, terminate_process

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import sys; sys.stdout.write('READY\\n' + 'x' * 160 * 1024); sys.stdout.flush()",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stderr = StderrCollector(proc.stderr)
    assert await asyncio.wait_for(proc.stdout.readline(), timeout=5.0) == b"READY\n"
    for _ in range(50):
        if proc.returncode is not None:
            break
        await asyncio.sleep(0.02)
    assert proc.returncode == 0
    await terminate_process(proc, stderr)
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        assert transport.is_closing()


def test_openai_disabled_mcp_overrides_skips_undefined_node_repl(tmp_path):
    from nocturnomath.backends.openai import _disabled_mcp_overrides

    empty_home = tmp_path / "empty"
    empty_home.mkdir()
    assert _disabled_mcp_overrides(empty_home) == []

    configured_home = tmp_path / "configured"
    configured_home.mkdir()
    (configured_home / "config.toml").write_text(
        '[mcp_servers.explore]\ncommand = "keep"\n'
        '[mcp_servers.node_repl]\ncommand = "node"\n'
        '[mcp_servers.other]\nurl = "https://example.test"\n',
        encoding="utf-8",
    )
    assert _disabled_mcp_overrides(configured_home) == [
        "-c",
        "mcp_servers.node_repl.enabled=false",
        "-c",
        "mcp_servers.other.enabled=false",
    ]


@pytest.mark.asyncio
async def test_anthropic_falsey_toggle_env_does_not_treat_cli_as_authenticated(
    tmp_path, monkeypatch
):
    from nocturnomath.backends.anthropic import AnthropicBackend

    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "0")
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "false")
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "off")

    fake_claude = _write_fake_cli(
        tmp_path / "claude_falsey_toggle",
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
    backend = AnthropicBackend()
    with patch(
        "nocturnomath.backends.anthropic.claude_cli_candidates",
        return_value=[fake_claude],
    ):
        assert await backend.discover_models(ProviderAuth("anthropic", "cli")) == []
        monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
        models = await backend.discover_models(ProviderAuth("anthropic", "cli"))
        assert [model.id for model in models] == ["claude-sonnet-5-5"]


@pytest.mark.asyncio
async def test_google_run_turn_clears_buffered_resume_usage_update(tmp_path):
    fake_agy = _write_fake_cli(
        tmp_path / "agy_buffered_turn",
        """
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"protocolVersion": 1}}), flush=True)
    elif method == "session/resume":
        sid = msg["params"]["sessionId"]
        print(json.dumps({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": sid, "update": {"sessionUpdate": "usage_update", "used": 180000, "size": 1000000}}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": sid}}), flush=True)
    elif method == "session/set_model":
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {}}), flush=True)
    elif method == "session/prompt":
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn", "usage": {"inputTokens": 450}}}), flush=True)
""",
    )
    backend = GoogleBackend(cli_path=fake_agy)
    events = [
        event
        async for event in backend.run_turn(
            "hi",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gemini-4-carbon",
            effort="high",
            auth=ProviderAuth("google", "cli"),
            resume_session_id="agy-resume",
        )
    ]
    result = next(event for event in events if isinstance(event, TurnResult))
    assert result.context_usage == {
        "used_tokens": 450,
        "window_tokens": None,
        "model": "gemini-4-carbon",
    }


@pytest.mark.asyncio
async def test_google_drains_post_rpc_setup_updates(tmp_path):
    prompt_log = tmp_path / "prompts.json"
    fake_agy = _write_fake_cli(
        tmp_path / "agy_post_rpc",
        f"""
import json, os, sys
log_file = {str(prompt_log)!r}
seen = []
for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"protocolVersion": 1}}}}), flush=True)
    elif method in ("session/new", "session/resume"):
        sid = (msg.get("params") or {{}}).get("sessionId") or "agy-s1"
        cmds = [] if sid == "no-compact" else [{{"name": "compact"}}]
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"sessionId": sid, "models": {{"availableModels": [{{"modelId": "gemini-4-carbon-low", "name": "Gemini 4 Carbon (Low)"}}]}}}}}}), flush=True)
        # Real agy writes available_commands_update and usage_update AFTER the RPC response.
        print(json.dumps({{"jsonrpc": "2.0", "method": "session/update", "params": {{"sessionId": sid, "update": {{"sessionUpdate": "available_commands_update", "availableCommands": cmds}}}}}}), flush=True)
        print(json.dumps({{"jsonrpc": "2.0", "method": "session/update", "params": {{"sessionId": sid, "update": {{"sessionUpdate": "usage_update", "used": 180000, "size": 1000000}}}}}}), flush=True)
    elif method == "session/set_model":
        sid = msg["params"]["sessionId"]
        cmds = [] if sid == "no-compact" else [{{"name": "compact"}}]
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{}}}}), flush=True)
        print(json.dumps({{"jsonrpc": "2.0", "method": "session/update", "params": {{"sessionId": sid, "update": {{"sessionUpdate": "available_commands_update", "availableCommands": cmds}}}}}}), flush=True)
        print(json.dumps({{"jsonrpc": "2.0", "method": "session/update", "params": {{"sessionId": sid, "update": {{"sessionUpdate": "usage_update", "used": 180000, "size": 1000000}}}}}}), flush=True)
    elif method == "session/prompt":
        seen.append(msg["params"]["prompt"][0]["text"])
        open(log_file, "w").write(json.dumps(seen))
        if msg["params"]["sessionId"] == "unconfirmed":
            print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"stopReason": "end_turn"}}}}), flush=True)
        else:
            print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"stopReason": "end_turn", "usage": {{"inputTokens": 320}}}}}}), flush=True)
""",
    )
    auth = ProviderAuth("google", "cli")
    backend = GoogleBackend(cli_path=fake_agy)

    with patch(
        "nocturnomath.backends.google.agy_cli_candidates", return_value=[fake_agy]
    ):
        models = await backend.discover_models(auth)
        assert [m.id for m in models] == ["gemini-4-carbon"]

    with pytest.raises(RuntimeError, match="does not support /compact"):
        await backend.compact(
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gemini-4-carbon",
            effort="low",
            auth=auth,
            resume_session_id="no-compact",
        )
    assert not prompt_log.exists()

    with pytest.raises(RuntimeError, match="did not confirm that compaction completed"):
        await backend.compact(
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gemini-4-carbon",
            effort="low",
            auth=auth,
            resume_session_id="unconfirmed",
        )

    events = [
        event
        async for event in backend.run_turn(
            "hello",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gemini-4-carbon",
            effort="low",
            auth=auth,
            resume_session_id="agy-s1",
        )
    ]
    turn_res = next(event for event in events if isinstance(event, TurnResult))
    assert turn_res.context_usage == {
        "used_tokens": 320,
        "window_tokens": None,
        "model": "gemini-4-carbon",
    }
