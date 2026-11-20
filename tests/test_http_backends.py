import asyncio
import base64
import json
from collections.abc import Callable
from contextlib import suppress
from unittest.mock import patch

import httpx
import pytest
from conftest import FakeEnvironment, FakeKernel

from nocturnomath.auth import ProviderAuth
from nocturnomath.backends import (
    AnthropicBackend,
    GoogleBackend,
    MessageUsage,
    OpenAIBackend,
    SessionResumeError,
    SessionStarted,
    TextBlockDone,
    TextDelta,
    TurnResult,
    TurnStarted,
)
from nocturnomath.runtime import Runtime
from nocturnomath.session import ExplorationSession
from nocturnomath.tools import tool

_TINY_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfakepng").decode("ascii")

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
def clean_env_and_hide_clis(monkeypatch):
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("nocturnomath.backends.anthropic.claude_cli_candidates", list)
    monkeypatch.setattr("nocturnomath.backends.openai.codex_cli_candidates", list)
    monkeypatch.setattr("nocturnomath.backends.google.agy_cli_candidates", list)


def _sse_body(events: list[tuple[str | None, dict | str]]) -> bytes:
    chunks: list[str] = []
    for event_name, payload in events:
        if event_name is not None:
            chunks.append(f"event: {event_name}\n")
        raw = payload if isinstance(payload, str) else json.dumps(payload)
        for line in raw.splitlines() or [""]:
            chunks.append(f"data: {line}\n")
        chunks.append("\n")
    return "".join(chunks).encode("utf-8")


def _client_factory(
    handler: Callable[[httpx.Request], object],
) -> Callable[..., httpx.AsyncClient]:
    transport = httpx.MockTransport(handler)
    return lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs)


@pytest.mark.asyncio
async def test_anthropic_http_discovery_turn_compact_cancel_and_cli_switch(tmp_path):
    requests: list[tuple[str, str, dict | None, dict[str, str]]] = []
    turn_round = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn_round
        body = json.loads(request.content.decode("utf-8")) if request.content else None
        requests.append((request.method, str(request.url), body, dict(request.headers)))
        assert request.headers["x-api-key"] == "sk-ant-http"

        if request.url.path == "/v1/models":
            after = request.url.params.get("after_id")
            if after is None:
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "id": "claude-opus-4-6",
                                "max_input_tokens": 200000,
                                "max_tokens": 32000,
                                "capabilities": {
                                    "thinking": {
                                        "types": {"adaptive": {"supported": True}}
                                    },
                                    "effort": {
                                        "supported": True,
                                        "low": {"supported": True},
                                        "medium": {"supported": True},
                                        "high": {"supported": True},
                                        "max": {"supported": True},
                                    },
                                },
                            },
                            {"id": "non-claude-model"},
                        ],
                        "has_more": True,
                        "last_id": "claude-opus-4-6",
                    },
                )
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "claude-sonnet-4-6",
                            "max_input_tokens": 1000000,
                            "max_tokens": 16384,
                        }
                    ],
                    "has_more": False,
                },
            )

        if request.url.path == "/v1/messages" and body and not body.get("stream"):
            # Compaction call: ensure plot image was stripped.
            assert '"type": "image"' not in json.dumps(body["messages"])
            assert _TINY_PNG_B64 not in json.dumps(body["messages"])
            return httpx.Response(
                200,
                json={
                    "id": "msg_cmp",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Compacted findings summary."}
                    ],
                    "usage": {"input_tokens": 50, "output_tokens": 80},
                },
            )

        turn_round += 1
        if turn_round == 1:
            # Reject adaptive thinking/effort once to exercise fallback retry.
            return httpx.Response(
                400,
                json={"error": {"message": "output_config is not supported here"}},
            )
        if turn_round == 2:
            sse = _sse_body(
                [
                    (
                        "message_start",
                        {
                            "type": "message_start",
                            "message": {
                                "id": "msg_1",
                                "model": "claude-sonnet-4-6",
                                "usage": {
                                    "input_tokens": 100,
                                    "cache_read_input_tokens": 20,
                                },
                            },
                        },
                    ),
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "thinking"},
                        },
                    ),
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "thinking_delta",
                                "thinking": "Plan probe.",
                            },
                        },
                    ),
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "signature_delta",
                                "signature": "sig_abc",
                            },
                        },
                    ),
                    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 1,
                            "content_block": {"type": "text"},
                        },
                    ),
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 1,
                            "delta": {"type": "text_delta", "text": "Running plot."},
                        },
                    ),
                    ("content_block_stop", {"type": "content_block_stop", "index": 1}),
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 2,
                            "content_block": {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "plot",
                            },
                        },
                    ),
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 2,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": '{"n": 3}',
                            },
                        },
                    ),
                    ("content_block_stop", {"type": "content_block_stop", "index": 2}),
                    (
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": "tool_use"},
                            "usage": {"output_tokens": 45},
                        },
                    ),
                    ("message_stop", {"type": "message_stop"}),
                ]
            )
            return httpx.Response(200, content=sse)

        if turn_round == 3:
            # Verify thinking + signature + base64 tool_result image are preserved cleanly.
            assistant_msg = body["messages"][1]
            assert assistant_msg["content"][0] == {
                "type": "thinking",
                "thinking": "Plan probe.",
                "signature": "sig_abc",
            }
            tool_res_msg = body["messages"][2]
            assert tool_res_msg["content"][0]["content"][1] == {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _TINY_PNG_B64,
                },
            }
            sse = _sse_body(
                [
                    (
                        "message_start",
                        {
                            "type": "message_start",
                            "message": {
                                "id": "msg_2",
                                "model": "claude-sonnet-4-6",
                                "usage": {
                                    "input_tokens": 350,
                                    "cache_creation_input_tokens": 10,
                                },
                            },
                        },
                    ),
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text"},
                        },
                    ),
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": "Plot confirmed."},
                        },
                    ),
                    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                    (
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": "end_turn"},
                            "usage": {"output_tokens": 20},
                        },
                    ),
                    ("message_stop", {"type": "message_stop"}),
                ]
            )
            return httpx.Response(200, content=sse)

        return httpx.Response(200, content=_sse_body([]))

    backend = AnthropicBackend(http_client_factory=_client_factory(handler))
    auth = ProviderAuth.interactive("anthropic", "api_key", "sk-ant-http")

    specs = await backend.discover_models(auth)
    assert [s.id for s in specs] == ["claude-opus-4-6", "claude-sonnet-4-6"]
    assert specs[0].efforts == ["low", "medium", "high", "max"]
    assert specs[0].is_default is False
    assert specs[1].efforts == ["low", "medium", "high"]
    assert specs[1].is_default is True

    @tool("plot", "Make a plot.", {"n": int})
    async def plot_tool(args):
        assert args == {"n": 3}
        return {
            "content": [
                {"type": "text", "text": "plotted"},
                {"type": "image", "data": _TINY_PNG_B64, "mimeType": "image/png"},
            ]
        }

    events = [
        event
        async for event in backend.run_turn(
            "run a plot",
            system_prompt="System prompt",
            tools=[plot_tool],
            workspace_path=tmp_path,
            model="claude-sonnet-4-6",
            effort="high",
            auth=auth,
        )
    ]
    started = next(e for e in events if isinstance(e, SessionStarted))
    assert started.session_id.startswith("http:anthropic:")
    assert any(isinstance(e, TurnStarted) for e in events)
    assert [e.text for e in events if isinstance(e, TextDelta)] == [
        "Running plot.",
        "Plot confirmed.",
    ]
    assert [e.text for e in events if isinstance(e, TextBlockDone)] == [
        "Running plot.",
        "Plot confirmed.",
    ]
    usages = [e for e in events if isinstance(e, MessageUsage)]
    assert len(usages) == 2
    res = next(e for e in events if isinstance(e, TurnResult))
    assert res.num_turns == 2
    assert res.context_usage == {
        "used_tokens": 360,
        "window_tokens": 1000000,
        "model": "claude-sonnet-4-6",
    }

    compact = await backend.compact(
        system_prompt="System prompt",
        tools=[plot_tool],
        workspace_path=tmp_path,
        model="claude-sonnet-4-6",
        effort="high",
        auth=auth,
        resume_session_id=started.session_id,
    )
    assert compact.session_id == started.session_id
    assert compact.record_reloaded is False
    assert compact.context_usage == {
        "used_tokens": 100,
        "window_tokens": 1000000,
        "model": "claude-sonnet-4-6",
    }

    # Missing HTTP session raises SessionResumeError.
    with pytest.raises(SessionResumeError):
        async for _ in backend.run_turn(
            "hi",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="claude-sonnet-4-6",
            effort=None,
            auth=auth,
            resume_session_id="http:anthropic:" + ("0" * 32),
        ):
            pass

    # Switching from HTTP back to CLI mode triggers SessionResumeError rather than CLI crash.
    with pytest.raises(SessionResumeError):
        async for _ in backend.run_turn(
            "hi",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="claude-sonnet-4-6",
            effort=None,
            auth=ProviderAuth("anthropic", "cli"),
            resume_session_id=started.session_id,
        ):
            pass

    # HTTP 401 during discovery returns [].
    bad_backend = AnthropicBackend(
        http_client_factory=_client_factory(
            lambda _req: httpx.Response(
                401, json={"error": {"message": "unauthorized"}}
            )
        )
    )
    assert await bad_backend.discover_models(auth) == []


@pytest.mark.asyncio
async def test_anthropic_http_cancellation_saves_valid_tool_pairs(tmp_path):
    tool_entered = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/messages":
            sse = _sse_body(
                [
                    (
                        "message_start",
                        {
                            "type": "message_start",
                            "message": {
                                "id": "m1",
                                "model": "claude-sonnet-4-6",
                                "usage": {"input_tokens": 20},
                            },
                        },
                    ),
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {
                                "type": "tool_use",
                                "id": "toolu_c",
                                "name": "slow",
                            },
                        },
                    ),
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "input_json_delta", "partial_json": "{}"},
                        },
                    ),
                    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                    ("message_stop", {"type": "message_stop"}),
                ]
            )
            return httpx.Response(200, content=sse)
        return httpx.Response(404)

    @tool("slow", "Wait.", {})
    async def slow_tool(_args):
        tool_entered.set()
        await asyncio.Event().wait()
        return {"content": [{"type": "text", "text": "done"}]}

    backend = AnthropicBackend(http_client_factory=_client_factory(handler))
    auth = ProviderAuth.interactive("anthropic", "api_key", "sk-ant-cancel")
    seen_session: list[str] = []

    async def runner():
        async for ev in backend.run_turn(
            "cancel during tool",
            system_prompt="Sys",
            tools=[slow_tool],
            workspace_path=tmp_path,
            model="claude-sonnet-4-6",
            effort=None,
            auth=auth,
        ):
            if isinstance(ev, SessionStarted):
                seen_session.append(ev.session_id)

    task = asyncio.create_task(runner())
    await tool_entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    from nocturnomath.backends.http import HttpThreadStore

    saved = HttpThreadStore(tmp_path, "anthropic").load(seen_session[0])
    msgs = saved["messages"]
    assert msgs[-2]["role"] == "user"
    assert msgs[-2]["content"][0]["tool_use_id"] == "toolu_c"
    assert msgs[-2]["content"][0]["is_error"] is True
    assert msgs[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_openai_http_discovery_responses_turn_compact_cancel_and_cli_switch(
    tmp_path,
):
    seen_bodies: list[dict] = []
    round_num = 0
    tool_entered = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal round_num
        assert request.headers["authorization"] == "Bearer sk-openai-http"
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "gpt-4o", "created": 100},
                        {"id": "gpt-5.4", "created": 200},
                        {"id": "text-embedding-3-large", "created": 300},
                        {"id": "gpt-4o-realtime-preview", "created": 250},
                        {"id": "gpt-5.2-tts", "created": 250},
                        {"id": "whisper-1", "created": 10},
                        {"id": "dall-e-3", "created": 10},
                    ]
                },
            )
        body = json.loads(request.content.decode("utf-8"))
        seen_bodies.append(body)
        if body.get("stream") is False:
            assert "input_image" not in json.dumps(body["input"])
            return httpx.Response(
                200,
                json={
                    "id": "resp_cmp",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "OpenAI compacted."}
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 30, "output_tokens": 55},
                },
            )

        round_num += 1
        if round_num == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unrecognized parameter: reasoning.encrypted_content"
                    }
                },
            )
        if round_num == 2:
            sse = _sse_body(
                [
                    (
                        "response.created",
                        {
                            "type": "response.created",
                            "response": {"id": "r1", "model": "gpt-5.4"},
                        },
                    ),
                    (
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "item": {
                                "type": "reasoning",
                                "id": "rs_1",
                                "summary": [],
                                "encrypted_content": "enc_tok_123",
                            },
                        },
                    ),
                    (
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "item_id": "msg_1",
                            "content_index": 0,
                            "delta": "Calling run.",
                        },
                    ),
                    (
                        "response.output_text.done",
                        {
                            "type": "response.output_text.done",
                            "item_id": "msg_1",
                            "content_index": 0,
                            "text": "Calling run.",
                        },
                    ),
                    (
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "item": {
                                "type": "message",
                                "id": "msg_1",
                                "role": "assistant",
                                "content": [
                                    {"type": "output_text", "text": "Calling run."}
                                ],
                            },
                        },
                    ),
                    (
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "item": {
                                "type": "function_call",
                                "call_id": "call_1",
                                "name": "plot",
                                "arguments": '{"k": 2}',
                            },
                        },
                    ),
                    (
                        "response.completed",
                        {
                            "type": "response.completed",
                            "response": {
                                "id": "r1",
                                "model": "gpt-5.4",
                                "usage": {"input_tokens": 110, "output_tokens": 30},
                            },
                        },
                    ),
                ]
            )
            return httpx.Response(200, content=sse)
        if round_num == 3:
            assert body["store"] is False
            assert any(
                item.get("type") == "reasoning"
                and item.get("encrypted_content") == "enc_tok_123"
                for item in body["input"]
            )
            fco = next(
                item
                for item in body["input"]
                if item.get("type") == "function_call_output"
            )
            assert fco["output"][1] == {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{_TINY_PNG_B64}",
                "detail": "auto",
            }
            sse = _sse_body(
                [
                    (
                        "response.created",
                        {
                            "type": "response.created",
                            "response": {"id": "r2", "model": "gpt-5.4"},
                        },
                    ),
                    (
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "item_id": "msg_2",
                            "content_index": 0,
                            "delta": "All good.",
                        },
                    ),
                    (
                        "response.output_text.done",
                        {
                            "type": "response.output_text.done",
                            "item_id": "msg_2",
                            "content_index": 0,
                            "text": "All good.",
                        },
                    ),
                    (
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "item": {
                                "type": "message",
                                "id": "msg_2",
                                "role": "assistant",
                                "content": [
                                    {"type": "output_text", "text": "All good."}
                                ],
                            },
                        },
                    ),
                    (
                        "response.completed",
                        {
                            "type": "response.completed",
                            "response": {
                                "id": "r2",
                                "model": "gpt-5.4",
                                "usage": {"input_tokens": 220, "output_tokens": 15},
                            },
                        },
                    ),
                ]
            )
            return httpx.Response(200, content=sse)

        # Round 4 is used by the cancellation test below.
        sse = _sse_body(
            [
                (
                    "response.created",
                    {
                        "type": "response.created",
                        "response": {"id": "r3", "model": "gpt-5.4"},
                    },
                ),
                (
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "function_call",
                            "call_id": "call_wait",
                            "name": "slow",
                            "arguments": "{}",
                        },
                    },
                ),
                (
                    "response.completed",
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "r3",
                            "model": "gpt-5.4",
                            "usage": {"input_tokens": 20, "output_tokens": 5},
                        },
                    },
                ),
            ]
        )
        return httpx.Response(200, content=sse)

    backend = OpenAIBackend(http_client_factory=_client_factory(handler))
    auth = ProviderAuth.interactive("openai", "api_key", "sk-openai-http")
    specs = await backend.discover_models(auth)
    assert [s.id for s in specs] == ["gpt-5.4", "gpt-4o"]
    assert specs[0].efforts == ["low", "medium", "high"]
    assert specs[0].is_default is True
    assert specs[1].efforts == []

    @tool("plot", "Plot.", {"k": int})
    async def plot_tool(args):
        assert args == {"k": 2}
        return {
            "content": [
                {"type": "text", "text": "plot ready"},
                {"type": "image", "data": _TINY_PNG_B64, "mimeType": "image/png"},
            ]
        }

    events = [
        e
        async for e in backend.run_turn(
            "test openai",
            system_prompt="Sys",
            tools=[plot_tool],
            workspace_path=tmp_path,
            model="gpt-5.4",
            effort="high",
            auth=auth,
        )
    ]
    started = next(e for e in events if isinstance(e, SessionStarted))
    assert started.session_id.startswith("http:openai:")
    assert [e.text for e in events if isinstance(e, TextBlockDone)] == [
        "Calling run.",
        "All good.",
    ]
    res = next(e for e in events if isinstance(e, TurnResult))
    assert res.context_usage == {
        "used_tokens": 220,
        "window_tokens": None,
        "model": "gpt-5.4",
    }

    compact = await backend.compact(
        system_prompt="Sys",
        tools=[plot_tool],
        workspace_path=tmp_path,
        model="gpt-5.4",
        effort="high",
        auth=auth,
        resume_session_id=started.session_id,
    )
    assert compact.record_reloaded is False
    assert compact.context_usage == {
        "used_tokens": 75,
        "window_tokens": None,
        "model": "gpt-5.4",
    }

    with pytest.raises(SessionResumeError):
        async for _ in backend.run_turn(
            "hi",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gpt-5.4",
            effort=None,
            auth=ProviderAuth("openai", "cli"),
            resume_session_id=started.session_id,
        ):
            pass

    @tool("slow", "Wait.", {})
    async def slow_tool(_args):
        tool_entered.set()
        await asyncio.Event().wait()
        return {"content": [{"type": "text", "text": "done"}]}

    cancel_sid: list[str] = []

    async def run_cancel():
        async for ev in backend.run_turn(
            "cancel openai",
            system_prompt="Sys",
            tools=[slow_tool],
            workspace_path=tmp_path,
            model="gpt-5.4",
            effort=None,
            auth=auth,
        ):
            if isinstance(ev, SessionStarted):
                cancel_sid.append(ev.session_id)

    task = asyncio.create_task(run_cancel())
    await tool_entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    from nocturnomath.backends.http import HttpThreadStore

    saved_items = HttpThreadStore(tmp_path, "openai").load(cancel_sid[0])["messages"]
    assert saved_items[-2]["type"] == "function_call_output"
    assert saved_items[-2]["call_id"] == "call_wait"
    assert saved_items[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_google_http_discovery_streaming_thought_signature_compact_and_cancel(
    tmp_path,
):
    seen_requests: list[dict] = []
    stream_call = 0
    tool_entered = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal stream_call
        # Never expose API key in query parameters!
        assert "key" not in request.url.params
        assert request.headers["x-goog-api-key"] == "AIza-http-key"

        if request.url.path == "/v1beta/models":
            token = request.url.params.get("pageToken")
            if token is None:
                return httpx.Response(
                    200,
                    json={
                        "models": [
                            {
                                "name": "models/gemini-2.5-pro",
                                "displayName": "Gemini 2.5 Pro",
                                "supportedGenerationMethods": ["generateContent"],
                                "inputTokenLimit": 1048576,
                            },
                            {
                                "name": "models/text-embedding-004",
                                "supportedGenerationMethods": ["embedContent"],
                            },
                        ],
                        "nextPageToken": "page2",
                    },
                )
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/gemini-3-pro-preview",
                            "displayName": "Gemini 3 Pro",
                            "supportedGenerationMethods": ["generateContent"],
                            "inputTokenLimit": 2000000,
                        }
                    ]
                },
            )

        body = json.loads(request.content.decode("utf-8"))
        seen_requests.append({"path": request.url.path, "body": body})

        if request.url.path.endswith(":generateContent"):
            assert "inlineData" not in json.dumps(body["contents"])
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": "Gemini compacted summary."}],
                            }
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 40,
                        "candidatesTokenCount": 60,
                    },
                },
            )

        stream_call += 1
        if stream_call == 1:
            assert body["generationConfig"] == {
                "thinkingConfig": {"thinkingLevel": "HIGH"}
            }
            sse = _sse_body(
                [
                    (
                        None,
                        {
                            "candidates": [
                                {
                                    "content": {
                                        "role": "model",
                                        "parts": [
                                            {"text": "hidden thought", "thought": True},
                                            {"text": "Let's test."},
                                        ],
                                    }
                                }
                            ],
                            "usageMetadata": {
                                "promptTokenCount": 90,
                                "candidatesTokenCount": 12,
                            },
                        },
                    ),
                    (
                        None,
                        {
                            "candidates": [
                                {
                                    "content": {
                                        "role": "model",
                                        "parts": [
                                            {
                                                "functionCall": {
                                                    "name": "plot",
                                                    "args": {"x": 5},
                                                    "id": "call_g1",
                                                },
                                                "thoughtSignature": "tsig_opaque_999",
                                            }
                                        ],
                                    }
                                }
                            ],
                            "usageMetadata": {
                                "promptTokenCount": 90,
                                "candidatesTokenCount": 25,
                            },
                        },
                    ),
                ]
            )
            return httpx.Response(200, content=sse)

        if stream_call == 2:
            # Thought signature and nested inlineData under functionResponse must be intact.
            model_turn = body["contents"][1]
            assert model_turn["role"] == "model"
            assert model_turn["parts"][-1]["thoughtSignature"] == "tsig_opaque_999"
            func_turn = body["contents"][2]
            assert func_turn["role"] == "user"
            fr = func_turn["parts"][0]["functionResponse"]
            assert fr["id"] == "call_g1"
            assert fr["parts"][0]["inlineData"] == {
                "mimeType": "image/png",
                "data": _TINY_PNG_B64,
            }
            sse = _sse_body(
                [
                    (
                        None,
                        {
                            "candidates": [
                                {
                                    "content": {
                                        "role": "model",
                                        "parts": [{"text": "Verified plot."}],
                                    }
                                }
                            ],
                            "usageMetadata": {
                                "promptTokenCount": 310,
                                "candidatesTokenCount": 18,
                            },
                        },
                    )
                ]
            )
            return httpx.Response(200, content=sse)

        if stream_call == 3:
            # Verify Gemini 2.5 uses thinkingBudget instead of thinkingLevel.
            assert body["generationConfig"] == {
                "thinkingConfig": {"thinkingBudget": 1024}
            }
            sse = _sse_body(
                [
                    (
                        None,
                        {
                            "candidates": [
                                {
                                    "content": {
                                        "role": "model",
                                        "parts": [
                                            {
                                                "functionCall": {
                                                    "name": "slow",
                                                    "args": {},
                                                    "id": "call_wait",
                                                }
                                            }
                                        ],
                                    }
                                }
                            ],
                            "usageMetadata": {
                                "promptTokenCount": 50,
                                "candidatesTokenCount": 5,
                            },
                        },
                    )
                ]
            )
            return httpx.Response(200, content=sse)
        return httpx.Response(500)

    backend = GoogleBackend(http_client_factory=_client_factory(handler))
    auth = ProviderAuth.interactive("google", "api_key", "AIza-http-key")

    specs = await backend.discover_models(auth)
    assert [s.id for s in specs] == ["gemini-2.5-pro", "gemini-3-pro-preview"]
    assert [s.label for s in specs] == ["gemini-2.5-pro", "gemini-3-pro-preview"]
    assert specs[0].efforts == ["low", "medium", "high"]
    assert specs[0].is_default is True

    @tool("plot", "Plot.", {"x": int})
    async def plot_tool(args):
        assert args == {"x": 5}
        return {
            "content": [
                {"type": "text", "text": "plotted"},
                {"type": "image", "data": _TINY_PNG_B64, "mimeType": "image/png"},
            ]
        }

    events = [
        e
        async for e in backend.run_turn(
            "check gemini",
            system_prompt="Sys",
            tools=[plot_tool],
            workspace_path=tmp_path,
            model="gemini-3-pro-preview",
            effort="high",
            auth=auth,
        )
    ]
    started = next(e for e in events if isinstance(e, SessionStarted))
    assert started.session_id.startswith("http:google:")
    assert [e.text for e in events if isinstance(e, TextDelta)] == [
        "Let's test.",
        "Verified plot.",
    ]
    assert [e.text for e in events if isinstance(e, TextBlockDone)] == [
        "Let's test.",
        "Verified plot.",
    ]
    res = next(e for e in events if isinstance(e, TurnResult))
    assert res.context_usage == {
        "used_tokens": 310,
        "window_tokens": 2000000,
        "model": "gemini-3-pro-preview",
    }

    compact = await backend.compact(
        system_prompt="Sys",
        tools=[plot_tool],
        workspace_path=tmp_path,
        model="gemini-3-pro-preview",
        effort="high",
        auth=auth,
        resume_session_id=started.session_id,
    )
    assert compact.record_reloaded is False
    assert compact.context_usage == {
        "used_tokens": 80,
        "window_tokens": 2000000,
        "model": "gemini-3-pro-preview",
    }

    with pytest.raises(SessionResumeError):
        async for _ in backend.run_turn(
            "hi",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gemini-3-pro-preview",
            effort=None,
            auth=ProviderAuth("google", "cli"),
            resume_session_id=started.session_id,
        ):
            pass

    @tool("slow", "Wait.", {})
    async def slow_tool(_args):
        tool_entered.set()
        await asyncio.Event().wait()
        return {"content": [{"type": "text", "text": "done"}]}

    cancel_sid: list[str] = []

    async def run_cancel():
        async for ev in backend.run_turn(
            "cancel gemini",
            system_prompt="Sys",
            tools=[slow_tool],
            workspace_path=tmp_path,
            model="gemini-2.5-pro",
            effort="low",
            auth=auth,
        ):
            if isinstance(ev, SessionStarted):
                cancel_sid.append(ev.session_id)

    task = asyncio.create_task(run_cancel())
    await tool_entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    from nocturnomath.backends.http import HttpThreadStore

    saved_contents = HttpThreadStore(tmp_path, "google").load(cancel_sid[0])["messages"]
    assert saved_contents[-2]["role"] == "user"
    assert saved_contents[-2]["parts"][0]["functionResponse"]["id"] == "call_wait"
    assert saved_contents[-1]["role"] == "model"


@pytest.mark.asyncio
async def test_session_and_runtime_end_to_end_http_no_clis_compaction_and_resume(
    tmp_path, monkeypatch
):
    turn_calls: list[dict] = []
    compact_calls = 0
    step = 0

    class PlotFakeKernel(FakeKernel):
        def execute(self, code, timeout):
            self.executed_codes.append(code)
            return ("42\n", [base64.b64decode(_TINY_PNG_B64)], None)

    async def anthropic_handler(request: httpx.Request) -> httpx.Response:
        nonlocal step, compact_calls
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "claude-sonnet-4-6",
                            "max_input_tokens": 200000,
                            "max_tokens": 8192,
                        }
                    ],
                    "has_more": False,
                },
            )
        body = json.loads(request.content.decode("utf-8"))
        if not body.get("stream"):
            compact_calls += 1
            return httpx.Response(
                200,
                json={
                    "id": "msg_cmp",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Summary before compact."}],
                    "usage": {"output_tokens": 40},
                },
            )

        turn_calls.append(body)
        step += 1
        if step == 1:
            sse = _sse_body(
                [
                    (
                        "message_start",
                        {
                            "type": "message_start",
                            "message": {
                                "id": "m1",
                                "model": "claude-sonnet-4-6",
                                "usage": {"input_tokens": 100},
                            },
                        },
                    ),
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {
                                "type": "tool_use",
                                "id": "tu_run",
                                "name": "run",
                            },
                        },
                    ),
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(
                                    {"code": "print(6*7)", "expected": "prints 42"}
                                ),
                            },
                        },
                    ),
                    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                    (
                        "message_delta",
                        {"type": "message_delta", "usage": {"output_tokens": 15}},
                    ),
                    ("message_stop", {"type": "message_stop"}),
                ]
            )
            return httpx.Response(200, content=sse)
        if step == 2:
            # Verify base64 PNG plot was forwarded to Anthropic from FakeKernel.
            tool_res_msg = body["messages"][-1]
            assert any(
                isinstance(part, dict) and part.get("type") == "image"
                for part in tool_res_msg["content"][0]["content"]
            )
            sse = _sse_body(
                [
                    (
                        "message_start",
                        {
                            "type": "message_start",
                            "message": {
                                "id": "m2",
                                "model": "claude-sonnet-4-6",
                                "usage": {"input_tokens": 150},
                            },
                        },
                    ),
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {
                                "type": "tool_use",
                                "id": "tu_verd",
                                "name": "verdict",
                            },
                        },
                    ),
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps({"text": "Matched 42."}),
                            },
                        },
                    ),
                    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                    (
                        "message_delta",
                        {"type": "message_delta", "usage": {"output_tokens": 10}},
                    ),
                    ("message_stop", {"type": "message_stop"}),
                ]
            )
            return httpx.Response(200, content=sse)
        if step == 3:
            sse = _sse_body(
                [
                    (
                        "message_start",
                        {
                            "type": "message_start",
                            "message": {
                                "id": "m3",
                                "model": "claude-sonnet-4-6",
                                "usage": {"input_tokens": 180},
                            },
                        },
                    ),
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {
                                "type": "tool_use",
                                "id": "tu_ev",
                                "name": "evidence",
                            },
                        },
                    ),
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(
                                    {
                                        "text": "6*7 printed 42.",
                                        "sources": ["S001/P001/plot-1.png"],
                                    }
                                ),
                            },
                        },
                    ),
                    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                    (
                        "message_delta",
                        {"type": "message_delta", "usage": {"output_tokens": 10}},
                    ),
                    ("message_stop", {"type": "message_stop"}),
                ]
            )
            return httpx.Response(200, content=sse)
        if step == 4:
            sse = _sse_body(
                [
                    (
                        "message_start",
                        {
                            "type": "message_start",
                            "message": {
                                "id": "m4",
                                "model": "claude-sonnet-4-6",
                                "usage": {"input_tokens": 210},
                            },
                        },
                    ),
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text"},
                        },
                    ),
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": "Recorded E001."},
                        },
                    ),
                    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                    (
                        "message_delta",
                        {"type": "message_delta", "usage": {"output_tokens": 12}},
                    ),
                    ("message_stop", {"type": "message_stop"}),
                ]
            )
            return httpx.Response(200, content=sse)
        # Step 5+: after compaction or session resume.
        sse = _sse_body(
            [
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": "m5",
                            "model": "claude-sonnet-4-6",
                            "usage": {"input_tokens": 230},
                        },
                    },
                ),
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text"},
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "Continued cleanly."},
                    },
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                (
                    "message_delta",
                    {"type": "message_delta", "usage": {"output_tokens": 8}},
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
        )
        return httpx.Response(200, content=sse)

    def make_backends():
        return {
            "anthropic": AnthropicBackend(
                http_client_factory=_client_factory(anthropic_handler)
            ),
            "openai": OpenAIBackend(),
            "google": GoogleBackend(),
        }

    with (
        patch("nocturnomath.session.Kernel", PlotFakeKernel),
        patch("nocturnomath.session.ResearchEnvironment", FakeEnvironment),
    ):
        session = ExplorationSession(tmp_path)
        runtime = Runtime(session, backends=make_backends())
        runtime.start()
        try:
            # Authenticate purely via API key with zero CLIs installed.
            await runtime.authenticate("anthropic", "api_key", "sk-ant-e2e")
            assert runtime.models == ["claude-sonnet-4-6"]

            assert runtime.start_query("Verify 6*7") == "started"
            await runtime.wait_for_query()
            assert (
                "6*7 printed 42."
                in (tmp_path / ".nocturnomath/kb/evidence.md").read_text()
            )

            # Compact in HTTP Anthropic mode -> must set record_reloaded=False so next turn reloads notes.
            await runtime.compact_agent(runtime.primary_agent_id)
            assert compact_calls == 1
            assert session._record_dirty is True

            assert runtime.start_query("What did E001 say?") == "started"
            await runtime.wait_for_query()
            last_user_text = turn_calls[-1]["messages"][-1]["content"]
            assert "Conversation history was compacted." in last_user_text
            assert "E001" in last_user_text
            assert session._record_dirty is False

            # Resume in a fresh session instance using the saved HTTP thread file + ambient env var.
            fresh_session = ExplorationSession(tmp_path, backends=make_backends())
            try:
                monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-e2e")
                fresh_session.model_providers = dict(runtime.model_providers)
                fresh_session.model_raw_ids = dict(runtime.model_raw_ids)
                fresh_session.model = "claude-sonnet-4-6"
                resumed = fresh_session.resume_session("S001")
                assert resumed["context_restored"] is True
                assert await fresh_session.query("One more follow-up") is True
                # Native HTTP thread was resumed with compacted summary + follow-up (no recap).
                assert (
                    turn_calls[-1]["messages"][0]["content"]
                    == "Summary of our earlier conversation before compaction:"
                )
                assert turn_calls[-1]["messages"][-1]["content"].endswith(
                    "One more follow-up"
                )
                assert "Me: Verify 6*7" not in turn_calls[-1]["messages"][-1]["content"]

                # Switching back to CLI mode raises SessionResumeError on the HTTP thread id
                # and rebuilds context from Workspace.recap automatically.
                monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
                seen_recap: list[str] = []
                real_run_turn = fresh_session.backends["anthropic"].run_turn

                async def switched_to_cli_run_turn(prompt, **kwargs):
                    if kwargs.get("resume_session_id"):
                        async for item in real_run_turn(prompt, **kwargs):
                            yield item
                        return
                    seen_recap.append(prompt)
                    yield TurnStarted()
                    yield TextBlockDone("Rebuilt from CLI fallback.")
                    yield TurnResult()

                with patch.object(
                    fresh_session.backends["anthropic"],
                    "run_turn",
                    side_effect=switched_to_cli_run_turn,
                ):
                    assert await fresh_session.query("Back to CLI") is True
                assert seen_recap and "Verify 6*7" in seen_recap[0]
            finally:
                with suppress(Exception):
                    fresh_session.shutdown()
        finally:
            await runtime.drain()
            runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "openai", "google"])
async def test_http_turn_aclosing_cancel_saves_synchronously(tmp_path, provider):
    import gc
    from contextlib import aclosing

    from nocturnomath.backends.http import HttpThreadStore

    if provider == "anthropic":
        sse = _sse_body(
            [
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {"id": "m1", "model": "claude-sonnet-4-6"},
                    },
                ),
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text"},
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "Halfway"},
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
        )
        backend = AnthropicBackend(
            http_client_factory=_client_factory(
                lambda _r: httpx.Response(200, content=sse)
            )
        )
        auth = ProviderAuth.interactive("anthropic", "api_key", "sk-ant-aclose")
        model = "claude-sonnet-4-6"
    elif provider == "openai":
        sse = _sse_body(
            [
                (
                    "response.created",
                    {
                        "type": "response.created",
                        "response": {"id": "r1", "model": "gpt-5.4"},
                    },
                ),
                (
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": "m1",
                        "content_index": 0,
                        "delta": "Halfway",
                    },
                ),
            ]
        )
        backend = OpenAIBackend(
            http_client_factory=_client_factory(
                lambda _r: httpx.Response(200, content=sse)
            )
        )
        auth = ProviderAuth.interactive("openai", "api_key", "sk-oai-aclose")
        model = "gpt-5.4"
    else:
        sse = _sse_body(
            [
                (
                    None,
                    {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "Halfway"}],
                                }
                            }
                        ]
                    },
                )
            ]
        )
        backend = GoogleBackend(
            http_client_factory=_client_factory(
                lambda _r: httpx.Response(200, content=sse)
            )
        )
        auth = ProviderAuth.interactive("google", "api_key", "AIza-aclose")
        model = "gemini-3-pro-preview"

    sid = None
    async with aclosing(
        backend.run_turn(
            "interrupt during consumer close",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model=model,
            effort=None,
            auth=auth,
        )
    ) as events:
        async for ev in events:
            if isinstance(ev, SessionStarted):
                sid = ev.session_id
            elif isinstance(ev, TextDelta):
                break

    assert sid is not None
    # Saved immediately upon aclosing() exit before any GC finalization.
    saved = HttpThreadStore(tmp_path, provider).load(sid)["messages"]
    assert saved
    gc.collect()
    await asyncio.sleep(0)
    assert "(Interrupted by user.)" in json.dumps(saved[-1])


@pytest.mark.asyncio
async def test_google_round2_error_and_max_tool_rounds_keep_valid_thread(
    tmp_path, monkeypatch
):
    from nocturnomath.backends import http as http_mod
    from nocturnomath.backends.http import HttpThreadStore

    call_num = 0

    async def google_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_num
        call_num += 1
        body = json.loads(request.content.decode("utf-8"))
        if call_num == 1:
            return httpx.Response(
                200,
                content=_sse_body(
                    [
                        (
                            None,
                            {
                                "candidates": [
                                    {
                                        "finishReason": "STOP",
                                        "content": {
                                            "role": "model",
                                            "parts": [
                                                {
                                                    "functionCall": {
                                                        "name": "ping",
                                                        "args": {},
                                                        "id": "c1",
                                                    }
                                                }
                                            ],
                                        },
                                    }
                                ]
                            },
                        )
                    ]
                ),
            )
        if call_num == 2:
            return httpx.Response(500, json={"error": {"message": "upstream boom"}})

        # Follow-up turn must never have consecutive user turns after failed Round 2.
        roles = [entry["role"] for entry in body["contents"]]
        assert all(
            not (roles[i] == "user" and roles[i + 1] == "user")
            for i in range(len(roles) - 1)
        )
        return httpx.Response(
            200,
            content=_sse_body(
                [
                    (
                        None,
                        {
                            "candidates": [
                                {
                                    "finishReason": "STOP",
                                    "content": {
                                        "role": "model",
                                        "parts": [{"text": "Recovered cleanly."}],
                                    },
                                }
                            ]
                        },
                    )
                ]
            ),
        )

    @tool("ping", "Ping.", {})
    async def ping_tool(_args):
        return {"content": [{"type": "text", "text": "pong"}]}

    g_backend = GoogleBackend(http_client_factory=_client_factory(google_handler))
    g_auth = ProviderAuth.interactive("google", "api_key", "AIza-recover")
    seen: list[str] = []
    with pytest.raises(RuntimeError, match="upstream boom"):
        async for ev in g_backend.run_turn(
            "round1 ok round2 fails",
            system_prompt="Sys",
            tools=[ping_tool],
            workspace_path=tmp_path,
            model="gemini-3-pro-preview",
            effort=None,
            auth=g_auth,
        ):
            if isinstance(ev, SessionStarted):
                seen.append(ev.session_id)

    saved = HttpThreadStore(tmp_path, "google").load(seen[0])["messages"]
    assert saved[-1]["role"] == "model"

    events = [
        e
        async for e in g_backend.run_turn(
            "try again",
            system_prompt="Sys",
            tools=[ping_tool],
            workspace_path=tmp_path,
            model="gemini-3-pro-preview",
            effort=None,
            auth=g_auth,
            resume_session_id=seen[0],
        )
    ]
    assert [e.text for e in events if isinstance(e, TextBlockDone)] == [
        "Recovered cleanly."
    ]

    # MAX_TOOL_ROUNDS exceeded refuses to execute final round tools and pairs tool_use.
    monkeypatch.setattr(http_mod, "MAX_TOOL_ROUNDS", 1)
    ran_tool = False

    @tool("never", "Should not execute.", {})
    async def never_tool(_args):
        nonlocal ran_tool
        ran_tool = True
        return {"content": [{"type": "text", "text": "ran"}]}

    ant_sse = _sse_body(
        [
            (
                "message_start",
                {"type": "message_start", "message": {"id": "m1", "model": "c"}},
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "tu_max",
                        "name": "never",
                    },
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    )
    a_backend = AnthropicBackend(
        http_client_factory=_client_factory(
            lambda _r: httpx.Response(200, content=ant_sse)
        )
    )
    a_seen: list[str] = []
    with pytest.raises(RuntimeError, match="maximum tool rounds"):
        async for ev in a_backend.run_turn(
            "loop",
            system_prompt="Sys",
            tools=[never_tool],
            workspace_path=tmp_path,
            model="claude-sonnet-4-6",
            effort=None,
            auth=ProviderAuth.interactive("anthropic", "api_key", "sk-ant-max"),
        ):
            if isinstance(ev, SessionStarted):
                a_seen.append(ev.session_id)
    assert ran_tool is False
    a_msgs = HttpThreadStore(tmp_path, "anthropic").load(a_seen[0])["messages"]
    assert a_msgs[-2]["role"] == "user"
    assert a_msgs[-2]["content"][0]["tool_use_id"] == "tu_max"
    assert a_msgs[-2]["content"][0]["is_error"] is True
    assert a_msgs[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_http_compaction_truncation_guards_and_openai_reasoning_tokens(tmp_path):
    from nocturnomath.backends.http import HttpThreadStore

    # Google compaction rejects SAFETY finishReason without overwriting saved thread.
    g_store = HttpThreadStore(tmp_path, "google")
    g_sid = g_store.new_session_id()
    g_orig = [
        {"role": "user", "parts": [{"text": "plot"}]},
        {
            "role": "model",
            "parts": [
                {"text": "plan", "thought": True},
                {
                    "functionCall": {"name": "p", "args": {}, "id": "1"},
                    "thoughtSignature": "s",
                },
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "p",
                        "id": "1",
                        "response": {"output": "ok"},
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": _TINY_PNG_B64,
                                }
                            }
                        ],
                    }
                }
            ],
        },
        {"role": "model", "parts": [{"text": "done"}]},
    ]
    g_store.save(g_sid, model="gemini-3-pro-preview", messages=g_orig)

    g_backend = GoogleBackend(
        http_client_factory=_client_factory(
            lambda _r: httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "finishReason": "SAFETY",
                            "content": {
                                "role": "model",
                                "parts": [{"text": "Partial"}],
                            },
                        }
                    ]
                },
            )
        )
    )
    with pytest.raises(RuntimeError, match="SAFETY"):
        await g_backend.compact(
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model=None,
            effort=None,
            auth=ProviderAuth.interactive("google", "api_key", "AIza-cmp"),
            resume_session_id=g_sid,
        )
    assert g_store.load(g_sid)["messages"] == g_orig

    # OpenAI compaction strips reasoning_tokens from post-compact usage and keeps no tools.
    o_store = HttpThreadStore(tmp_path, "openai")
    o_sid = o_store.new_session_id()
    o_store.save(
        o_sid,
        model="gpt-5.4",
        messages=[
            {"role": "user", "content": "hi"},
            {"type": "reasoning", "encrypted_content": "enc"},
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "run",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "c1", "output": "done"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "42"}],
            },
        ],
    )
    seen_o_body = []

    async def o_handler(req: httpx.Request) -> httpx.Response:
        seen_o_body.append(json.loads(req.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Compact OAI."}],
                    }
                ],
                "usage": {
                    "output_tokens": 500,
                    "output_tokens_details": {"reasoning_tokens": 450},
                },
            },
        )

    o_backend = OpenAIBackend(http_client_factory=_client_factory(o_handler))
    res = await o_backend.compact(
        system_prompt="Sys",
        tools=[],
        workspace_path=tmp_path,
        model=None,
        effort=None,
        auth=ProviderAuth.interactive("openai", "api_key", "sk-oai-cmp"),
        resume_session_id=o_sid,
    )
    assert "tools" not in seen_o_body[0]
    assert "reasoning" not in json.dumps(seen_o_body[0]["input"])
    assert res.context_usage["used_tokens"] == 70  # (500 - 450) + 20


@pytest.mark.asyncio
async def test_gemini_25_plot_images_and_same_provider_model_switching(tmp_path):
    # Gemini 2.5 sends returned plot images in a separate user turn, and switching to Gemini 3
    # adds skip_thought_signature_validator on historical functionCalls without carrying window_tokens.
    seen_bodies: list[dict] = []
    step = 0

    async def handler(req: httpx.Request) -> httpx.Response:
        nonlocal step
        body = json.loads(req.content.decode("utf-8"))
        seen_bodies.append(body)
        step += 1
        if step == 1:
            return httpx.Response(
                200,
                content=_sse_body(
                    [
                        (
                            None,
                            {
                                "candidates": [
                                    {
                                        "finishReason": "STOP",
                                        "content": {
                                            "role": "model",
                                            "parts": [
                                                {"text": "secret", "thought": True},
                                                {
                                                    "functionCall": {
                                                        "name": "plot",
                                                        "args": {},
                                                        "id": "g25",
                                                    }
                                                },
                                            ],
                                        },
                                    }
                                ]
                            },
                        )
                    ]
                ),
            )
        if step == 2:
            fr_turn = body["contents"][-2]
            img_turn = body["contents"][-1]
            assert "parts" not in fr_turn["parts"][0]["functionResponse"]
            assert img_turn["role"] == "user"
            assert img_turn["parts"][1]["inlineData"]["data"] == _TINY_PNG_B64
            return httpx.Response(
                200,
                content=_sse_body(
                    [
                        (
                            None,
                            {
                                "candidates": [
                                    {
                                        "finishReason": "STOP",
                                        "content": {
                                            "role": "model",
                                            "parts": [{"text": "Gemini 2.5 saw plot."}],
                                        },
                                    }
                                ],
                                "usageMetadata": {"promptTokenCount": 120},
                            },
                        )
                    ]
                ),
            )
        # Step 3: resumed under gemini-3-pro-preview
        model_hist = body["contents"][1]
        assert all(not p.get("thought") for p in model_hist["parts"])
        fc_part = next(p for p in model_hist["parts"] if "functionCall" in p)
        assert fc_part["thoughtSignature"] == "skip_thought_signature_validator"
        return httpx.Response(
            200,
            content=_sse_body(
                [
                    (
                        None,
                        {
                            "candidates": [
                                {
                                    "finishReason": "STOP",
                                    "content": {
                                        "role": "model",
                                        "parts": [{"text": "Switched cleanly."}],
                                    },
                                }
                            ],
                            "usageMetadata": {"promptTokenCount": 90},
                        },
                    )
                ]
            ),
        )

    @tool("plot", "Plot.", {})
    async def plot_tool(_args):
        return {
            "content": [
                {"type": "text", "text": "plotted"},
                {"type": "image", "data": _TINY_PNG_B64, "mimeType": "image/png"},
            ]
        }

    backend = GoogleBackend(http_client_factory=_client_factory(handler))
    backend._http_model_info["gemini-2.5-pro"] = {"context_window": 1000000}
    auth = ProviderAuth.interactive("google", "api_key", "AIza-switch")

    events1 = [
        e
        async for e in backend.run_turn(
            "run on 2.5",
            system_prompt="Sys",
            tools=[plot_tool],
            workspace_path=tmp_path,
            model="gemini-2.5-pro",
            effort="low",
            auth=auth,
        )
    ]
    sid = next(e.session_id for e in events1 if isinstance(e, SessionStarted))

    events2 = [
        e
        async for e in backend.run_turn(
            "switch to 3",
            system_prompt="Sys",
            tools=[plot_tool],
            workspace_path=tmp_path,
            model="gemini-3-pro-preview",
            effort="high",
            auth=auth,
            resume_session_id=sid,
        )
    ]
    res2 = next(e for e in events2 if isinstance(e, TurnResult))
    # Does not inherit gemini-2.5-pro's 1M window after switching models.
    assert res2.context_usage["window_tokens"] is None


@pytest.mark.asyncio
async def test_sse_unicode_separators_bom_and_stream_failure_guards(tmp_path):

    payload_text = "line\u2028sep\u2029nel\x85ok"
    sse_bytes = (
        "\ufeff"
        + "event: message_start\n"
        + 'data: {"type": "message_start", "message": {"id": "m1", "model": "claude-sonnet-4-6"}}\n\n'
        + "event: content_block_start\n"
        + 'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}\n\n'
        + "event: content_block_delta\n"
        + f'data: {{"type": "content_block_delta", "index": 0, "delta": {{"type": "text_delta", "text": {json.dumps(payload_text, ensure_ascii=False)}}}}}\n\n'
        + "event: content_block_stop\n"
        + 'data: {"type": "content_block_stop", "index": 0}\n\n'
        + "event: message_delta\n"
        + 'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}\n\n'
        + "event: message_stop\n"
        + 'data: {"type": "message_stop"}\n\n'
    ).encode("utf-8")

    # Deliver 5 bytes at a time to test chunked UTF-8 BOM and separator handling.
    class ChunkedStream(httpx.AsyncByteStream):
        def __init__(self, raw: bytes):
            self.raw = raw

        async def __aiter__(self):
            for i in range(0, len(self.raw), 5):
                yield self.raw[i : i + 5]

    a_backend = AnthropicBackend(
        http_client_factory=_client_factory(
            lambda _r: httpx.Response(200, stream=ChunkedStream(sse_bytes))
        )
    )
    events = [
        e
        async for e in a_backend.run_turn(
            "unicode",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="claude-sonnet-4-6",
            effort=None,
            auth=ProviderAuth.interactive("anthropic", "api_key", "sk-ant-u"),
        )
    ]
    assert [e.text for e in events if isinstance(e, TextBlockDone)] == [payload_text]

    # Google chunk-1 promptFeedback blockReason must NOT emit TurnStarted or save session.
    g_block_sse = _sse_body([(None, {"promptFeedback": {"blockReason": "SAFETY"}})])
    g_backend = GoogleBackend(
        http_client_factory=_client_factory(
            lambda _r: httpx.Response(200, content=g_block_sse)
        )
    )
    seen_before_block: list = []
    with pytest.raises(RuntimeError, match="blocked the prompt.*SAFETY"):
        async for ev in g_backend.run_turn(
            "blocked",
            system_prompt="Sys",
            tools=[],
            workspace_path=tmp_path,
            model="gemini-3-pro-preview",
            effort=None,
            auth=ProviderAuth.interactive("google", "api_key", "AIza-blk"),
        ):
            seen_before_block.append(ev)
    assert not any(isinstance(ev, TurnStarted) for ev in seen_before_block)


@pytest.mark.asyncio
async def test_openai_model_inference_base_url_key_redaction_and_store_errors(
    tmp_path, monkeypatch
):
    from nocturnomath.backends.http import HttpThreadStore, discover_openai_http_models

    # OpenAI HTTP model IDs (o3, o4-mini, codex-*, chatgpt-*) map to provider="openai" even pre-discovery.
    async def oai_models_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "o3", "created": 300},
                    {"id": "o4-mini", "created": 250},
                    {"id": "codex-mini-latest", "created": 200},
                    {"id": "chatgpt-4o-latest", "created": 100},
                ]
            },
        )

    specs, err = await discover_openai_http_models(
        "sk-oai-test", http_client_factory=_client_factory(oai_models_handler)
    )
    assert err is None
    assert {s.id for s in specs} == {
        "o3",
        "o4-mini",
        "codex-mini-latest",
        "chatgpt-4o-latest",
    }

    with (
        patch("nocturnomath.session.Kernel", FakeKernel),
        patch("nocturnomath.session.ResearchEnvironment", FakeEnvironment),
    ):
        session = ExplorationSession(tmp_path)
        try:
            for mid in ("o3", "o4-mini", "codex-mini-latest", "chatgpt-4o-latest"):
                assert session.provider_for_model(mid) == "openai"
        finally:
            session.shutdown()

    # ANTHROPIC_BASE_URL with trailing /v1 normalizes without /v1/v1/.
    seen_urls: list[str] = []
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example/v1")

    def ant_proxy_handler(req: httpx.Request) -> httpx.Response:
        seen_urls.append(str(req.url))
        return httpx.Response(
            200, json={"data": [{"id": "claude-sonnet-4-6"}], "has_more": False}
        )

    a_backend = AnthropicBackend(http_client_factory=_client_factory(ant_proxy_handler))
    await a_backend.discover_models(
        ProviderAuth.interactive("anthropic", "api_key", "sk-ant-proxy")
    )
    assert seen_urls == ["https://proxy.example/v1/models?limit=100"]

    # Control chars in ambient env key never leak secret in discovery error.
    secret = "sk-ant-secret\r\nleak"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    await a_backend.discover_models(ProviderAuth("anthropic", "cli"))
    assert a_backend.last_discovery_error is not None
    assert "sk-ant-secret" not in a_backend.last_discovery_error

    # Invalid UTF-8 in saved thread file yields SessionResumeError.
    store = HttpThreadStore(tmp_path, "anthropic")
    bad_sid = store.new_session_id()
    store._path_for(bad_sid).parent.mkdir(parents=True, exist_ok=True)
    store._path_for(bad_sid).write_bytes(b"\xff\xfe\xfd")
    with pytest.raises(SessionResumeError):
        store.load(bad_sid)


@pytest.mark.asyncio
async def test_http_batched_tools_short_circuit_after_steer_and_discovery_error(
    tmp_path,
):
    tool_runs: list[str] = []

    with (
        patch("nocturnomath.session.Kernel", FakeKernel),
        patch("nocturnomath.session.ResearchEnvironment", FakeEnvironment),
    ):
        session = ExplorationSession(tmp_path)
        runtime = Runtime(session)

        async def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content.decode("utf-8"))
            if len(body["messages"]) == 1:
                return httpx.Response(
                    200,
                    content=_sse_body(
                        [
                            (
                                "message_start",
                                {
                                    "type": "message_start",
                                    "message": {
                                        "id": "m1",
                                        "model": "claude-sonnet-4-6",
                                    },
                                },
                            ),
                            (
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": 0,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": "tu1",
                                        "name": "probe_a",
                                        "input": {},
                                    },
                                },
                            ),
                            (
                                "content_block_stop",
                                {"type": "content_block_stop", "index": 0},
                            ),
                            (
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": 1,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": "tu2",
                                        "name": "probe_b",
                                        "input": {},
                                    },
                                },
                            ),
                            (
                                "content_block_stop",
                                {"type": "content_block_stop", "index": 1},
                            ),
                            (
                                "message_delta",
                                {
                                    "type": "message_delta",
                                    "delta": {"stop_reason": "tool_use"},
                                },
                            ),
                            ("message_stop", {"type": "message_stop"}),
                        ]
                    ),
                )
            # Second round receives steered result for tu1 and skipped error for tu2.
            res_blocks = body["messages"][-1]["content"]
            assert res_blocks[0]["content"][0]["text"].startswith(
                "Nothing ran: the user sent a new message"
            )
            assert res_blocks[1]["is_error"] is True
            assert (
                "Skipped: the user sent a new message"
                in res_blocks[1]["content"][0]["text"]
            )
            return httpx.Response(
                200,
                content=_sse_body(
                    [
                        (
                            "message_start",
                            {
                                "type": "message_start",
                                "message": {"id": "m2", "model": "claude-sonnet-4-6"},
                            },
                        ),
                        (
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": 0,
                                "content_block": {"type": "text"},
                            },
                        ),
                        (
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {
                                    "type": "text_delta",
                                    "text": "Steer handled.",
                                },
                            },
                        ),
                        (
                            "content_block_stop",
                            {"type": "content_block_stop", "index": 0},
                        ),
                        (
                            "message_delta",
                            {
                                "type": "message_delta",
                                "delta": {"stop_reason": "end_turn"},
                            },
                        ),
                        ("message_stop", {"type": "message_stop"}),
                    ]
                ),
            )

        @tool("probe_a", "A", {})
        async def probe_a(_args):
            tool_runs.append("a")
            session.enqueue_prompt("steer now", mode="steer")
            steered = await session.consume_steer_message()
            assert steered is not None
            return steered

        @tool("probe_b", "B", {})
        async def probe_b(_args):
            tool_runs.append("b")
            return {"content": [{"type": "text", "text": "b ran"}]}

        try:
            backend = AnthropicBackend(http_client_factory=_client_factory(handler))
            session._turn_selection = ("anthropic", "claude-sonnet-4-6", None)
            events = [
                e
                async for e in backend.run_turn(
                    "go",
                    system_prompt="Sys",
                    tools=[probe_a, probe_b],
                    workspace_path=tmp_path,
                    model="claude-sonnet-4-6",
                    effort=None,
                    auth=ProviderAuth.interactive(
                        "anthropic", "api_key", "sk-ant-steer"
                    ),
                )
            ]
            assert tool_runs == ["a"]
            assert [e.text for e in events if isinstance(e, TextBlockDone)] == [
                "Steer handled."
            ]

            # Discovery 401 surfaces in Runtime.auth_public().
            bad_google = GoogleBackend(
                http_client_factory=_client_factory(
                    lambda _r: httpx.Response(
                        401, json=[{"error": {"message": "API key expired"}}]
                    )
                )
            )
            rt = Runtime(
                session,
                backends={
                    "anthropic": AnthropicBackend(),
                    "openai": OpenAIBackend(),
                    "google": bad_google,
                },
            )
            pub = await rt.authenticate("google", "api_key", "AIza-expired")
            assert pub["providers"]["google"]["error"] == "API key expired"
        finally:
            runtime.shutdown()
