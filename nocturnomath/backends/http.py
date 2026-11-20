"""Direct HTTP API backends for Anthropic, OpenAI, and Google Gemini."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx

from ..auth import PROVIDER_TITLES, ProviderId
from ..mcp_bridge import _validate_arguments
from ..tools import ToolSpec
from .base import (
    EFFORT_LEVELS,
    CompactResult,
    MessageUsage,
    ModelSpec,
    SessionResumeError,
    SessionStarted,
    TextBlockDone,
    TextDelta,
    TurnEvent,
    TurnResult,
    TurnStarted,
)

logger = logging.getLogger("nocturnomath.backends.http")

HTTP_SESSION_PREFIX = "http:"
HTTP_DISCOVERY_TIMEOUT_S = 15.0
HTTP_TIMEOUT = httpx.Timeout(connect=15.0, read=300.0, write=30.0, pool=15.0)
MAX_TOOL_ROUNDS = 100
COMPACTION_PROMPT = (
    "Summarize our conversation so far clearly and concisely for continuing this "
    "scientific investigation. Include:\n"
    "1. The scientific question and hypotheses tested.\n"
    "2. Key probe IDs, exact numerical results, errors, and recorded verdicts.\n"
    "3. Open questions, next steps, and any user instructions or preferences.\n"
    "4. Important variables still relevant in Python memory or files created.\n"
    "Do not call any tools; reply with the summary text only."
)

_HTTP_SESSION_ID_RE = re.compile(r"http:(anthropic|openai|google):([0-9a-f]{32})")
_OPENAI_MODEL_ALLOW_RE = re.compile(
    r"^(?:gpt-(?:[4-9]|[1-9]\d)|o[1-9]|codex-|chatgpt-)"
)
_OPENAI_MODEL_EXCLUDE = (
    "-audio",
    "-realtime",
    "-transcribe",
    "-tts",
    "-image",
    "-search",
    "-instruct",
    "embedding",
    "whisper",
    "dall-e",
    "moderation",
    "sora",
    "babbage",
    "davinci",
    "computer-use",
    "deep-research",
    "diarize",
)
_OPENAI_REASONING_RE = re.compile(r"^(?:o[1-9]|gpt-(?:[5-9]|[1-9]\d)|codex-)")
_OPENAI_DEFAULT_ORDER = ("gpt-5.4", "gpt-5.2", "gpt-5", "gpt-4.1", "gpt-4o")
_GOOGLE_MODEL_EXCLUDE = (
    "embedding",
    "imagen",
    "veo",
    "aqa",
    "tts",
    "audio",
    "robotics",
    "computer-use",
    "deep-research",
    "-live-",
)
_GOOGLE_EFFORT_RE = re.compile(r"^gemini-(?:2\.5|[3-9]|[1-9]\d)")
_ANTHROPIC_VERSION = "2023-06-01"
_GEMINI_25_THINKING_BUDGETS = {"low": 1024, "medium": 8192, "high": 24576}


def is_http_session_id(session_id: str | None) -> bool:
    """Return whether ``session_id`` identifies a direct HTTP thread."""
    return bool(session_id and session_id.startswith(HTTP_SESSION_PREFIX))


class HttpThreadStore:
    """Persist provider-native HTTP message histories under ``.nocturnomath/runtime``."""

    def __init__(self, workspace_path: Path, provider: ProviderId):
        self.root = workspace_path / ".nocturnomath" / "runtime" / "http_sessions"
        self.provider = provider

    def new_session_id(self) -> str:
        """Allocate a new opaque HTTP session identifier."""
        return f"http:{self.provider}:{uuid.uuid4().hex}"

    def _path_for(self, session_id: str) -> Path:
        match = _HTTP_SESSION_ID_RE.fullmatch(session_id)
        if not match:
            raise SessionResumeError(f"Unrecognized HTTP session id: {session_id}")
        provider, hex_id = match.groups()
        if provider != self.provider:
            raise SessionResumeError(
                f"Session {session_id} belongs to {provider}, not {self.provider}."
            )
        return self.root / f"{provider}-{hex_id}.json"

    def load(self, session_id: str) -> dict[str, Any]:
        """Load a saved HTTP thread or raise ``SessionResumeError`` if unavailable."""
        path = self._path_for(session_id)
        if not path.is_file():
            raise SessionResumeError(f"Saved session {session_id} was not found.")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SessionResumeError(
                f"Saved session {session_id} could not be read."
            ) from exc
        if (
            not isinstance(data, dict)
            or data.get("provider") != self.provider
            or not isinstance(data.get("messages"), list)
        ):
            raise SessionResumeError(f"Saved session {session_id} is invalid.")
        return data

    def save(
        self,
        session_id: str,
        *,
        model: str | None,
        messages: list[Any],
        context_window: int | None = None,
    ) -> None:
        """Atomically persist thread messages for subsequent turns or compaction."""
        path = self._path_for(session_id)
        self.root.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "version": 1,
                "provider": self.provider,
                "model": model,
                "messages": messages,
                "context_window": context_window,
            }
        )
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.stem}-", suffix=".tmp", dir=str(self.root)
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp_path, path)
        except Exception:
            with suppress(OSError):
                tmp_path.unlink()
            raise


def make_http_client(
    factory: Callable[..., httpx.AsyncClient] | None,
    *,
    timeout: httpx.Timeout | float,
) -> httpx.AsyncClient:
    """Instantiate an ``httpx.AsyncClient`` using an optional test factory."""
    if factory is None:
        return httpx.AsyncClient(timeout=timeout)
    return factory(timeout=timeout)


async def iter_sse_events(response: httpx.Response) -> AsyncIterator[tuple[str, str]]:
    """Yield ``(event_name, data)`` tuples from an SSE HTTP response."""
    event_name: str | None = None
    data_lines: list[str] = []
    async for raw_line in response.aiter_lines():
        if raw_line == "":
            if data_lines:
                yield (event_name or "message", "\n".join(data_lines))
                event_name = None
                data_lines.clear()
            else:
                event_name = None
            continue
        if raw_line.startswith(":"):
            continue
        if raw_line.startswith("event:"):
            event_name = raw_line[6:].removeprefix(" ").strip()
        elif raw_line.startswith("data:"):
            data_lines.append(raw_line[5:].removeprefix(" "))
    if data_lines:
        yield (event_name or "message", "\n".join(data_lines))


def http_error_message(
    provider_title: str, response: httpx.Response, body: bytes | str
) -> str:
    """Extract a readable API error message from a non-2xx HTTP response."""
    text = (
        body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body)
    ).strip()
    if text:
        with suppress(json.JSONDecodeError):
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                err = parsed.get("error")
                if isinstance(err, dict) and isinstance(err.get("message"), str):
                    if err["message"].strip():
                        return err["message"].strip()
                elif isinstance(err, str) and err.strip():
                    return err.strip()
                msg = parsed.get("message")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
    base = f"{provider_title} API request failed (HTTP {response.status_code})."
    if text:
        snippet = text[:300].replace("\n", " ").strip()
        return f"{base} {snippet}"
    return base


async def execute_tool_call(
    tool_by_name: Mapping[str, ToolSpec],
    name: str,
    raw_args: Any,
    tool_timeout_s: int,
) -> dict[str, Any]:
    """Validate and run an in-process ``ToolSpec`` without swallowing cancellations."""
    spec = tool_by_name.get(name)
    if spec is None:
        return {
            "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
            "is_error": True,
        }
    if not isinstance(raw_args, dict):
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Tool '{name}' arguments must be a JSON object.",
                }
            ],
            "is_error": True,
        }
    try:
        cleaned = _validate_arguments(spec.input_schema, raw_args)
    except ValueError as exc:
        return {"content": [{"type": "text", "text": str(exc)}], "is_error": True}
    try:
        return await asyncio.wait_for(spec.handler(cleaned), timeout=tool_timeout_s)
    except TimeoutError:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Tool '{name}' timed out after {tool_timeout_s}s.",
                }
            ],
            "is_error": True,
        }
    except Exception as exc:
        return {
            "content": [{"type": "text", "text": str(exc) or type(exc).__name__}],
            "is_error": True,
        }


def _tool_text_and_images(
    result: dict[str, Any], *, is_error: bool
) -> tuple[str, list[dict[str, Any]]]:
    text_parts: list[str] = []
    images: list[dict[str, Any]] = []
    for item in result.get("content") or ():
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text":
            text = str(item.get("text") or "")
            if text:
                text_parts.append(text)
        elif kind == "image" and isinstance(item.get("data"), str):
            images.append(item)
    text_out = (
        "\n".join(text_parts) if text_parts else ("Error" if is_error else "(ok)")
    )
    if is_error and not text_out.lower().startswith("error"):
        text_out = f"Error: {text_out}"
    return text_out, images


def _add_int_usage(target: dict[str, int], source: Mapping[str, Any] | None) -> None:
    if not isinstance(source, Mapping):
        return
    for key, value in source.items():
        if isinstance(value, int) and not isinstance(value, bool):
            target[key] = target.get(key, 0) + value


def _resume_state(
    store: HttpThreadStore, resume_session_id: str | None
) -> tuple[str, list[Any], str | None, int | None]:
    if resume_session_id:
        saved = store.load(resume_session_id)
        ctx_window = saved.get("context_window")
        return (
            resume_session_id,
            list(saved["messages"]),
            saved.get("model") if isinstance(saved.get("model"), str) else None,
            ctx_window if isinstance(ctx_window, int) and ctx_window > 0 else None,
        )
    return store.new_session_id(), [], None, None


# ---------------------------------------------------------------------------
# Anthropic Messages API
# ---------------------------------------------------------------------------


def _anthropic_base_url() -> str:
    return (os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip(
        "/"
    )


def _anthropic_headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def _anthropic_capabilities(model_id: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    caps = (
        entry.get("capabilities")
        if isinstance(entry.get("capabilities"), Mapping)
        else {}
    )
    efforts: list[str] = []
    cap_effort = caps.get("effort")
    if isinstance(cap_effort, Mapping):
        if cap_effort.get("supported") is not False:
            efforts = [
                level
                for level in EFFORT_LEVELS
                if isinstance(cap_effort.get(level), Mapping)
                and cap_effort[level].get("supported") is True
            ]
    elif "effort" not in caps:
        if re.search(r"claude-opus-(?:4-[6-9]|[5-9])", model_id):
            efforts = ["low", "medium", "high", "max"]
        elif re.search(r"claude-(?:opus|sonnet)-(?:4-[5-9]|[5-9])", model_id):
            efforts = ["low", "medium", "high"]

    cap_thinking = caps.get("thinking")
    adaptive: bool | None = None
    if isinstance(cap_thinking, Mapping):
        types = cap_thinking.get("types")
        if isinstance(types, Mapping) and isinstance(types.get("adaptive"), Mapping):
            adaptive_spec = types["adaptive"]
            if "supported" in adaptive_spec:
                adaptive = bool(adaptive_spec["supported"])
    if adaptive is None:
        adaptive = bool(
            re.search(r"claude-(?:opus|sonnet)-(?:4-[6-9]|[5-9])", model_id)
        )

    max_input = entry.get("max_input_tokens")
    max_tokens = entry.get("max_tokens")
    return {
        "efforts": efforts,
        "thinking_adaptive": adaptive,
        "context_window": max_input
        if isinstance(max_input, int) and max_input > 0
        else None,
        "max_tokens": max_tokens
        if isinstance(max_tokens, int) and max_tokens > 0
        else None,
    }


async def discover_anthropic_http_models(
    api_key: str,
    *,
    http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
) -> tuple[list[ModelSpec], dict[str, dict[str, Any]]]:
    """List Claude models via ``GET /v1/models``."""
    base = _anthropic_base_url()
    headers = _anthropic_headers(api_key)
    raw_items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    after_id: str | None = None
    try:
        async with (
            asyncio.timeout(HTTP_DISCOVERY_TIMEOUT_S),
            make_http_client(
                http_client_factory, timeout=HTTP_DISCOVERY_TIMEOUT_S
            ) as client,
        ):
            for _ in range(5):
                params: dict[str, str] = {"limit": "100"}
                if after_id is not None:
                    params["after_id"] = after_id
                response = await client.get(
                    f"{base}/v1/models", headers=headers, params=params
                )
                if response.status_code != 200:
                    logger.warning(
                        "Anthropic HTTP model discovery failed: %s",
                        http_error_message("Anthropic", response, response.content),
                    )
                    return [], {}
                payload = response.json()
                data = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(data, list):
                    break
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    mid = item.get("id")
                    if (
                        isinstance(mid, str)
                        and mid.startswith("claude-")
                        and mid not in seen_ids
                    ):
                        seen_ids.add(mid)
                        raw_items.append(item)
                if not payload.get("has_more"):
                    break
                after_id = payload.get("last_id")
                if not isinstance(after_id, str) or not after_id:
                    break
    except Exception as exc:
        logger.warning("Anthropic HTTP model discovery failed: %s", exc)
        return [], {}

    if not raw_items:
        return [], {}

    model_info: dict[str, dict[str, Any]] = {}
    default_id = next(
        (item["id"] for item in raw_items if "sonnet" in item["id"]),
        raw_items[0]["id"],
    )
    specs: list[ModelSpec] = []
    for item in raw_items:
        mid = item["id"]
        info = _anthropic_capabilities(mid, item)
        model_info[mid] = info
        specs.append(
            ModelSpec(
                id=mid,
                raw_id=mid,
                provider="anthropic",
                label=mid,
                efforts=list(info["efforts"]),
                is_default=(mid == default_id),
            )
        )
    return specs, model_info


def _anthropic_prompt_tokens(usage: Mapping[str, Any]) -> int | None:
    keys = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    if not any(key in usage for key in keys):
        return None
    total = 0
    for key in keys:
        val = usage.get(key, 0)
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            return None
        total += val
    return total


def _anthropic_tool_result_block(
    tool_use_id: str, result: dict[str, Any]
) -> dict[str, Any]:
    is_error = bool(result.get("is_error"))
    text_out, images = _tool_text_and_images(result, is_error=is_error)
    content: list[dict[str, Any]] = [{"type": "text", "text": text_out}]
    for img in images:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.get("mimeType") or "image/png",
                    "data": img.get("data") or "",
                },
            }
        )
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if is_error:
        block["is_error"] = True
    return block


def _finalize_anthropic_cancel(
    working_messages: list[Any],
    *,
    partial_text: str,
    completed_tool_results: list[dict[str, Any]],
    pending_tool_ids: list[str],
) -> None:
    last = working_messages[-1] if working_messages else None
    if isinstance(last, dict) and last.get("role") == "assistant" and pending_tool_ids:
        done_ids = {
            block.get("tool_use_id")
            for block in completed_tool_results
            if isinstance(block, dict)
        }
        results = list(completed_tool_results)
        for tc_id in pending_tool_ids:
            if tc_id not in done_ids:
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc_id,
                        "is_error": True,
                        "content": [{"type": "text", "text": "Cancelled by user."}],
                    }
                )
        working_messages.append({"role": "user", "content": results})
        working_messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "(Interrupted by user.)"}],
            }
        )
        return
    if isinstance(last, dict) and last.get("role") == "user":
        suffix = (
            f"{partial_text}\n\n(Interrupted by user.)"
            if partial_text
            else "(Interrupted by user.)"
        )
        working_messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": suffix}]}
        )


async def run_anthropic_http_turn(
    prompt: str,
    *,
    api_key: str,
    system_prompt: str,
    tools: Sequence[ToolSpec],
    workspace_path: Path,
    model: str,
    effort: str | None,
    resume_session_id: str | None = None,
    tool_timeout_s: int = 660,
    model_info: Mapping[str, Any] | None = None,
    http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
) -> AsyncIterator[TurnEvent]:
    """Run one streaming Anthropic Messages API turn with multi-step tool calls."""
    store = HttpThreadStore(workspace_path, "anthropic")
    session_id, saved_messages, _, saved_window = _resume_state(
        store, resume_session_id
    )
    info = (
        dict(model_info)
        if model_info is not None
        else _anthropic_capabilities(model, {})
    )
    context_window = (
        info.get("context_window")
        if isinstance(info.get("context_window"), int) and info["context_window"] > 0
        else saved_window
    )
    max_tokens_meta = info.get("max_tokens")
    max_tokens = (
        min(max_tokens_meta, 16384)
        if isinstance(max_tokens_meta, int) and max_tokens_meta > 0
        else 16384
    )
    base = _anthropic_base_url()
    headers = _anthropic_headers(api_key)
    tool_by_name = {spec.name: spec for spec in tools}
    tool_defs = [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
        }
        for spec in tools
    ]

    yield SessionStarted(session_id)

    working_messages: list[Any] = [*saved_messages, {"role": "user", "content": prompt}]
    total_usage: dict[str, int] = {}
    last_context_usage: dict[str, Any] | None = None
    made_progress = False
    partial_text_chunks: list[str] = []
    completed_tool_results: list[dict[str, Any]] = []
    pending_tool_ids: list[str] = []

    try:
        async with make_http_client(
            http_client_factory, timeout=HTTP_TIMEOUT
        ) as client:
            for round_idx in range(1, MAX_TOOL_ROUNDS + 1):
                include_effort = bool(effort)
                partial_text_chunks = []
                completed_tool_results = []
                pending_tool_ids = []
                assistant_blocks: list[dict[str, Any]] = []
                round_usage: dict[str, Any] = {}
                round_msg_id: str | None = None
                round_model: str = model

                for effort_attempt in range(2):
                    body: dict[str, Any] = {
                        "model": model,
                        "max_tokens": max_tokens,
                        "system": system_prompt,
                        "messages": working_messages,
                        "stream": True,
                    }
                    if tool_defs:
                        body["tools"] = tool_defs
                    if include_effort and effort:
                        if info.get("thinking_adaptive"):
                            body["thinking"] = {"type": "adaptive"}
                        body["output_config"] = {"effort": effort}

                    blocks_by_index: dict[int, dict[str, Any]] = {}
                    open_text_indices: set[int] = set()
                    round_started = False
                    retry_without_effort = False

                    async with client.stream(
                        "POST", f"{base}/v1/messages", headers=headers, json=body
                    ) as response:
                        if response.status_code != 200:
                            raw_err = await response.aread()
                            err_msg = http_error_message("Anthropic", response, raw_err)
                            lowered = err_msg.lower()
                            if (
                                response.status_code == 400
                                and effort_attempt == 0
                                and include_effort
                                and any(
                                    k in lowered
                                    for k in ("output_config", "thinking", "effort")
                                )
                            ):
                                include_effort = False
                                retry_without_effort = True
                                continue
                            raise RuntimeError(err_msg)

                        async for event_name, raw_data in iter_sse_events(response):
                            if not raw_data or raw_data == "[DONE]":
                                continue
                            try:
                                data = json.loads(raw_data)
                            except json.JSONDecodeError:
                                continue
                            if not isinstance(data, dict):
                                continue
                            ev_type = data.get("type") or event_name

                            if ev_type == "error":
                                err = data.get("error")
                                detail = (
                                    err.get("message")
                                    if isinstance(err, dict)
                                    else str(err or "Anthropic stream error.")
                                )
                                raise RuntimeError(detail or "Anthropic stream error.")

                            if ev_type == "message_start":
                                msg_obj = data.get("message") or {}
                                if isinstance(msg_obj, dict):
                                    round_msg_id = msg_obj.get("id") or round_msg_id
                                    round_model = msg_obj.get("model") or round_model
                                    _merge_usage(round_usage, msg_obj.get("usage"))
                                if not round_started:
                                    round_started = True
                                    made_progress = True
                                    yield TurnStarted()
                            elif ev_type == "content_block_start":
                                if not round_started:
                                    round_started = True
                                    made_progress = True
                                    yield TurnStarted()
                                idx = data.get("index")
                                cb = data.get("content_block")
                                if isinstance(idx, int) and isinstance(cb, dict):
                                    cb_type = cb.get("type")
                                    if cb_type == "text":
                                        blocks_by_index[idx] = {
                                            "type": "text",
                                            "text": str(cb.get("text") or ""),
                                        }
                                        open_text_indices.add(idx)
                                    elif cb_type == "thinking":
                                        blocks_by_index[idx] = {
                                            "type": "thinking",
                                            "thinking": str(cb.get("thinking") or ""),
                                            "signature": str(cb.get("signature") or ""),
                                        }
                                    elif cb_type == "redacted_thinking":
                                        blocks_by_index[idx] = {
                                            "type": "redacted_thinking",
                                            "data": cb.get("data") or "",
                                        }
                                    elif cb_type == "tool_use":
                                        initial_input = cb.get("input")
                                        blocks_by_index[idx] = {
                                            "type": "tool_use",
                                            "id": cb.get("id") or "",
                                            "name": cb.get("name") or "",
                                            "input": initial_input
                                            if isinstance(initial_input, dict)
                                            else {},
                                            "_partial_json": [],
                                        }
                            elif ev_type == "content_block_delta":
                                idx = data.get("index")
                                delta = data.get("delta")
                                if not isinstance(idx, int) or not isinstance(
                                    delta, dict
                                ):
                                    continue
                                block = blocks_by_index.get(idx)
                                if block is None:
                                    continue
                                d_type = delta.get("type")
                                if d_type == "text_delta" and block["type"] == "text":
                                    chunk = delta.get("text")
                                    if isinstance(chunk, str) and chunk:
                                        block["text"] += chunk
                                        partial_text_chunks.append(chunk)
                                        yield TextDelta(chunk)
                                elif (
                                    d_type == "thinking_delta"
                                    and block["type"] == "thinking"
                                ):
                                    chunk = delta.get("thinking")
                                    if isinstance(chunk, str):
                                        block["thinking"] += chunk
                                elif (
                                    d_type == "signature_delta"
                                    and block["type"] == "thinking"
                                ):
                                    sig = delta.get("signature")
                                    if isinstance(sig, str):
                                        block["signature"] += sig
                                elif (
                                    d_type == "input_json_delta"
                                    and block["type"] == "tool_use"
                                ):
                                    part = delta.get("partial_json")
                                    if isinstance(part, str):
                                        block["_partial_json"].append(part)
                            elif ev_type == "content_block_stop":
                                idx = data.get("index")
                                if not isinstance(idx, int):
                                    continue
                                block = blocks_by_index.get(idx)
                                if block is None:
                                    continue
                                if block["type"] == "text" and idx in open_text_indices:
                                    open_text_indices.discard(idx)
                                    if block["text"]:
                                        yield TextBlockDone(block["text"])
                                elif block["type"] == "tool_use":
                                    raw_json = "".join(block.pop("_partial_json", []))
                                    if raw_json:
                                        try:
                                            parsed = json.loads(raw_json)
                                            if isinstance(parsed, dict):
                                                block["input"] = parsed
                                            else:
                                                block["_json_error"] = (
                                                    "Tool arguments must decode to a JSON object."
                                                )
                                        except json.JSONDecodeError as exc:
                                            block["_json_error"] = (
                                                f"Invalid JSON tool arguments: {exc}"
                                            )
                            elif ev_type == "message_delta":
                                _merge_usage(round_usage, data.get("usage"))

                    if not retry_without_effort:
                        for idx in sorted(open_text_indices):
                            block = blocks_by_index[idx]
                            if block.get("text"):
                                yield TextBlockDone(block["text"])
                        for idx in sorted(blocks_by_index):
                            block = blocks_by_index[idx]
                            if (
                                block.get("type") == "tool_use"
                                and "_partial_json" in block
                            ):
                                raw_json = "".join(block.pop("_partial_json", []))
                                if raw_json:
                                    try:
                                        parsed = json.loads(raw_json)
                                        if isinstance(parsed, dict):
                                            block["input"] = parsed
                                        else:
                                            block["_json_error"] = (
                                                "Tool arguments must decode to a JSON object."
                                            )
                                    except json.JSONDecodeError as exc:
                                        block["_json_error"] = (
                                            f"Invalid JSON tool arguments: {exc}"
                                        )
                            assistant_blocks.append(block)
                        break

                if round_usage:
                    yield MessageUsage(
                        message_id=round_msg_id,
                        model=round_model,
                        usage=dict(round_usage),
                    )
                    _add_int_usage(total_usage, round_usage)
                    prompt_tokens = _anthropic_prompt_tokens(round_usage)
                    if prompt_tokens is not None:
                        last_context_usage = {
                            "used_tokens": prompt_tokens,
                            "window_tokens": context_window,
                            "model": round_model,
                        }

                tool_blocks = [
                    block
                    for block in assistant_blocks
                    if block.get("type") == "tool_use"
                ]
                stored_blocks = [
                    {k: v for k, v in block.items() if not k.startswith("_")}
                    for block in assistant_blocks
                ]
                if not stored_blocks:
                    stored_blocks = [{"type": "text", "text": ""}]
                working_messages.append({"role": "assistant", "content": stored_blocks})

                if not tool_blocks:
                    store.save(
                        session_id,
                        model=model,
                        messages=working_messages,
                        context_window=context_window,
                    )
                    yield TurnResult(
                        usage=total_usage or None,
                        context_usage=last_context_usage,
                        num_turns=round_idx,
                    )
                    return

                pending_tool_ids = [str(block.get("id") or "") for block in tool_blocks]
                completed_tool_results = []
                for block in tool_blocks:
                    tc_id = str(block.get("id") or "")
                    tc_name = str(block.get("name") or "")
                    if "_json_error" in block:
                        result = {
                            "content": [{"type": "text", "text": block["_json_error"]}],
                            "is_error": True,
                        }
                    else:
                        result = await execute_tool_call(
                            tool_by_name, tc_name, block.get("input"), tool_timeout_s
                        )
                    completed_tool_results.append(
                        _anthropic_tool_result_block(tc_id, result)
                    )
                pending_tool_ids = []
                working_messages.append(
                    {"role": "user", "content": completed_tool_results}
                )
                completed_tool_results = []
                store.save(
                    session_id,
                    model=model,
                    messages=working_messages,
                    context_window=context_window,
                )
        raise RuntimeError(
            f"Anthropic turn exceeded maximum tool rounds ({MAX_TOOL_ROUNDS})."
        )
    except (asyncio.CancelledError, GeneratorExit):
        if made_progress:
            _finalize_anthropic_cancel(
                working_messages,
                partial_text="".join(partial_text_chunks),
                completed_tool_results=completed_tool_results,
                pending_tool_ids=pending_tool_ids,
            )
            store.save(
                session_id,
                model=model,
                messages=working_messages,
                context_window=context_window,
            )
        raise


async def compact_anthropic_http(
    *,
    api_key: str,
    system_prompt: str,
    workspace_path: Path,
    model: str | None,
    resume_session_id: str,
    model_info: Mapping[str, Any] | None = None,
    http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
) -> CompactResult:
    """Compact an HTTP Anthropic conversation by summarizing saved messages."""
    store = HttpThreadStore(workspace_path, "anthropic")
    saved = store.load(resume_session_id)
    target_model = model or saved.get("model")
    if not isinstance(target_model, str) or not target_model:
        raise RuntimeError("No model is available to compact this session.")
    context_window = (
        model_info.get("context_window")
        if model_info and isinstance(model_info.get("context_window"), int)
        else saved.get("context_window")
        if isinstance(saved.get("context_window"), int)
        else None
    )

    sanitized: list[dict[str, Any]] = []
    for msg in saved["messages"]:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            sanitized.append(msg)
            continue
        clean_blocks: list[Any] = []
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and isinstance(block.get("content"), list)
            ):
                new_items = [
                    (
                        {
                            "type": "text",
                            "text": "[plot image omitted during compaction]",
                        }
                        if isinstance(item, dict) and item.get("type") == "image"
                        else item
                    )
                    for item in block["content"]
                ]
                clean_blocks.append({**block, "content": new_items})
            else:
                clean_blocks.append(block)
        sanitized.append({**msg, "content": clean_blocks})

    base = _anthropic_base_url()
    headers = _anthropic_headers(api_key)
    body = {
        "model": target_model,
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": [*sanitized, {"role": "user", "content": COMPACTION_PROMPT}],
    }
    async with make_http_client(http_client_factory, timeout=HTTP_TIMEOUT) as client:
        response = await client.post(f"{base}/v1/messages", headers=headers, json=body)
        if response.status_code != 200:
            raise RuntimeError(
                http_error_message("Anthropic", response, response.content)
            )
        payload = response.json()

    content = payload.get("content") if isinstance(payload, dict) else None
    summary_parts = [
        str(block.get("text") or "")
        for block in (content or [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    summary = "\n\n".join(part for part in summary_parts if part.strip()).strip()
    if not summary:
        raise RuntimeError("Anthropic did not return a compaction summary.")

    compacted_messages = [
        {
            "role": "user",
            "content": "Summary of our earlier conversation before compaction:",
        },
        {"role": "assistant", "content": [{"type": "text", "text": summary}]},
    ]
    store.save(
        resume_session_id,
        model=target_model,
        messages=compacted_messages,
        context_window=context_window,
    )
    output_tokens = (
        (payload.get("usage") or {}).get("output_tokens")
        if isinstance(payload, dict)
        else None
    )
    context_usage = (
        {
            "used_tokens": output_tokens + 20,
            "window_tokens": context_window,
            "model": target_model,
        }
        if isinstance(output_tokens, int) and output_tokens >= 0
        else None
    )
    return CompactResult(
        session_id=resume_session_id,
        context_usage=context_usage,
        record_reloaded=False,
    )


def _merge_usage(target: dict[str, Any], update: Any) -> None:
    if isinstance(update, Mapping):
        for key, value in update.items():
            if value is not None:
                target[key] = value


# ---------------------------------------------------------------------------
# OpenAI Responses API
# ---------------------------------------------------------------------------


def _openai_base_url() -> str:
    return (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip(
        "/"
    )


def _openai_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


async def discover_openai_http_models(
    api_key: str,
    *,
    http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
) -> list[ModelSpec]:
    """List chat/reasoning models via ``GET /models``."""
    base = _openai_base_url()
    headers = _openai_headers(api_key)
    try:
        async with (
            asyncio.timeout(HTTP_DISCOVERY_TIMEOUT_S),
            make_http_client(
                http_client_factory, timeout=HTTP_DISCOVERY_TIMEOUT_S
            ) as client,
        ):
            response = await client.get(f"{base}/models", headers=headers)
            if response.status_code != 200:
                logger.warning(
                    "OpenAI HTTP model discovery failed: %s",
                    http_error_message("OpenAI", response, response.content),
                )
                return []
            payload = response.json()
    except Exception as exc:
        logger.warning("OpenAI HTTP model discovery failed: %s", exc)
        return []

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []

    filtered: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        if not isinstance(mid, str) or not _OPENAI_MODEL_ALLOW_RE.match(mid):
            continue
        lowered = mid.lower()
        if any(marker in lowered for marker in _OPENAI_MODEL_EXCLUDE):
            continue
        if mid not in seen_ids:
            seen_ids.add(mid)
            filtered.append(item)

    if not filtered:
        return []

    filtered.sort(
        key=lambda item: (
            -(item.get("created") if isinstance(item.get("created"), int) else 0),
            item["id"],
        )
    )
    ids = [item["id"] for item in filtered]
    default_id = next(
        (cand for cand in _OPENAI_DEFAULT_ORDER if cand in seen_ids), ids[0]
    )

    specs: list[ModelSpec] = []
    for mid in ids:
        efforts = (
            ["low", "medium", "high"]
            if _OPENAI_REASONING_RE.match(mid) and "chat-latest" not in mid
            else []
        )
        specs.append(
            ModelSpec(
                id=mid,
                raw_id=mid,
                provider="openai",
                label=mid,
                efforts=efforts,
                is_default=(mid == default_id),
            )
        )
    return specs


def _openai_tool_output(call_id: str, result: dict[str, Any]) -> dict[str, Any]:
    is_error = bool(result.get("is_error"))
    text_out, images = _tool_text_and_images(result, is_error=is_error)
    if images:
        payload: Any = [{"type": "input_text", "text": text_out}]
        for img in images:
            mime = img.get("mimeType") or "image/png"
            b64 = img.get("data") or ""
            payload.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime};base64,{b64}",
                    "detail": "auto",
                }
            )
    else:
        payload = text_out
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": payload,
    }


def _store_openai_output_item(
    item: Mapping[str, Any],
    *,
    working_items: list[Any],
    pending_calls: list[dict[str, Any]],
) -> None:
    item_type = item.get("type")
    if item_type == "reasoning":
        encrypted = item.get("encrypted_content")
        if isinstance(encrypted, str) and encrypted:
            stored: dict[str, Any] = {
                "type": "reasoning",
                "summary": item.get("summary")
                if isinstance(item.get("summary"), list)
                else [],
                "encrypted_content": encrypted,
            }
            if isinstance(item.get("id"), str) and item["id"]:
                stored["id"] = item["id"]
            working_items.append(stored)
    elif item_type == "message":
        content = item.get("content")
        working_items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": list(content) if isinstance(content, list) else [],
            }
        )
    elif item_type == "function_call":
        call_item = {
            "type": "function_call",
            "call_id": str(item.get("call_id") or ""),
            "name": str(item.get("name") or ""),
            "arguments": str(item.get("arguments") or "{}"),
        }
        working_items.append(call_item)
        pending_calls.append(call_item)


def _finalize_openai_cancel(
    working_items: list[Any],
    *,
    partial_text: str,
    completed_outputs: list[dict[str, Any]],
    pending_calls: list[dict[str, Any]],
) -> None:
    if pending_calls:
        done_ids = {
            item.get("call_id") for item in completed_outputs if isinstance(item, dict)
        }
        working_items.extend(completed_outputs)
        for call in pending_calls:
            cid = call.get("call_id")
            if cid not in done_ids:
                working_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": cid,
                        "output": "Error: Cancelled by user.",
                    }
                )
        working_items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "(Interrupted by user.)"}],
            }
        )
        return
    last = working_items[-1] if working_items else None
    needs_reply = not isinstance(last, dict) or last.get("role") != "assistant"
    if needs_reply:
        suffix = (
            f"{partial_text}\n\n(Interrupted by user.)"
            if partial_text
            else "(Interrupted by user.)"
        )
        working_items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": suffix}],
            }
        )


async def run_openai_http_turn(
    prompt: str,
    *,
    api_key: str,
    system_prompt: str,
    tools: Sequence[ToolSpec],
    workspace_path: Path,
    model: str,
    effort: str | None,
    resume_session_id: str | None = None,
    tool_timeout_s: int = 660,
    http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
) -> AsyncIterator[TurnEvent]:
    """Run one streaming OpenAI Responses API turn with ``store=False``."""
    store = HttpThreadStore(workspace_path, "openai")
    session_id, saved_items, _, _ = _resume_state(store, resume_session_id)
    base = _openai_base_url()
    headers = _openai_headers(api_key)
    tool_by_name = {spec.name: spec for spec in tools}
    tool_defs = [
        {
            "type": "function",
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
            "strict": False,
        }
        for spec in tools
    ]

    yield SessionStarted(session_id)

    working_items: list[Any] = [*saved_items, {"role": "user", "content": prompt}]
    total_usage: dict[str, int] = {}
    last_context_usage: dict[str, Any] | None = None
    made_progress = False
    partial_text_chunks: list[str] = []
    pending_calls: list[dict[str, Any]] = []
    completed_outputs: list[dict[str, Any]] = []

    try:
        async with make_http_client(
            http_client_factory, timeout=HTTP_TIMEOUT
        ) as client:
            for round_idx in range(1, MAX_TOOL_ROUNDS + 1):
                include_effort = bool(effort)
                partial_text_chunks = []
                pending_calls = []
                completed_outputs = []
                round_usage: dict[str, Any] | None = None
                round_msg_id: str | None = None
                round_model = model

                for effort_attempt in range(2):
                    body: dict[str, Any] = {
                        "model": model,
                        "instructions": system_prompt,
                        "input": working_items,
                        "store": False,
                        "stream": True,
                    }
                    if tool_defs:
                        body["tools"] = tool_defs
                    if include_effort and effort:
                        body["reasoning"] = {"effort": effort}
                        body["include"] = ["reasoning.encrypted_content"]

                    round_started = False
                    retry_without_reasoning = False
                    text_buffers: dict[tuple[str, int], list[str]] = {}
                    emitted_parts: set[tuple[str, int]] = set()
                    emitted_fallback_texts: set[str] = set()
                    saw_done_items = False

                    async with client.stream(
                        "POST", f"{base}/responses", headers=headers, json=body
                    ) as response:
                        if response.status_code != 200:
                            raw_err = await response.aread()
                            err_msg = http_error_message("OpenAI", response, raw_err)
                            lowered = err_msg.lower()
                            if (
                                response.status_code == 400
                                and effort_attempt == 0
                                and include_effort
                                and any(
                                    k in lowered
                                    for k in (
                                        "reasoning",
                                        "encrypted_content",
                                        "include",
                                        "effort",
                                    )
                                )
                            ):
                                include_effort = False
                                retry_without_reasoning = True
                                continue
                            raise RuntimeError(err_msg)

                        async for event_name, raw_data in iter_sse_events(response):
                            if not raw_data or raw_data == "[DONE]":
                                continue
                            try:
                                data = json.loads(raw_data)
                            except json.JSONDecodeError:
                                continue
                            if not isinstance(data, dict):
                                continue
                            ev_type = data.get("type") or event_name

                            if ev_type == "error":
                                err = data.get("error")
                                msg_text = (
                                    err.get("message")
                                    if isinstance(err, dict)
                                    else data.get("message")
                                )
                                raise RuntimeError(
                                    str(msg_text or "OpenAI API stream error.")
                                )
                            if ev_type in ("response.failed", "response.incomplete"):
                                resp = data.get("response") or {}
                                if ev_type == "response.failed":
                                    err = (
                                        resp.get("error")
                                        if isinstance(resp, dict)
                                        else None
                                    )
                                    detail = (
                                        err.get("message")
                                        if isinstance(err, dict)
                                        else None
                                    )
                                    raise RuntimeError(
                                        str(detail or "OpenAI response failed.")
                                    )
                                details = (
                                    resp.get("incomplete_details")
                                    if isinstance(resp, dict)
                                    else None
                                )
                                reason = (
                                    details.get("reason")
                                    if isinstance(details, dict)
                                    else "incomplete"
                                )
                                raise RuntimeError(
                                    f"OpenAI response stopped early ({reason})."
                                )

                            if ev_type in ("response.created", "response.in_progress"):
                                resp = data.get("response") or {}
                                if isinstance(resp, dict):
                                    round_msg_id = resp.get("id") or round_msg_id
                                    round_model = resp.get("model") or round_model
                                if not round_started:
                                    round_started = True
                                    made_progress = True
                                    yield TurnStarted()
                            elif ev_type == "response.output_text.delta":
                                if not round_started:
                                    round_started = True
                                    made_progress = True
                                    yield TurnStarted()
                                delta = data.get("delta")
                                if isinstance(delta, str) and delta:
                                    key = (
                                        str(data.get("item_id") or ""),
                                        int(data.get("content_index") or 0),
                                    )
                                    text_buffers.setdefault(key, []).append(delta)
                                    partial_text_chunks.append(delta)
                                    yield TextDelta(delta)
                            elif ev_type == "response.output_text.done":
                                key = (
                                    str(data.get("item_id") or ""),
                                    int(data.get("content_index") or 0),
                                )
                                buf_text = "".join(text_buffers.pop(key, []))
                                full_text = (
                                    data.get("text")
                                    if isinstance(data.get("text"), str)
                                    and data["text"]
                                    else buf_text
                                )
                                if full_text and key not in emitted_parts:
                                    emitted_parts.add(key)
                                    emitted_fallback_texts.add(full_text)
                                    yield TextBlockDone(full_text)
                            elif ev_type == "response.output_item.done":
                                item = data.get("item")
                                if not isinstance(item, dict):
                                    continue
                                saw_done_items = True
                                item_id = str(item.get("id") or "")
                                if item.get("type") == "message":
                                    content = item.get("content") or []
                                    if isinstance(content, list):
                                        for c_idx, part in enumerate(content):
                                            if (
                                                isinstance(part, dict)
                                                and part.get("type") == "output_text"
                                            ):
                                                key = (item_id, c_idx)
                                                part_text = str(part.get("text") or "")
                                                if (
                                                    part_text
                                                    and key not in emitted_parts
                                                ):
                                                    emitted_parts.add(key)
                                                    emitted_fallback_texts.add(
                                                        part_text
                                                    )
                                                    yield TextBlockDone(part_text)
                                _store_openai_output_item(
                                    item,
                                    working_items=working_items,
                                    pending_calls=pending_calls,
                                )
                            elif ev_type == "response.completed":
                                resp = data.get("response") or {}
                                if isinstance(resp, dict):
                                    round_msg_id = resp.get("id") or round_msg_id
                                    round_model = resp.get("model") or round_model
                                    if isinstance(resp.get("usage"), dict):
                                        round_usage = dict(resp["usage"])
                                    if not saw_done_items and isinstance(
                                        resp.get("output"), list
                                    ):
                                        for out_item in resp["output"]:
                                            if not isinstance(out_item, dict):
                                                continue
                                            if out_item.get("type") == "message":
                                                for part in (
                                                    out_item.get("content") or ()
                                                ):
                                                    if (
                                                        isinstance(part, dict)
                                                        and part.get("type")
                                                        == "output_text"
                                                    ):
                                                        part_text = str(
                                                            part.get("text") or ""
                                                        )
                                                        if (
                                                            part_text
                                                            and part_text
                                                            not in emitted_fallback_texts
                                                        ):
                                                            emitted_fallback_texts.add(
                                                                part_text
                                                            )
                                                            yield TextBlockDone(
                                                                part_text
                                                            )
                                            _store_openai_output_item(
                                                out_item,
                                                working_items=working_items,
                                                pending_calls=pending_calls,
                                            )

                    if not retry_without_reasoning:
                        for key, chunks in sorted(text_buffers.items()):
                            leftover = "".join(chunks)
                            if leftover and key not in emitted_parts:
                                emitted_parts.add(key)
                                yield TextBlockDone(leftover)
                        break

                if round_usage is not None:
                    yield MessageUsage(
                        message_id=round_msg_id,
                        model=round_model,
                        usage=round_usage,
                    )
                    _add_int_usage(total_usage, round_usage)
                    inp = round_usage.get("input_tokens")
                    if isinstance(inp, int) and not isinstance(inp, bool) and inp >= 0:
                        last_context_usage = {
                            "used_tokens": inp,
                            "window_tokens": None,
                            "model": round_model,
                        }

                if not pending_calls:
                    store.save(session_id, model=model, messages=working_items)
                    yield TurnResult(
                        usage=total_usage or None,
                        context_usage=last_context_usage,
                        num_turns=round_idx,
                    )
                    return

                completed_outputs = []
                for call in pending_calls:
                    call_id = call["call_id"]
                    call_name = call["name"]
                    raw_arguments = call["arguments"]
                    try:
                        parsed_args = (
                            json.loads(raw_arguments) if raw_arguments.strip() else {}
                        )
                    except json.JSONDecodeError as exc:
                        result = {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Invalid JSON tool arguments: {exc}",
                                }
                            ],
                            "is_error": True,
                        }
                    else:
                        result = await execute_tool_call(
                            tool_by_name, call_name, parsed_args, tool_timeout_s
                        )
                    completed_outputs.append(_openai_tool_output(call_id, result))
                working_items.extend(completed_outputs)
                completed_outputs = []
                pending_calls = []
                store.save(session_id, model=model, messages=working_items)
        raise RuntimeError(
            f"OpenAI turn exceeded maximum tool rounds ({MAX_TOOL_ROUNDS})."
        )
    except (asyncio.CancelledError, GeneratorExit):
        if made_progress:
            _finalize_openai_cancel(
                working_items,
                partial_text="".join(partial_text_chunks),
                completed_outputs=completed_outputs,
                pending_calls=pending_calls,
            )
            store.save(session_id, model=model, messages=working_items)
        raise


def _strip_openai_images(items: list[Any]) -> list[Any]:
    cleaned: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            cleaned.append(item)
            continue
        if item.get("type") == "function_call_output" and isinstance(
            item.get("output"), list
        ):
            new_out = [
                (
                    {
                        "type": "input_text",
                        "text": "[plot image omitted during compaction]",
                    }
                    if isinstance(part, dict) and part.get("type") == "input_image"
                    else part
                )
                for part in item["output"]
            ]
            cleaned.append({**item, "output": new_out})
        else:
            cleaned.append(item)
    return cleaned


async def compact_openai_http(
    *,
    api_key: str,
    system_prompt: str,
    workspace_path: Path,
    model: str | None,
    resume_session_id: str,
    http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
) -> CompactResult:
    """Compact an HTTP OpenAI conversation via stateless summary replacement."""
    store = HttpThreadStore(workspace_path, "openai")
    saved = store.load(resume_session_id)
    target_model = model or saved.get("model")
    if not isinstance(target_model, str) or not target_model:
        raise RuntimeError("No model is available to compact this session.")

    sanitized = _strip_openai_images(saved["messages"])
    base = _openai_base_url()
    headers = _openai_headers(api_key)
    body = {
        "model": target_model,
        "instructions": system_prompt,
        "input": [*sanitized, {"role": "user", "content": COMPACTION_PROMPT}],
        "store": False,
        "stream": False,
    }
    async with make_http_client(http_client_factory, timeout=HTTP_TIMEOUT) as client:
        response = await client.post(f"{base}/responses", headers=headers, json=body)
        if response.status_code != 200:
            raise RuntimeError(http_error_message("OpenAI", response, response.content))
        payload = response.json()

    summary_parts: list[str] = []
    for item in payload.get("output") or ():
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or ():
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = str(part.get("text") or "")
                if text:
                    summary_parts.append(text)
    if not summary_parts and isinstance(payload.get("output_text"), str):
        summary_parts.append(payload["output_text"])
    summary = "\n\n".join(part for part in summary_parts if part.strip()).strip()
    if not summary:
        raise RuntimeError("OpenAI did not return a compaction summary.")

    compacted_items = [
        {
            "role": "user",
            "content": "Summary of our earlier conversation before compaction:",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": summary}],
        },
    ]
    store.save(resume_session_id, model=target_model, messages=compacted_items)
    output_tokens = (
        (payload.get("usage") or {}).get("output_tokens")
        if isinstance(payload, dict)
        else None
    )
    context_usage = (
        {
            "used_tokens": output_tokens + 20,
            "window_tokens": None,
            "model": target_model,
        }
        if isinstance(output_tokens, int) and output_tokens >= 0
        else None
    )
    return CompactResult(
        session_id=resume_session_id,
        context_usage=context_usage,
        record_reloaded=False,
    )


# ---------------------------------------------------------------------------
# Google Gemini REST API
# ---------------------------------------------------------------------------


def _google_base_url() -> str:
    return (
        os.environ.get("GEMINI_API_BASE_URL")
        or "https://generativelanguage.googleapis.com/v1beta"
    ).rstrip("/")


def _google_headers(api_key: str) -> dict[str, str]:
    return {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }


async def discover_google_http_models(
    api_key: str,
    *,
    http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
) -> tuple[list[ModelSpec], dict[str, dict[str, Any]]]:
    """List Gemini models via ``GET /models``."""
    base = _google_base_url()
    headers = _google_headers(api_key)
    raw_items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    page_token: str | None = None
    try:
        async with (
            asyncio.timeout(HTTP_DISCOVERY_TIMEOUT_S),
            make_http_client(
                http_client_factory, timeout=HTTP_DISCOVERY_TIMEOUT_S
            ) as client,
        ):
            for _ in range(5):
                params: dict[str, str] = {"pageSize": "100"}
                if page_token is not None:
                    params["pageToken"] = page_token
                response = await client.get(
                    f"{base}/models", headers=headers, params=params
                )
                if response.status_code != 200:
                    logger.warning(
                        "Google HTTP model discovery failed: %s",
                        http_error_message("Google", response, response.content),
                    )
                    return [], {}
                payload = response.json()
                models = payload.get("models") if isinstance(payload, dict) else None
                if not isinstance(models, list):
                    break
                for item in models:
                    if not isinstance(item, dict):
                        continue
                    methods = item.get("supportedGenerationMethods") or ()
                    if "generateContent" not in methods:
                        continue
                    raw_name = item.get("name")
                    if not isinstance(raw_name, str):
                        continue
                    raw_id = raw_name.removeprefix("models/")
                    if not raw_id.startswith("gemini-"):
                        continue
                    lowered = raw_id.lower()
                    if any(marker in lowered for marker in _GOOGLE_MODEL_EXCLUDE):
                        continue
                    if raw_id not in seen_ids:
                        seen_ids.add(raw_id)
                        raw_items.append({"raw_id": raw_id, **item})
                next_token = payload.get("nextPageToken")
                if not isinstance(next_token, str) or not next_token:
                    break
                page_token = next_token
    except Exception as exc:
        logger.warning("Google HTTP model discovery failed: %s", exc)
        return [], {}

    if not raw_items:
        return [], {}

    default_id = next(
        (item["raw_id"] for item in raw_items if "lite" not in item["raw_id"].lower()),
        raw_items[0]["raw_id"],
    )
    specs: list[ModelSpec] = []
    model_info: dict[str, dict[str, Any]] = {}
    for item in raw_items:
        raw_id = item["raw_id"]
        efforts = ["low", "medium", "high"] if _GOOGLE_EFFORT_RE.match(raw_id) else []
        limit = item.get("inputTokenLimit")
        context_window = (
            limit
            if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0
            else None
        )
        model_info[raw_id] = {"context_window": context_window}
        label = item.get("displayName")
        specs.append(
            ModelSpec(
                id=raw_id,
                raw_id=raw_id,
                provider="google",
                label=str(label).strip()
                if isinstance(label, str) and label.strip()
                else raw_id,
                efforts=efforts,
                is_default=(raw_id == default_id),
            )
        )
    return specs, model_info


def _append_google_model_part(
    parts: list[dict[str, Any]], part: dict[str, Any]
) -> None:
    if (
        parts
        and set(part.keys()) == {"text"}
        and set(parts[-1].keys()) == {"text"}
        and isinstance(part["text"], str)
    ):
        parts[-1]["text"] += part["text"]
        return
    if (
        parts
        and set(part.keys()) == {"text", "thought"}
        and part.get("thought") is True
        and set(parts[-1].keys()) == {"text", "thought"}
        and parts[-1].get("thought") is True
        and isinstance(part["text"], str)
    ):
        parts[-1]["text"] += part["text"]
        return
    parts.append(dict(part))


def _google_function_response_part(
    call: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    is_error = bool(result.get("is_error"))
    text_out, images = _tool_text_and_images(result, is_error=is_error)
    resp_obj: dict[str, Any] = {
        "name": call["name"],
        "response": {"error": text_out} if is_error else {"output": text_out},
    }
    if call.get("id") is not None:
        resp_obj["id"] = call["id"]
    if images:
        resp_obj["parts"] = [
            {
                "inlineData": {
                    "mimeType": img.get("mimeType") or "image/png",
                    "data": img.get("data") or "",
                }
            }
            for img in images
        ]
    return {"functionResponse": resp_obj}


def _finalize_google_cancel(
    working_contents: list[Any],
    *,
    partial_text: str,
    completed_responses: list[dict[str, Any]],
    pending_calls: list[dict[str, Any]],
) -> None:
    last = working_contents[-1] if working_contents else None
    if isinstance(last, dict) and last.get("role") == "model" and pending_calls:
        responses = list(completed_responses)
        for call in pending_calls[len(completed_responses) :]:
            err_resp: dict[str, Any] = {
                "name": call["name"],
                "response": {"error": "Error: Cancelled by user."},
            }
            if call.get("id") is not None:
                err_resp["id"] = call["id"]
            responses.append({"functionResponse": err_resp})
        working_contents.append({"role": "user", "parts": responses})
        working_contents.append(
            {"role": "model", "parts": [{"text": "(Interrupted by user.)"}]}
        )
        return
    if isinstance(last, dict) and last.get("role") == "user":
        suffix = (
            f"{partial_text}\n\n(Interrupted by user.)"
            if partial_text
            else "(Interrupted by user.)"
        )
        working_contents.append({"role": "model", "parts": [{"text": suffix}]})


async def run_google_http_turn(
    prompt: str,
    *,
    api_key: str,
    system_prompt: str,
    tools: Sequence[ToolSpec],
    workspace_path: Path,
    model: str,
    effort: str | None,
    resume_session_id: str | None = None,
    tool_timeout_s: int = 660,
    model_info: Mapping[str, Any] | None = None,
    http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
) -> AsyncIterator[TurnEvent]:
    """Run one streaming Gemini REST API turn with multi-step tool calls."""
    store = HttpThreadStore(workspace_path, "google")
    session_id, saved_contents, _, saved_window = _resume_state(
        store, resume_session_id
    )
    info_window = (
        model_info.get("context_window") if isinstance(model_info, Mapping) else None
    )
    context_window = (
        info_window
        if isinstance(info_window, int) and info_window > 0
        else saved_window
    )
    base = _google_base_url()
    headers = _google_headers(api_key)
    tool_by_name = {spec.name: spec for spec in tools}
    tool_defs = (
        [
            {
                "functionDeclarations": [
                    {
                        "name": spec.name,
                        "description": spec.description,
                        "parametersJsonSchema": spec.input_schema,
                    }
                    for spec in tools
                ]
            }
        ]
        if tools
        else None
    )

    yield SessionStarted(session_id)

    working_contents: list[Any] = [
        *saved_contents,
        {"role": "user", "parts": [{"text": prompt}]},
    ]
    total_usage: dict[str, int] = {}
    last_context_usage: dict[str, Any] | None = None
    made_progress = False
    partial_text_chunks: list[str] = []
    pending_calls: list[dict[str, Any]] = []
    completed_responses: list[dict[str, Any]] = []

    try:
        async with make_http_client(
            http_client_factory, timeout=HTTP_TIMEOUT
        ) as client:
            for round_idx in range(1, MAX_TOOL_ROUNDS + 1):
                include_thinking = bool(effort)
                partial_text_chunks = []
                pending_calls = []
                completed_responses = []
                round_model_parts: list[dict[str, Any]] = []
                round_usage: dict[str, Any] | None = None

                for thinking_attempt in range(2):
                    body: dict[str, Any] = {
                        "systemInstruction": {"parts": [{"text": system_prompt}]},
                        "contents": working_contents,
                    }
                    if tool_defs is not None:
                        body["tools"] = tool_defs
                    if include_thinking and effort:
                        if model.startswith("gemini-2.5"):
                            budget = _GEMINI_25_THINKING_BUDGETS.get(effort, 8192)
                            body["generationConfig"] = {
                                "thinkingConfig": {"thinkingBudget": budget}
                            }
                        else:
                            body["generationConfig"] = {
                                "thinkingConfig": {"thinkingLevel": effort.upper()}
                            }

                    round_started = False
                    retry_without_thinking = False
                    current_text_block: list[str] = []
                    round_model_parts = []
                    pending_calls = []
                    round_usage = None

                    async with client.stream(
                        "POST",
                        f"{base}/models/{model}:streamGenerateContent?alt=sse",
                        headers=headers,
                        json=body,
                    ) as response:
                        if response.status_code != 200:
                            raw_err = await response.aread()
                            err_msg = http_error_message("Google", response, raw_err)
                            lowered = err_msg.lower()
                            if (
                                response.status_code == 400
                                and thinking_attempt == 0
                                and include_thinking
                                and any(
                                    k in lowered
                                    for k in (
                                        "thinkingconfig",
                                        "thinkinglevel",
                                        "thinkingbudget",
                                        "thinking_config",
                                        "thinking_level",
                                        "thinking_budget",
                                    )
                                )
                            ):
                                include_thinking = False
                                retry_without_thinking = True
                                continue
                            raise RuntimeError(err_msg)

                        async for _, raw_data in iter_sse_events(response):
                            if not raw_data or raw_data == "[DONE]":
                                continue
                            try:
                                data = json.loads(raw_data)
                            except json.JSONDecodeError:
                                continue
                            if not isinstance(data, dict):
                                continue
                            if "error" in data and data["error"] is not None:
                                err = data["error"]
                                msg_text = (
                                    err.get("message")
                                    if isinstance(err, dict)
                                    else str(err)
                                )
                                raise RuntimeError(
                                    str(msg_text or "Google Gemini stream error.")
                                )

                            if not round_started:
                                round_started = True
                                made_progress = True
                                yield TurnStarted()

                            if isinstance(data.get("usageMetadata"), dict):
                                round_usage = dict(data["usageMetadata"])

                            candidates = data.get("candidates")
                            if not candidates:
                                prompt_fb = data.get("promptFeedback")
                                if isinstance(prompt_fb, dict) and prompt_fb.get(
                                    "blockReason"
                                ):
                                    raise RuntimeError(
                                        f"Google Gemini blocked the prompt ({prompt_fb['blockReason']})."
                                    )
                                continue

                            cand = (
                                candidates[0] if isinstance(candidates[0], dict) else {}
                            )
                            content = cand.get("content") or {}
                            parts = content.get("parts") or []
                            for part in parts:
                                if not isinstance(part, dict):
                                    continue
                                _append_google_model_part(round_model_parts, part)
                                if part.get("thought") is True:
                                    continue
                                if isinstance(part.get("text"), str) and part["text"]:
                                    chunk = part["text"]
                                    current_text_block.append(chunk)
                                    partial_text_chunks.append(chunk)
                                    yield TextDelta(chunk)
                                if isinstance(part.get("functionCall"), dict):
                                    if current_text_block:
                                        yield TextBlockDone("".join(current_text_block))
                                        current_text_block.clear()
                                    fc = part["functionCall"]
                                    pending_calls.append(
                                        {
                                            "name": str(fc.get("name") or ""),
                                            "args": fc.get("args"),
                                            "id": fc.get("id"),
                                        }
                                    )

                    if not retry_without_thinking:
                        if current_text_block:
                            yield TextBlockDone("".join(current_text_block))
                            current_text_block.clear()
                        break

                if round_usage is not None:
                    yield MessageUsage(
                        message_id=None, model=model, usage=dict(round_usage)
                    )
                    _add_int_usage(total_usage, round_usage)
                    prompt_tokens = round_usage.get("promptTokenCount")
                    if (
                        isinstance(prompt_tokens, int)
                        and not isinstance(prompt_tokens, bool)
                        and prompt_tokens >= 0
                    ):
                        last_context_usage = {
                            "used_tokens": prompt_tokens,
                            "window_tokens": context_window,
                            "model": model,
                        }

                working_contents.append(
                    {
                        "role": "model",
                        "parts": round_model_parts or [{"text": ""}],
                    }
                )

                if not pending_calls:
                    store.save(
                        session_id,
                        model=model,
                        messages=working_contents,
                        context_window=context_window,
                    )
                    yield TurnResult(
                        usage=total_usage or None,
                        context_usage=last_context_usage,
                        num_turns=round_idx,
                    )
                    return

                completed_responses = []
                for call in pending_calls:
                    raw_args = call.get("args")
                    if raw_args is None:
                        raw_args = {}
                    elif isinstance(raw_args, str):
                        with suppress(json.JSONDecodeError):
                            raw_args = json.loads(raw_args)
                    result = await execute_tool_call(
                        tool_by_name, call["name"], raw_args, tool_timeout_s
                    )
                    completed_responses.append(
                        _google_function_response_part(call, result)
                    )
                working_contents.append({"role": "user", "parts": completed_responses})
                completed_responses = []
                pending_calls = []
                store.save(
                    session_id,
                    model=model,
                    messages=working_contents,
                    context_window=context_window,
                )
        raise RuntimeError(
            f"Google Gemini turn exceeded maximum tool rounds ({MAX_TOOL_ROUNDS})."
        )
    except (asyncio.CancelledError, GeneratorExit):
        if made_progress:
            _finalize_google_cancel(
                working_contents,
                partial_text="".join(partial_text_chunks),
                completed_responses=completed_responses,
                pending_calls=pending_calls,
            )
            store.save(
                session_id,
                model=model,
                messages=working_contents,
                context_window=context_window,
            )
        raise


def _strip_google_images(contents: list[Any]) -> list[Any]:
    cleaned: list[Any] = []
    for entry in contents:
        if not isinstance(entry, dict) or not isinstance(entry.get("parts"), list):
            cleaned.append(entry)
            continue
        new_parts: list[Any] = []
        for part in entry["parts"]:
            if not isinstance(part, dict):
                new_parts.append(part)
                continue
            if "inlineData" in part:
                new_parts.append({"text": "[image omitted during compaction]"})
                continue
            if isinstance(part.get("functionResponse"), dict):
                fr = part["functionResponse"]
                if isinstance(fr.get("parts"), list):
                    stripped_sub = [
                        (
                            {"text": "[plot image omitted during compaction]"}
                            if isinstance(sub, dict) and "inlineData" in sub
                            else sub
                        )
                        for sub in fr["parts"]
                    ]
                    new_parts.append(
                        {"functionResponse": {**fr, "parts": stripped_sub}}
                    )
                    continue
            new_parts.append(part)
        cleaned.append({**entry, "parts": new_parts})
    return cleaned


async def compact_google_http(
    *,
    api_key: str,
    system_prompt: str,
    workspace_path: Path,
    model: str | None,
    resume_session_id: str,
    model_info: Mapping[str, Any] | None = None,
    http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
) -> CompactResult:
    """Compact an HTTP Google Gemini conversation by summarizing saved contents."""
    store = HttpThreadStore(workspace_path, "google")
    saved = store.load(resume_session_id)
    target_model = model or saved.get("model")
    if not isinstance(target_model, str) or not target_model:
        raise RuntimeError("No model is available to compact this session.")
    context_window = (
        model_info.get("context_window")
        if model_info and isinstance(model_info.get("context_window"), int)
        else saved.get("context_window")
        if isinstance(saved.get("context_window"), int)
        else None
    )

    sanitized = _strip_google_images(saved["messages"])
    base = _google_base_url()
    headers = _google_headers(api_key)
    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            *sanitized,
            {"role": "user", "parts": [{"text": COMPACTION_PROMPT}]},
        ],
    }
    async with make_http_client(http_client_factory, timeout=HTTP_TIMEOUT) as client:
        response = await client.post(
            f"{base}/models/{target_model}:generateContent",
            headers=headers,
            json=body,
        )
        if response.status_code != 200:
            raise RuntimeError(http_error_message("Google", response, response.content))
        payload = response.json()

    summary_parts: list[str] = []
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if isinstance(candidates, list) and candidates:
        parts = ((candidates[0] or {}).get("content") or {}).get("parts") or ()
        for part in parts:
            if (
                isinstance(part, dict)
                and part.get("thought") is not True
                and isinstance(part.get("text"), str)
                and part["text"]
            ):
                summary_parts.append(part["text"])
    summary = "\n\n".join(part for part in summary_parts if part.strip()).strip()
    if not summary:
        raise RuntimeError("Google Gemini did not return a compaction summary.")

    compacted_contents = [
        {
            "role": "user",
            "parts": [
                {"text": "Summary of our earlier conversation before compaction:"}
            ],
        },
        {"role": "model", "parts": [{"text": summary}]},
    ]
    store.save(
        resume_session_id,
        model=target_model,
        messages=compacted_contents,
        context_window=context_window,
    )
    candidates_tokens = (
        (payload.get("usageMetadata") or {}).get("candidatesTokenCount")
        if isinstance(payload, dict)
        else None
    )
    context_usage = (
        {
            "used_tokens": candidates_tokens + 20,
            "window_tokens": context_window,
            "model": target_model,
        }
        if isinstance(candidates_tokens, int) and candidates_tokens >= 0
        else None
    )
    return CompactResult(
        session_id=resume_session_id,
        context_usage=context_usage,
        record_reloaded=False,
    )
