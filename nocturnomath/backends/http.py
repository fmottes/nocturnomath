"""Direct HTTP API backends for Anthropic, OpenAI, and Google Gemini."""

from __future__ import annotations

import asyncio
import codecs
import json
import logging
import os
import re
import tempfile
import urllib.parse
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import aclosing, suppress
from pathlib import Path
from typing import Any

import httpx

from ..auth import ProviderId
from ..tools import ToolSpec, validate_tool_arguments
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
    "-image",
    "image-generation",
)
_GOOGLE_EFFORT_RE = re.compile(r"^gemini-(?:2\.5|[3-9]|[1-9]\d)")
_GOOGLE_GEMINI_3_RE = re.compile(r"^gemini-(?:[3-9]|[1-9]\d)")
_GOOGLE_DEFAULT_ORDER = (
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
)
_ANTHROPIC_VERSION = "2023-06-01"
_GEMINI_25_THINKING_BUDGETS = {"low": 1024, "medium": 8192, "high": 24576}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_pos_int(value: Any) -> bool:
    return _is_int(value) and value > 0


def _validate_http_api_key(provider_title: str, api_key: str) -> str:
    cleaned = api_key.strip() if isinstance(api_key, str) else ""
    if not cleaned or any(ord(ch) < 33 or ord(ch) > 126 for ch in cleaned):
        raise RuntimeError(
            f"The {provider_title} API key contains invalid whitespace, control, or non-ASCII characters."
        )
    return cleaned


def _sanitize_error_text(text: str, secret: str | None) -> str:
    if not secret or not text:
        return text
    out = text.replace(secret, "[REDACTED]")
    stripped = secret.strip()
    if stripped:
        out = out.replace(stripped, "[REDACTED]")
    return out


def is_http_session_id(session_id: str | None) -> bool:
    """Return whether ``session_id`` identifies a direct HTTP thread."""
    return bool(session_id and session_id.startswith(HTTP_SESSION_PREFIX))


def reject_http_session_for_cli(session_id: str | None) -> None:
    """Raise ``SessionResumeError`` when CLI mode receives a direct HTTP session ID."""
    if is_http_session_id(session_id):
        raise SessionResumeError(
            "Session was created in HTTP API mode and cannot be resumed by the CLI."
        )


class HttpThreadStore:
    """Persist provider-native HTTP message histories under ``.nocturnomath/runtime``."""

    def __init__(self, workspace_path: Path, provider: ProviderId):
        self.root = workspace_path / ".nocturnomath" / "runtime" / "http_sessions"
        self.provider = provider

    def new_session_id(self) -> str:
        """Allocate a new opaque HTTP session identifier."""
        return f"http:{self.provider}:{uuid.uuid4().hex}"

    def _path_for(self, session_id: str) -> Path:
        if not isinstance(session_id, str):
            raise SessionResumeError(f"Unrecognized HTTP session id: {session_id!r}")
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
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SessionResumeError(
                f"Saved session {session_id} could not be read."
            ) from exc
        if (
            not isinstance(data, dict)
            or data.get("version") != 1
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
        except BaseException:
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


async def _iter_sse_lines(response: httpx.Response) -> AsyncIterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf = ""
    first_text = True

    def drain(*, final: bool) -> list[str]:
        nonlocal buf
        lines: list[str] = []
        pos = 0
        length = len(buf)
        while pos < length:
            idx_n = buf.find("\n", pos)
            idx_r = buf.find("\r", pos)
            if idx_n == -1 and idx_r == -1:
                break
            if idx_r != -1 and (idx_n == -1 or idx_r < idx_n):
                if not final and idx_r == length - 1:
                    break
                lines.append(buf[pos:idx_r])
                pos = idx_r + (
                    2 if idx_r + 1 < length and buf[idx_r + 1] == "\n" else 1
                )
            else:
                lines.append(buf[pos:idx_n])
                pos = idx_n + 1
        buf = buf[pos:]
        if final and buf:
            lines.append(buf)
            buf = ""
        return lines

    async for chunk in response.aiter_bytes():
        text = decoder.decode(chunk, final=False)
        if first_text and text:
            first_text = False
            text = text.removeprefix("\ufeff")
        buf += text
        for line in drain(final=False):
            yield line

    tail = decoder.decode(b"", final=True)
    if first_text and tail:
        first_text = False
        tail = tail.removeprefix("\ufeff")
    buf += tail
    for line in drain(final=True):
        yield line


async def iter_sse_events(response: httpx.Response) -> AsyncIterator[tuple[str, str]]:
    """Yield ``(event_name, data)`` tuples from an SSE HTTP response."""
    event_name: str | None = None
    data_lines: list[str] = []
    async with aclosing(_iter_sse_lines(response)) as lines:
        async for raw_line in lines:
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
            if ":" in raw_line:
                field, value = raw_line.split(":", 1)
                value = value.removeprefix(" ")
            else:
                field, value = raw_line, ""
            if field == "event":
                event_name = value.strip()
            elif field == "data":
                data_lines.append(value)


async def _iter_sse_json(
    response: httpx.Response,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    async with aclosing(iter_sse_events(response)) as sse_events:
        async for event_name, raw_data in sse_events:
            if not raw_data or raw_data == "[DONE]":
                continue
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                yield event_name, data


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
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                parsed = parsed[0]
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


def _sanitize_exception(exc: Exception, secret: str) -> RuntimeError | None:
    raw = str(exc)
    sanitized = _sanitize_error_text(raw or type(exc).__name__, secret)
    return RuntimeError(sanitized) if sanitized != raw else None


def _should_retry_without_effort(
    status_code: int,
    attempt: int,
    enabled: bool,
    err_msg: str,
    keywords: Sequence[str],
) -> bool:
    if status_code != 400 or attempt != 0 or not enabled:
        return False
    lowered = err_msg.lower()
    return any(keyword in lowered for keyword in keywords)


def _tool_error(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "is_error": True}


def _parse_tool_arguments(raw_json: str) -> tuple[dict[str, Any] | None, str | None]:
    if not raw_json.strip():
        return {}, None
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON tool arguments: {exc}"
    if parsed is None:
        return {}, None
    if not isinstance(parsed, dict):
        return None, "Tool arguments must decode to a JSON object."
    return parsed, None


_SKIPPED_STEER_RESULT: dict[str, Any] = _tool_error(
    "Skipped: the user sent a new message. Read and address it before running further tools."
)


async def execute_tool_call(
    tool_by_name: Mapping[str, ToolSpec],
    name: str,
    raw_args: Any,
    tool_timeout_s: int,
) -> dict[str, Any]:
    """Validate and run an in-process ``ToolSpec`` without swallowing cancellations."""
    spec = tool_by_name.get(name)
    if spec is None:
        return _tool_error(f"Unknown tool: {name}")
    if raw_args is None:
        raw_args = {}
    if not isinstance(raw_args, dict):
        return _tool_error(f"Tool '{name}' arguments must be a JSON object.")
    try:
        cleaned = validate_tool_arguments(spec.input_schema, raw_args)
    except ValueError as exc:
        return _tool_error(str(exc))
    try:
        return await asyncio.wait_for(spec.handler(cleaned), timeout=tool_timeout_s)
    except TimeoutError:
        return _tool_error(f"Tool '{name}' timed out after {tool_timeout_s}s.")
    except Exception as exc:
        return _tool_error(str(exc) or type(exc).__name__)


async def _iter_tool_batch_results(
    calls: Sequence[Any],
    *,
    tool_by_name: Mapping[str, ToolSpec],
    tool_timeout_s: int,
    extract_call: Callable[[Any], tuple[str, Any, str | None]],
) -> AsyncIterator[tuple[Any, dict[str, Any]]]:
    """Execute a round's tool calls sequentially, short-circuiting after steer."""
    steered_out = False
    for call in calls:
        if steered_out:
            yield call, _SKIPPED_STEER_RESULT
            continue
        name, raw_args, json_err = extract_call(call)
        if json_err is not None:
            result = _tool_error(json_err)
        else:
            result = await execute_tool_call(
                tool_by_name, name, raw_args, tool_timeout_s
            )
        if result.get("steered"):
            steered_out = True
        yield call, result


def _tool_text_and_images(
    result: dict[str, Any], *, is_error: bool, prefix_error: bool = False
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
    if prefix_error and is_error and not text_out.lower().startswith("error"):
        text_out = f"Error: {text_out}"
    return text_out, images


def _add_int_usage(target: dict[str, int], source: Mapping[str, Any] | None) -> None:
    if not isinstance(source, Mapping):
        return
    for key, value in source.items():
        if _is_int(value):
            target[key] = target.get(key, 0) + value


def _interrupted_text(partial_text: str, assistant_note: str) -> str:
    return (
        f"{partial_text}\n\n{assistant_note}"
        if partial_text.strip()
        else assistant_note
    )


def _resolve_context_window(
    info: Mapping[str, Any] | None,
    *,
    saved_model: str | None,
    target_model: str,
    saved_window: Any,
) -> int | None:
    info_window = info.get("context_window") if isinstance(info, Mapping) else None
    if _is_pos_int(info_window):
        return info_window
    same_model = saved_model is None or saved_model == target_model
    return saved_window if same_model and _is_pos_int(saved_window) else None


async def _fetch_http_model_pages(
    provider_title: str,
    url: str,
    *,
    headers_fn: Callable[[str], dict[str, str]],
    api_key: str,
    http_client_factory: Callable[..., httpx.AsyncClient] | None,
    params_for_token: Callable[[str | None], dict[str, str] | None] | None = None,
    next_token_fn: Callable[[dict[str, Any]], str | None] | None = None,
    max_pages: int = 5,
) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch one or more paginated model-list JSON payloads with sanitized errors."""
    pages: list[dict[str, Any]] = []
    token: str | None = None
    try:
        headers = headers_fn(api_key)
        async with (
            asyncio.timeout(HTTP_DISCOVERY_TIMEOUT_S),
            make_http_client(
                http_client_factory, timeout=HTTP_DISCOVERY_TIMEOUT_S
            ) as client,
        ):
            for _ in range(max_pages):
                params = params_for_token(token) if params_for_token else None
                response = await client.get(url, headers=headers, params=params)
                if response.status_code != 200:
                    err_msg = _sanitize_error_text(
                        http_error_message(provider_title, response, response.content),
                        api_key,
                    )
                    logger.warning(
                        "%s HTTP model discovery failed: %s", provider_title, err_msg
                    )
                    return [], err_msg
                payload = response.json()
                if not isinstance(payload, dict):
                    break
                pages.append(payload)
                if next_token_fn is None:
                    break
                token = next_token_fn(payload)
                if not token:
                    break
    except Exception as exc:
        err_msg = _sanitize_error_text(str(exc) or type(exc).__name__, api_key)
        logger.warning("%s HTTP model discovery failed: %s", provider_title, err_msg)
        return [], err_msg
    return pages, None


async def _http_post_json(
    provider_title: str,
    url: str,
    *,
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    api_key: str,
    http_client_factory: Callable[..., httpx.AsyncClient] | None,
) -> Any:
    try:
        async with make_http_client(
            http_client_factory, timeout=HTTP_TIMEOUT
        ) as client:
            response = await client.post(url, headers=headers, json=body)
            if response.status_code != 200:
                raise RuntimeError(
                    _sanitize_error_text(
                        http_error_message(provider_title, response, response.content),
                        api_key,
                    )
                )
            return response.json()
    except Exception as exc:
        if (sanitized := _sanitize_exception(exc, api_key)) is not None:
            raise sanitized from None
        raise


def _resume_state(
    store: HttpThreadStore, resume_session_id: str | None
) -> tuple[str, list[Any], str | None, int | None]:
    if resume_session_id:
        saved = store.load(resume_session_id)
        ctx_window = saved.get("context_window")
        messages = list(saved["messages"])
        if store.provider == "anthropic":
            _repair_anthropic_messages(messages)
        elif store.provider == "openai":
            _repair_openai_items(messages)
        elif store.provider == "google":
            _repair_google_contents(messages)
        return (
            resume_session_id,
            messages,
            saved.get("model") if isinstance(saved.get("model"), str) else None,
            ctx_window if _is_pos_int(ctx_window) else None,
        )
    return store.new_session_id(), [], None, None


async def _compact_http_thread(
    *,
    provider: ProviderId,
    provider_title: str,
    headers: Mapping[str, str],
    api_key: str,
    workspace_path: Path,
    model: str | None,
    resume_session_id: str,
    resolve_context: Callable[[str, str | None, Any], int | None],
    repair_history: Callable[[list[Any]], None],
    render_history: Callable[[list[Any]], str],
    build_request: Callable[[str, str], tuple[str, dict[str, Any]]],
    parse_response: Callable[[Any], tuple[str, int | None]],
    format_compacted: Callable[[str], list[Any]],
    save_context_window: bool = True,
    http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
) -> CompactResult:
    """Shared stateless compaction flow across direct HTTP providers."""
    store = HttpThreadStore(workspace_path, provider)
    saved = store.load(resume_session_id)
    target_model = model or saved.get("model")
    if not isinstance(target_model, str) or not target_model:
        raise RuntimeError("No model is available to compact this session.")
    saved_model = saved.get("model") if isinstance(saved.get("model"), str) else None
    context_window = resolve_context(
        target_model, saved_model, saved.get("context_window")
    )

    history = saved["messages"]
    if isinstance(history, list):
        history = list(history)
        repair_history(history)
    transcript = render_history(history)
    compact_prompt = (
        f"Earlier conversation transcript:\n\n{transcript}\n\n{COMPACTION_PROMPT}"
        if transcript
        else COMPACTION_PROMPT
    )

    url, body = build_request(target_model, compact_prompt)
    payload = await _http_post_json(
        provider_title,
        url,
        headers=headers,
        body=body,
        api_key=api_key,
        http_client_factory=http_client_factory,
    )
    summary, token_count = parse_response(payload)
    if not summary:
        raise RuntimeError(f"{provider_title} did not return a compaction summary.")

    store.save(
        resume_session_id,
        model=target_model,
        messages=format_compacted(summary),
        context_window=context_window if save_context_window else None,
    )
    context_usage = (
        {
            "used_tokens": token_count + 20,
            "window_tokens": context_window,
            "model": target_model,
        }
        if token_count is not None
        else None
    )
    return CompactResult(
        session_id=resume_session_id,
        context_usage=context_usage,
        record_reloaded=False,
    )


# ---------------------------------------------------------------------------
# Anthropic Messages API
# ---------------------------------------------------------------------------


def _anthropic_base_url() -> str:
    raw = (os.environ.get("ANTHROPIC_BASE_URL") or "").strip()
    base = (raw or "https://api.anthropic.com").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base or "https://api.anthropic.com"


def _anthropic_headers(api_key: str) -> dict[str, str]:
    cleaned = _validate_http_api_key("Anthropic", api_key)
    return {
        "x-api-key": cleaned,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def _anthropic_default_max_tokens(model_id: str) -> int:
    if re.search(
        r"claude-(?:opus|sonnet|haiku)-(?:[4-9]|[1-9]\d)|claude-3-7", model_id
    ):
        return 16384
    if "claude-3-5" in model_id:
        return 8192
    return 4096


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
        "context_window": max_input if _is_pos_int(max_input) else None,
        "max_tokens": max_tokens if _is_pos_int(max_tokens) else None,
    }


async def discover_anthropic_http_models(
    api_key: str,
    *,
    http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
) -> tuple[list[ModelSpec], dict[str, dict[str, Any]], str | None]:
    """List Claude models via ``GET /v1/models``."""
    base = _anthropic_base_url()
    pages, err_msg = await _fetch_http_model_pages(
        "Anthropic",
        f"{base}/v1/models",
        headers_fn=_anthropic_headers,
        api_key=api_key,
        http_client_factory=http_client_factory,
        params_for_token=lambda tok: {
            "limit": "100",
            **({"after_id": tok} if tok else {}),
        },
        next_token_fn=lambda payload: (
            payload.get("last_id")
            if payload.get("has_more") and isinstance(payload.get("last_id"), str)
            else None
        ),
    )
    if err_msg is not None:
        return [], {}, err_msg

    raw_items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for payload in pages:
        data = payload.get("data")
        if not isinstance(data, list):
            continue
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

    if not raw_items:
        return [], {}, None

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
    return specs, model_info, None


def _anthropic_prompt_tokens(usage: Mapping[str, Any]) -> int | None:
    keys = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    if not any(key in usage for key in keys):
        return None
    total = 0
    for key in keys:
        val = usage.get(key, 0)
        if not _is_int(val) or val < 0:
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


def _finalize_anthropic_tool_block(block: dict[str, Any]) -> None:
    if block.get("type") != "tool_use" or "_partial_json" not in block:
        return
    raw_json = "".join(block.pop("_partial_json", []))
    if raw_json.strip():
        parsed, json_err = _parse_tool_arguments(raw_json)
        if json_err is not None:
            block["_json_error"] = json_err
        elif parsed is not None:
            block["input"] = parsed


def _finalize_anthropic_interrupted(
    working_messages: list[Any],
    *,
    partial_text: str,
    completed_tool_results: list[dict[str, Any]],
    pending_tool_ids: list[str],
    tool_reason: str = "Cancelled by user.",
    assistant_note: str = "(Interrupted by user.)",
) -> None:
    last = working_messages[-1] if working_messages else None
    if isinstance(last, dict) and last.get("role") == "assistant" and pending_tool_ids:
        results = list(completed_tool_results)
        for tc_id in pending_tool_ids[len(completed_tool_results) :]:
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tc_id,
                    "is_error": True,
                    "content": [{"type": "text", "text": tool_reason}],
                }
            )
        working_messages.append({"role": "user", "content": results})
        working_messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_note}],
            }
        )
        return
    if isinstance(last, dict) and last.get("role") == "user":
        suffix = _interrupted_text(partial_text, assistant_note)
        working_messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": suffix}]}
        )


def _repair_anthropic_messages(messages: list[Any]) -> None:
    if not messages or not isinstance(messages[-1], dict):
        return
    last = messages[-1]
    if last.get("role") == "assistant":
        content = last.get("content")
        tool_ids = (
            [
                str(block.get("id") or "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "tool_use"
            ]
            if isinstance(content, list)
            else []
        )
        if tool_ids:
            _finalize_anthropic_interrupted(
                messages,
                partial_text="",
                completed_tool_results=[],
                pending_tool_ids=tool_ids,
                tool_reason="Previous turn was interrupted.",
                assistant_note="(Previous turn was interrupted.)",
            )
    elif last.get("role") == "user":
        _finalize_anthropic_interrupted(
            messages,
            partial_text="",
            completed_tool_results=[],
            pending_tool_ids=[],
            assistant_note="(Previous turn was interrupted.)",
        )


def _sanitize_anthropic_for_model_switch(messages: list[Any]) -> list[Any]:
    cleaned: list[Any] = []
    for msg in messages:
        if not isinstance(msg, dict):
            cleaned.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            cleaned.append(msg)
            continue
        role = msg.get("role")
        new_blocks: list[Any] = []
        for block in content:
            if not isinstance(block, dict):
                new_blocks.append(block)
                continue
            btype = block.get("type")
            if btype in ("thinking", "redacted_thinking"):
                continue
            if btype == "text" and not str(block.get("text") or "").strip():
                continue
            new_blocks.append(block)
        if role == "assistant" and not new_blocks:
            new_blocks = [{"type": "text", "text": "(continued)"}]
        elif role == "user" and not new_blocks:
            new_blocks = [{"type": "text", "text": "(user message)"}]
        cleaned.append({**msg, "content": new_blocks})
    return cleaned


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
    headers = _anthropic_headers(api_key)
    store = HttpThreadStore(workspace_path, "anthropic")
    session_id, saved_messages, saved_model, saved_window = _resume_state(
        store, resume_session_id
    )
    if saved_model and saved_model != model:
        saved_messages = _sanitize_anthropic_for_model_switch(saved_messages)
    info = (
        dict(model_info)
        if model_info is not None
        else _anthropic_capabilities(model, {})
    )
    context_window = _resolve_context_window(
        info, saved_model=saved_model, target_model=model, saved_window=saved_window
    )
    max_tokens_meta = info.get("max_tokens")
    max_tokens = (
        min(max_tokens_meta, 16384)
        if _is_pos_int(max_tokens_meta)
        else _anthropic_default_max_tokens(model)
    )
    base = _anthropic_base_url()
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
            include_effort = bool(effort)
            for round_idx in range(1, MAX_TOOL_ROUNDS + 1):
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
                    saw_message_stop = False
                    stop_reason: str | None = None

                    async with client.stream(
                        "POST", f"{base}/v1/messages", headers=headers, json=body
                    ) as response:
                        if response.status_code != 200:
                            raw_err = await response.aread()
                            err_msg = _sanitize_error_text(
                                http_error_message("Anthropic", response, raw_err),
                                api_key,
                            )
                            if _should_retry_without_effort(
                                response.status_code,
                                effort_attempt,
                                include_effort,
                                err_msg,
                                ("output_config", "thinking", "effort"),
                            ):
                                include_effort = False
                                retry_without_effort = True
                                continue
                            raise RuntimeError(err_msg)

                        async with aclosing(_iter_sse_json(response)) as events:
                            async for event_name, data in events:
                                ev_type = data.get("type") or event_name

                                if ev_type == "error":
                                    err = data.get("error")
                                    detail = (
                                        err.get("message")
                                        if isinstance(err, dict)
                                        else str(err or "Anthropic stream error.")
                                    )
                                    raise RuntimeError(
                                        detail or "Anthropic stream error."
                                    )

                                if ev_type == "message_start":
                                    msg_obj = data.get("message") or {}
                                    if isinstance(msg_obj, dict):
                                        round_msg_id = msg_obj.get("id") or round_msg_id
                                        round_model = (
                                            msg_obj.get("model") or round_model
                                        )
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
                                            initial_text = str(cb.get("text") or "")
                                            blocks_by_index[idx] = {
                                                "type": "text",
                                                "text": initial_text,
                                            }
                                            open_text_indices.add(idx)
                                            if initial_text:
                                                partial_text_chunks.append(initial_text)
                                                yield TextDelta(initial_text)
                                        elif cb_type == "thinking":
                                            blocks_by_index[idx] = {
                                                "type": "thinking",
                                                "thinking": str(
                                                    cb.get("thinking") or ""
                                                ),
                                                "signature": str(
                                                    cb.get("signature") or ""
                                                ),
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
                                    if (
                                        d_type == "text_delta"
                                        and block["type"] == "text"
                                    ):
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
                                    if (
                                        block["type"] == "text"
                                        and idx in open_text_indices
                                    ):
                                        open_text_indices.discard(idx)
                                        if block["text"]:
                                            yield TextBlockDone(block["text"])
                                    else:
                                        _finalize_anthropic_tool_block(block)
                                elif ev_type == "message_delta":
                                    _merge_usage(round_usage, data.get("usage"))
                                    delta = data.get("delta")
                                    if isinstance(delta, Mapping) and isinstance(
                                        delta.get("stop_reason"), str
                                    ):
                                        stop_reason = delta["stop_reason"]
                                elif ev_type == "message_stop":
                                    saw_message_stop = True

                    if not retry_without_effort:
                        if not saw_message_stop:
                            raise RuntimeError(
                                "Anthropic API stream ended before completing the response."
                            )
                        if stop_reason == "refusal":
                            raise RuntimeError(
                                "Anthropic refused to complete the response."
                            )
                        if stop_reason in (
                            "max_tokens",
                            "model_context_window_exceeded",
                        ):
                            raise RuntimeError(
                                f"Anthropic response stopped early ({stop_reason})."
                            )
                        for idx in sorted(open_text_indices):
                            block = blocks_by_index[idx]
                            if block.get("text"):
                                yield TextBlockDone(block["text"])
                        for idx in sorted(blocks_by_index):
                            block = blocks_by_index[idx]
                            _finalize_anthropic_tool_block(block)
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
                stored_blocks: list[dict[str, Any]] = []
                for block in assistant_blocks:
                    btype = block.get("type")
                    if btype == "text":
                        if str(block.get("text") or "").strip():
                            stored_blocks.append(
                                {"type": "text", "text": block["text"]}
                            )
                    elif btype == "thinking":
                        if (
                            isinstance(block.get("signature"), str)
                            and block["signature"]
                        ):
                            stored_blocks.append(
                                {
                                    k: v
                                    for k, v in block.items()
                                    if not k.startswith("_")
                                }
                            )
                    else:
                        stored_blocks.append(
                            {k: v for k, v in block.items() if not k.startswith("_")}
                        )
                if not stored_blocks:
                    stored_blocks = [{"type": "text", "text": "(no response)"}]
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
                if round_idx >= MAX_TOOL_ROUNDS:
                    raise RuntimeError(
                        f"Anthropic turn exceeded maximum tool rounds ({MAX_TOOL_ROUNDS})."
                    )
                completed_tool_results = []
                async for block, result in _iter_tool_batch_results(
                    tool_blocks,
                    tool_by_name=tool_by_name,
                    tool_timeout_s=tool_timeout_s,
                    extract_call=lambda b: (
                        str(b.get("name") or ""),
                        b.get("input"),
                        b.get("_json_error"),
                    ),
                ):
                    completed_tool_results.append(
                        _anthropic_tool_result_block(str(block.get("id") or ""), result)
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
    except (asyncio.CancelledError, GeneratorExit):
        if made_progress:
            _finalize_anthropic_interrupted(
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
    except Exception as exc:
        if made_progress:
            _finalize_anthropic_interrupted(
                working_messages,
                partial_text="".join(partial_text_chunks),
                completed_tool_results=completed_tool_results,
                pending_tool_ids=pending_tool_ids,
                tool_reason="Turn failed before tool execution completed.",
                assistant_note="(Error before response completed.)",
            )
            store.save(
                session_id,
                model=model,
                messages=working_messages,
                context_window=context_window,
            )
        if (sanitized := _sanitize_exception(exc, api_key)) is not None:
            raise sanitized from None
        raise


def _render_anthropic_history_for_compaction(messages: list[Any]) -> str:
    lines: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content")
        if isinstance(content, str):
            if content.strip():
                lines.append(f"{role}: {content}")
            continue
        if not isinstance(content, list):
            continue
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in ("thinking", "redacted_thinking"):
                continue
            if btype == "text":
                text = str(block.get("text") or "")
                if text.strip():
                    parts.append(text)
            elif btype == "image":
                parts.append("[image omitted during compaction]")
            elif btype == "tool_use":
                args_json = json.dumps(block.get("input") or {}, ensure_ascii=False)
                parts.append(f"[Tool call {block.get('name') or ''}: {args_json}]")
            elif btype == "tool_result":
                inner_parts: list[str] = []
                inner = block.get("content")
                if isinstance(inner, list):
                    for item in inner:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "text":
                            t = str(item.get("text") or "")
                            if t:
                                inner_parts.append(t)
                        elif item.get("type") == "image":
                            inner_parts.append("[plot image omitted]")
                elif isinstance(inner, str) and inner:
                    inner_parts.append(inner)
                tag = "Tool result (error)" if block.get("is_error") else "Tool result"
                parts.append(f"[{tag}: {' '.join(inner_parts)}]")
        if parts:
            lines.append(f"{role}: " + "\n".join(parts))
    return "\n\n".join(lines)


async def compact_anthropic_http(
    *,
    api_key: str,
    system_prompt: str,
    workspace_path: Path,
    model: str | None,
    resume_session_id: str,
    model_info: Mapping[str, Any] | None = None,
    model_info_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
) -> CompactResult:
    """Compact an HTTP Anthropic conversation by summarizing saved messages."""
    headers = _anthropic_headers(api_key)
    base = _anthropic_base_url()

    def info_for(target_model: str) -> dict[str, Any]:
        resolved = (model_info_by_id or {}).get(target_model) or model_info
        return (
            dict(resolved)
            if resolved is not None
            else _anthropic_capabilities(target_model, {})
        )

    def build_request(
        target_model: str, compact_prompt: str
    ) -> tuple[str, dict[str, Any]]:
        info = info_for(target_model)
        max_tokens = (
            min(info["max_tokens"], 8192)
            if _is_pos_int(info.get("max_tokens"))
            else min(_anthropic_default_max_tokens(target_model), 4096)
        )
        return f"{base}/v1/messages", {
            "model": target_model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": compact_prompt}],
        }

    def parse_response(payload: Any) -> tuple[str, int | None]:
        stop_reason = payload.get("stop_reason") if isinstance(payload, dict) else None
        if stop_reason in ("max_tokens", "refusal", "model_context_window_exceeded"):
            raise RuntimeError(f"Anthropic compaction stopped early ({stop_reason}).")
        content = payload.get("content") if isinstance(payload, dict) else None
        summary_parts = [
            str(block.get("text") or "")
            for block in (content or [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        summary = "\n\n".join(part for part in summary_parts if part.strip()).strip()
        output_tokens = (
            (payload.get("usage") or {}).get("output_tokens")
            if isinstance(payload, dict)
            else None
        )
        return summary, (
            output_tokens if _is_int(output_tokens) and output_tokens >= 0 else None
        )

    return await _compact_http_thread(
        provider="anthropic",
        provider_title="Anthropic",
        headers=headers,
        api_key=api_key,
        workspace_path=workspace_path,
        model=model,
        resume_session_id=resume_session_id,
        resolve_context=lambda target_model, saved_model, saved_window: (
            _resolve_context_window(
                info_for(target_model),
                saved_model=saved_model,
                target_model=target_model,
                saved_window=saved_window,
            )
        ),
        repair_history=_repair_anthropic_messages,
        render_history=_render_anthropic_history_for_compaction,
        build_request=build_request,
        parse_response=parse_response,
        format_compacted=lambda summary: [
            {
                "role": "user",
                "content": "Summary of our earlier conversation before compaction:",
            },
            {"role": "assistant", "content": [{"type": "text", "text": summary}]},
        ],
        http_client_factory=http_client_factory,
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
    raw = (os.environ.get("OPENAI_BASE_URL") or "").strip()
    return (raw or "https://api.openai.com/v1").rstrip("/")


def _openai_headers(api_key: str) -> dict[str, str]:
    cleaned = _validate_http_api_key("OpenAI", api_key)
    return {
        "Authorization": f"Bearer {cleaned}",
        "Content-Type": "application/json",
    }


async def discover_openai_http_models(
    api_key: str,
    *,
    http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
) -> tuple[list[ModelSpec], str | None]:
    """List chat/reasoning models via ``GET /models``."""
    base = _openai_base_url()
    pages, err_msg = await _fetch_http_model_pages(
        "OpenAI",
        f"{base}/models",
        headers_fn=_openai_headers,
        api_key=api_key,
        http_client_factory=http_client_factory,
        max_pages=1,
    )
    if err_msg is not None:
        return [], err_msg

    data = pages[0].get("data") if pages else None
    if not isinstance(data, list):
        return [], None

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
        return [], None

    filtered.sort(
        key=lambda item: (
            -(item.get("created") if _is_int(item.get("created")) else 0),
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
    return specs, None


def _openai_tool_output(call_id: str, result: dict[str, Any]) -> dict[str, Any]:
    is_error = bool(result.get("is_error"))
    text_out, images = _tool_text_and_images(
        result, is_error=is_error, prefix_error=True
    )
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


def _drop_trailing_openai_reasoning(items: list[Any]) -> None:
    while (
        items and isinstance(items[-1], dict) and items[-1].get("type") == "reasoning"
    ):
        items.pop()


def _finalize_openai_interrupted(
    working_items: list[Any],
    *,
    partial_text: str,
    completed_outputs: list[dict[str, Any]],
    pending_calls: list[dict[str, Any]],
    tool_reason: str = "Error: Cancelled by user.",
    assistant_note: str = "(Interrupted by user.)",
) -> None:
    if pending_calls:
        working_items.extend(completed_outputs)
        for call in pending_calls[len(completed_outputs) :]:
            working_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.get("call_id"),
                    "output": tool_reason,
                }
            )
        working_items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": assistant_note}],
            }
        )
        return
    _drop_trailing_openai_reasoning(working_items)
    last = working_items[-1] if working_items else None
    needs_reply = not isinstance(last, dict) or last.get("role") != "assistant"
    if needs_reply:
        suffix = _interrupted_text(partial_text, assistant_note)
        working_items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": suffix}],
            }
        )


def _repair_openai_items(items: list[Any]) -> None:
    _drop_trailing_openai_reasoning(items)
    if not items:
        return
    seen_outputs = {
        item.get("call_id")
        for item in items
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    }
    unanswered_calls = [
        item
        for item in items
        if isinstance(item, dict)
        and item.get("type") == "function_call"
        and item.get("call_id") not in seen_outputs
    ]
    if unanswered_calls:
        _finalize_openai_interrupted(
            items,
            partial_text="",
            completed_outputs=[],
            pending_calls=unanswered_calls,
            tool_reason="Error: Previous turn was interrupted.",
            assistant_note="(Previous turn was interrupted.)",
        )
        return
    last = items[-1]
    if isinstance(last, dict) and (
        last.get("role") == "user" or last.get("type") == "function_call_output"
    ):
        _finalize_openai_interrupted(
            items,
            partial_text="",
            completed_outputs=[],
            pending_calls=[],
            assistant_note="(Previous turn was interrupted.)",
        )


def _sanitize_openai_for_model_switch(items: list[Any]) -> list[Any]:
    return [
        item
        for item in items
        if not (isinstance(item, dict) and item.get("type") == "reasoning")
    ]


def _safe_content_index(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _openai_refusal_in_item(item: Mapping[str, Any]) -> str | None:
    if item.get("type") != "message":
        return None
    for part in item.get("content") or ():
        if isinstance(part, dict) and part.get("type") == "refusal":
            return str(part.get("refusal") or "OpenAI refused the request.")
    return None


def _process_openai_output_item(
    item: Mapping[str, Any],
    *,
    emitted_parts: set[tuple[str, int]],
    working_items: list[Any],
    pending_calls: list[dict[str, Any]],
) -> list[str]:
    """Validate, store, and return any unemitted text blocks for one output item."""
    refusal = _openai_refusal_in_item(item)
    if refusal is not None:
        raise RuntimeError(f"OpenAI refused the request: {refusal}")
    texts: list[str] = []
    if item.get("type") == "message":
        item_id = str(item.get("id") or "")
        content = item.get("content")
        if isinstance(content, list):
            for c_idx, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "output_text":
                    key = (item_id, c_idx)
                    part_text = str(part.get("text") or "")
                    if part_text and key not in emitted_parts:
                        emitted_parts.add(key)
                        texts.append(part_text)
    _store_openai_output_item(
        item, working_items=working_items, pending_calls=pending_calls
    )
    return texts


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
    headers = _openai_headers(api_key)
    store = HttpThreadStore(workspace_path, "openai")
    session_id, saved_items, saved_model, _ = _resume_state(store, resume_session_id)
    if saved_model and saved_model != model:
        saved_items = _sanitize_openai_for_model_switch(saved_items)
    base = _openai_base_url()
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
            include_effort = bool(effort)
            include_reasoning = bool(
                _OPENAI_REASONING_RE.match(model) and "chat-latest" not in model
            )
            for round_idx in range(1, MAX_TOOL_ROUNDS + 1):
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
                    if include_reasoning:
                        body["include"] = ["reasoning.encrypted_content"]

                    round_started = False
                    retry_without_reasoning = False
                    saw_completed = False
                    text_buffers: dict[tuple[str, int], list[str]] = {}
                    emitted_parts: set[tuple[str, int]] = set()
                    round_output_items: list[Any] = []
                    round_pending_calls: list[dict[str, Any]] = []
                    saw_done_items = False

                    async with client.stream(
                        "POST", f"{base}/responses", headers=headers, json=body
                    ) as response:
                        if response.status_code != 200:
                            raw_err = await response.aread()
                            err_msg = _sanitize_error_text(
                                http_error_message("OpenAI", response, raw_err), api_key
                            )
                            if _should_retry_without_effort(
                                response.status_code,
                                effort_attempt,
                                include_effort or include_reasoning,
                                err_msg,
                                ("reasoning", "encrypted_content", "include", "effort"),
                            ):
                                include_effort = False
                                include_reasoning = False
                                retry_without_reasoning = True
                                continue
                            raise RuntimeError(err_msg)

                        async with aclosing(_iter_sse_json(response)) as events:
                            async for event_name, data in events:
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
                                if ev_type in (
                                    "response.failed",
                                    "response.incomplete",
                                ):
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
                                if ev_type == "response.refusal.done":
                                    refusal = str(
                                        data.get("refusal")
                                        or "OpenAI refused the request."
                                    )
                                    raise RuntimeError(
                                        f"OpenAI refused the request: {refusal}"
                                    )

                                if ev_type in (
                                    "response.created",
                                    "response.in_progress",
                                ):
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
                                            _safe_content_index(
                                                data.get("content_index")
                                            ),
                                        )
                                        text_buffers.setdefault(key, []).append(delta)
                                        partial_text_chunks.append(delta)
                                        yield TextDelta(delta)
                                elif ev_type == "response.output_text.done":
                                    key = (
                                        str(data.get("item_id") or ""),
                                        _safe_content_index(data.get("content_index")),
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
                                        yield TextBlockDone(full_text)
                                elif ev_type == "response.output_item.done":
                                    if not round_started:
                                        round_started = True
                                        made_progress = True
                                        yield TurnStarted()
                                    item = data.get("item")
                                    if not isinstance(item, dict):
                                        continue
                                    saw_done_items = True
                                    for text in _process_openai_output_item(
                                        item,
                                        emitted_parts=emitted_parts,
                                        working_items=round_output_items,
                                        pending_calls=round_pending_calls,
                                    ):
                                        yield TextBlockDone(text)
                                elif ev_type == "response.completed":
                                    saw_completed = True
                                    if not round_started:
                                        round_started = True
                                        made_progress = True
                                        yield TurnStarted()
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
                                                for text in _process_openai_output_item(
                                                    out_item,
                                                    emitted_parts=emitted_parts,
                                                    working_items=round_output_items,
                                                    pending_calls=round_pending_calls,
                                                ):
                                                    yield TextBlockDone(text)

                    if not retry_without_reasoning:
                        if not saw_completed:
                            raise RuntimeError(
                                "OpenAI API stream ended before completing the response."
                            )
                        for key, chunks in sorted(text_buffers.items()):
                            leftover = "".join(chunks)
                            if leftover and key not in emitted_parts:
                                emitted_parts.add(key)
                                yield TextBlockDone(leftover)
                        working_items.extend(round_output_items)
                        pending_calls = round_pending_calls
                        break

                if round_usage is not None:
                    yield MessageUsage(
                        message_id=round_msg_id,
                        model=round_model,
                        usage=round_usage,
                    )
                    _add_int_usage(total_usage, round_usage)
                    inp = round_usage.get("input_tokens")
                    if _is_int(inp) and inp >= 0:
                        last_context_usage = {
                            "used_tokens": inp,
                            "window_tokens": None,
                            "model": round_model,
                        }

                if not pending_calls:
                    _drop_trailing_openai_reasoning(working_items)
                    store.save(session_id, model=model, messages=working_items)
                    yield TurnResult(
                        usage=total_usage or None,
                        context_usage=last_context_usage,
                        num_turns=round_idx,
                    )
                    return

                if round_idx >= MAX_TOOL_ROUNDS:
                    raise RuntimeError(
                        f"OpenAI turn exceeded maximum tool rounds ({MAX_TOOL_ROUNDS})."
                    )

                completed_outputs = []

                def extract_openai_call(
                    call: dict[str, Any],
                ) -> tuple[str, Any, str | None]:
                    parsed_args, json_err = _parse_tool_arguments(call["arguments"])
                    return call["name"], parsed_args, json_err

                async for call, result in _iter_tool_batch_results(
                    pending_calls,
                    tool_by_name=tool_by_name,
                    tool_timeout_s=tool_timeout_s,
                    extract_call=extract_openai_call,
                ):
                    completed_outputs.append(
                        _openai_tool_output(call["call_id"], result)
                    )
                working_items.extend(completed_outputs)
                completed_outputs = []
                pending_calls = []
                store.save(session_id, model=model, messages=working_items)
    except (asyncio.CancelledError, GeneratorExit):
        if made_progress:
            _finalize_openai_interrupted(
                working_items,
                partial_text="".join(partial_text_chunks),
                completed_outputs=completed_outputs,
                pending_calls=pending_calls,
            )
            store.save(session_id, model=model, messages=working_items)
        raise
    except Exception as exc:
        if made_progress:
            _finalize_openai_interrupted(
                working_items,
                partial_text="".join(partial_text_chunks),
                completed_outputs=completed_outputs,
                pending_calls=pending_calls,
                tool_reason="Error: Turn failed before tool execution completed.",
                assistant_note="(Error before response completed.)",
            )
            store.save(session_id, model=model, messages=working_items)
        if (sanitized := _sanitize_exception(exc, api_key)) is not None:
            raise sanitized from None
        raise


def _render_openai_history_for_compaction(items: list[Any]) -> str:
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "reasoning":
            continue
        if item.get("role") in ("user", "assistant"):
            role = "User" if item.get("role") == "user" else "Assistant"
            content = item.get("content")
            if isinstance(content, str):
                if content.strip():
                    lines.append(f"{role}: {content}")
            elif isinstance(content, list):
                text_parts = [
                    str(part.get("text") or "")
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") in ("text", "input_text", "output_text")
                    and str(part.get("text") or "").strip()
                ]
                if text_parts:
                    lines.append(f"{role}: " + "\n".join(text_parts))
        elif itype == "function_call":
            lines.append(
                f"Assistant: [Tool call {item.get('name') or ''}: {item.get('arguments') or '{}'}]"
            )
        elif itype == "function_call_output":
            out = item.get("output")
            if isinstance(out, list):
                rendered_parts: list[str] = []
                for part in out:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "input_text":
                        rendered_parts.append(str(part.get("text") or ""))
                    elif part.get("type") == "input_image":
                        rendered_parts.append("[plot image omitted]")
                text_val = " ".join(p for p in rendered_parts if p)
            else:
                text_val = str(out or "")
            lines.append(f"User: [Tool result: {text_val}]")
    return "\n\n".join(lines)


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
    headers = _openai_headers(api_key)
    base = _openai_base_url()

    def parse_response(payload: Any) -> tuple[str, int | None]:
        if isinstance(payload, dict):
            if payload.get("error"):
                err = payload["error"]
                detail = err.get("message") if isinstance(err, dict) else str(err)
                raise RuntimeError(str(detail or "OpenAI compaction failed."))
            status = payload.get("status")
            if status in ("failed", "incomplete"):
                details = payload.get("incomplete_details")
                reason = details.get("reason") if isinstance(details, dict) else status
                raise RuntimeError(f"OpenAI compaction stopped early ({reason}).")
        summary_parts: list[str] = []
        for item in (payload.get("output") or ()) if isinstance(payload, dict) else ():
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for part in item.get("content") or ():
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text = str(part.get("text") or "")
                    if text:
                        summary_parts.append(text)
        if (
            not summary_parts
            and isinstance(payload, dict)
            and isinstance(payload.get("output_text"), str)
        ):
            summary_parts.append(payload["output_text"])
        summary = "\n\n".join(part for part in summary_parts if part.strip()).strip()
        usage = payload.get("usage") if isinstance(payload, dict) else None
        output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
        reasoning_tokens = (
            (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
            if isinstance(usage, dict)
            and isinstance(usage.get("output_tokens_details"), dict)
            else None
        )
        visible_tokens = (
            max(
                0,
                output_tokens
                - (
                    reasoning_tokens
                    if _is_int(reasoning_tokens)
                    and 0 <= reasoning_tokens <= output_tokens
                    else 0
                ),
            )
            if _is_int(output_tokens) and output_tokens >= 0
            else None
        )
        return summary, visible_tokens

    return await _compact_http_thread(
        provider="openai",
        provider_title="OpenAI",
        headers=headers,
        api_key=api_key,
        workspace_path=workspace_path,
        model=model,
        resume_session_id=resume_session_id,
        resolve_context=lambda *_args: None,
        repair_history=_repair_openai_items,
        render_history=_render_openai_history_for_compaction,
        build_request=lambda target_model, compact_prompt: (
            f"{base}/responses",
            {
                "model": target_model,
                "instructions": system_prompt,
                "input": [{"role": "user", "content": compact_prompt}],
                "store": False,
                "stream": False,
            },
        ),
        parse_response=parse_response,
        format_compacted=lambda summary: [
            {
                "role": "user",
                "content": "Summary of our earlier conversation before compaction:",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": summary}],
            },
        ],
        save_context_window=False,
        http_client_factory=http_client_factory,
    )


# ---------------------------------------------------------------------------
# Google Gemini REST API
# ---------------------------------------------------------------------------


def _google_base_url() -> str:
    raw = (os.environ.get("GEMINI_API_BASE_URL") or "").strip()
    return (raw or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")


def _google_headers(api_key: str) -> dict[str, str]:
    cleaned = _validate_http_api_key("Google", api_key)
    return {
        "x-goog-api-key": cleaned,
        "Content-Type": "application/json",
    }


def _google_supports_multimodal_function_response(model: str) -> bool:
    return bool(_GOOGLE_GEMINI_3_RE.match(model))


def _google_default_sort_key(raw_id: str) -> tuple[int, tuple[int, int], int, str]:
    lowered = raw_id.lower()
    is_lite = 1 if "lite" in lowered else 0
    is_preview = 1 if ("preview" in lowered or "exp" in lowered) else 0
    match = re.match(r"^gemini-(\d+)(?:\.(\d+))?", lowered)
    if match:
        version_key = (-int(match.group(1)), -int(match.group(2) or 0))
    else:
        version_key = (0, 0)
    return (is_lite, version_key, is_preview, lowered)


async def discover_google_http_models(
    api_key: str,
    *,
    http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
) -> tuple[list[ModelSpec], dict[str, dict[str, Any]], str | None]:
    """List Gemini models via ``GET /models``."""
    base = _google_base_url()
    pages, err_msg = await _fetch_http_model_pages(
        "Google",
        f"{base}/models",
        headers_fn=_google_headers,
        api_key=api_key,
        http_client_factory=http_client_factory,
        params_for_token=lambda tok: {
            "pageSize": "100",
            **({"pageToken": tok} if tok else {}),
        },
        next_token_fn=lambda payload: (
            payload.get("nextPageToken")
            if isinstance(payload.get("nextPageToken"), str)
            and payload.get("nextPageToken")
            else None
        ),
    )
    if err_msg is not None:
        return [], {}, err_msg

    raw_items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for payload in pages:
        models = payload.get("models")
        if not isinstance(models, list):
            continue
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

    if not raw_items:
        return [], {}, None

    default_id = next(
        (cand for cand in _GOOGLE_DEFAULT_ORDER if cand in seen_ids), None
    )
    if default_id is None:
        default_id = min(
            (item["raw_id"] for item in raw_items), key=_google_default_sort_key
        )
    specs: list[ModelSpec] = []
    model_info: dict[str, dict[str, Any]] = {}
    for item in raw_items:
        raw_id = item["raw_id"]
        efforts = ["low", "medium", "high"] if _GOOGLE_EFFORT_RE.match(raw_id) else []
        limit = item.get("inputTokenLimit")
        context_window = limit if _is_pos_int(limit) else None
        model_info[raw_id] = {"context_window": context_window}
        specs.append(
            ModelSpec(
                id=raw_id,
                raw_id=raw_id,
                provider="google",
                label=raw_id,
                efforts=efforts,
                is_default=(raw_id == default_id),
            )
        )
    return specs, model_info, None


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
    call: dict[str, Any], result: dict[str, Any], *, multimodal: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    is_error = bool(result.get("is_error"))
    text_out, images = _tool_text_and_images(result, is_error=is_error)
    resp_obj: dict[str, Any] = {
        "name": call["name"],
        "response": {"error": text_out} if is_error else {"output": text_out},
    }
    if call.get("id") is not None:
        resp_obj["id"] = call["id"]
    image_parts = [
        {
            "inlineData": {
                "mimeType": img.get("mimeType") or "image/png",
                "data": img.get("data") or "",
            }
        }
        for img in images
    ]
    if image_parts and multimodal:
        resp_obj["parts"] = image_parts
        return {"functionResponse": resp_obj}, []
    return {"functionResponse": resp_obj}, image_parts


def _finalize_google_interrupted(
    working_contents: list[Any],
    *,
    partial_text: str,
    completed_responses: list[dict[str, Any]],
    pending_calls: list[dict[str, Any]],
    tool_reason: str = "Error: Cancelled by user.",
    assistant_note: str = "(Interrupted by user.)",
) -> None:
    last = working_contents[-1] if working_contents else None
    if isinstance(last, dict) and last.get("role") == "model" and pending_calls:
        responses = list(completed_responses)
        for call in pending_calls[len(completed_responses) :]:
            err_resp: dict[str, Any] = {
                "name": call["name"],
                "response": {"error": tool_reason},
            }
            if call.get("id") is not None:
                err_resp["id"] = call["id"]
            responses.append({"functionResponse": err_resp})
        working_contents.append({"role": "user", "parts": responses})
        working_contents.append({"role": "model", "parts": [{"text": assistant_note}]})
        return
    if isinstance(last, dict) and last.get("role") == "user":
        suffix = _interrupted_text(partial_text, assistant_note)
        working_contents.append({"role": "model", "parts": [{"text": suffix}]})


def _repair_google_contents(contents: list[Any]) -> None:
    if not contents or not isinstance(contents[-1], dict):
        return
    last = contents[-1]
    if last.get("role") == "model":
        parts = last.get("parts")
        pending_calls = (
            [
                {
                    "name": str(part["functionCall"].get("name") or ""),
                    "id": part["functionCall"].get("id"),
                }
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("functionCall"), dict)
            ]
            if isinstance(parts, list)
            else []
        )
        if pending_calls:
            _finalize_google_interrupted(
                contents,
                partial_text="",
                completed_responses=[],
                pending_calls=pending_calls,
                tool_reason="Error: Previous turn was interrupted.",
                assistant_note="(Previous turn was interrupted.)",
            )
    elif last.get("role") == "user":
        _finalize_google_interrupted(
            contents,
            partial_text="",
            completed_responses=[],
            pending_calls=[],
            assistant_note="(Previous turn was interrupted.)",
        )


def _sanitize_google_for_model_switch(
    contents: list[Any], target_model: str
) -> list[Any]:
    multimodal = _google_supports_multimodal_function_response(target_model)
    cleaned: list[Any] = []
    for entry in contents:
        if not isinstance(entry, dict) or not isinstance(entry.get("parts"), list):
            cleaned.append(entry)
            continue
        role = entry.get("role")
        new_parts: list[Any] = []
        extracted_images: list[dict[str, Any]] = []
        for part in entry["parts"]:
            if not isinstance(part, dict):
                new_parts.append(part)
                continue
            if role == "model":
                if part.get("thought") is True:
                    continue
                new_part = {
                    k: v
                    for k, v in part.items()
                    if k not in ("thoughtSignature", "thought")
                }
                if multimodal and isinstance(new_part.get("functionCall"), dict):
                    new_part["thoughtSignature"] = "skip_thought_signature_validator"
                new_parts.append(new_part)
            elif (
                role == "user"
                and not multimodal
                and isinstance(part.get("functionResponse"), dict)
            ):
                fr = dict(part["functionResponse"])
                sub_parts = fr.pop("parts", None)
                if isinstance(sub_parts, list):
                    for sub in sub_parts:
                        if isinstance(sub, dict) and "inlineData" in sub:
                            extracted_images.append(sub)
                new_parts.append({"functionResponse": fr})
            else:
                new_parts.append(part)
        if role == "model" and not new_parts:
            new_parts = [{"text": "(continued)"}]
        cleaned.append({**entry, "parts": new_parts})
        if extracted_images:
            cleaned.append(
                {
                    "role": "user",
                    "parts": [
                        {"text": "Plots returned by the tool call(s) above:"},
                        *extracted_images,
                    ],
                }
            )
    return cleaned


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
    headers = _google_headers(api_key)
    store = HttpThreadStore(workspace_path, "google")
    session_id, saved_contents, saved_model, saved_window = _resume_state(
        store, resume_session_id
    )
    if saved_model and saved_model != model:
        saved_contents = _sanitize_google_for_model_switch(saved_contents, model)
    context_window = _resolve_context_window(
        model_info,
        saved_model=saved_model,
        target_model=model,
        saved_window=saved_window,
    )
    base = _google_base_url()
    model_slug = urllib.parse.quote(model.removeprefix("models/"), safe="")
    multimodal_fr = _google_supports_multimodal_function_response(model)
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
            include_thinking = bool(effort)
            for round_idx in range(1, MAX_TOOL_ROUNDS + 1):
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
                    saw_candidate = False
                    finish_reason: str | None = None
                    current_text_block: list[str] = []
                    round_model_parts = []
                    pending_calls = []
                    round_usage = None

                    async with client.stream(
                        "POST",
                        f"{base}/models/{model_slug}:streamGenerateContent?alt=sse",
                        headers=headers,
                        json=body,
                    ) as response:
                        if response.status_code != 200:
                            raw_err = await response.aread()
                            err_msg = _sanitize_error_text(
                                http_error_message("Google", response, raw_err), api_key
                            )
                            if _should_retry_without_effort(
                                response.status_code,
                                thinking_attempt,
                                include_thinking,
                                err_msg,
                                (
                                    "thinking",
                                    "thinkingconfig",
                                    "thinkinglevel",
                                    "thinkingbudget",
                                    "thinking_config",
                                    "thinking_level",
                                    "thinking_budget",
                                ),
                            ):
                                include_thinking = False
                                retry_without_thinking = True
                                continue
                            raise RuntimeError(err_msg)

                        async with aclosing(_iter_sse_json(response)) as events:
                            async for _, data in events:
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

                                prompt_fb = data.get("promptFeedback")
                                if isinstance(prompt_fb, dict) and prompt_fb.get(
                                    "blockReason"
                                ):
                                    raise RuntimeError(
                                        f"Google Gemini blocked the prompt ({prompt_fb['blockReason']})."
                                    )

                                if isinstance(data.get("usageMetadata"), dict):
                                    round_usage = dict(data["usageMetadata"])

                                candidates = data.get("candidates")
                                if not candidates:
                                    if (
                                        isinstance(data.get("usageMetadata"), dict)
                                        and not round_started
                                    ):
                                        round_started = True
                                        made_progress = True
                                        yield TurnStarted()
                                    continue

                                saw_candidate = True
                                if not round_started:
                                    round_started = True
                                    made_progress = True
                                    yield TurnStarted()

                                cand = (
                                    candidates[0]
                                    if isinstance(candidates[0], dict)
                                    else {}
                                )
                                cand_finish = cand.get("finishReason")
                                if isinstance(cand_finish, str) and cand_finish:
                                    finish_reason = cand_finish
                                content = cand.get("content") or {}
                                parts = content.get("parts") or []
                                for part in parts:
                                    if not isinstance(part, dict):
                                        continue
                                    _append_google_model_part(round_model_parts, part)
                                    if part.get("thought") is True:
                                        continue
                                    if (
                                        isinstance(part.get("text"), str)
                                        and part["text"]
                                    ):
                                        chunk = part["text"]
                                        current_text_block.append(chunk)
                                        partial_text_chunks.append(chunk)
                                        yield TextDelta(chunk)
                                    if isinstance(part.get("functionCall"), dict):
                                        if current_text_block:
                                            yield TextBlockDone(
                                                "".join(current_text_block)
                                            )
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
                        if not saw_candidate:
                            raise RuntimeError(
                                "Google Gemini stream ended without a response."
                            )
                        if finish_reason and finish_reason not in (
                            "STOP",
                            "FINISH_REASON_UNSPECIFIED",
                        ):
                            raise RuntimeError(
                                f"Google Gemini stopped early ({finish_reason})."
                            )
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
                    if _is_int(prompt_tokens) and prompt_tokens >= 0:
                        last_context_usage = {
                            "used_tokens": prompt_tokens,
                            "window_tokens": context_window,
                            "model": model,
                        }

                working_contents.append(
                    {
                        "role": "model",
                        "parts": round_model_parts or [{"text": "(no response)"}],
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

                if round_idx >= MAX_TOOL_ROUNDS:
                    raise RuntimeError(
                        f"Google Gemini turn exceeded maximum tool rounds ({MAX_TOOL_ROUNDS})."
                    )

                completed_responses = []
                extra_image_parts: list[dict[str, Any]] = []

                def extract_google_call(
                    call: dict[str, Any],
                ) -> tuple[str, Any, str | None]:
                    raw_args = call.get("args")
                    if raw_args is None:
                        return call["name"], {}, None
                    if isinstance(raw_args, str):
                        parsed_args, json_err = _parse_tool_arguments(raw_args)
                        return call["name"], parsed_args, json_err
                    return call["name"], raw_args, None

                async for call, result in _iter_tool_batch_results(
                    pending_calls,
                    tool_by_name=tool_by_name,
                    tool_timeout_s=tool_timeout_s,
                    extract_call=extract_google_call,
                ):
                    resp_part, img_parts = _google_function_response_part(
                        call, result, multimodal=multimodal_fr
                    )
                    completed_responses.append(resp_part)
                    extra_image_parts.extend(img_parts)
                working_contents.append({"role": "user", "parts": completed_responses})
                if extra_image_parts:
                    working_contents.append(
                        {
                            "role": "user",
                            "parts": [
                                {"text": "Plots returned by the tool call(s) above:"},
                                *extra_image_parts,
                            ],
                        }
                    )
                completed_responses = []
                pending_calls = []
                store.save(
                    session_id,
                    model=model,
                    messages=working_contents,
                    context_window=context_window,
                )
    except (asyncio.CancelledError, GeneratorExit):
        if made_progress:
            _finalize_google_interrupted(
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
    except Exception as exc:
        if made_progress:
            _finalize_google_interrupted(
                working_contents,
                partial_text="".join(partial_text_chunks),
                completed_responses=completed_responses,
                pending_calls=pending_calls,
                tool_reason="Error: Turn failed before tool execution completed.",
                assistant_note="(Error before response completed.)",
            )
            store.save(
                session_id,
                model=model,
                messages=working_contents,
                context_window=context_window,
            )
        if (sanitized := _sanitize_exception(exc, api_key)) is not None:
            raise sanitized from None
        raise


def _render_google_history_for_compaction(contents: list[Any]) -> str:
    lines: list[str] = []
    for entry in contents:
        if not isinstance(entry, dict) or not isinstance(entry.get("parts"), list):
            continue
        role = "User" if entry.get("role") == "user" else "Assistant"
        parts_text: list[str] = []
        for part in entry["parts"]:
            if not isinstance(part, dict):
                continue
            if part.get("thought") is True:
                continue
            if isinstance(part.get("text"), str) and part["text"].strip():
                parts_text.append(part["text"])
            elif "inlineData" in part:
                parts_text.append("[image omitted during compaction]")
            elif isinstance(part.get("functionCall"), dict):
                fc = part["functionCall"]
                args_json = json.dumps(fc.get("args") or {}, ensure_ascii=False)
                parts_text.append(f"[Tool call {fc.get('name') or ''}: {args_json}]")
            elif isinstance(part.get("functionResponse"), dict):
                fr = part["functionResponse"]
                resp_val = json.dumps(fr.get("response") or {}, ensure_ascii=False)
                sub = fr.get("parts") or ()
                omitted = " [plot image omitted]" if sub else ""
                parts_text.append(
                    f"[Tool result {fr.get('name') or ''}: {resp_val}{omitted}]"
                )
        if parts_text:
            lines.append(f"{role}: " + "\n".join(parts_text))
    return "\n\n".join(lines)


async def compact_google_http(
    *,
    api_key: str,
    system_prompt: str,
    workspace_path: Path,
    model: str | None,
    resume_session_id: str,
    model_info: Mapping[str, Any] | None = None,
    model_info_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
) -> CompactResult:
    """Compact an HTTP Google Gemini conversation by summarizing saved contents."""
    headers = _google_headers(api_key)
    base = _google_base_url()

    def build_request(
        target_model: str, compact_prompt: str
    ) -> tuple[str, dict[str, Any]]:
        model_slug = urllib.parse.quote(target_model.removeprefix("models/"), safe="")
        return f"{base}/models/{model_slug}:generateContent", {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": compact_prompt}]}],
        }

    def parse_response(payload: Any) -> tuple[str, int | None]:
        prompt_fb = payload.get("promptFeedback") if isinstance(payload, dict) else None
        if isinstance(prompt_fb, dict) and prompt_fb.get("blockReason"):
            raise RuntimeError(
                f"Google Gemini blocked compaction ({prompt_fb['blockReason']})."
            )
        summary_parts: list[str] = []
        candidates = payload.get("candidates") if isinstance(payload, dict) else None
        if isinstance(candidates, list) and candidates:
            cand = candidates[0] if isinstance(candidates[0], dict) else {}
            finish_reason = cand.get("finishReason")
            if finish_reason and finish_reason not in (
                "STOP",
                "FINISH_REASON_UNSPECIFIED",
            ):
                raise RuntimeError(
                    f"Google Gemini compaction stopped early ({finish_reason})."
                )
            for part in (cand.get("content") or {}).get("parts") or ():
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
        candidates_tokens = (
            (payload.get("usageMetadata") or {}).get("candidatesTokenCount")
            if isinstance(payload, dict)
            else None
        )
        return summary, (
            candidates_tokens
            if _is_int(candidates_tokens) and candidates_tokens >= 0
            else None
        )

    return await _compact_http_thread(
        provider="google",
        provider_title="Google",
        headers=headers,
        api_key=api_key,
        workspace_path=workspace_path,
        model=model,
        resume_session_id=resume_session_id,
        resolve_context=lambda target_model, saved_model, saved_window: (
            _resolve_context_window(
                (model_info_by_id or {}).get(target_model) or model_info,
                saved_model=saved_model,
                target_model=target_model,
                saved_window=saved_window,
            )
        ),
        repair_history=_repair_google_contents,
        render_history=_render_google_history_for_compaction,
        build_request=build_request,
        parse_response=parse_response,
        format_compacted=lambda summary: [
            {
                "role": "user",
                "parts": [
                    {"text": "Summary of our earlier conversation before compaction:"}
                ],
            },
            {"role": "model", "parts": [{"text": summary}]},
        ],
        http_client_factory=http_client_factory,
    )
