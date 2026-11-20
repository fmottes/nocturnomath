"""Google Antigravity CLI backend speaking ACP JSON-RPC over stdio."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import tempfile
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from ..auth import ProviderAuth, ProviderId
from ..mcp_bridge import McpToolBridge
from ..tools import ToolSpec
from .base import (
    DEFAULT_TOOL_TIMEOUT_S,
    EFFORT_LEVELS,
    CompactResult,
    JsonMessageStream,
    ModelSpec,
    SessionResumeError,
    SessionStarted,
    StderrCollector,
    TextBlockDone,
    TextDelta,
    TurnEvent,
    TurnResult,
    TurnStarted,
    find_cli_executables,
    is_usable_cli_binary,
    terminate_process,
)
from .http import (
    compact_google_http,
    discover_google_http_models,
    is_http_session_id,
    run_google_http_turn,
)

logger = logging.getLogger("nocturnomath.backends.google")

AGY_CLI_NAME = "agy.exe" if sys.platform == "win32" else "agy"
DISCOVERY_TIMEOUT_S = 15.0
COMPACT_TIMEOUT_S = 600.0
_EFFORT_SUFFIX_RE = re.compile(r"^(gemini-.+)-(low|medium|high|xhigh|max|ultra)$")
_EFFORT_LABEL_RE = re.compile(r"\s*\((Low|Medium|High|XHigh|Max|Ultra)\)\s*$")
_AGY_EXECUTOR_ERROR_PREFIXES = (
    "⚠ agent executor error:",
    "⚠ failed to construct executor:",
)


def _is_missing_session_error(text: str) -> bool:
    lowered = text.lower()
    if any(
        marker in lowered
        for marker in (
            "failed to read session",
            "session not found",
            "unknown session",
            "no session found",
            "no such session",
        )
    ):
        return True
    return "does not exist" in lowered and "session" in lowered


def _extract_agy_executor_error(text: Any) -> str | None:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if stripped.lower().startswith(_AGY_EXECUTOR_ERROR_PREFIXES):
        return stripped.removeprefix("⚠").strip()
    return None


def _usage_to_context(usage: Any, model: str | None) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    inp = usage.get("inputTokens")
    cached = usage.get("cachedReadTokens", 0)
    if isinstance(inp, int) and inp >= 0:
        return {
            "used_tokens": inp + (cached if isinstance(cached, int) else 0),
            "window_tokens": None,
            "model": model,
        }
    return None


def agy_cli_candidates() -> list[Path]:
    """Locate an installed Antigravity CLI (``agy``) on PATH."""
    return find_cli_executables(AGY_CLI_NAME)


def _parse_gemini_models(
    available_models: list[Any],
    *,
    current_model_id: str | None = None,
) -> tuple[list[ModelSpec], dict[tuple[str, str], str]]:
    grouped: dict[str, dict[str, Any]] = {}
    slug_map: dict[tuple[str, str], str] = {}
    for item in available_models:
        if not isinstance(item, dict):
            continue
        model_id = item.get("modelId")
        name = item.get("name") or model_id
        if not isinstance(model_id, str) or not model_id.startswith("gemini-"):
            continue
        match = _EFFORT_SUFFIX_RE.fullmatch(model_id)
        if match:
            base_id, effort = match.group(1), match.group(2)
            base_label = _EFFORT_LABEL_RE.sub("", str(name)).strip() or base_id
            slug_map[(base_id, effort)] = model_id
        else:
            base_id, effort = model_id, None
            base_label = str(name).strip() or base_id

        entry = grouped.setdefault(base_id, {"label": base_label, "efforts": []})
        if effort and effort in EFFORT_LEVELS and effort not in entry["efforts"]:
            entry["efforts"].append(effort)

    default_base: str | None = None
    if isinstance(current_model_id, str) and current_model_id.startswith("gemini-"):
        match = _EFFORT_SUFFIX_RE.fullmatch(current_model_id)
        default_base = match.group(1) if match else current_model_id
    if default_base not in grouped:
        default_base = None

    specs: list[ModelSpec] = []
    for index, (base_id, entry) in enumerate(grouped.items()):
        ordered_efforts = [e for e in EFFORT_LEVELS if e in entry["efforts"]]
        specs.append(
            ModelSpec(
                id=base_id,
                raw_id=base_id,
                provider="google",
                label=entry["label"],
                efforts=ordered_efforts,
                is_default=(
                    (base_id == default_base) if default_base else (index == 0)
                ),
            )
        )
    return specs, slug_map


def _mcp_server_entry(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "explore",
        "command": cfg["command"],
        "args": list(cfg.get("args") or ()),
        "env": [
            {"name": str(key), "value": str(value)}
            for key, value in (cfg.get("env") or {}).items()
        ],
    }


class GoogleBackend:
    """Drive the local ``agy acp`` server over JSON-RPC stdio."""

    provider: ProviderId = "google"
    uses_compact_hook: bool = False

    def __init__(
        self,
        cli_path: Path | None = None,
        *,
        http_client_factory: Callable[..., Any] | None = None,
    ):
        self.cli_path = cli_path
        self._http_client_factory = http_client_factory
        self._http_model_info: dict[str, dict[str, Any]] = {}
        self._slug_map: dict[tuple[str, str], str] = {}
        self._next_rpc_id = 0

    def _resolve_cli(self) -> Path:
        if self.cli_path is not None and is_usable_cli_binary(self.cli_path):
            return self.cli_path
        candidates = agy_cli_candidates()
        if not candidates:
            raise RuntimeError(
                "Antigravity CLI (`agy`) was not found on PATH. Install agy or sign in."
            )
        self.cli_path = candidates[0]
        return self.cli_path

    @staticmethod
    def _prepare_env_and_flags(auth: ProviderAuth) -> tuple[dict[str, str], list[str]]:
        return auth.process_env(), ["--include-default-customizations=false"]

    def _next_id(self) -> int:
        self._next_rpc_id += 1
        return self._next_rpc_id

    @staticmethod
    async def _send(proc: asyncio.subprocess.Process, payload: dict[str, Any]) -> None:
        if proc.stdin is None:
            raise RuntimeError("Antigravity ACP stdin is not open.")
        proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        await proc.stdin.drain()

    async def _handle_request(
        self,
        proc: asyncio.subprocess.Process,
        msg: dict[str, Any],
        allowed_tools: set[str],
    ) -> None:
        req_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}
        if method == "session/request_permission":
            tool_call = params.get("toolCall") or {}
            tool_name = str(tool_call.get("name") or "")
            options = params.get("options") or []
            allowed = tool_name in allowed_tools
            target_kinds = (
                ("allow_once", "allow_always")
                if allowed
                else ("reject_once", "reject_always")
            )
            chosen = next(
                (
                    opt.get("optionId")
                    for opt in options
                    if isinstance(opt, dict)
                    and opt.get("kind") in target_kinds
                    and isinstance(opt.get("optionId"), str)
                    and opt.get("optionId")
                ),
                None,
            )
            outcome = (
                {"outcome": "selected", "optionId": chosen}
                if chosen is not None
                else {"outcome": "cancelled"}
            )
            await self._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"outcome": outcome},
                },
            )
            return
        await self._send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unsupported method: {method}"},
            },
        )

    async def _rpc(
        self,
        proc: asyncio.subprocess.Process,
        messages: JsonMessageStream,
        method: str,
        params: dict[str, Any],
        allowed_tools: set[str] = frozenset(),
        stderr: StderrCollector | None = None,
    ) -> dict[str, Any]:
        req_id = self._next_id()
        try:
            await self._send(
                proc,
                {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params},
            )
        except (BrokenPipeError, ConnectionResetError) as exc:
            detail = await stderr.text() if stderr is not None else ""
            raise RuntimeError(
                detail or f"Antigravity ACP closed before replying to {method}."
            ) from exc
        while (msg := await messages.next_wire()) is not None:
            if msg.get("id") is not None and "method" in msg:
                await self._handle_request(proc, msg, allowed_tools)
                continue
            if msg.get("id") == req_id:
                if "error" in msg and msg["error"] is not None:
                    err = msg["error"]
                    text = err.get("message") if isinstance(err, dict) else str(err)
                    raise RuntimeError(text or f"Antigravity ACP {method} failed.")
                return msg.get("result") or {}
            if msg.get("id") is None:
                messages.buffer_notification(msg)
        detail = await stderr.text() if stderr is not None else ""
        raise RuntimeError(
            detail or f"Antigravity ACP closed before replying to {method}."
        )

    async def _initialize_acp(
        self,
        proc: asyncio.subprocess.Process,
        messages: JsonMessageStream,
        stderr: StderrCollector | None = None,
    ) -> None:
        await self._rpc(
            proc,
            messages,
            "initialize",
            {"protocolVersion": 1, "clientCapabilities": {}},
            stderr=stderr,
        )

    async def discover_models_staged(
        self, auth: ProviderAuth
    ) -> tuple[list[ModelSpec], Callable[[], None]]:
        api_key = auth.direct_api_key()
        if api_key is not None:
            specs, info = await discover_google_http_models(
                api_key, http_client_factory=self._http_client_factory
            )

            def commit_http(
                discovered_info: dict[str, dict[str, Any]] = info,
            ) -> None:
                self._http_model_info = dict(discovered_info)

            return specs, commit_http

        candidates = agy_cli_candidates()
        if not candidates:
            return [], lambda: None
        env, flags = self._prepare_env_and_flags(auth)
        for candidate in candidates:
            with tempfile.TemporaryDirectory(
                prefix="nocturnomath-agy-cwd-", ignore_cleanup_errors=True
            ) as isolated_cwd:
                proc: asyncio.subprocess.Process | None = None
                stderr: StderrCollector | None = None
                try:
                    async with asyncio.timeout(DISCOVERY_TIMEOUT_S):
                        proc = await asyncio.create_subprocess_exec(
                            str(candidate),
                            "acp",
                            *flags,
                            stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd=isolated_cwd,
                            env=env,
                            start_new_session=(os.name == "posix"),
                        )
                        stderr = StderrCollector(proc.stderr)
                        messages = JsonMessageStream(proc.stdout)
                        await self._initialize_acp(proc, messages, stderr)
                        session_info = await self._rpc(
                            proc,
                            messages,
                            "session/new",
                            {
                                "cwd": isolated_cwd,
                                "mcpServers": [],
                                "_meta": {"disableBuiltInTools": True},
                            },
                            stderr=stderr,
                        )
                        models_state = session_info.get("models") or {}
                        available = models_state.get("availableModels") or []
                        current_model_id = models_state.get("currentModelId")
                        specs, slug_map = _parse_gemini_models(
                            available, current_model_id=current_model_id
                        )
                        if specs:

                            def commit(
                                chosen: Path = candidate,
                                discovered_slugs: dict[tuple[str, str], str] = slug_map,
                            ) -> None:
                                self._slug_map = dict(discovered_slugs)
                                self.cli_path = chosen

                            return specs, commit
                except Exception as exc:
                    logger.warning(
                        "Google Antigravity model discovery with %s failed: %s",
                        candidate,
                        exc,
                    )
                finally:
                    if proc is not None:
                        await terminate_process(proc, stderr)
                    elif stderr is not None:
                        await stderr.close()
        return [], lambda: None

    async def discover_models(self, auth: ProviderAuth) -> list[ModelSpec]:
        specs, commit = await self.discover_models_staged(auth)
        commit()
        return specs

    async def _configure_model(
        self,
        proc: asyncio.subprocess.Process,
        messages: JsonMessageStream,
        session_id: str,
        model: str,
        effort: str | None,
        stderr: StderrCollector | None = None,
    ) -> None:
        slug = self._slug_map.get((model, effort or "")) or (
            f"{model}-{effort}" if effort else model
        )
        try:
            await self._rpc(
                proc,
                messages,
                "session/set_model",
                {"sessionId": session_id, "modelId": slug},
                stderr=stderr,
            )
            return
        except RuntimeError:
            pass
        await self._rpc(
            proc,
            messages,
            "session/set_config_option",
            {"sessionId": session_id, "configId": "model", "value": model},
            stderr=stderr,
        )
        if effort:
            await self._rpc(
                proc,
                messages,
                "session/set_config_option",
                {
                    "sessionId": session_id,
                    "configId": "thought_level",
                    "value": effort,
                },
                stderr=stderr,
            )

    async def _drain_setup_updates(
        self,
        proc: asyncio.subprocess.Process,
        messages: JsonMessageStream,
        allowed_tools: set[str] = frozenset(),
    ) -> set[str] | None:
        available_commands: set[str] | None = None

        def inspect(msg: dict[str, Any]) -> None:
            nonlocal available_commands
            if msg.get("method") != "session/update":
                return
            update = (msg.get("params") or {}).get("update") or {}
            if update.get("sessionUpdate") == "available_commands_update":
                available_commands = {
                    str(cmd.get("name"))
                    for cmd in (update.get("availableCommands") or [])
                    if isinstance(cmd, dict) and cmd.get("name")
                }

        while messages._buffered:
            inspect(messages._buffered.popleft())
        while True:
            try:
                msg = await messages.next_wire(timeout=0.02)
            except TimeoutError:
                break
            if msg is None:
                break
            if msg.get("id") is not None and "method" in msg:
                await self._handle_request(proc, msg, allowed_tools)
                continue
            inspect(msg)
        return available_commands

    async def run_turn(
        self,
        prompt: str,
        *,
        system_prompt: str,
        tools: Sequence[ToolSpec],
        workspace_path: Path,
        model: str,
        effort: str | None,
        auth: ProviderAuth,
        resume_session_id: str | None = None,
        tool_timeout_s: int = DEFAULT_TOOL_TIMEOUT_S,
    ) -> AsyncIterator[TurnEvent]:
        api_key = auth.direct_api_key()
        if api_key is not None:
            async for event in run_google_http_turn(
                prompt,
                api_key=api_key,
                system_prompt=system_prompt,
                tools=tools,
                workspace_path=workspace_path,
                model=model,
                effort=effort,
                resume_session_id=resume_session_id,
                tool_timeout_s=tool_timeout_s,
                model_info=self._http_model_info.get(model),
                http_client_factory=self._http_client_factory,
            ):
                yield event
            return
        if is_http_session_id(resume_session_id):
            raise SessionResumeError(
                "Session was created in HTTP API mode and cannot be resumed by the CLI."
            )
        del tool_timeout_s
        cli_path = self._resolve_cli()
        env, flags = self._prepare_env_and_flags(auth)
        allowed_tools = {
            name
            for spec in tools
            for name in (spec.name, f"mcp_explore_{spec.name}", f"explore/{spec.name}")
        }
        with tempfile.TemporaryDirectory(
            prefix="nocturnomath-agy-cwd-", ignore_cleanup_errors=True
        ) as isolated_cwd:
            async with McpToolBridge(tools) as bridge:
                mcp_servers = [_mcp_server_entry(bridge.stdio_server_config())]
                proc = await asyncio.create_subprocess_exec(
                    str(cli_path),
                    "acp",
                    *flags,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=isolated_cwd,
                    env=env,
                    start_new_session=(os.name == "posix"),
                )
                stderr = StderrCollector(proc.stderr)
                session_id = resume_session_id
                try:
                    messages = JsonMessageStream(proc.stdout)
                    await self._initialize_acp(proc, messages, stderr)
                    session_params: dict[str, Any] = {
                        "cwd": isolated_cwd,
                        "mcpServers": mcp_servers,
                        "_meta": {
                            "disableBuiltInTools": True,
                            "antigravity": {"systemPrompt": {"append": system_prompt}},
                        },
                    }
                    if resume_session_id:
                        try:
                            started = await self._rpc(
                                proc,
                                messages,
                                "session/resume",
                                {"sessionId": resume_session_id, **session_params},
                                allowed_tools,
                                stderr,
                            )
                        except RuntimeError as exc:
                            if _is_missing_session_error(str(exc)):
                                raise SessionResumeError(str(exc)) from exc
                            raise
                    else:
                        started = await self._rpc(
                            proc,
                            messages,
                            "session/new",
                            session_params,
                            allowed_tools,
                            stderr,
                        )
                    session_id = started.get("sessionId") or resume_session_id
                    if not isinstance(session_id, str) or not session_id:
                        raise RuntimeError(
                            "Antigravity ACP did not return a sessionId."
                        )

                    await self._configure_model(
                        proc, messages, session_id, model, effort, stderr
                    )
                    await self._drain_setup_updates(proc, messages, allowed_tools)

                    prompt_req_id = self._next_id()
                    try:
                        await self._send(
                            proc,
                            {
                                "jsonrpc": "2.0",
                                "id": prompt_req_id,
                                "method": "session/prompt",
                                "params": {
                                    "sessionId": session_id,
                                    "prompt": [{"type": "text", "text": prompt}],
                                },
                            },
                        )
                    except (BrokenPipeError, ConnectionResetError) as exc:
                        detail = await stderr.text()
                        raise RuntimeError(
                            detail
                            or "Antigravity ACP disconnected before receiving prompt."
                        ) from exc
                    yield SessionStarted(session_id)

                    current_block: list[str] = []
                    last_context_usage: dict[str, Any] | None = None
                    prompt_done = False

                    async for msg in messages:
                        if msg.get("id") is not None and "method" in msg:
                            await self._handle_request(proc, msg, allowed_tools)
                            continue
                        if msg.get("method") == "session/update":
                            update = (msg.get("params") or {}).get("update") or {}
                            kind = update.get("sessionUpdate")
                            if kind == "agent_thought_chunk":
                                yield TurnStarted()
                            elif kind == "agent_message_chunk":
                                content = update.get("content") or {}
                                chunk = (
                                    content.get("text")
                                    if isinstance(content, dict)
                                    else None
                                )
                                if isinstance(chunk, str) and chunk:
                                    err_text = _extract_agy_executor_error(chunk)
                                    if err_text is not None:
                                        raise RuntimeError(err_text)
                                    current_block.append(chunk)
                                    yield TextDelta(chunk)
                            elif kind == "tool_call":
                                yield TurnStarted()
                                if current_block:
                                    yield TextBlockDone("".join(current_block))
                                    current_block.clear()
                            elif kind == "usage_update":
                                used = update.get("used")
                                size = update.get("size")
                                if isinstance(used, int) and used > 0:
                                    last_context_usage = {
                                        "used_tokens": used,
                                        "window_tokens": size
                                        if isinstance(size, int) and size > 0
                                        else None,
                                        "model": model,
                                    }
                        elif msg.get("id") == prompt_req_id:
                            prompt_done = True
                            if "error" in msg and msg["error"] is not None:
                                err = msg["error"]
                                text = (
                                    err.get("message")
                                    if isinstance(err, dict)
                                    else str(err)
                                )
                                raise RuntimeError(
                                    text
                                    or "Google Antigravity could not finish this turn."
                                )
                            if current_block:
                                yield TextBlockDone("".join(current_block))
                                current_block.clear()
                            result = msg.get("result") or {}
                            if result.get("stopReason") == "cancelled":
                                raise asyncio.CancelledError(
                                    "Google Antigravity turn was cancelled."
                                )
                            usage = result.get("usage")
                            if last_context_usage is None:
                                last_context_usage = _usage_to_context(usage, model)
                            yield TurnResult(
                                usage=usage if isinstance(usage, dict) else None,
                                context_usage=last_context_usage,
                                num_turns=1,
                            )
                            break

                    if not prompt_done:
                        detail = await stderr.text()
                        raise RuntimeError(
                            detail
                            or "Antigravity ACP disconnected before completing the turn."
                        )
                except (asyncio.CancelledError, GeneratorExit):
                    if session_id:
                        with suppress(Exception, TimeoutError):
                            await asyncio.wait_for(
                                self._send(
                                    proc,
                                    {
                                        "jsonrpc": "2.0",
                                        "method": "session/cancel",
                                        "params": {"sessionId": session_id},
                                    },
                                ),
                                timeout=0.5,
                            )
                    await bridge.cancel_in_flight()
                    raise
                finally:
                    await terminate_process(proc, stderr)

    async def compact(
        self,
        *,
        system_prompt: str,
        tools: Sequence[ToolSpec],
        workspace_path: Path,
        model: str | None,
        effort: str | None,
        auth: ProviderAuth,
        resume_session_id: str,
    ) -> CompactResult:
        api_key = auth.direct_api_key()
        if api_key is not None:
            return await compact_google_http(
                api_key=api_key,
                system_prompt=system_prompt,
                workspace_path=workspace_path,
                model=model,
                resume_session_id=resume_session_id,
                model_info=self._http_model_info.get(model or ""),
                http_client_factory=self._http_client_factory,
            )
        if is_http_session_id(resume_session_id):
            raise SessionResumeError(
                "Session was created in HTTP API mode and cannot be resumed by the CLI."
            )
        cli_path = self._resolve_cli()
        env, flags = self._prepare_env_and_flags(auth)
        with tempfile.TemporaryDirectory(
            prefix="nocturnomath-agy-cwd-", ignore_cleanup_errors=True
        ) as isolated_cwd:
            proc = await asyncio.create_subprocess_exec(
                str(cli_path),
                "acp",
                *flags,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=isolated_cwd,
                env=env,
                start_new_session=(os.name == "posix"),
            )
            stderr = StderrCollector(proc.stderr)
            try:
                messages = JsonMessageStream(proc.stdout)
                context_usage: dict[str, Any] | None = None
                done = False
                unexpected_chunks: list[str] = []
                try:
                    async with asyncio.timeout(COMPACT_TIMEOUT_S):
                        await self._initialize_acp(proc, messages, stderr)
                        try:
                            await self._rpc(
                                proc,
                                messages,
                                "session/resume",
                                {
                                    "sessionId": resume_session_id,
                                    "cwd": isolated_cwd,
                                    "mcpServers": [],
                                    "_meta": {
                                        "disableBuiltInTools": True,
                                        "antigravity": {
                                            "systemPrompt": {"append": system_prompt}
                                        },
                                    },
                                },
                                stderr=stderr,
                            )
                        except RuntimeError as exc:
                            if _is_missing_session_error(str(exc)):
                                raise SessionResumeError(str(exc)) from exc
                            raise
                        if model:
                            await self._configure_model(
                                proc, messages, resume_session_id, model, effort, stderr
                            )
                        commands = await self._drain_setup_updates(proc, messages)
                        if commands is not None and "compact" not in commands:
                            raise RuntimeError(
                                "Antigravity ACP does not support /compact for this session."
                            )
                        req_id = self._next_id()
                        try:
                            await self._send(
                                proc,
                                {
                                    "jsonrpc": "2.0",
                                    "id": req_id,
                                    "method": "session/prompt",
                                    "params": {
                                        "sessionId": resume_session_id,
                                        "prompt": [
                                            {"type": "text", "text": "/compact"}
                                        ],
                                    },
                                },
                            )
                        except (BrokenPipeError, ConnectionResetError) as exc:
                            detail = await stderr.text()
                            raise RuntimeError(
                                detail
                                or "Antigravity ACP disconnected before compacting."
                            ) from exc
                        async for msg in messages:
                            if msg.get("id") is not None and "method" in msg:
                                await self._handle_request(proc, msg, set())
                                continue
                            if msg.get("method") == "session/update":
                                update = (msg.get("params") or {}).get("update") or {}
                                kind = update.get("sessionUpdate")
                                if kind == "available_commands_update":
                                    commands = {
                                        str(cmd.get("name"))
                                        for cmd in (
                                            update.get("availableCommands") or []
                                        )
                                        if isinstance(cmd, dict) and cmd.get("name")
                                    }
                                    if "compact" not in commands:
                                        raise RuntimeError(
                                            "Antigravity ACP does not support /compact for this session."
                                        )
                                elif kind == "agent_message_chunk":
                                    content = update.get("content") or {}
                                    chunk = (
                                        content.get("text")
                                        if isinstance(content, dict)
                                        else None
                                    )
                                    err_text = _extract_agy_executor_error(chunk)
                                    if err_text is not None:
                                        raise RuntimeError(err_text)
                                    if isinstance(chunk, str) and chunk.strip():
                                        unexpected_chunks.append(chunk.strip())
                                elif kind == "usage_update":
                                    used = update.get("used")
                                    size = update.get("size")
                                    if isinstance(used, int) and used >= 0:
                                        context_usage = {
                                            "used_tokens": used,
                                            "window_tokens": size
                                            if isinstance(size, int) and size > 0
                                            else None,
                                            "model": model,
                                        }
                            elif msg.get("id") == req_id:
                                done = True
                                if "error" in msg and msg["error"] is not None:
                                    err = msg["error"]
                                    text = (
                                        err.get("message")
                                        if isinstance(err, dict)
                                        else str(err)
                                    )
                                    raise RuntimeError(
                                        text or "Antigravity compaction failed."
                                    )
                                if unexpected_chunks:
                                    raise RuntimeError(" ".join(unexpected_chunks))
                                result = msg.get("result") or {}
                                if result.get("stopReason") == "cancelled":
                                    raise asyncio.CancelledError(
                                        "Google Antigravity compaction was cancelled."
                                    )
                                raw_usage = result.get("usage")
                                if context_usage is None:
                                    context_usage = _usage_to_context(raw_usage, model)
                                if context_usage is None and not isinstance(
                                    raw_usage, dict
                                ):
                                    raise RuntimeError(
                                        "Antigravity did not confirm that compaction completed."
                                    )
                                break
                except TimeoutError as exc:
                    raise RuntimeError(
                        f"Google Antigravity compaction timed out after {int(COMPACT_TIMEOUT_S)}s."
                    ) from exc
                if not done:
                    detail = await stderr.text()
                    raise RuntimeError(
                        detail
                        or "Antigravity did not confirm that compaction completed."
                    )
                return CompactResult(
                    session_id=resume_session_id, context_usage=context_usage
                )
            finally:
                await terminate_process(proc, stderr)
