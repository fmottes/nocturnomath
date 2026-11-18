"""Anthropic Claude CLI backend speaking stream-json over stdio."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import shlex
import sys
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from ..auth import ProviderAuth, ProviderId
from ..mcp_bridge import McpToolBridge
from ..tools import ToolSpec
from .base import (
    DEFAULT_TOOL_TIMEOUT_S,
    EFFORT_LEVELS,
    CompactionNotice,
    CompactResult,
    MessageUsage,
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
    read_jsonl,
    terminate_process,
)

logger = logging.getLogger("nocturnomath.backends.anthropic")

CLAUDE_CLI_NAME = "claude.exe" if sys.platform == "win32" else "claude"
_CLAUDE_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
DISCOVERY_TIMEOUT_S = 15.0
COMPACT_TIMEOUT_S = 600.0
CONTEXT_USAGE_TIMEOUT_S = 10.0

_CLAUDE_CREDENTIAL_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)


def _has_claude_credentials(
    resp: dict[str, Any], auth: ProviderAuth, env: Mapping[str, str]
) -> bool:
    if auth.method != "cli":
        return True
    if any(env.get(name) for name in _CLAUDE_CREDENTIAL_VARS):
        return True
    account = resp.get("account")
    if isinstance(account, dict):
        api_key_source = account.get("apiKeySource")
        if (
            isinstance(api_key_source, str)
            and api_key_source
            and api_key_source != "none"
        ):
            return True
    return not (
        isinstance(account, dict)
        and account.get("tokenSource") == "none"
        and account.get("apiProvider") in (None, "firstParty")
    )


def _is_missing_session_error(text: str) -> bool:
    return "no conversation found" in text.lower()


def _bundled_claude_cli() -> Path | None:
    """Locate a bundled Claude binary if claude_agent_sdk happens to be installed."""
    with suppress(ImportError, ValueError):
        spec = importlib.util.find_spec("claude_agent_sdk")
        if spec and spec.origin:
            candidate = Path(spec.origin).with_name("_bundled") / CLAUDE_CLI_NAME
            if candidate.is_file():
                return candidate
    return None


def claude_cli_candidates() -> list[Path]:
    """Prefer an installed Claude CLI on PATH, falling back to any bundled binary."""
    bundled = _bundled_claude_cli()
    extra = (bundled,) if bundled is not None else ()
    return find_cli_executables(CLAUDE_CLI_NAME, extra_candidates=extra)


def _error_result_text(message: dict[str, Any]) -> str:
    errors = message.get("errors")
    if isinstance(errors, list):
        parts = [str(item).strip() for item in errors if str(item).strip()]
        if parts:
            return "; ".join(parts)
    result = message.get("result")
    if isinstance(result, str) and result.strip():
        return result.strip()
    subtype = message.get("subtype")
    if isinstance(subtype, str) and subtype and subtype != "success":
        return subtype
    return "Claude could not finish this turn."


def _context_usage_from_turn(
    last_prompt_tokens: int | None,
    last_prompt_model: str | None,
    models: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if last_prompt_tokens is None:
        return None
    models = models or {}
    model_usage = models.get(last_prompt_model or "")
    if model_usage is None:
        model_usage = next(
            (
                value
                for value in models.values()
                if isinstance(value, dict)
                and value.get("canonicalModel") == last_prompt_model
            ),
            None,
        )
    if model_usage is None and len(models) == 1:
        only = next(iter(models.values()))
        if isinstance(only, dict):
            model_usage = only
    window = (model_usage or {}).get("contextWindow")
    return {
        "used_tokens": last_prompt_tokens,
        "window_tokens": window if isinstance(window, int) and window > 0 else None,
        "model": last_prompt_model,
    }


def _parse_models(entries: list[Any]) -> list[ModelSpec]:
    specs: dict[str, ModelSpec] = {}
    default_id: str | None = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        resolved = entry.get("resolvedModel")
        if not isinstance(value, str) or not value:
            continue
        identifier = resolved if isinstance(resolved, str) and resolved else None
        if value == "default":
            default_id = identifier
        if identifier is None and value != "default":
            identifier = value
        if identifier is None:
            continue
        raw_levels = entry.get("supportedEffortLevels")
        supported = (
            [level for level in raw_levels if level in EFFORT_LEVELS]
            if isinstance(raw_levels, list)
            else list(_CLAUDE_EFFORT_LEVELS)
            if entry.get("supportsEffort")
            else []
        )
        existing = specs.get(identifier)
        merged_efforts = list(existing.efforts) if existing else []
        merged_efforts.extend(
            level for level in supported if level not in merged_efforts
        )
        specs[identifier] = ModelSpec(
            id=identifier,
            raw_id=identifier,
            provider="anthropic",
            label=identifier,
            efforts=merged_efforts,
            is_default=(identifier == default_id)
            or (existing.is_default if existing else False),
        )
    if default_id and default_id in specs:
        target = specs[default_id]
        specs[default_id] = ModelSpec(
            id=target.id,
            raw_id=target.raw_id,
            provider="anthropic",
            label=target.label,
            efforts=target.efforts,
            is_default=True,
        )
    return list(specs.values())


class AnthropicBackend:
    """Drive the local ``claude`` CLI over ``stream-json`` stdio."""

    provider: ProviderId = "anthropic"
    uses_compact_hook: bool = True

    def __init__(self, cli_path: Path | None = None):
        self.cli_path = cli_path

    def _resolve_cli(self) -> Path:
        if self.cli_path is not None and is_usable_cli_binary(self.cli_path):
            return self.cli_path
        candidates = claude_cli_candidates()
        if not candidates:
            raise RuntimeError(
                "Claude CLI (`claude`) was not found on PATH. Install Claude Code or sign in."
            )
        self.cli_path = candidates[0]
        return self.cli_path

    @staticmethod
    async def _write_json(proc: asyncio.subprocess.Process, payload: dict[str, Any]):
        if proc.stdin is None:
            raise RuntimeError("Claude CLI stdin is not open.")
        proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        await proc.stdin.drain()

    async def _await_initialize(
        self,
        proc: asyncio.subprocess.Process,
        messages: AsyncIterator[dict[str, Any]],
        init_id: str,
        stderr: StderrCollector,
        *,
        resume_session_id: str | None = None,
        allowed_tools: set[str] = frozenset(),
    ) -> dict[str, Any]:
        async for msg in messages:
            msg_type = msg.get("type")
            if (
                msg_type == "control_response"
                and (msg.get("response") or {}).get("request_id") == init_id
            ):
                resp = msg.get("response") or {}
                if resp.get("subtype") == "error":
                    raise RuntimeError(
                        str(resp.get("error") or "Claude CLI initialization failed.")
                    )
                return resp.get("response") or {}
            if msg_type == "control_request":
                await self._handle_control_request(proc, msg, allowed_tools)
            elif msg_type == "result" and msg.get("is_error"):
                detail = _error_result_text(msg)
                if resume_session_id and _is_missing_session_error(detail):
                    raise SessionResumeError(detail)
                raise RuntimeError(detail)
        detail = await stderr.text()
        rc = await proc.wait()
        if resume_session_id and _is_missing_session_error(detail):
            raise SessionResumeError(detail)
        raise RuntimeError(
            detail
            or f"Claude CLI exited before completing initialization (status {rc})."
        )

    async def discover_models_staged(
        self, auth: ProviderAuth
    ) -> tuple[list[ModelSpec], Callable[[], None]]:
        env = auth.process_env()
        for candidate in claude_cli_candidates():
            proc: asyncio.subprocess.Process | None = None
            stderr: StderrCollector | None = None
            try:
                async with asyncio.timeout(DISCOVERY_TIMEOUT_S):
                    proc = await asyncio.create_subprocess_exec(
                        str(candidate),
                        "--output-format",
                        "stream-json",
                        "--input-format",
                        "stream-json",
                        "--verbose",
                        "--tools",
                        "",
                        "--strict-mcp-config",
                        "--setting-sources",
                        "user",
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=env,
                        start_new_session=(os.name == "posix"),
                    )
                    stderr = StderrCollector(proc.stderr)
                    req_id = await self._send_control(
                        proc, "initialize", {"skills": []}
                    )
                    resp = await self._await_initialize(
                        proc, read_jsonl(proc.stdout), req_id, stderr
                    )
                    if not _has_claude_credentials(resp, auth, env):

                        def commit(chosen: Path = candidate) -> None:
                            self.cli_path = chosen

                        return [], commit
                    raw_models = resp.get("models")
                    entries = raw_models if isinstance(raw_models, list) else []
                    models = _parse_models(entries)
                    if models:

                        def commit(chosen: Path = candidate) -> None:
                            self.cli_path = chosen

                        return models, commit
                    logger.warning(
                        "Claude CLI at %s reported no usable models", candidate
                    )
            except Exception as exc:
                logger.warning(
                    "Claude model discovery with %s failed: %s", candidate, exc
                )
            finally:
                if proc is not None:
                    await terminate_process(proc, stderr)
                elif stderr is not None:
                    await stderr.close()
        return [], lambda: None

    async def discover_models(self, auth: ProviderAuth) -> list[ModelSpec]:
        models, commit = await self.discover_models_staged(auth)
        commit()
        return models

    def _build_turn_cmd(
        self,
        cli_path: Path,
        *,
        system_prompt: str,
        mcp_config: dict[str, Any] | None,
        model: str | None,
        effort: str | None,
        resume_session_id: str | None,
    ) -> list[str]:
        compact_hook = shlex.join([sys.executable, "-m", "nocturnomath.compaction"])
        settings = json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "matcher": "compact",
                            "hooks": [{"type": "command", "command": compact_hook}],
                        }
                    ]
                }
            }
        )
        cmd = [
            str(cli_path),
            "--output-format",
            "stream-json",
            "--input-format",
            "stream-json",
            "--verbose",
            "--system-prompt",
            system_prompt,
            "--tools",
            "",
            "--permission-mode",
            "bypassPermissions",
            "--setting-sources",
            "user",
            "--strict-mcp-config",
            "--settings",
            settings,
            "--include-partial-messages",
        ]
        if mcp_config is not None:
            cmd.extend(["--mcp-config", json.dumps({"mcpServers": mcp_config})])
        if model:
            cmd.extend(["--model", model])
        if effort:
            cmd.extend(["--effort", effort])
        if resume_session_id:
            cmd.append(f"--resume={resume_session_id}")
        return cmd

    async def _send_control(
        self,
        proc: asyncio.subprocess.Process,
        subtype: str,
        payload: dict[str, Any] | None = None,
    ) -> str:
        req_id = f"{subtype}_{uuid.uuid4().hex[:8]}"
        await self._write_json(
            proc,
            {
                "type": "control_request",
                "request_id": req_id,
                "request": {"subtype": subtype, **(payload or {})},
            },
        )
        return req_id

    async def _handle_control_request(
        self,
        proc: asyncio.subprocess.Process,
        msg: dict[str, Any],
        allowed_tools: set[str] = frozenset(),
    ) -> None:
        req_id = msg.get("request_id")
        request = msg.get("request") or {}
        if not req_id:
            return
        subtype = request.get("subtype")
        if subtype == "can_use_tool":
            tool_name = str(
                request.get("tool_name") or request.get("name") or ""
            ).strip()
            response_body: dict[str, Any]
            if tool_name and tool_name in allowed_tools:
                response_body = {
                    "behavior": "allow",
                    "updatedInput": request.get("input") or {},
                }
            else:
                response_body = {
                    "behavior": "deny",
                    "message": "Built-in tools are disabled.",
                }
            await self._write_json(
                proc,
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": req_id,
                        "response": response_body,
                    },
                },
            )
            return
        await self._write_json(
            proc,
            {
                "type": "control_response",
                "response": {
                    "subtype": "error",
                    "request_id": req_id,
                    "error": f"Unsupported control request: {subtype}",
                },
            },
        )

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
        cli_path = self._resolve_cli()
        env = {
            **auth.process_env(),
            "NOCTURNOMATH_WORKSPACE_PATH": str(workspace_path),
        }
        allowed_tools = {
            name
            for spec in tools
            for name in (
                spec.name,
                f"mcp__explore__{spec.name}",
                f"explore/{spec.name}",
            )
        }
        required_timeout_ms = max(1, int(tool_timeout_s)) * 1000
        try:
            configured_ms = int(env.get("MCP_TOOL_TIMEOUT") or 0)
        except ValueError:
            configured_ms = 0
        env["MCP_TOOL_TIMEOUT"] = str(max(required_timeout_ms, configured_ms))
        async with McpToolBridge(tools) as bridge:
            cmd = self._build_turn_cmd(
                cli_path,
                system_prompt=system_prompt,
                mcp_config={"explore": bridge.stdio_server_config()},
                model=model,
                effort=effort,
                resume_session_id=resume_session_id,
            )
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace_path,
                env=env,
                start_new_session=(os.name == "posix"),
            )
            stderr = StderrCollector(proc.stderr)
            try:
                messages = read_jsonl(proc.stdout)
                try:
                    init_id = await self._send_control(
                        proc, "initialize", {"skills": []}
                    )
                except (BrokenPipeError, ConnectionResetError) as exc:
                    detail = await stderr.text()
                    if resume_session_id and _is_missing_session_error(detail):
                        raise SessionResumeError(detail) from exc
                    raise RuntimeError(detail or str(exc)) from exc
                await self._await_initialize(
                    proc,
                    messages,
                    init_id,
                    stderr,
                    resume_session_id=resume_session_id,
                    allowed_tools=allowed_tools,
                )

                await self._write_json(
                    proc,
                    {
                        "type": "user",
                        "message": {"role": "user", "content": prompt},
                        "parent_tool_use_id": None,
                        "session_id": resume_session_id or "default",
                    },
                )

                seen_session_id = resume_session_id
                last_prompt_tokens: int | None = None
                last_prompt_model: str | None = None
                saw_result = False
                progressed = False
                emitted_turn_started = False

                async for msg in messages:
                    msg_type = msg.get("type")
                    if msg_type == "result" and msg.get("is_error"):
                        detail = _error_result_text(msg)
                        if (
                            resume_session_id
                            and not progressed
                            and _is_missing_session_error(detail)
                        ):
                            raise SessionResumeError(detail)
                        raise RuntimeError(detail)

                    session_id = msg.get("session_id")
                    if (
                        isinstance(session_id, str)
                        and session_id
                        and session_id != seen_session_id
                    ):
                        seen_session_id = session_id
                        yield SessionStarted(session_id)

                    if msg_type == "control_request":
                        await self._handle_control_request(proc, msg, allowed_tools)
                        continue

                    if msg_type == "stream_event":
                        if msg.get("parent_tool_use_id"):
                            continue
                        raw = msg.get("event") or {}
                        raw_type = raw.get("type")
                        if not emitted_turn_started and raw_type in (
                            "message_start",
                            "content_block_start",
                            "content_block_delta",
                        ):
                            emitted_turn_started = True
                            progressed = True
                            yield TurnStarted()
                        if raw_type == "content_block_delta":
                            delta = raw.get("delta") or {}
                            if delta.get("type") == "text_delta":
                                chunk = delta.get("text")
                                if isinstance(chunk, str) and chunk:
                                    progressed = True
                                    yield TextDelta(chunk)
                    elif (
                        msg_type == "system"
                        and msg.get("subtype") == "compact_boundary"
                    ):
                        progressed = True
                        yield CompactionNotice()
                    elif msg_type == "assistant":
                        inner = (
                            msg.get("message")
                            if isinstance(msg.get("message"), dict)
                            else msg
                        )
                        usage = inner.get("usage") or msg.get("usage") or {}
                        parent_tool = msg.get("parent_tool_use_id")
                        msg_model = inner.get("model") or msg.get("model")
                        msg_id = inner.get("id") or msg.get("message_id")
                        if parent_tool is None and isinstance(usage, dict) and usage:
                            progressed = True
                            yield MessageUsage(
                                message_id=msg_id,
                                model=msg_model,
                                usage=usage,
                            )
                            keys = (
                                "input_tokens",
                                "cache_read_input_tokens",
                                "cache_creation_input_tokens",
                            )
                            counts = tuple(usage.get(k, 0) for k in keys)
                            if any(k in usage for k in keys) and all(
                                isinstance(c, int) and c >= 0 for c in counts
                            ):
                                last_prompt_tokens = sum(counts)
                                last_prompt_model = msg_model
                        content = inner.get("content") or []
                        if parent_tool is None and isinstance(content, list):
                            for block in content:
                                if (
                                    isinstance(block, dict)
                                    and block.get("type") == "text"
                                    and isinstance(block.get("text"), str)
                                ):
                                    progressed = True
                                    yield TextBlockDone(block["text"])
                    elif msg_type == "result":
                        saw_result = True
                        model_usage = msg.get("modelUsage") or msg.get("model_usage")
                        context_usage = _context_usage_from_turn(
                            last_prompt_tokens, last_prompt_model, model_usage
                        )
                        progressed = True
                        yield TurnResult(
                            usage=msg.get("usage"),
                            model_usage=model_usage,
                            context_usage=context_usage,
                            total_cost_usd=msg.get("total_cost_usd"),
                            duration_ms=msg.get("duration_ms"),
                            duration_api_ms=msg.get("duration_api_ms"),
                            num_turns=msg.get("num_turns"),
                        )
                        break

                if not saw_result:
                    detail = await stderr.text()
                    rc = await proc.wait()
                    raise RuntimeError(
                        detail
                        or f"Claude CLI exited without completing the turn (status {rc})."
                    )
            except (asyncio.CancelledError, GeneratorExit):
                with suppress(Exception):
                    await self._send_control(proc, "interrupt")
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
        del tools
        cli_path = self._resolve_cli()
        env = {
            **auth.process_env(),
            "NOCTURNOMATH_WORKSPACE_PATH": str(workspace_path),
        }
        cmd = self._build_turn_cmd(
            cli_path,
            system_prompt=system_prompt,
            mcp_config=None,
            model=model,
            effort=effort,
            resume_session_id=resume_session_id,
        )
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workspace_path,
            env=env,
            start_new_session=(os.name == "posix"),
        )
        stderr = StderrCollector(proc.stderr)
        try:
            messages = read_jsonl(proc.stdout)
            compacted = False
            latest_session_id: str | None = resume_session_id
            try:
                async with asyncio.timeout(COMPACT_TIMEOUT_S):
                    try:
                        init_id = await self._send_control(
                            proc, "initialize", {"skills": []}
                        )
                    except (BrokenPipeError, ConnectionResetError) as exc:
                        detail = await stderr.text()
                        if _is_missing_session_error(detail):
                            raise SessionResumeError(detail) from exc
                        raise RuntimeError(detail or str(exc)) from exc
                    await self._await_initialize(
                        proc,
                        messages,
                        init_id,
                        stderr,
                        resume_session_id=resume_session_id,
                    )

                    await self._write_json(
                        proc,
                        {
                            "type": "user",
                            "message": {"role": "user", "content": "/compact"},
                            "parent_tool_use_id": None,
                            "session_id": resume_session_id,
                        },
                    )

                    async for msg in messages:
                        sid = msg.get("session_id")
                        if isinstance(sid, str) and sid:
                            latest_session_id = sid
                        msg_type = msg.get("type")
                        if msg_type == "control_request":
                            await self._handle_control_request(proc, msg)
                            continue
                        if (
                            msg_type == "system"
                            and msg.get("subtype") == "compact_boundary"
                        ):
                            compacted = True
                        elif msg_type == "result":
                            if msg.get("is_error"):
                                detail = _error_result_text(msg)
                                if not compacted and _is_missing_session_error(detail):
                                    raise SessionResumeError(detail)
                                raise RuntimeError(detail)
                            break
            except TimeoutError as exc:
                raise RuntimeError(
                    f"Claude compaction timed out after {int(COMPACT_TIMEOUT_S)}s."
                ) from exc

            if not compacted:
                detail = await stderr.text()
                raise RuntimeError(
                    detail or "Claude did not confirm that compaction completed."
                )

            context_usage: dict[str, Any] | None = None
            try:
                async with asyncio.timeout(CONTEXT_USAGE_TIMEOUT_S):
                    ctx_id = await self._send_control(proc, "get_context_usage")
                    async for msg in messages:
                        if msg.get("type") == "control_request":
                            await self._handle_control_request(proc, msg)
                            continue
                        if (
                            msg.get("type") == "control_response"
                            and (msg.get("response") or {}).get("request_id") == ctx_id
                        ):
                            resp = msg.get("response") or {}
                            if resp.get("subtype") == "error":
                                raise RuntimeError(
                                    resp.get("error") or "get_context_usage failed"
                                )
                            usage = resp.get("response") or {}
                            used = usage.get("totalTokens")
                            window = usage.get("rawMaxTokens")
                            if isinstance(used, int) and used >= 0:
                                context_usage = {
                                    "used_tokens": used,
                                    "window_tokens": window
                                    if isinstance(window, int) and window > 0
                                    else None,
                                    "model": usage.get("model"),
                                }
                            break
            except TimeoutError:
                logger.warning(
                    "Context usage refresh timed out after %ss",
                    CONTEXT_USAGE_TIMEOUT_S,
                )
            except Exception as exc:
                logger.warning(
                    "Could not refresh context usage after compaction: %s", exc
                )
            return CompactResult(
                session_id=latest_session_id, context_usage=context_usage
            )
        finally:
            await terminate_process(proc, stderr)
