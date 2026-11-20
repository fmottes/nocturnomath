"""OpenAI backend supporting direct HTTP Responses API and local ``codex`` CLI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import tempfile
import tomllib
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import aclosing, suppress
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
    package_version,
    terminate_process,
)
from .http import (
    compact_openai_http,
    discover_openai_http_models,
    is_http_session_id,
    run_openai_http_turn,
)

logger = logging.getLogger("nocturnomath.backends.openai")

CODEX_CLI_NAME = "codex.exe" if sys.platform == "win32" else "codex"
DISCOVERY_TIMEOUT_S = 15.0
COMPACT_TIMEOUT_S = 600.0

CODEX_DISABLE_FLAGS = (
    "--disable",
    "shell_tool",
    "--disable",
    "unified_exec",
    "--disable",
    "browser_use",
    "--disable",
    "browser_use_external",
    "--disable",
    "computer_use",
    "--disable",
    "image_generation",
    "--disable",
    "view_image",
    "--disable",
    "apps",
    "--disable",
    "plugins",
    "--disable",
    "multi_agent",
    "--disable",
    "memories",
    "--disable",
    "code_mode",
    "--disable",
    "hooks",
    "--disable",
    "sleep_tool",
    "-c",
    'web_search="disabled"',
)


def codex_cli_candidates() -> list[Path]:
    """Locate an installed Codex CLI on PATH."""
    return find_cli_executables(CODEX_CLI_NAME)


def _is_missing_thread_error(text: str) -> bool:
    lowered = text.lower()
    if any(
        marker in lowered
        for marker in (
            "no rollout found",
            "thread not found",
            "conversation not found",
        )
    ):
        return True
    return "does not exist" in lowered and any(
        noun in lowered for noun in ("thread", "rollout", "conversation")
    )


_TOML_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _toml_quote(value: str) -> str:
    out = ['"']
    for ch in value:
        code = ord(ch)
        if ch in _TOML_ESCAPES:
            out.append(_TOML_ESCAPES[ch])
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\u{code:04x}")
        elif code > 0xFFFF:
            out.append(f"\\U{code:08x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _toml_mcp_override(
    cfg: dict[str, Any], *, tool_timeout_s: int = DEFAULT_TOOL_TIMEOUT_S
) -> str:
    env_items = ", ".join(
        f"{key if re.fullmatch(r'[A-Za-z0-9_-]+', str(key)) else _toml_quote(str(key))}={_toml_quote(str(val))}"
        for key, val in (cfg.get("env") or {}).items()
    )
    args_toml = ", ".join(_toml_quote(str(arg)) for arg in (cfg.get("args") or ()))
    timeout_sec = max(1, int(tool_timeout_s))
    return (
        "mcp_servers={explore={"
        "enabled=true, "
        f"command={_toml_quote(str(cfg['command']))}, "
        f"args=[{args_toml}], "
        f"env={{{env_items}}}, "
        "startup_timeout_sec=30, "
        f"tool_timeout_sec={timeout_sec}, "
        'default_tools_approval_mode="approve"'
        "}}"
    )


def _disabled_mcp_overrides(codex_home: Path) -> list[str]:
    names: set[str] = set()
    config_file = codex_home / "config.toml"
    if config_file.is_file():
        with suppress(Exception):
            parsed = tomllib.loads(config_file.read_text(encoding="utf-8"))
            mcp_servers = parsed.get("mcp_servers")
            if isinstance(mcp_servers, dict):
                names.update(
                    str(k)
                    for k, v in mcp_servers.items()
                    if k != "explore" and isinstance(v, dict)
                )
    flags: list[str] = []
    for name in sorted(names):
        if name == "explore":
            continue
        key = name if re.fullmatch(r"[A-Za-z0-9_-]+", name) else _toml_quote(name)
        flags.extend(["-c", f"mcp_servers.{key}.enabled=false"])
    return flags


def _extract_context_usage(
    tu: dict[str, Any], model: str | None
) -> dict[str, Any] | None:
    last = tu.get("last") or {}
    if not isinstance(last, dict):
        return None
    window = tu.get("modelContextWindow")
    used = last.get("totalTokens")
    if not isinstance(used, int) or used < 0:
        inp = last.get("inputTokens")
        cached = last.get("cachedInputTokens", 0)
        if isinstance(inp, int) and inp >= 0:
            used = inp + (cached if isinstance(cached, int) else 0)
    if not isinstance(used, int) or used < 0:
        return None
    return {
        "used_tokens": used,
        "window_tokens": window if isinstance(window, int) and window > 0 else None,
        "model": model,
    }


class OpenAIBackend:
    """Drive OpenAI via direct HTTP Responses API or local ``codex app-server``."""

    provider: ProviderId = "openai"
    uses_compact_hook: bool = False

    def __init__(
        self,
        cli_path: Path | None = None,
        *,
        http_client_factory: Callable[..., Any] | None = None,
    ):
        self.cli_path = cli_path
        self._http_client_factory = http_client_factory
        self.last_discovery_error: str | None = None
        self._next_rpc_id = 0

    def _resolve_cli(self) -> Path:
        if self.cli_path is not None and is_usable_cli_binary(self.cli_path):
            return self.cli_path
        candidates = codex_cli_candidates()
        if not candidates:
            raise RuntimeError(
                "Codex CLI (`codex`) was not found on PATH. Install Codex or configure an OpenAI API key."
            )
        self.cli_path = candidates[0]
        return self.cli_path

    @staticmethod
    def _prepare_env(auth: ProviderAuth) -> tuple[dict[str, str], Path]:
        env = auth.process_env()
        codex_home = Path(env.get("CODEX_HOME") or (Path.home() / ".codex"))
        return env, codex_home

    def _next_id(self) -> int:
        self._next_rpc_id += 1
        return self._next_rpc_id

    @staticmethod
    async def _send(proc: asyncio.subprocess.Process, payload: dict[str, Any]) -> None:
        if proc.stdin is None:
            raise RuntimeError("Codex app-server stdin is not open.")
        proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        await proc.stdin.drain()

    async def _rpc(
        self,
        proc: asyncio.subprocess.Process,
        messages: JsonMessageStream,
        method: str,
        params: dict[str, Any],
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
                detail or f"Codex app-server closed before replying to {method}."
            ) from exc
        while (msg := await messages.next_wire()) is not None:
            if msg.get("id") is not None and "method" in msg:
                await self._handle_server_request(proc, msg)
                continue
            if msg.get("id") == req_id:
                if "error" in msg and msg["error"] is not None:
                    err = msg["error"]
                    text = err.get("message") if isinstance(err, dict) else str(err)
                    raise RuntimeError(text or f"Codex RPC {method} failed.")
                return msg.get("result") or {}
            if msg.get("id") is None:
                messages.buffer_notification(msg)
        detail = await stderr.text() if stderr is not None else ""
        raise RuntimeError(
            detail or f"Codex app-server closed before replying to {method}."
        )

    async def _handle_server_request(
        self, proc: asyncio.subprocess.Process, msg: dict[str, Any]
    ) -> None:
        req_id = msg.get("id")
        method = msg.get("method")
        if method in (
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
        ):
            await self._send(
                proc,
                {"jsonrpc": "2.0", "id": req_id, "result": {"decision": "decline"}},
            )
        elif method in ("execCommandApproval", "applyPatchApproval"):
            await self._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "decision": {
                            "denied": {"rejection": "Built-in tools are disabled."}
                        }
                    },
                },
            )
        elif method == "item/permissions/requestApproval":
            await self._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"permissions": {}, "scope": "turn"},
                },
            )
        elif method == "item/tool/requestUserInput":
            await self._send(
                proc, {"jsonrpc": "2.0", "id": req_id, "result": {"answers": {}}}
            )
        elif method == "mcpServer/elicitation/request":
            await self._send(
                proc, {"jsonrpc": "2.0", "id": req_id, "result": {"action": "decline"}}
            )
        else:
            await self._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unsupported: {method}"},
                },
            )

    async def _initialize_session(
        self,
        proc: asyncio.subprocess.Process,
        messages: JsonMessageStream,
        stderr: StderrCollector | None = None,
    ) -> None:
        await self._rpc(
            proc,
            messages,
            "initialize",
            {
                "clientInfo": {"name": "nocturnomath", "version": package_version()},
                "capabilities": {"experimentalApi": True},
            },
            stderr,
        )
        await self._send(
            proc, {"jsonrpc": "2.0", "method": "initialized", "params": {}}
        )

    async def discover_models_staged(
        self, auth: ProviderAuth
    ) -> tuple[list[ModelSpec], Callable[[], None]]:
        api_key = auth.direct_api_key()
        if api_key is not None:
            specs, err = await discover_openai_http_models(
                api_key, http_client_factory=self._http_client_factory
            )

            def commit_http(discovery_error: str | None = err) -> None:
                self.last_discovery_error = discovery_error

            return specs, commit_http

        def clear_error() -> None:
            self.last_discovery_error = None

        candidates = codex_cli_candidates()
        if not candidates:
            return [], clear_error
        env, codex_home = self._prepare_env(auth)
        fallback_commit: Callable[[], None] | None = None
        for candidate in candidates:
            proc: asyncio.subprocess.Process | None = None
            stderr: StderrCollector | None = None
            try:
                async with asyncio.timeout(DISCOVERY_TIMEOUT_S):
                    proc = await asyncio.create_subprocess_exec(
                        str(candidate),
                        "app-server",
                        *CODEX_DISABLE_FLAGS,
                        "-c",
                        "mcp_servers={}",
                        *_disabled_mcp_overrides(codex_home),
                        "--listen",
                        "stdio://",
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=env,
                        start_new_session=(os.name == "posix"),
                    )
                    stderr = StderrCollector(proc.stderr)
                    messages = JsonMessageStream(proc.stdout)
                    await self._initialize_session(proc, messages, stderr)
                    acct = await self._rpc(
                        proc,
                        messages,
                        "account/read",
                        {"refreshToken": False},
                        stderr,
                    )
                    has_env_key = bool(
                        (env.get("CODEX_API_KEY") or "").strip()
                        or (env.get("OPENAI_API_KEY") or "").strip()
                        or (env.get("CODEX_ACCESS_TOKEN") or "").strip()
                    )
                    if (
                        acct.get("requiresOpenaiAuth", True)
                        and acct.get("account") is None
                        and not has_env_key
                    ):

                        def commit(chosen: Path = candidate) -> None:
                            self.cli_path = chosen
                            self.last_discovery_error = None

                        return [], commit

                    raw_models: list[dict[str, Any]] = []
                    cursor: str | None = None
                    while True:
                        params: dict[str, Any] = {"includeHidden": False}
                        if cursor is not None:
                            params["cursor"] = cursor
                        page = await self._rpc(
                            proc, messages, "model/list", params, stderr
                        )
                        data = page.get("data") or []
                        if isinstance(data, list):
                            raw_models.extend(
                                item for item in data if isinstance(item, dict)
                            )
                        next_cursor = page.get("nextCursor")
                        if not isinstance(next_cursor, str) or not next_cursor:
                            break
                        cursor = next_cursor

                    specs: list[ModelSpec] = []
                    for item in raw_models:
                        raw_id = item.get("model") or item.get("id")
                        if not isinstance(raw_id, str) or not raw_id:
                            continue
                        raw_efforts = item.get("supportedReasoningEfforts") or []
                        efforts: list[str] = []
                        for entry in raw_efforts:
                            val = (
                                entry.get("reasoningEffort")
                                if isinstance(entry, dict)
                                else entry
                            )
                            if (
                                isinstance(val, str)
                                and val in EFFORT_LEVELS
                                and val not in efforts
                            ):
                                efforts.append(val)
                        specs.append(
                            ModelSpec(
                                id=raw_id,
                                raw_id=raw_id,
                                provider="openai",
                                label=raw_id,
                                efforts=efforts,
                                is_default=bool(item.get("isDefault")),
                            )
                        )

                    def commit(chosen: Path = candidate) -> None:
                        self.cli_path = chosen
                        self.last_discovery_error = None

                    if specs:
                        return specs, commit
                    if fallback_commit is None:
                        fallback_commit = commit
            except Exception as exc:
                logger.warning(
                    "Codex model discovery with %s failed: %s", candidate, exc
                )
            finally:
                if proc is not None:
                    await terminate_process(proc, stderr)
                elif stderr is not None:
                    await stderr.close()
        if fallback_commit is not None:
            return [], fallback_commit
        return [], clear_error

    async def discover_models(self, auth: ProviderAuth) -> list[ModelSpec]:
        specs, commit = await self.discover_models_staged(auth)
        commit()
        return specs

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
            async with aclosing(
                run_openai_http_turn(
                    prompt,
                    api_key=api_key,
                    system_prompt=system_prompt,
                    tools=tools,
                    workspace_path=workspace_path,
                    model=model,
                    effort=effort,
                    resume_session_id=resume_session_id,
                    tool_timeout_s=tool_timeout_s,
                    http_client_factory=self._http_client_factory,
                )
            ) as events:
                async for event in events:
                    yield event
            return
        if is_http_session_id(resume_session_id):
            raise SessionResumeError(
                "Session was created in HTTP API mode and cannot be resumed by the CLI."
            )
        cli_path = self._resolve_cli()
        env, codex_home = self._prepare_env(auth)
        with tempfile.TemporaryDirectory(
            prefix="nocturnomath-codex-cwd-", ignore_cleanup_errors=True
        ) as isolated_cwd:
            async with McpToolBridge(tools) as bridge:
                cfg = bridge.stdio_server_config()
                proc = await asyncio.create_subprocess_exec(
                    str(cli_path),
                    "app-server",
                    *CODEX_DISABLE_FLAGS,
                    "-c",
                    _toml_mcp_override(cfg, tool_timeout_s=tool_timeout_s),
                    *_disabled_mcp_overrides(codex_home),
                    "--listen",
                    "stdio://",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=isolated_cwd,
                    env=env,
                    start_new_session=(os.name == "posix"),
                )
                stderr = StderrCollector(proc.stderr)
                thread_id = resume_session_id
                turn_id: str | None = None
                try:
                    messages = JsonMessageStream(proc.stdout)
                    await self._initialize_session(proc, messages, stderr)
                    thread_params = {
                        "cwd": isolated_cwd,
                        "baseInstructions": system_prompt,
                        "model": model,
                        "sandbox": "read-only",
                        "approvalPolicy": "never",
                    }
                    if resume_session_id:
                        try:
                            res = await self._rpc(
                                proc,
                                messages,
                                "thread/resume",
                                {"threadId": resume_session_id, **thread_params},
                                stderr,
                            )
                        except RuntimeError as exc:
                            if _is_missing_thread_error(str(exc)):
                                raise SessionResumeError(str(exc)) from exc
                            raise
                    else:
                        res = await self._rpc(
                            proc, messages, "thread/start", thread_params, stderr
                        )
                    thread_id = (res.get("thread") or {}).get("id") or resume_session_id
                    if not isinstance(thread_id, str) or not thread_id:
                        raise RuntimeError(
                            "Codex app-server did not return a thread ID."
                        )

                    turn_params: dict[str, Any] = {
                        "threadId": thread_id,
                        "input": [{"type": "text", "text": prompt}],
                        "model": model,
                    }
                    if effort:
                        turn_params["effort"] = effort
                    turn_res = await self._rpc(
                        proc, messages, "turn/start", turn_params, stderr
                    )
                    turn_id = (turn_res.get("turn") or {}).get("id")
                    yield SessionStarted(thread_id)

                    last_error: str | None = None
                    last_context_usage: dict[str, Any] | None = None
                    baseline_total: dict[str, Any] | None = None
                    latest_total: dict[str, Any] | None = None
                    accumulated_last: dict[str, int] = {}
                    saw_compaction_item = False
                    turn_finished = False

                    def _is_other_turn(notif_params: dict[str, Any]) -> bool:
                        if not turn_id:
                            return False
                        nt = notif_params.get("turnId") or (
                            (notif_params.get("turn") or {}).get("id")
                            if isinstance(notif_params.get("turn"), dict)
                            else None
                        )
                        return isinstance(nt, str) and bool(nt) and nt != turn_id

                    async for msg in messages:
                        if msg.get("id") is not None and "method" in msg:
                            await self._handle_server_request(proc, msg)
                            continue
                        method = msg.get("method")
                        params = msg.get("params") or {}
                        if method != "thread/tokenUsage/updated" and _is_other_turn(
                            params
                        ):
                            continue
                        if method == "item/started":
                            item = params.get("item") or {}
                            if item.get("type") in (
                                "reasoning",
                                "mcpToolCall",
                                "agentMessage",
                            ):
                                yield TurnStarted()
                        elif method == "item/agentMessage/delta":
                            delta = params.get("delta")
                            if isinstance(delta, str) and delta:
                                yield TextDelta(delta)
                        elif method == "item/completed":
                            item = params.get("item") or {}
                            item_type = item.get("type")
                            if item_type == "agentMessage":
                                text = item.get("text")
                                if isinstance(text, str) and text:
                                    yield TextBlockDone(text)
                            elif item_type == "contextCompaction":
                                saw_compaction_item = True
                                yield CompactionNotice()
                        elif method == "thread/compacted":
                            if not saw_compaction_item:
                                yield CompactionNotice()
                        elif method == "thread/tokenUsage/updated":
                            tu = params.get("tokenUsage") or {}
                            notif_turn_id = params.get("turnId")
                            total = tu.get("total") or {}
                            last = tu.get("last") or {}
                            if (
                                turn_id
                                and isinstance(notif_turn_id, str)
                                and notif_turn_id
                                and notif_turn_id != turn_id
                            ):
                                if isinstance(total, dict) and total:
                                    baseline_total = total
                            else:
                                ctx = _extract_context_usage(tu, model)
                                if ctx is not None:
                                    last_context_usage = ctx
                                if isinstance(total, dict) and total:
                                    latest_total = total
                                if isinstance(last, dict):
                                    for key, val in last.items():
                                        if isinstance(val, int):
                                            accumulated_last[key] = (
                                                accumulated_last.get(key, 0) + val
                                            )
                        elif method == "error":
                            err = params.get("error") or {}
                            message = (
                                err.get("message")
                                if isinstance(err, dict)
                                else str(err)
                            )
                            if message:
                                last_error = message
                        elif method == "turn/completed":
                            turn_finished = True
                            turn = params.get("turn") or {}
                            status = turn.get("status")
                            if status == "interrupted":
                                raise asyncio.CancelledError(
                                    "OpenAI Codex turn was interrupted."
                                )
                            if status == "failed":
                                err = turn.get("error") or {}
                                message = (
                                    err.get("message")
                                    if isinstance(err, dict)
                                    else None
                                )
                                raise RuntimeError(
                                    message
                                    or last_error
                                    or "OpenAI Codex could not finish this turn."
                                )
                            if latest_total and baseline_total:
                                turn_usage: dict[str, Any] | None = {
                                    key: max(0, val - baseline_total.get(key, 0))
                                    for key, val in latest_total.items()
                                    if isinstance(val, int)
                                    and isinstance(baseline_total.get(key, 0), int)
                                }
                            elif resume_session_id is None and latest_total:
                                turn_usage = latest_total
                            elif accumulated_last:
                                turn_usage = accumulated_last
                            else:
                                turn_usage = latest_total
                            yield TurnResult(
                                usage=turn_usage,
                                context_usage=last_context_usage,
                                num_turns=1,
                            )
                            break

                    if not turn_finished:
                        detail = await stderr.text()
                        raise RuntimeError(
                            last_error
                            or detail
                            or "Codex app-server disconnected unexpectedly."
                        )
                except (asyncio.CancelledError, GeneratorExit):
                    if thread_id and turn_id:
                        with suppress(Exception, TimeoutError):
                            await asyncio.wait_for(
                                self._send(
                                    proc,
                                    {
                                        "jsonrpc": "2.0",
                                        "id": self._next_id(),
                                        "method": "turn/interrupt",
                                        "params": {
                                            "threadId": thread_id,
                                            "turnId": turn_id,
                                        },
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
            return await compact_openai_http(
                api_key=api_key,
                system_prompt=system_prompt,
                workspace_path=workspace_path,
                model=model,
                resume_session_id=resume_session_id,
                http_client_factory=self._http_client_factory,
            )
        if is_http_session_id(resume_session_id):
            raise SessionResumeError(
                "Session was created in HTTP API mode and cannot be resumed by the CLI."
            )
        cli_path = self._resolve_cli()
        env, codex_home = self._prepare_env(auth)
        with tempfile.TemporaryDirectory(
            prefix="nocturnomath-codex-cwd-", ignore_cleanup_errors=True
        ) as isolated_cwd:
            proc = await asyncio.create_subprocess_exec(
                str(cli_path),
                "app-server",
                *CODEX_DISABLE_FLAGS,
                "-c",
                "mcp_servers={}",
                *_disabled_mcp_overrides(codex_home),
                "--listen",
                "stdio://",
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
                compacted = False
                context_usage: dict[str, Any] | None = None
                try:
                    async with asyncio.timeout(COMPACT_TIMEOUT_S):
                        await self._initialize_session(proc, messages, stderr)
                        try:
                            await self._rpc(
                                proc,
                                messages,
                                "thread/resume",
                                {
                                    "threadId": resume_session_id,
                                    "cwd": isolated_cwd,
                                    "baseInstructions": system_prompt,
                                    **({"model": model} if model else {}),
                                    "sandbox": "read-only",
                                    "approvalPolicy": "never",
                                },
                                stderr,
                            )
                        except RuntimeError as exc:
                            if _is_missing_thread_error(str(exc)):
                                raise SessionResumeError(str(exc)) from exc
                            raise
                        messages.clear_buffered()
                        await self._rpc(
                            proc,
                            messages,
                            "thread/compact/start",
                            {"threadId": resume_session_id},
                            stderr,
                        )
                        deadline: float | None = None
                        while True:
                            if deadline is None:
                                try:
                                    msg = await messages.__anext__()
                                except StopAsyncIteration:
                                    break
                            else:
                                remaining = deadline - asyncio.get_running_loop().time()
                                if remaining <= 0:
                                    break
                                try:
                                    msg = await asyncio.wait_for(
                                        messages.__anext__(), timeout=remaining
                                    )
                                except (TimeoutError, StopAsyncIteration):
                                    break
                            if msg.get("id") is not None and "method" in msg:
                                await self._handle_server_request(proc, msg)
                                continue
                            method = msg.get("method")
                            params = msg.get("params") or {}
                            if method == "thread/tokenUsage/updated":
                                ctx = _extract_context_usage(
                                    params.get("tokenUsage") or {}, model
                                )
                                if ctx is not None:
                                    context_usage = ctx
                            elif method == "item/completed":
                                item = params.get("item") or {}
                                if item.get("type") == "contextCompaction":
                                    compacted = True
                                    if deadline is None:
                                        deadline = (
                                            asyncio.get_running_loop().time() + 2.0
                                        )
                            elif method == "thread/compacted":
                                compacted = True
                                break
                            elif method == "turn/completed":
                                turn = params.get("turn") or {}
                                if turn.get("status") == "failed":
                                    err = turn.get("error") or {}
                                    message = (
                                        err.get("message")
                                        if isinstance(err, dict)
                                        else None
                                    )
                                    raise RuntimeError(
                                        message or "Codex compaction failed."
                                    )
                                break
                            elif method == "error":
                                err = params.get("error") or {}
                                message = (
                                    err.get("message")
                                    if isinstance(err, dict)
                                    else str(err)
                                )
                                if not params.get("willRetry"):
                                    raise RuntimeError(
                                        message or "Codex compaction failed."
                                    )
                except TimeoutError as exc:
                    if not compacted:
                        raise RuntimeError(
                            f"OpenAI Codex compaction timed out after {int(COMPACT_TIMEOUT_S)}s."
                        ) from exc
                if not compacted:
                    detail = await stderr.text()
                    raise RuntimeError(
                        detail or "Codex did not confirm that compaction completed."
                    )
                return CompactResult(
                    session_id=resume_session_id, context_usage=context_usage
                )
            finally:
                await terminate_process(proc, stderr)
