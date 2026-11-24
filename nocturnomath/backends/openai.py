"""OpenAI backend supporting direct HTTP Responses API and local ``codex`` CLI."""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import tempfile
import tomllib
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import aclosing, suppress
from pathlib import Path
from typing import Any

from ..auth import ProviderAuth, ProviderId, inherited_credential
from ..mcp_bridge import McpToolBridge
from ..tools import ToolSpec
from .base import (
    DEFAULT_TOOL_TIMEOUT_S,
    EFFORT_LEVELS,
    CompactionNotice,
    CompactResult,
    JsonMessageStream,
    JsonRpcClient,
    ModelSpec,
    SessionResumeError,
    SessionStarted,
    StderrCollector,
    TextBlockDone,
    TextDelta,
    TurnEvent,
    TurnResult,
    TurnStarted,
    context_usage_payload,
    find_cli_executables,
    is_int,
    managed_subprocess,
    package_version,
    resolve_cli_binary,
    rpc_error_message,
    write_json_line,
)
from .http import (
    compact_openai_http,
    discover_openai_http_models,
    reject_http_session_for_cli,
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


def _toml_key(name: str) -> str:
    return name if re.fullmatch(r"[A-Za-z0-9_-]+", name) else _toml_quote(name)


def _toml_mcp_override(
    cfg: dict[str, Any], *, tool_timeout_s: int = DEFAULT_TOOL_TIMEOUT_S
) -> str:
    env_items = ", ".join(
        f"{_toml_key(str(key))}={_toml_quote(str(val))}"
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
        flags.extend(["-c", f"mcp_servers.{_toml_key(name)}.enabled=false"])
    return flags


def _extract_context_usage(
    tu: dict[str, Any], model: str | None
) -> dict[str, Any] | None:
    last = tu.get("last") or {}
    if not isinstance(last, dict):
        return None
    used = last.get("totalTokens")
    if not is_int(used) or used < 0:
        inp = last.get("inputTokens")
        cached = last.get("cachedInputTokens", 0)
        if is_int(inp) and inp >= 0:
            used = inp + (cached if is_int(cached) else 0)
    return context_usage_payload(used, tu.get("modelContextWindow"), model)


_SERVER_REQUEST_RESULTS: dict[str, dict[str, Any]] = {
    "item/commandExecution/requestApproval": {"decision": "decline"},
    "item/fileChange/requestApproval": {"decision": "decline"},
    "execCommandApproval": {
        "decision": {"denied": {"rejection": "Built-in tools are disabled."}}
    },
    "applyPatchApproval": {
        "decision": {"denied": {"rejection": "Built-in tools are disabled."}}
    },
    "item/permissions/requestApproval": {"permissions": {}, "scope": "turn"},
    "item/tool/requestUserInput": {"answers": {}},
    "mcpServer/elicitation/request": {"action": "decline"},
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

    def _resolve_cli(self) -> Path:
        self.cli_path = resolve_cli_binary(
            self.cli_path,
            codex_cli_candidates,
            "Codex CLI (`codex`) was not found on PATH. Install Codex or configure an OpenAI API key.",
        )
        return self.cli_path

    @staticmethod
    def _prepare_env(auth: ProviderAuth) -> tuple[dict[str, str], Path]:
        env = auth.process_env()
        codex_home = Path(env.get("CODEX_HOME") or (Path.home() / ".codex"))
        return env, codex_home

    @staticmethod
    def _app_server_cmd(
        cli_path: Path,
        disabled_flags: Sequence[str],
        *,
        mcp_override: str = "mcp_servers={}",
    ) -> list[str]:
        return [
            str(cli_path),
            "app-server",
            *CODEX_DISABLE_FLAGS,
            "-c",
            mcp_override,
            *disabled_flags,
            "--listen",
            "stdio://",
        ]

    @staticmethod
    async def _send(proc: asyncio.subprocess.Process, payload: dict[str, Any]) -> None:
        await write_json_line(proc, payload, "Codex app-server")

    def _rpc_client(
        self,
        proc: asyncio.subprocess.Process,
        messages: JsonMessageStream,
        stderr: StderrCollector | None = None,
    ) -> JsonRpcClient:
        return JsonRpcClient(
            proc,
            messages,
            stderr,
            label="Codex app-server",
            on_request=lambda msg: self._handle_server_request(proc, msg),
        )

    def _discovery_commit(
        self,
        *,
        cli_path: Path | None = None,
        error: str | None = None,
    ) -> Callable[[], None]:
        def commit() -> None:
            if cli_path is not None:
                self.cli_path = cli_path
            self.last_discovery_error = error

        return commit

    async def _handle_server_request(
        self, proc: asyncio.subprocess.Process, msg: dict[str, Any]
    ) -> None:
        req_id = msg.get("id")
        method = msg.get("method")
        result = _SERVER_REQUEST_RESULTS.get(str(method))
        if result is not None:
            await self._send(proc, {"jsonrpc": "2.0", "id": req_id, "result": result})
        else:
            await self._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unsupported: {method}"},
                },
            )

    async def _initialize_session(self, client: JsonRpcClient) -> None:
        await client.rpc(
            "initialize",
            {
                "clientInfo": {"name": "nocturnomath", "version": package_version()},
                "capabilities": {"experimentalApi": True},
            },
        )
        await client.send({"jsonrpc": "2.0", "method": "initialized", "params": {}})

    @staticmethod
    async def _open_thread(
        client: JsonRpcClient,
        *,
        cwd: str,
        system_prompt: str,
        model: str | None,
        resume_session_id: str | None = None,
    ) -> dict[str, Any]:
        thread_params: dict[str, Any] = {
            "cwd": cwd,
            "baseInstructions": system_prompt,
            **({"model": model} if model else {}),
            "sandbox": "read-only",
            "approvalPolicy": "never",
        }
        if resume_session_id:
            try:
                return await client.rpc(
                    "thread/resume", {"threadId": resume_session_id, **thread_params}
                )
            except RuntimeError as exc:
                if _is_missing_thread_error(str(exc)):
                    raise SessionResumeError(str(exc)) from exc
                raise
        return await client.rpc("thread/start", thread_params)

    async def discover_models_staged(
        self, auth: ProviderAuth
    ) -> tuple[list[ModelSpec], Callable[[], None]]:
        api_key = auth.direct_api_key()
        if api_key is not None:
            specs, err = await discover_openai_http_models(
                api_key, http_client_factory=self._http_client_factory
            )
            return specs, self._discovery_commit(error=err)

        candidates = codex_cli_candidates()
        if not candidates:
            return [], self._discovery_commit()
        env, codex_home = self._prepare_env(auth)
        disabled_flags = _disabled_mcp_overrides(codex_home)
        fallback_commit: Callable[[], None] | None = None
        for candidate in candidates:
            try:
                async with managed_subprocess(
                    *self._app_server_cmd(candidate, disabled_flags), env=env
                ) as (proc, stderr):
                    async with asyncio.timeout(DISCOVERY_TIMEOUT_S):
                        messages = JsonMessageStream(proc.stdout)
                        client = self._rpc_client(proc, messages, stderr)
                        await self._initialize_session(client)
                        acct = await client.rpc("account/read", {"refreshToken": False})
                        has_env_key = inherited_credential("openai", env) is not None
                        if (
                            acct.get("requiresOpenaiAuth", True)
                            and acct.get("account") is None
                            and not has_env_key
                        ):
                            return [], self._discovery_commit(cli_path=candidate)

                        raw_models: list[dict[str, Any]] = []
                        cursor: str | None = None
                        while True:
                            params: dict[str, Any] = {"includeHidden": False}
                            if cursor is not None:
                                params["cursor"] = cursor
                            page = await client.rpc("model/list", params)
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

                        commit = self._discovery_commit(cli_path=candidate)
                        if specs:
                            return specs, commit
                        if fallback_commit is None:
                            fallback_commit = commit
            except Exception as exc:
                logger.warning(
                    "Codex model discovery with %s failed: %s", candidate, exc
                )
        return [], fallback_commit or self._discovery_commit()

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
        reject_http_session_for_cli(resume_session_id)
        cli_path = self._resolve_cli()
        env, codex_home = self._prepare_env(auth)
        disabled_flags = _disabled_mcp_overrides(codex_home)
        with tempfile.TemporaryDirectory(
            prefix="nocturnomath-codex-cwd-", ignore_cleanup_errors=True
        ) as isolated_cwd:
            async with McpToolBridge(tools) as bridge:
                mcp_override = _toml_mcp_override(
                    bridge.stdio_server_config(), tool_timeout_s=tool_timeout_s
                )
                async with managed_subprocess(
                    *self._app_server_cmd(
                        cli_path, disabled_flags, mcp_override=mcp_override
                    ),
                    cwd=isolated_cwd,
                    env=env,
                ) as (proc, stderr):
                    thread_id = resume_session_id
                    turn_id: str | None = None
                    client: JsonRpcClient | None = None
                    try:
                        messages = JsonMessageStream(proc.stdout)
                        client = self._rpc_client(proc, messages, stderr)
                        await self._initialize_session(client)
                        res = await self._open_thread(
                            client,
                            cwd=isolated_cwd,
                            system_prompt=system_prompt,
                            model=model,
                            resume_session_id=resume_session_id,
                        )
                        thread_id = (res.get("thread") or {}).get(
                            "id"
                        ) or resume_session_id
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
                        turn_res = await client.rpc("turn/start", turn_params)
                        turn_id = (turn_res.get("turn") or {}).get("id")
                        yield SessionStarted(thread_id)

                        last_error: str | None = None
                        last_context_usage: dict[str, Any] | None = None
                        baseline_total: dict[str, Any] | None = None
                        latest_total: dict[str, Any] | None = None
                        accumulated_last: dict[str, int] = {}
                        saw_compaction_item = False
                        emitted_turn_started = False
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
                                if not emitted_turn_started and item.get("type") in (
                                    "reasoning",
                                    "mcpToolCall",
                                    "agentMessage",
                                ):
                                    emitted_turn_started = True
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
                                total = tu.get("total") or {}
                                last = tu.get("last") or {}
                                if _is_other_turn(params):
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
                                            if is_int(val):
                                                accumulated_last[key] = (
                                                    accumulated_last.get(key, 0) + val
                                                )
                            elif method == "error":
                                message = rpc_error_message(params.get("error"))
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
                                    raise RuntimeError(
                                        rpc_error_message(turn.get("error"))
                                        or last_error
                                        or "OpenAI Codex could not finish this turn."
                                    )
                                if latest_total and baseline_total:
                                    turn_usage: dict[str, Any] | None = {
                                        key: max(0, val - baseline_total.get(key, 0))
                                        for key, val in latest_total.items()
                                        if is_int(val)
                                        and is_int(baseline_total.get(key, 0))
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
                        if client is not None and thread_id and turn_id:
                            with suppress(Exception, TimeoutError):
                                await asyncio.wait_for(
                                    client.send_request(
                                        "turn/interrupt",
                                        {"threadId": thread_id, "turnId": turn_id},
                                    ),
                                    timeout=0.5,
                                )
                        await bridge.cancel_in_flight()
                        raise

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
        reject_http_session_for_cli(resume_session_id)
        cli_path = self._resolve_cli()
        env, codex_home = self._prepare_env(auth)
        disabled_flags = _disabled_mcp_overrides(codex_home)
        with tempfile.TemporaryDirectory(
            prefix="nocturnomath-codex-cwd-", ignore_cleanup_errors=True
        ) as isolated_cwd:
            async with managed_subprocess(
                *self._app_server_cmd(cli_path, disabled_flags),
                cwd=isolated_cwd,
                env=env,
            ) as (proc, stderr):
                messages = JsonMessageStream(proc.stdout)
                client = self._rpc_client(proc, messages, stderr)
                compacted = False
                context_usage: dict[str, Any] | None = None
                last_error: str | None = None
                try:
                    async with asyncio.timeout(COMPACT_TIMEOUT_S):
                        await self._initialize_session(client)
                        await self._open_thread(
                            client,
                            cwd=isolated_cwd,
                            system_prompt=system_prompt,
                            model=model,
                            resume_session_id=resume_session_id,
                        )
                        messages.clear_buffered()
                        await client.rpc(
                            "thread/compact/start",
                            {"threadId": resume_session_id},
                        )
                        deadline: float | None = None
                        while True:
                            timeout: float | None = None
                            if deadline is not None:
                                timeout = deadline - asyncio.get_running_loop().time()
                                if timeout <= 0:
                                    break
                            try:
                                msg = await messages.next_message(timeout=timeout)
                            except TimeoutError:
                                break
                            if msg is None:
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
                                    raise RuntimeError(
                                        rpc_error_message(turn.get("error"))
                                        or last_error
                                        or "Codex compaction failed."
                                    )
                                break
                            elif method == "error":
                                message = rpc_error_message(params.get("error"))
                                if message:
                                    last_error = message
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
                        last_error
                        or detail
                        or "Codex did not confirm that compaction completed."
                    )
                return CompactResult(
                    session_id=resume_session_id, context_usage=context_usage
                )
