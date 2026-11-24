"""Provider-agnostic agent backend interface and shared subprocess helpers."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from ..auth import ProviderAuth, ProviderId
from ..tools import ToolSpec

DEFAULT_EFFORT = "high"
DEFAULT_TOOL_TIMEOUT_S = 660
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max", "ultra")


def package_version() -> str:
    """Return the installed Nocturnomath package version."""
    try:
        return version("nocturnomath")
    except PackageNotFoundError:
        return "0.0.0"


class SessionResumeError(RuntimeError):
    """Raised when a saved native provider session/thread cannot be resumed."""


@dataclass(frozen=True)
class ModelSpec:
    """One model discovered from a provider backend."""

    id: str
    raw_id: str
    provider: ProviderId
    label: str
    efforts: list[str] = field(default_factory=list)
    is_default: bool = False


@dataclass(frozen=True)
class SessionStarted:
    """Native provider conversation/thread identifier emitted during a turn."""

    session_id: str


@dataclass(frozen=True)
class TurnStarted:
    """Signal that the provider accepted the turn and model execution has begun."""


@dataclass(frozen=True)
class TextDelta:
    """Incremental assistant text token/chunk."""

    text: str


@dataclass(frozen=True)
class TextBlockDone:
    """Complete assistant text block ready for transcript persistence."""

    text: str


@dataclass(frozen=True)
class CompactionNotice:
    """Notice that the provider compacted conversation history during a turn."""


@dataclass(frozen=True)
class MessageUsage:
    """Per-assistant-message token telemetry."""

    message_id: str | None
    model: str | None
    usage: dict[str, Any]


@dataclass(frozen=True)
class TurnResult:
    """Final turn summary including optional usage and context window metrics."""

    usage: dict[str, Any] | None = None
    model_usage: dict[str, Any] | None = None
    context_usage: dict[str, Any] | None = None
    total_cost_usd: float | None = None
    duration_ms: int | None = None
    duration_api_ms: int | None = None
    num_turns: int | None = None


TurnEvent = (
    SessionStarted
    | TurnStarted
    | TextDelta
    | TextBlockDone
    | CompactionNotice
    | MessageUsage
    | TurnResult
)


@dataclass(frozen=True)
class CompactResult:
    """Outcome of an explicit conversation compaction."""

    session_id: str | None
    context_usage: dict[str, Any] | None
    record_reloaded: bool | None = None


class AgentBackend(Protocol):
    """Uniform contract for driving a provider backend via HTTP API or local CLI."""

    provider: ProviderId
    uses_compact_hook: bool

    async def discover_models(self, auth: ProviderAuth) -> list[ModelSpec]: ...

    async def discover_models_staged(
        self, auth: ProviderAuth
    ) -> tuple[list[ModelSpec], Callable[[], None]]: ...

    def run_turn(
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
    ) -> AsyncIterator[TurnEvent]: ...

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
    ) -> CompactResult: ...


def _is_windows_batch(candidate: Path) -> bool:
    return sys.platform == "win32" and candidate.suffix.lower() in (".bat", ".cmd")


def is_usable_cli_binary(candidate: Path) -> bool:
    """Return whether ``candidate`` is an executable non-batch binary file."""
    return (
        os.path.isfile(candidate)
        and os.access(candidate, os.X_OK)
        and not _is_windows_batch(candidate)
    )


def find_cli_executables(
    binary_name: str, extra_candidates: Sequence[Path] = ()
) -> list[Path]:
    """Locate executable binaries on PATH/user bin, ignoring relative and cwd paths."""
    launch_dir = Path.cwd().resolve()
    directories = [Path(entry) for entry in os.get_exec_path()]
    home = Path(os.path.expanduser("~"))
    if home.is_absolute():
        directories.append(home / ".local" / "bin")

    candidates: list[Path] = []
    seen: set[Path] = set()

    def add_if_usable(candidate: Path, *, ignore_cwd: bool) -> None:
        if not is_usable_cli_binary(candidate):
            return
        resolved = candidate.resolve()
        if ignore_cwd and launch_dir in (candidate.parent.resolve(), resolved.parent):
            return
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(candidate)

    for directory in directories:
        if not directory.is_absolute():
            continue
        add_if_usable(directory / binary_name, ignore_cwd=True)

    for extra in extra_candidates:
        add_if_usable(extra, ignore_cwd=False)

    return candidates


def is_int(value: Any) -> bool:
    """Return True for integers while excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool)


def context_usage_payload(
    used: Any,
    window: Any,
    model: str | None,
    *,
    min_used: int = 0,
) -> dict[str, Any] | None:
    """Build a normalized context-usage dict when ``used`` is a valid integer."""
    if not is_int(used) or used < min_used:
        return None
    return {
        "used_tokens": used,
        "window_tokens": window if is_int(window) and window > 0 else None,
        "model": model,
    }


def rpc_error_message(err: Any) -> str | None:
    """Extract a readable string from a JSON-RPC or provider error object."""
    if err is None:
        return None
    if isinstance(err, dict):
        msg = err.get("message")
        return str(msg) if msg else None
    text = str(err)
    return text or None


async def read_lines(stream: asyncio.StreamReader | None) -> AsyncIterator[str]:
    """Read newline-delimited UTF-8 lines of arbitrary length without 64 KiB limits."""
    if stream is None:
        return
    buffer = bytearray()
    search_from = 0
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            if buffer:
                line = buffer.decode("utf-8", errors="replace").strip()
                if line:
                    yield line
            break
        buffer.extend(chunk)
        start = 0
        while True:
            newline = buffer.find(b"\n", search_from)
            if newline == -1:
                break
            line = buffer[start:newline].decode("utf-8", errors="replace").strip()
            start = newline + 1
            search_from = start
            if line:
                yield line
        if start:
            del buffer[:start]
        search_from = len(buffer)


async def read_jsonl(
    stream: asyncio.StreamReader | None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield decoded JSON objects from a stream of NDJSON lines."""
    async for line in read_lines(stream):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            yield item


class StderrCollector:
    """Concurrently drain subprocess stderr while retaining a bounded tail."""

    def __init__(self, stream: asyncio.StreamReader | None, max_bytes: int = 32768):
        self._buffer = bytearray()
        self._max_bytes = max_bytes
        self._task: asyncio.Task[None] | None = (
            asyncio.create_task(self._drain(stream)) if stream is not None else None
        )

    async def _drain(self, stream: asyncio.StreamReader) -> None:
        with suppress(Exception):
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                self._buffer.extend(chunk)
                if len(self._buffer) > self._max_bytes:
                    del self._buffer[: len(self._buffer) - self._max_bytes]

    async def text(self) -> str:
        """Return the collected stderr tail, waiting briefly for EOF."""
        if self._task is not None:
            with suppress(Exception):
                await asyncio.wait_for(asyncio.shield(self._task), timeout=1.0)
        return self._buffer.decode("utf-8", errors="replace").strip()

    async def close(self) -> None:
        """Cancel and await the background stderr reader task."""
        if self._task is None:
            return
        if not self._task.done():
            self._task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await self._task


async def _discard_stream(stream: asyncio.StreamReader | None) -> None:
    if stream is None:
        return
    with suppress(Exception):
        while await stream.read(65536):
            pass


def _signal_process_or_group(proc: asyncio.subprocess.Process, sig: int) -> None:
    if os.name == "posix" and proc.pid is not None:
        with suppress(ProcessLookupError, PermissionError, OSError):
            pgid = os.getpgid(proc.pid)
            if pgid == proc.pid:
                os.killpg(pgid, sig)
                return
    with suppress(ProcessLookupError):
        if sig == getattr(signal, "SIGKILL", None):
            proc.kill()
        else:
            proc.terminate()


async def terminate_process(
    proc: asyncio.subprocess.Process, stderr: StderrCollector | None = None
) -> None:
    """Terminate a subprocess gracefully, killing it if it ignores SIGTERM."""
    stdout_drain: asyncio.Task[None] | None = None
    cancelled_exc: asyncio.CancelledError | None = None
    try:
        if proc.returncode is None:
            if proc.stdout is not None:
                stdout_drain = asyncio.create_task(_discard_stream(proc.stdout))
            if proc.stdin is not None and not proc.stdin.is_closing():
                with suppress(Exception):
                    proc.stdin.close()
            _signal_process_or_group(proc, signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.CancelledError as exc:
                cancelled_exc = exc
            except TimeoutError:
                pass
            if proc.returncode is None:
                kill_sig = getattr(signal, "SIGKILL", signal.SIGTERM)
                _signal_process_or_group(proc, kill_sig)
                wait_task = asyncio.create_task(proc.wait())
                try:
                    await asyncio.wait_for(asyncio.shield(wait_task), timeout=2.0)
                except asyncio.CancelledError as exc:
                    cancelled_exc = cancelled_exc or exc
                    with suppress(TimeoutError, Exception):
                        await asyncio.wait_for(wait_task, timeout=0.5)
                except TimeoutError:
                    wait_task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await wait_task
    finally:
        if stdout_drain is not None:
            if not stdout_drain.done():
                stdout_drain.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await stdout_drain
        if stderr is not None:
            await stderr.close()
        transport = getattr(proc, "_transport", None)
        if transport is not None:
            with suppress(Exception):
                transport.close()
    if cancelled_exc is not None:
        raise cancelled_exc


class JsonMessageStream:
    """JSON-RPC stream allowing RPC calls to buffer concurrent notifications."""

    def __init__(self, stream: asyncio.StreamReader | None):
        self._stream = stream
        self._raw = bytearray()
        self._search_from = 0
        self._eof = stream is None
        self._buffered: deque[dict[str, Any]] = deque()

    def clear_buffered(self) -> None:
        """Discard notifications buffered before the current phase begins."""
        self._buffered.clear()

    def drain_buffered(self) -> list[dict[str, Any]]:
        """Pop and return all currently buffered notifications in FIFO order."""
        items = list(self._buffered)
        self._buffered.clear()
        return items

    def _pop_raw_object(self) -> dict[str, Any] | None:
        start = 0
        result: dict[str, Any] | None = None
        while True:
            newline = self._raw.find(b"\n", self._search_from)
            if newline == -1:
                if start:
                    del self._raw[:start]
                    start = 0
                self._search_from = len(self._raw)
                if not self._eof or not self._raw:
                    return None
                line = self._raw.decode("utf-8", errors="replace").strip()
                self._raw.clear()
                self._search_from = 0
            else:
                line = (
                    self._raw[start:newline].decode("utf-8", errors="replace").strip()
                )
                start = newline + 1
                self._search_from = start
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                result = item
                break
        if start:
            del self._raw[:start]
            self._search_from = max(0, self._search_from - start)
        return result

    async def _next_wire_unbounded(self) -> dict[str, Any] | None:
        while True:
            item = self._pop_raw_object()
            if item is not None or self._eof:
                return item
            assert self._stream is not None
            chunk = await self._stream.read(65536)
            if not chunk:
                self._eof = True
            else:
                self._raw.extend(chunk)

    async def next_wire(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Read the next object directly from the underlying stdout stream."""
        if timeout is None:
            return await self._next_wire_unbounded()
        async with asyncio.timeout(timeout):
            return await self._next_wire_unbounded()

    async def next_message(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Return the next buffered notification or read the next wire object."""
        if self._buffered:
            return self._buffered.popleft()
        return await self.next_wire(timeout=timeout)

    def buffer_notification(self, message: dict[str, Any]) -> None:
        """Queue a notification observed during an RPC so later loops process it."""
        self._buffered.append(message)

    def __aiter__(self) -> JsonMessageStream:
        return self

    async def __anext__(self) -> dict[str, Any]:
        item = await self.next_message()
        if item is None:
            raise StopAsyncIteration
        return item


def resolve_cli_binary(
    cached: Path | None,
    candidates_fn: Callable[[], list[Path]],
    missing_message: str,
) -> Path:
    """Return a cached usable CLI binary or resolve the first candidate on PATH."""
    if cached is not None and is_usable_cli_binary(cached):
        return cached
    candidates = candidates_fn()
    if not candidates:
        raise RuntimeError(missing_message)
    return candidates[0]


async def write_json_line(
    proc: asyncio.subprocess.Process, payload: dict[str, Any], label: str
) -> None:
    """Write one newline-delimited JSON object to a subprocess's stdin."""
    if proc.stdin is None:
        raise RuntimeError(f"{label} stdin is not open.")
    proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
    await proc.stdin.drain()


@asynccontextmanager
async def managed_subprocess(
    *cmd: str,
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> AsyncIterator[tuple[asyncio.subprocess.Process, StderrCollector]]:
    """Spawn a piped subprocess in its own POSIX session and always terminate it."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
        start_new_session=(os.name == "posix"),
    )
    stderr = StderrCollector(proc.stderr)
    try:
        yield proc, stderr
    finally:
        await terminate_process(proc, stderr)


class JsonRpcClient:
    """Minimal newline-delimited JSON-RPC 2.0 client over subprocess stdio."""

    def __init__(
        self,
        proc: asyncio.subprocess.Process,
        messages: JsonMessageStream,
        stderr: StderrCollector | None,
        *,
        label: str,
        on_request: Callable[[dict[str, Any]], Awaitable[None]],
    ):
        self.proc = proc
        self.messages = messages
        self.stderr = stderr
        self.label = label
        self._on_request = on_request
        self._counter = 0

    def next_id(self) -> int:
        self._counter += 1
        return self._counter

    async def send(self, payload: dict[str, Any]) -> None:
        await write_json_line(self.proc, payload, self.label)

    async def send_request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        disconnect_message: str | None = None,
    ) -> int:
        req_id = self.next_id()
        try:
            await self.send(
                {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
            )
        except (BrokenPipeError, ConnectionResetError) as exc:
            detail = await self.stderr.text() if self.stderr is not None else ""
            raise RuntimeError(
                detail
                or disconnect_message
                or f"{self.label} closed before replying to {method}."
            ) from exc
        return req_id

    async def rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        req_id = await self.send_request(method, params)
        while (msg := await self.messages.next_wire()) is not None:
            if msg.get("id") is not None and "method" in msg:
                await self._on_request(msg)
                continue
            if msg.get("id") == req_id:
                if "error" in msg and msg["error"] is not None:
                    raise RuntimeError(
                        rpc_error_message(msg["error"])
                        or f"{self.label} {method} failed."
                    )
                return msg.get("result") or {}
            if msg.get("id") is None:
                self.messages.buffer_notification(msg)
        detail = await self.stderr.text() if self.stderr is not None else ""
        raise RuntimeError(
            detail or f"{self.label} closed before replying to {method}."
        )
