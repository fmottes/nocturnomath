"""Provider-agnostic agent backend interface and shared subprocess helpers."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import deque
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from ..auth import ProviderAuth, ProviderId
from ..tools import ToolSpec

DEFAULT_EFFORT = "high"
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


class AgentBackend(Protocol):
    """Uniform contract for driving a provider CLI over stdio."""

    provider: ProviderId
    uses_compact_hook: bool

    async def discover_models(self, auth: ProviderAuth) -> list[ModelSpec]: ...

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
        if not os.path.isfile(candidate) or not os.access(candidate, os.X_OK):
            return
        if _is_windows_batch(candidate):
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


async def read_lines(stream: asyncio.StreamReader | None) -> AsyncIterator[str]:
    """Read newline-delimited UTF-8 lines of arbitrary length without 64 KiB limits."""
    if stream is None:
        return
    buffer = bytearray()
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            if buffer:
                line = buffer.decode("utf-8", errors="replace").strip()
                if line:
                    yield line
            break
        buffer.extend(chunk)
        while True:
            newline = buffer.find(b"\n")
            if newline == -1:
                break
            line = bytes(buffer[:newline]).decode("utf-8", errors="replace").strip()
            del buffer[: newline + 1]
            if line:
                yield line


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


async def terminate_process(
    proc: asyncio.subprocess.Process, stderr: StderrCollector | None = None
) -> None:
    """Terminate a subprocess gracefully, killing it if it ignores SIGTERM."""
    try:
        if proc.returncode is None:
            if proc.stdin is not None and not proc.stdin.is_closing():
                with suppress(Exception):
                    proc.stdin.close()
            with suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
    finally:
        if stderr is not None:
            await stderr.close()


class JsonMessageStream:
    """JSON-RPC stream allowing RPC calls to buffer concurrent notifications."""

    def __init__(self, stream: asyncio.StreamReader | None):
        self._iter = read_jsonl(stream)
        self._buffered: deque[dict[str, Any]] = deque()

    async def next_wire(self) -> dict[str, Any] | None:
        """Read the next object directly from the underlying stdout stream."""
        return await anext(self._iter, None)

    def buffer_notification(self, message: dict[str, Any]) -> None:
        """Queue a notification observed during an RPC so later loops process it."""
        self._buffered.append(message)

    def __aiter__(self) -> JsonMessageStream:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._buffered:
            return self._buffered.popleft()
        return await self._iter.__anext__()
