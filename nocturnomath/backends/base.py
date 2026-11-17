"""Provider-agnostic agent backend interface and shared subprocess helpers."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..auth import ProviderAuth, ProviderId
from ..tools import ToolSpec

DEFAULT_EFFORT = "high"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max", "ultra")


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
