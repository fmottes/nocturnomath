"""Provider backends for driving Anthropic, OpenAI, and Google via HTTP APIs or local CLIs."""

from ..auth import ProviderId
from .anthropic import AnthropicBackend, claude_cli_candidates
from .base import (
    DEFAULT_EFFORT,
    DEFAULT_TOOL_TIMEOUT_S,
    EFFORT_LEVELS,
    AgentBackend,
    CompactionNotice,
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
    find_cli_executables,
)
from .google import GoogleBackend, agy_cli_candidates
from .openai import OpenAIBackend, codex_cli_candidates


def default_backends() -> dict[ProviderId, AgentBackend]:
    """Create the default provider backends (direct HTTP APIs and local CLIs)."""
    return {
        "anthropic": AnthropicBackend(),
        "openai": OpenAIBackend(),
        "google": GoogleBackend(),
    }


__all__ = [
    "DEFAULT_EFFORT",
    "DEFAULT_TOOL_TIMEOUT_S",
    "EFFORT_LEVELS",
    "AgentBackend",
    "AnthropicBackend",
    "CompactResult",
    "CompactionNotice",
    "GoogleBackend",
    "MessageUsage",
    "ModelSpec",
    "OpenAIBackend",
    "SessionResumeError",
    "SessionStarted",
    "TextBlockDone",
    "TextDelta",
    "TurnEvent",
    "TurnResult",
    "TurnStarted",
    "agy_cli_candidates",
    "claude_cli_candidates",
    "codex_cli_candidates",
    "default_backends",
    "find_cli_executables",
]
