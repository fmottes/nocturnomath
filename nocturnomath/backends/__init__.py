"""Provider backends for driving Anthropic, OpenAI, and Google CLIs directly."""

from ..auth import ProviderId
from .anthropic import AnthropicBackend, claude_cli_candidates
from .base import (
    DEFAULT_EFFORT,
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
    find_cli_executables,
)
from .google import GoogleBackend, agy_cli_candidates
from .openai import OpenAIBackend, codex_cli_candidates


def default_backends() -> dict[ProviderId, AgentBackend]:
    """Create the default direct-CLI backends for all supported providers."""
    return {
        "anthropic": AnthropicBackend(),
        "openai": OpenAIBackend(),
        "google": GoogleBackend(),
    }


__all__ = [
    "DEFAULT_EFFORT",
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
    "agy_cli_candidates",
    "claude_cli_candidates",
    "codex_cli_candidates",
    "default_backends",
    "find_cli_executables",
]
