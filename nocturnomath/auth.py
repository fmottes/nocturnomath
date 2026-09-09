"""In-memory authentication configuration for Claude Agent SDK clients."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

AuthMethod = Literal["claude_code", "subscription", "api_key"]

LABELS: dict[str, str] = {
    "claude_code": "Claude Code (automatic)",
    "subscription": "Claude subscription",
    "api_key": "Claude API key",
}

# Environment credentials the Claude CLI would prefer over one typed into
# Nocturnomath. Empty values read as unset, so a manual credential wins.
_CREDENTIAL_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
)

# Inherited settings that decide where automatic mode sends requests, roughly
# in the order the CLI consults them. Reported to the user, never overridden.
_INHERITED_ENV = (
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
)


def inherited_credential() -> str | None:
    """Name the environment variable Claude Code will use instead of its saved login."""
    for name in _INHERITED_ENV:
        if os.environ.get(name):
            return name
    return None


@dataclass(frozen=True)
class ClaudeAuth:
    """One credential choice, kept only in the running process.

    The default instance changes nothing: the SDK resolves Claude Code's saved
    login and environment exactly as it would without Nocturnomath.
    """

    method: AuthMethod = "claude_code"
    credential: str | None = field(default=None, repr=False)

    @classmethod
    def interactive(cls, method: AuthMethod, credential: str | None) -> ClaudeAuth:
        """Validate a credential entered through one of Nocturnomath's UIs."""
        if method not in LABELS:
            raise ValueError("Choose Claude Code, a subscription token, or an API key.")
        if method == "claude_code":
            if credential and credential.strip():
                raise ValueError("Claude Code login does not take a credential.")
            return cls()
        value = (credential or "").strip()
        if not value:
            noun = "subscription token" if method == "subscription" else "API key"
            raise ValueError(f"Enter a Claude {noun}.")
        if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError(
                "The credential must not contain whitespace or control characters."
            )
        return cls(method, value)

    def sdk_env(self) -> dict[str, str]:
        """Return the environment overlay passed directly to ClaudeAgentOptions."""
        if self.method == "claude_code":
            return {}
        env = dict.fromkeys(_CREDENTIAL_ENV, "")
        name = (
            "CLAUDE_CODE_OAUTH_TOKEN"
            if self.method == "subscription"
            else "ANTHROPIC_API_KEY"
        )
        env[name] = self.credential or ""
        return env

    def public(self) -> dict[str, str | None]:
        """Describe the selection without exposing the secret."""
        note = None
        if self.method == "claude_code":
            name = inherited_credential()
            if name:
                note = (
                    f"{name} is set in the environment and takes precedence "
                    "over Claude Code's saved login."
                )
        return {"method": self.method, "label": LABELS[self.method], "note": note}
