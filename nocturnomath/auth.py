"""In-memory authentication configuration for Claude Agent SDK clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AuthMethod = Literal["claude_code", "subscription", "api_key"]

# Explicit credentials must not accidentally lose to a higher-priority credential
# inherited by the Claude subprocess. Empty values disable those inherited sources.
_COMPETING_AUTH_ENV = {
    "ANTHROPIC_AUTH_TOKEN": "",
    "ANTHROPIC_API_KEY": "",
    "CLAUDE_CODE_OAUTH_TOKEN": "",
    "CLAUDE_CODE_USE_BEDROCK": "",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS": "",
    "CLAUDE_CODE_USE_VERTEX": "",
    "CLAUDE_CODE_USE_FOUNDRY": "",
    # A manually entered Claude credential must go to Anthropic, even if the
    # inherited setup uses a gateway. Automatic mode preserves that setup.
    "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
}


@dataclass(frozen=True)
class ClaudeAuth:
    """One credential choice, kept only in the running process."""

    method: AuthMethod = "claude_code"
    credential: str | None = field(default=None, repr=False)
    source: Literal["claude_code", "interactive"] = "claude_code"

    @classmethod
    def from_environment(cls) -> ClaudeAuth:
        """Let Claude Code resolve its own login and environment precedence."""
        return cls()

    @classmethod
    def interactive(cls, method: AuthMethod, credential: str | None) -> ClaudeAuth:
        """Validate a credential entered through one of Nocturnomath's UIs."""
        if method not in ("claude_code", "subscription", "api_key"):
            raise ValueError("Choose Claude Code, a subscription token, or an API key.")
        if method == "claude_code":
            if credential and credential.strip():
                raise ValueError("Claude Code login does not take a credential.")
            return cls.from_environment()
        value = (credential or "").strip()
        if not value:
            noun = "subscription token" if method == "subscription" else "API key"
            raise ValueError(f"Enter a Claude {noun}.")
        if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError(
                "The credential must not contain whitespace or control characters."
            )
        return cls(method, value, "interactive")

    def sdk_env(self) -> dict[str, str]:
        """Return the environment overlay passed directly to ClaudeAgentOptions."""
        if self.method == "claude_code":
            return {}
        env = dict(_COMPETING_AUTH_ENV)
        if self.method == "subscription":
            env["CLAUDE_CODE_OAUTH_TOKEN"] = self.credential or ""
        else:
            env["ANTHROPIC_API_KEY"] = self.credential or ""
        return env

    def public(self) -> dict[str, str | bool]:
        """Describe the selection without exposing the secret."""
        labels = {
            "claude_code": "Claude Code (automatic)",
            "subscription": "Claude subscription",
            "api_key": "Claude API key",
        }
        return {
            "method": self.method,
            "source": self.source,
            "label": labels[self.method],
        }
