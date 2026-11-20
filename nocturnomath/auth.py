"""In-memory multi-provider authentication configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

ProviderId = Literal["anthropic", "openai", "google"]
PROVIDERS: tuple[ProviderId, ...] = ("anthropic", "openai", "google")

PROVIDER_TITLES: dict[ProviderId, str] = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
}

PROVIDER_METHODS: dict[ProviderId, dict[str, str]] = {
    "anthropic": {
        "cli": "Claude CLI (automatic)",
        "subscription": "Claude subscription",
        "api_key": "Anthropic API key",
    },
    "openai": {
        "cli": "Codex CLI (automatic)",
        "api_key": "OpenAI API key",
    },
    "google": {
        "cli": "Antigravity CLI (automatic)",
        "api_key": "Google Gemini API key",
    },
}

_CREDENTIAL_ENV: dict[ProviderId, tuple[str, ...]] = {
    "anthropic": (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
    ),
    "openai": (
        "CODEX_API_KEY",
        "OPENAI_API_KEY",
    ),
    "google": (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ),
}

_INHERITED_ENV: dict[ProviderId, tuple[str, ...]] = {
    "anthropic": (
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ),
    "openai": (
        "CODEX_API_KEY",
        "OPENAI_API_KEY",
        "CODEX_ACCESS_TOKEN",
    ),
    "google": (
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ),
}

_PROVIDER_SECRET_ENV: dict[ProviderId, tuple[str, ...]] = {
    "anthropic": _CREDENTIAL_ENV["anthropic"],
    "openai": (*_CREDENTIAL_ENV["openai"], "CODEX_ACCESS_TOKEN"),
    "google": _CREDENTIAL_ENV["google"],
}

_BOOL_TOGGLE_ENV = frozenset(
    {
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "GOOGLE_GENAI_USE_VERTEXAI",
    }
)


def env_flag_enabled(name: str, env: Mapping[str, str] | None = None) -> bool:
    """Return True if an environment toggle variable is set to an enabled value."""
    source = os.environ if env is None else env
    raw = (source.get(name) or "").strip().lower()
    return bool(raw) and raw not in ("0", "false", "no", "off")


def inherited_credential(provider: ProviderId = "anthropic") -> str | None:
    """Name the environment variable a CLI uses instead of its saved login."""
    for name in _INHERITED_ENV[provider]:
        if name in _BOOL_TOGGLE_ENV:
            if env_flag_enabled(name):
                return name
        elif os.environ.get(name):
            return name
    return None


@dataclass(frozen=True)
class ProviderAuth:
    """One provider credential choice, kept only in the running process."""

    provider: ProviderId
    method: str = "cli"
    credential: str | None = field(default=None, repr=False)

    @classmethod
    def interactive(
        cls, provider: str, method: str, credential: str | None
    ) -> ProviderAuth:
        """Validate a credential entered through one of Nocturnomath's UIs."""
        if provider not in PROVIDERS:
            raise ValueError("Choose Anthropic, OpenAI, or Google.")
        methods = PROVIDER_METHODS[provider]
        if method not in methods:
            choices = ", ".join(methods)
            raise ValueError(
                f"Choose a valid {PROVIDER_TITLES[provider]} authentication method: {choices}."
            )
        if method == "cli":
            if credential and credential.strip():
                raise ValueError(f"{methods['cli']} does not take a credential.")
            return cls(provider=provider, method="cli")
        value = (credential or "").strip()
        if not value:
            noun = "subscription token" if method == "subscription" else "API key"
            raise ValueError(f"Enter a {PROVIDER_TITLES[provider]} {noun}.")
        if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError(
                "The credential must not contain whitespace or control characters."
            )
        return cls(provider=provider, method=method, credential=value)

    def direct_api_key(self, env: Mapping[str, str] | None = None) -> str | None:
        """Return the plain API key to use over direct HTTP, or ``None`` for CLI mode."""
        source = os.environ if env is None else env
        if self.method == "api_key":
            return (self.credential or "").strip() or None
        if self.method != "cli":
            return None
        if self.provider == "anthropic":
            if any(
                env_flag_enabled(name, source)
                for name in (
                    "CLAUDE_CODE_USE_BEDROCK",
                    "CLAUDE_CODE_USE_VERTEX",
                    "CLAUDE_CODE_USE_FOUNDRY",
                )
            ) or any(
                (source.get(name) or "").strip()
                for name in (
                    "ANTHROPIC_AUTH_TOKEN",
                    "CLAUDE_CODE_OAUTH_TOKEN",
                    "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
                )
            ):
                return None
            return (source.get("ANTHROPIC_API_KEY") or "").strip() or None
        if self.provider == "openai":
            if (source.get("CODEX_ACCESS_TOKEN") or "").strip():
                return None
            return (
                (source.get("CODEX_API_KEY") or "").strip()
                or (source.get("OPENAI_API_KEY") or "").strip()
                or None
            )
        if self.provider == "google":
            if env_flag_enabled("GOOGLE_GENAI_USE_VERTEXAI", source):
                return None
            return (
                (source.get("GEMINI_API_KEY") or "").strip()
                or (source.get("GOOGLE_API_KEY") or "").strip()
                or None
            )
        return None

    def sdk_env(self) -> dict[str, str]:
        """Return the environment overlay passed directly to the provider CLI."""
        if self.method == "cli":
            return {}
        env = dict.fromkeys(_CREDENTIAL_ENV[self.provider], "")
        secret = self.credential or ""
        if self.provider == "anthropic":
            key = (
                "CLAUDE_CODE_OAUTH_TOKEN"
                if self.method == "subscription"
                else "ANTHROPIC_API_KEY"
            )
            env[key] = secret
        elif self.provider == "openai":
            env["CODEX_API_KEY"] = secret
            env["OPENAI_API_KEY"] = secret
        elif self.provider == "google":
            env["GEMINI_API_KEY"] = secret
        return env

    def process_env(self, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
        """Build a subprocess environment isolated from other providers' secrets."""
        env = dict(os.environ if base_env is None else base_env)
        for other, keys in _PROVIDER_SECRET_ENV.items():
            if other != self.provider:
                for key in keys:
                    env.pop(key, None)
        if self.method != "cli":
            if self.provider == "openai":
                env["CODEX_ACCESS_TOKEN"] = ""
            elif self.provider == "google":
                env["GOOGLE_GENAI_USE_VERTEXAI"] = ""
            env.update(self.sdk_env())
        elif (
            self.provider == "google"
            and not env_flag_enabled("GOOGLE_GENAI_USE_VERTEXAI", env)
            and not env.get("GEMINI_API_KEY")
            and env.get("GOOGLE_API_KEY")
        ):
            env["GEMINI_API_KEY"] = env["GOOGLE_API_KEY"]
        return env

    def public(self) -> dict[str, Any]:
        """Describe the selection without exposing the secret."""
        note = None
        if self.method == "cli":
            name = inherited_credential(self.provider)
            if name:
                if self.direct_api_key():
                    note = (
                        f"{name} is set in the environment and uses the direct "
                        f"{PROVIDER_TITLES[self.provider]} HTTP API (no CLI required)."
                    )
                else:
                    note = (
                        f"{name} is set in the environment and takes precedence "
                        f"over {PROVIDER_TITLES[self.provider]}'s saved CLI login."
                    )
        return {
            "provider": self.provider,
            "title": PROVIDER_TITLES[self.provider],
            "method": self.method,
            "label": PROVIDER_METHODS[self.provider][self.method],
            "note": note,
        }


class AuthStore:
    """In-memory credential configuration across all supported providers."""

    def __init__(self, providers: Mapping[ProviderId, ProviderAuth] | None = None):
        self._providers: dict[ProviderId, ProviderAuth] = {
            provider: ProviderAuth(provider=provider, method="cli")
            for provider in PROVIDERS
        }
        if providers:
            for provider, auth in providers.items():
                self._providers[provider] = auth

    def get(self, provider: ProviderId) -> ProviderAuth:
        return self._providers[provider]

    def set(
        self, provider: ProviderId, method: str, credential: str | None = None
    ) -> ProviderAuth:
        auth = ProviderAuth.interactive(provider, method, credential)
        self._providers[auth.provider] = auth
        return auth

    def copy(self) -> AuthStore:
        return AuthStore(self._providers)

    def summary_label(self) -> str:
        custom = [
            f"{PROVIDER_TITLES[p]}: {PROVIDER_METHODS[p][self._providers[p].method]}"
            for p in PROVIDERS
            if self._providers[p].method != "cli"
        ]
        return ", ".join(custom) if custom else "CLI logins (automatic)"

    def public(
        self, provider_models: Mapping[ProviderId, Sequence[str]] | None = None
    ) -> dict[str, Any]:
        return {
            "label": self.summary_label(),
            "providers": {
                provider: {
                    **self._providers[provider].public(),
                    "models_count": (
                        len(provider_models.get(provider, ()))
                        if provider_models is not None
                        else None
                    ),
                }
                for provider in PROVIDERS
            },
        }
