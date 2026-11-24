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

_DIRECT_API_KEY_RULES: dict[
    ProviderId, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]
] = {
    # (blocking boolean toggles, blocking non-empty vars, direct API key vars)
    "anthropic": (
        (
            "CLAUDE_CODE_USE_BEDROCK",
            "CLAUDE_CODE_USE_VERTEX",
            "CLAUDE_CODE_USE_FOUNDRY",
        ),
        (
            "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
        ),
        ("ANTHROPIC_API_KEY",),
    ),
    "openai": (
        (),
        ("CODEX_ACCESS_TOKEN",),
        ("CODEX_API_KEY", "OPENAI_API_KEY"),
    ),
    "google": (
        ("GOOGLE_GENAI_USE_VERTEXAI",),
        (),
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    ),
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
    provider: (*toggles, *blockers, *keys)
    for provider, (toggles, blockers, keys) in _DIRECT_API_KEY_RULES.items()
}

_PROVIDER_SECRET_ENV: dict[ProviderId, tuple[str, ...]] = {
    "anthropic": _CREDENTIAL_ENV["anthropic"],
    "openai": (*_CREDENTIAL_ENV["openai"], "CODEX_ACCESS_TOKEN"),
    "google": _CREDENTIAL_ENV["google"],
}

ALL_PROVIDER_SECRET_ENV: tuple[str, ...] = tuple(
    dict.fromkeys(
        key for provider in PROVIDERS for key in _PROVIDER_SECRET_ENV[provider]
    )
)

_BOOL_TOGGLE_ENV = frozenset(
    name for toggles, _, _ in _DIRECT_API_KEY_RULES.values() for name in toggles
)


def env_flag_enabled(name: str, env: Mapping[str, str] | None = None) -> bool:
    """Return True if an environment toggle variable is set to an enabled value."""
    source = os.environ if env is None else env
    raw = (source.get(name) or "").strip().lower()
    return bool(raw) and raw not in ("0", "false", "no", "off")


def inherited_credential(
    provider: ProviderId = "anthropic", env: Mapping[str, str] | None = None
) -> str | None:
    """Name the environment variable a CLI or direct HTTP client uses."""
    source = os.environ if env is None else env
    for name in _INHERITED_ENV[provider]:
        if name in _BOOL_TOGGLE_ENV:
            if env_flag_enabled(name, source):
                return name
        elif (source.get(name) or "").strip():
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
        if any(ord(char) < 33 or ord(char) > 126 for char in value):
            raise ValueError(
                "The credential must not contain whitespace, control, or non-ASCII characters."
            )
        return cls(provider=provider, method=method, credential=value)

    def direct_api_key(self, env: Mapping[str, str] | None = None) -> str | None:
        """Return the plain API key to use over direct HTTP, or ``None`` for CLI mode."""
        source = os.environ if env is None else env
        if self.method == "api_key":
            return (self.credential or "").strip() or None
        if self.method != "cli":
            return None
        rule = _DIRECT_API_KEY_RULES.get(self.provider)
        if rule is None:
            return None
        toggles, blockers, keys = rule
        if any(env_flag_enabled(name, source) for name in toggles) or any(
            (source.get(name) or "").strip() for name in blockers
        ):
            return None
        for name in keys:
            value = (source.get(name) or "").strip()
            if value:
                return value
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
            and not (env.get("GEMINI_API_KEY") or "").strip()
            and (env.get("GOOGLE_API_KEY") or "").strip()
        ):
            env["GEMINI_API_KEY"] = env["GOOGLE_API_KEY"].strip()
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
        entries: list[str] = []
        for provider in PROVIDERS:
            auth = self._providers[provider]
            if auth.method != "cli":
                entries.append(
                    f"{PROVIDER_TITLES[provider]}: {PROVIDER_METHODS[provider][auth.method]}"
                )
            elif auth.direct_api_key():
                entries.append(f"{PROVIDER_TITLES[provider]}: API key (env)")
        return ", ".join(entries) if entries else "CLI logins (automatic)"

    def short_summary_label(self) -> str:
        entries: list[str] = []
        for provider in PROVIDERS:
            auth = self._providers[provider]
            if auth.method != "cli":
                entries.append(f"{provider}:{auth.method}")
            elif auth.direct_api_key():
                entries.append(f"{provider}:api_key(env)")
        return ", ".join(entries) if entries else "cli"

    def public(
        self,
        provider_models: Mapping[ProviderId, Sequence[str]] | None = None,
        provider_errors: Mapping[ProviderId, str] | None = None,
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
                    "error": (
                        provider_errors.get(provider)
                        if provider_errors is not None
                        else None
                    ),
                }
                for provider in PROVIDERS
            },
        }
