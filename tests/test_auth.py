import pytest

from nocturnomath.auth import AuthStore, ProviderAuth

INHERITED = (
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CODEX_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_USE_VERTEXAI",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in INHERITED:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("provider", "expected_label"),
    [
        ("anthropic", "Claude CLI (automatic)"),
        ("openai", "Codex CLI (automatic)"),
        ("google", "Antigravity CLI (automatic)"),
    ],
)
def test_automatic_mode_changes_nothing(provider, expected_label):
    auth = ProviderAuth(provider)
    assert auth.method == "cli"
    assert auth.sdk_env() == {}
    assert auth.public() == {
        "provider": provider,
        "title": {"anthropic": "Anthropic", "openai": "OpenAI", "google": "Google"}[
            provider
        ],
        "method": "cli",
        "label": expected_label,
        "note": None,
    }


def test_automatic_mode_reports_inherited_credentials(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-secret")
    public = ProviderAuth("anthropic").public()
    assert "ANTHROPIC_API_KEY" in public["note"]
    assert "api-secret" not in repr(public)
    assert ProviderAuth("anthropic").sdk_env() == {}

    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    assert "CLAUDE_CODE_USE_BEDROCK" in ProviderAuth("anthropic").public()["note"]

    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    assert "OPENAI_API_KEY" in ProviderAuth("openai").public()["note"]
    assert ProviderAuth("openai").sdk_env() == {}

    monkeypatch.setenv("GEMINI_API_KEY", "gem-secret")
    assert "GEMINI_API_KEY" in ProviderAuth("google").public()["note"]
    assert ProviderAuth("google").sdk_env() == {}


def test_manual_credentials_replace_only_provider_credentials(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")

    subscription = ProviderAuth.interactive(
        "anthropic", "subscription", " oauth-secret "
    )
    env = subscription.sdk_env()
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-secret"
    assert env["ANTHROPIC_API_KEY"] == ""
    assert env["ANTHROPIC_AUTH_TOKEN"] == ""
    assert "ANTHROPIC_BASE_URL" not in env
    assert not any(name.startswith("CLAUDE_CODE_USE_") for name in env)
    assert "oauth-secret" not in repr(subscription)

    api_key = ProviderAuth.interactive("anthropic", "api_key", "api-secret")
    env = api_key.sdk_env()
    assert env["ANTHROPIC_API_KEY"] == "api-secret"
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == ""
    assert api_key.public()["label"] == "Anthropic API key"

    openai_key = ProviderAuth.interactive("openai", "api_key", "sk-openai")
    assert openai_key.sdk_env() == {
        "CODEX_API_KEY": "sk-openai",
        "OPENAI_API_KEY": "sk-openai",
    }
    assert "sk-openai" not in repr(openai_key)

    google_key = ProviderAuth.interactive("google", "api_key", "AIza-secret")
    assert google_key.sdk_env() == {
        "GEMINI_API_KEY": "AIza-secret",
        "GOOGLE_API_KEY": "",
    }
    assert "AIza-secret" not in repr(google_key)


def test_returning_to_automatic_is_the_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "inherited-key")
    assert ProviderAuth.interactive("anthropic", "cli", None) == ProviderAuth(
        "anthropic"
    )
    assert ProviderAuth.interactive("anthropic", "cli", "  ").sdk_env() == {}
    assert ProviderAuth.interactive("openai", "cli", None).sdk_env() == {}
    assert ProviderAuth.interactive("google", "cli", None).sdk_env() == {}


def test_interactive_validation_messages_never_echo_the_secret():
    with pytest.raises(ValueError, match="API key"):
        ProviderAuth.interactive("anthropic", "api_key", "   ")
    with pytest.raises(ValueError, match="whitespace") as info:
        ProviderAuth.interactive("anthropic", "subscription", "top secret")
    assert "top secret" not in str(info.value)
    with pytest.raises(ValueError, match="does not take"):
        ProviderAuth.interactive("openai", "cli", "stray")
    with pytest.raises(ValueError, match="Anthropic, OpenAI, or Google"):
        ProviderAuth.interactive("other", "cli", None)
    with pytest.raises(ValueError, match="OpenAI"):
        ProviderAuth.interactive("openai", "subscription", "x")


def test_auth_store_tracks_multiple_providers_and_redacts_secrets():
    store = AuthStore()
    assert store.summary_label() == "CLI logins (automatic)"
    store.set("openai", "api_key", "sk-openai-secret")
    store.set("google", "api_key", "gem-secret")

    public = store.public({"anthropic": ["a"], "openai": ["o1", "o2"], "google": []})
    assert "sk-openai-secret" not in repr(public)
    assert "gem-secret" not in repr(public)
    assert public["providers"]["anthropic"]["method"] == "cli"
    assert public["providers"]["anthropic"]["models_count"] == 1
    assert public["providers"]["openai"]["method"] == "api_key"
    assert public["providers"]["openai"]["models_count"] == 2
    assert public["providers"]["google"]["models_count"] == 0


def test_falsey_provider_toggles_do_not_report_inherited_credentials(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    assert ProviderAuth("google").public()["note"] is None

    monkeypatch.setenv("GEMINI_API_KEY", "gem-secret")
    assert "GEMINI_API_KEY" in ProviderAuth("google").public()["note"]

    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "0")
    assert ProviderAuth("anthropic").public()["note"] is None
