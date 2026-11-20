import pytest

from nocturnomath.auth import AuthStore, ProviderAuth

INHERITED = (
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CODEX_API_KEY",
    "OPENAI_API_KEY",
    "CODEX_ACCESS_TOKEN",
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
    assert "direct Anthropic HTTP API" in public["note"]
    assert "api-secret" not in repr(public)
    assert ProviderAuth("anthropic").sdk_env() == {}

    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    bedrock_note = ProviderAuth("anthropic").public()["note"]
    assert "CLAUDE_CODE_USE_BEDROCK" in bedrock_note
    assert "saved CLI login" in bedrock_note

    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    openai_note = ProviderAuth("openai").public()["note"]
    assert "OPENAI_API_KEY" in openai_note
    assert "direct OpenAI HTTP API" in openai_note
    assert ProviderAuth("openai").sdk_env() == {}

    monkeypatch.setenv("GEMINI_API_KEY", "gem-secret")
    google_note = ProviderAuth("google").public()["note"]
    assert "GEMINI_API_KEY" in google_note
    assert "direct Google HTTP API" in google_note
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


def test_google_cli_mode_forwards_google_api_key_when_gemini_api_key_is_unset(
    monkeypatch,
):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-shell-key")
    auth = ProviderAuth("google", "cli")
    assert "GOOGLE_API_KEY" in auth.public()["note"]
    env = auth.process_env()
    assert env["GEMINI_API_KEY"] == "google-shell-key"

    monkeypatch.setenv("GEMINI_API_KEY", "explicit-gemini-key")
    assert "GEMINI_API_KEY" in auth.public()["note"]
    assert auth.process_env()["GEMINI_API_KEY"] == "explicit-gemini-key"

    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    assert "GEMINI_API_KEY" not in auth.process_env()


def test_direct_api_key_routing_and_precedence(monkeypatch):
    # Manual API key overrides everything and bypasses CLIs.
    assert (
        ProviderAuth.interactive("anthropic", "api_key", "sk-manual").direct_api_key()
        == "sk-manual"
    )
    assert (
        ProviderAuth.interactive("openai", "api_key", "sk-oai").direct_api_key()
        == "sk-oai"
    )
    assert (
        ProviderAuth.interactive("google", "api_key", "AIza-manual").direct_api_key()
        == "AIza-manual"
    )
    assert (
        ProviderAuth.interactive(
            "anthropic", "subscription", "oauth-tok"
        ).direct_api_key()
        is None
    )

    # Anthropic CLI mode honors ambient ANTHROPIC_API_KEY unless gateway/OAuth vars override it.
    assert ProviderAuth("anthropic", "cli").direct_api_key() is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    assert ProviderAuth("anthropic", "cli").direct_api_key() == "sk-ant-env"
    for blocking_toggle in (
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
    ):
        monkeypatch.setenv(blocking_toggle, "true")
        assert ProviderAuth("anthropic", "cli").direct_api_key() is None
        monkeypatch.delenv(blocking_toggle)
    for blocking_var in (
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
    ):
        monkeypatch.setenv(blocking_var, "token-or-fd")
        assert ProviderAuth("anthropic", "cli").direct_api_key() is None
        monkeypatch.delenv(blocking_var)

    # OpenAI CLI mode honors CODEX_API_KEY then OPENAI_API_KEY unless CODEX_ACCESS_TOKEN is set.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env")
    assert ProviderAuth("openai", "cli").direct_api_key() == "sk-openai-env"
    monkeypatch.setenv("CODEX_API_KEY", "sk-codex-env")
    assert ProviderAuth("openai", "cli").direct_api_key() == "sk-codex-env"
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "codex-oauth")
    assert ProviderAuth("openai", "cli").direct_api_key() is None
    monkeypatch.delenv("CODEX_ACCESS_TOKEN")

    # Google CLI mode honors GEMINI_API_KEY then GOOGLE_API_KEY unless Vertex AI is enabled.
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-google-env")
    assert ProviderAuth("google", "cli").direct_api_key() == "AIza-google-env"
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-gemini-env")
    assert ProviderAuth("google", "cli").direct_api_key() == "AIza-gemini-env"
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "1")
    assert ProviderAuth("google", "cli").direct_api_key() is None
    assert (
        ProviderAuth.interactive("google", "api_key", "AIza-manual").direct_api_key()
        == "AIza-manual"
    )
