import pytest

from nocturnomath.auth import ClaudeAuth

INHERITED = (
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in INHERITED:
        monkeypatch.delenv(name, raising=False)


def test_automatic_mode_changes_nothing():
    auth = ClaudeAuth()
    assert auth.method == "claude_code"
    assert auth.sdk_env() == {}
    assert auth.public() == {
        "method": "claude_code",
        "label": "Claude Code (automatic)",
        "note": None,
    }


def test_automatic_mode_reports_inherited_credentials(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-secret")
    public = ClaudeAuth().public()
    assert "ANTHROPIC_API_KEY" in public["note"]
    assert "api-secret" not in repr(public)
    assert ClaudeAuth().sdk_env() == {}

    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    assert "CLAUDE_CODE_USE_BEDROCK" in ClaudeAuth().public()["note"]


def test_manual_credentials_replace_only_environment_credentials(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")

    subscription = ClaudeAuth.interactive("subscription", " oauth-secret ")
    env = subscription.sdk_env()
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-secret"
    assert env["ANTHROPIC_API_KEY"] == ""
    assert env["ANTHROPIC_AUTH_TOKEN"] == ""
    assert "ANTHROPIC_BASE_URL" not in env
    assert not any(name.startswith("CLAUDE_CODE_USE_") for name in env)
    assert subscription.public() == {
        "method": "subscription",
        "label": "Claude subscription",
        "note": None,
    }
    assert "oauth-secret" not in repr(subscription)

    api_key = ClaudeAuth.interactive("api_key", "api-secret")
    env = api_key.sdk_env()
    assert env["ANTHROPIC_API_KEY"] == "api-secret"
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == ""
    assert api_key.public()["label"] == "Claude API key"


def test_returning_to_automatic_is_the_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "inherited-key")
    assert ClaudeAuth.interactive("claude_code", None) == ClaudeAuth()
    assert ClaudeAuth.interactive("claude_code", "  ").sdk_env() == {}


def test_interactive_validation_messages_never_echo_the_secret():
    with pytest.raises(ValueError, match="API key"):
        ClaudeAuth.interactive("api_key", "   ")
    with pytest.raises(ValueError, match="whitespace") as info:
        ClaudeAuth.interactive("subscription", "top secret")
    assert "top secret" not in str(info.value)
    with pytest.raises(ValueError, match="does not take"):
        ClaudeAuth.interactive("claude_code", "stray")
    with pytest.raises(ValueError):
        ClaudeAuth.interactive("other", "x")
