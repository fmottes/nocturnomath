from nocturnomath.auth import ClaudeAuth


def test_environment_credentials_are_detected_without_exposing_them(monkeypatch):
    for name in (
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_ANTHROPIC_AWS",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-secret")
    auth = ClaudeAuth.from_environment()
    assert auth.method == "claude_code"
    assert auth.sdk_env() == {}
    assert "oauth-secret" not in repr(auth.public())

    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-secret")
    auth = ClaudeAuth.from_environment()
    assert auth.method == "claude_code"
    assert auth.sdk_env() == {}
    assert auth.credential is None


def test_higher_priority_environment_auth_is_preserved(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "gateway-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-secret")
    auth = ClaudeAuth.from_environment()
    assert auth.sdk_env() == {}
    assert auth.public()["label"] == "Claude Code (automatic)"
    assert "gateway-secret" not in repr(auth)


def test_explicit_credentials_clear_competing_inherited_authentication():
    subscription = ClaudeAuth.interactive("subscription", " oauth-secret ")
    subscription_env = subscription.sdk_env()
    assert subscription_env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-secret"
    assert subscription_env["ANTHROPIC_API_KEY"] == ""
    assert subscription_env["ANTHROPIC_AUTH_TOKEN"] == ""
    assert subscription.public()["label"] == "Claude subscription"
    assert "oauth-secret" not in repr(subscription.public())

    api_key = ClaudeAuth.interactive("api_key", "api-secret")
    api_env = api_key.sdk_env()
    assert api_env["ANTHROPIC_API_KEY"] == "api-secret"
    assert api_env["CLAUDE_CODE_OAUTH_TOKEN"] == ""


def test_returning_to_automatic_preserves_environment_credentials(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "inherited-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example")
    auth = ClaudeAuth.interactive("claude_code", None)
    assert auth.sdk_env() == {}
    assert auth == ClaudeAuth.from_environment()


def test_manual_credentials_do_not_use_inherited_gateway(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example")
    for method in ("subscription", "api_key"):
        auth = ClaudeAuth.interactive(method, "secret")
        assert auth.sdk_env()["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
