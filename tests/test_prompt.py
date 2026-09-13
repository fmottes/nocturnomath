from nocturnomath.prompt import (
    DEFAULT_EXPLORATION_PROMPT,
    RUNTIME_PROMPT,
    build_system_prompt,
)


def test_default_system_prompt_combines_behavior_and_runtime_contract():
    prompt = build_system_prompt()

    assert prompt == (
        f"{DEFAULT_EXPLORATION_PROMPT.strip()}\n\n{RUNTIME_PROMPT.strip()}\n"
    )


def test_custom_behavior_keeps_runtime_contract():
    prompt = build_system_prompt("Explore cautiously.")

    assert prompt.startswith("Explore cautiously.\n\n")
    assert "# Exploration mode" not in prompt
    assert ".nocturnomath/kb/evidence.md" in prompt
    assert "KaTeX" in prompt
    assert r"\qquad \text{(B1)}" in prompt
