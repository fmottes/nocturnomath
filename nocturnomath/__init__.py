"""Hypothesis-driven scientific exploration with a persistent Python kernel."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .session import ExplorationSession

__all__ = ["ExplorationSession"]


def __getattr__(name: str) -> Any:
    if name == "ExplorationSession":
        from .session import ExplorationSession

        return ExplorationSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
