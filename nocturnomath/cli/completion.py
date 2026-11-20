"""Complete slash commands and their arguments in the terminal client."""

import logging
from collections.abc import Callable, Iterable, Iterator

from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
    PathCompleter,
)
from prompt_toolkit.document import Document

from ..auth import PROVIDER_METHODS, PROVIDER_TITLES
from ..runtime import Runtime

logger = logging.getLogger("nocturnomath.cli")


class CommandCompleter(Completer):
    """Suggest the slash commands, then the arguments each of them accepts."""

    def __init__(
        self,
        runtime: Runtime,
        commands: Iterable[tuple[str, str, str]],
        *,
        pending_model: Callable[[], str | None] | None = None,
    ):
        self.runtime = runtime
        self.commands = list(commands)
        self._pending_model = pending_model
        self.files = PathCompleter(expanduser=True)
        self.directories = PathCompleter(only_directories=True, expanduser=True)

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/") or "\n" in text:
            return
        name, separator, argument = text.partition(" ")
        if not separator:
            yield from self.command_names(name)
            return
        single_path_commands = {"/docs", "/env", "/workspace"}
        word = argument if name in single_path_commands else argument.rsplit(" ", 1)[-1]
        for value, meta in self.arguments(name, argument):
            if value.startswith(word):
                yield Completion(
                    value, start_position=-len(word), display_meta=meta or ""
                )
        delegate = {"/env": self.files, "/workspace": self.directories}.get(name)
        if delegate is not None:
            yield from delegate.get_completions(
                Document(argument, len(argument)), complete_event
            )

    def command_names(self, prefix: str) -> Iterator[Completion]:
        for name, _, description in self.commands:
            if name.startswith(prefix):
                yield Completion(
                    name, start_position=-len(prefix), display_meta=description
                )

    def arguments(self, name: str, argument: str = "") -> list[tuple[str, str]]:
        try:
            return self._arguments(name, argument)
        except Exception as exc:
            logger.debug(f"No completions for {name}: {exc}")
            return []

    def _arguments(self, name: str, argument: str = "") -> list[tuple[str, str]]:
        if name == "/model":
            if " " in argument.lstrip():
                return []
            items = []
            for model in self.runtime.models:
                provider = self.runtime.model_providers.get(model)
                label = self.runtime.model_labels.get(model, "")
                if provider:
                    title = PROVIDER_TITLES.get(provider, provider)
                    meta = f"{title} · {label}" if label and label != model else title
                else:
                    meta = label
                items.append((model, meta))
            return items
        if name == "/effort":
            if " " in argument.lstrip():
                return []
            current = (self._pending_model() if self._pending_model else None) or (
                self.runtime.session.model
                if self.runtime.session
                else self.runtime.model
            )
            return [
                (level, f"reasoning effort for {current}")
                for level in self.runtime.model_efforts.get(current or "", [])
            ]
        if name == "/auth":
            head, space, rest = argument.lstrip().partition(" ")
            if not space:
                return [
                    ("status", "show current authentication across providers"),
                    ("anthropic", "configure Anthropic / Claude credentials"),
                    ("openai", "configure OpenAI / Codex credentials"),
                    ("google", "configure Google / Gemini / Antigravity credentials"),
                ]
            provider = head.lower()
            if provider not in PROVIDER_METHODS or " " in rest:
                return []
            return [
                ("api-key" if key == "api_key" else key, label)
                for key, label in PROVIDER_METHODS[provider].items()
            ]
        if name == "/env":
            return [("managed", "the environment prepared in the workspace")]
        session = self.runtime.session
        if session is None:
            return []
        if name == "/resume":
            return [
                (record["id"], record["title"]) for record in session.list_sessions()
            ] + [("--kernel", "replay the stored probes")]
        if name == "/docs":
            return [
                (document["name"], "included" if document["included"] else "")
                for document in session.list_documents()
            ]
        return []
