"""Complete slash commands and their arguments in the terminal client."""

import logging
from collections.abc import Iterable, Iterator

from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
    PathCompleter,
)
from prompt_toolkit.document import Document

from ..runtime import Runtime

logger = logging.getLogger("nocturnomath.cli")


class CommandCompleter(Completer):
    """Suggest the slash commands, then the arguments each of them accepts."""

    def __init__(self, runtime: Runtime, commands: Iterable[tuple[str, str, str]]):
        self.runtime = runtime
        self.commands = list(commands)
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
        for value, meta in self.arguments(name):
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

    def arguments(self, name: str) -> list[tuple[str, str]]:
        try:
            return self._arguments(name)
        except Exception as exc:
            logger.debug(f"No completions for {name}: {exc}")
            return []

    def _arguments(self, name: str) -> list[tuple[str, str]]:
        if name == "/model":
            return [
                (model, self.runtime.model_labels.get(model, ""))
                for model in self.runtime.models
            ]
        if name == "/context":
            return [
                ("on", "carry chat context"),
                ("off", "start every message fresh"),
            ]
        if name == "/auth":
            return [
                ("status", "show the current authentication method"),
                ("subscription", "enter a Claude subscription token securely"),
                ("api-key", "enter a Claude API key securely"),
                ("claude-code", "use the existing Claude Code login"),
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
                (document["path"], "") for document in session.list_markdown_files()
            ]
        return []
