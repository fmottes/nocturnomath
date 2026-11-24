"""Numbered scientific records stored directly in two Markdown files."""

import os
import re
import threading
import uuid
from pathlib import Path

ENTRY = re.compile(r"^## ([ET]\d+)\n", re.MULTILINE)
CITATION = re.compile(r"\[([ET]\d+)\](?!\()")
_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(directory: Path) -> threading.RLock:
    """Return the process-wide lock for one shared scientific record."""
    directory = directory.resolve()
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(directory, threading.RLock())


def write_text(path: Path, text: str):
    """Publish a whole file without exposing a partially written record."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ResearchNotes:
    def __init__(self, directory: Path):
        self._lock = _lock_for(directory)
        self.evidence_path = directory / "evidence.md"
        self.thoughts_path = directory / "thoughts.md"
        for path, title in (
            (self.evidence_path, "Evidence"),
            (self.thoughts_path, "Thoughts"),
        ):
            if not path.exists():
                write_text(path, f"# {title}\n\n")

    def path_for(self, entry_id: str) -> Path:
        if not re.fullmatch(r"[ET]\d{3,}", entry_id):
            raise ValueError("Use an entry ID such as E001 or T001.")
        return self.evidence_path if entry_id.startswith("E") else self.thoughts_path

    @staticmethod
    def _parse_entries(text: str) -> dict[str, str]:
        pieces = ENTRY.split(text)
        return dict(zip(pieces[1::2], pieces[2::2]))

    @staticmethod
    def _next_entry_id(entries: dict[str, str], prefix: str) -> str:
        numbers = [int(key[1:]) for key in entries]
        return f"{prefix}{max(numbers, default=0) + 1:03d}"

    def entries(self, path: Path) -> dict[str, str]:
        return self._parse_entries(path.read_text(encoding="utf-8"))

    def entry(self, entry_id: str) -> str:
        entries = self.entries(self.path_for(entry_id))
        if entry_id not in entries:
            raise ValueError(f"Unknown entry: {entry_id}.")
        return entries[entry_id]

    def next_id(self, path: Path, prefix: str) -> str:
        return self._next_entry_id(self.entries(path), prefix)

    @staticmethod
    def block(entry_id: str, body: str) -> str:
        return f'## {entry_id}\n\n<a id="{entry_id}"></a>\n\n{body}\n\n'

    @staticmethod
    def strike(body: str) -> str:
        # Preserve the anchor, wording, equations, and links verbatim.
        anchor, content = body.split("</a>", 1)
        return anchor + "</a>\n\n<del>\n\n" + content.strip() + "\n\n</del>\n\n"

    def add_evidence(self, text: str, sources: list[str]) -> str:
        with self._lock:
            text = text.strip()
            if not text or len(text) > 250 or "\n" in text or "\r" in text:
                raise ValueError(
                    "Evidence must be one non-empty line, at most 250 characters excluding sources."
                )
            if re.search(r"\b[ET]\d{3,}\b", text) or re.search(
                r"\[[^\]]*\]\(|<[A-Za-z/!]|~~", text
            ):
                raise ValueError(
                    "Evidence must be plain factual text without citations, HTML, or strike markup."
                )
            if not sources:
                raise ValueError(
                    "Evidence needs at least one saved probe output or plot."
                )
            document = self.evidence_path.read_text(encoding="utf-8")
            entry_id = self._next_entry_id(self._parse_entries(document), "E")
            body = text + "\n\nSources: " + " · ".join(sources)
            write_text(self.evidence_path, document + self.block(entry_id, body))
            return entry_id

    def add_thought(self, text: str, replaces: list[str]) -> str:
        with self._lock:
            return self._add_thought(text, replaces)

    def _add_thought(self, text: str, replaces: list[str]) -> str:
        text = text.strip()
        if not text or ENTRY.search(text) or re.search(r"</?(?:del|a)\b|~~", text):
            raise ValueError(
                "Thoughts need text; entry headers, anchors, and strike markup are managed automatically."
            )
        references = set(CITATION.findall(text))
        if not references:
            raise ValueError(
                "Cite supporting evidence or thoughts using [E001] or [T001]."
            )
        document = self.thoughts_path.read_text(encoding="utf-8")
        thoughts_entries = self._parse_entries(document)
        evidence_entries: dict[str, str] | None = None
        for reference in references:
            path = self.path_for(reference)
            if path == self.evidence_path:
                if evidence_entries is None:
                    evidence_entries = self.entries(self.evidence_path)
                pool = evidence_entries
            else:
                pool = thoughts_entries
            if reference not in pool:
                raise ValueError(f"Unknown entry: {reference}.")
        # Validate every replacement before changing either entry.
        replaces = list(dict.fromkeys(replaces))
        for old_id in replaces:
            if not old_id.startswith("T"):
                raise ValueError(
                    "A replacement thought can only supersede thoughts. Use strike_evidence for evidence."
                )
            self.path_for(old_id)
            if old_id not in thoughts_entries:
                raise ValueError(f"Unknown entry: {old_id}.")
            if "<del>" in thoughts_entries[old_id]:
                raise ValueError(f"{old_id} is already struck.")
        entry_id = self._next_entry_id(thoughts_entries, "T")
        text = CITATION.sub(lambda match: self.link(match[1]), text)
        if replaces:
            text += "\n\nReplaces: " + " · ".join(self.link(key) for key in replaces)
        for old_id in replaces:
            old_body = thoughts_entries[old_id]
            struck = self.strike(old_body) + f"Replaced by {self.link(entry_id)}.\n\n"
            document = document.replace(
                f"## {old_id}\n{old_body}", f"## {old_id}\n{struck}", 1
            )
        write_text(self.thoughts_path, document + self.block(entry_id, text))
        return entry_id

    def strike_evidence(self, entry_id: str):
        with self._lock:
            if not entry_id.startswith("E"):
                raise ValueError(
                    "Only evidence can be struck without a replacement thought."
                )
            self.path_for(entry_id)
            document = self.evidence_path.read_text(encoding="utf-8")
            entries = self._parse_entries(document)
            if entry_id not in entries:
                raise ValueError(f"Unknown entry: {entry_id}.")
            body = entries[entry_id]
            if "<del>" in body:
                raise ValueError(f"{entry_id} is already struck.")
            write_text(
                self.evidence_path,
                document.replace(
                    f"## {entry_id}\n{body}", f"## {entry_id}\n{self.strike(body)}", 1
                ),
            )

    @staticmethod
    def link(entry_id: str) -> str:
        filename = "evidence.md" if entry_id.startswith("E") else "thoughts.md"
        return f"[{entry_id}]({filename}#{entry_id})"

    def read(self) -> str:
        return self.evidence_path.read_text() + "\n" + self.thoughts_path.read_text()
