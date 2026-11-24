"""Structured append-only JSONL scientific record with rendered Markdown views."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENTRY = re.compile(r"^## ([ET]\d+)\r?\n", re.MULTILINE)
_EVIDENCE_ENTRY = re.compile(r"^## (E\d{3,})\n", re.MULTILINE)
_THOUGHT_ENTRY = re.compile(r"^## (T\d{3,})\n", re.MULTILINE)
CITATION = re.compile(r"\[([ET]\d+)\](?!\()")
_RENDERED_LINK = re.compile(r"\[([ET]\d{3,})\]\((?:evidence|thoughts)\.md#\1\)")
_DEL_BLOCK = re.compile(r"<del>\s*(.*?)\s*</del>(.*)", re.DOTALL)
_REPLACED_BY = re.compile(r"Replaced by \[([T]\d{3,})\]")
_THOUGHT_ID = re.compile(r"\bT\d{3,}\b")

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


def _entry_num(entry_id: str) -> int:
    return int(entry_id[1:])


@dataclass
class EvidenceRecord:
    id: str
    text: str
    sources: list[str]
    struck: bool = False
    t: float | None = None
    session: str | None = None

    def copy(self) -> EvidenceRecord:
        return EvidenceRecord(
            self.id, self.text, list(self.sources), self.struck, self.t, self.session
        )


@dataclass
class ThoughtRecord:
    id: str
    text: str
    citations: list[str] = field(default_factory=list)
    replaces: list[str] = field(default_factory=list)
    struck: bool = False
    replaced_by: str | None = None
    t: float | None = None
    session: str | None = None

    def copy(self) -> ThoughtRecord:
        return ThoughtRecord(
            self.id,
            self.text,
            list(self.citations),
            list(self.replaces),
            self.struck,
            self.replaced_by,
            self.t,
            self.session,
        )


def _apply_event(
    event: dict[str, Any],
    evidence: dict[str, EvidenceRecord],
    thoughts: dict[str, ThoughtRecord],
) -> None:
    op = event.get("op")
    if op == "add_evidence":
        rec = EvidenceRecord(
            id=event["id"],
            text=event["text"],
            sources=list(event.get("sources") or []),
            t=event.get("t"),
            session=event.get("session"),
        )
        evidence[rec.id] = rec
    elif op == "strike_evidence":
        if target := evidence.get(event["id"]):
            target.struck = True
    elif op == "add_thought":
        replaces = list(event.get("replaces") or [])
        rec = ThoughtRecord(
            id=event["id"],
            text=event["text"],
            citations=list(event.get("citations") or []),
            replaces=replaces,
            t=event.get("t"),
            session=event.get("session"),
        )
        for old_id in replaces:
            if old := thoughts.get(old_id):
                old.struck = True
                old.replaced_by = rec.id
        thoughts[rec.id] = rec
    elif op == "strike_thought" and (target := thoughts.get(event["id"])):
        target.struck = True
        if event.get("replaced_by"):
            target.replaced_by = event["replaced_by"]


def _split_legacy_blocks(path: Path, pattern: re.Pattern[str]) -> list[tuple[str, str]]:
    if not path.is_file():
        return []
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    pieces = pattern.split(raw)
    return sorted(zip(pieces[1::2], pieces[2::2]), key=lambda item: _entry_num(item[0]))


def _split_trailing(content: str, label: str) -> tuple[str, str]:
    for sep in (f"\n\n{label}: ", f"\n{label}: "):
        if sep in content:
            head, tail = content.rsplit(sep, 1)
            return head.strip(), tail.strip()
    return content, ""


def _unwrap_legacy_body(raw_body: str) -> tuple[str, bool, str]:
    content = (
        raw_body.split("</a>", 1)[1].strip() if "</a>" in raw_body else raw_body.strip()
    )
    if match := _DEL_BLOCK.match(content):
        return match.group(1).strip(), True, match.group(2).strip()
    return content, False, ""


def _parse_legacy_markdown(
    evidence_path: Path, thoughts_path: Path
) -> list[dict[str, Any]]:
    """Convert legacy Markdown kb files into canonical record.jsonl events."""
    events: list[dict[str, Any]] = []
    struck_evidence: list[str] = []

    for entry_id, raw_body in _split_legacy_blocks(evidence_path, _EVIDENCE_ENTRY):
        content, struck, _ = _unwrap_legacy_body(raw_body)
        content, raw_sources = _split_trailing(content, "Sources")
        sources = [s.strip() for s in raw_sources.split(" · ") if s.strip()]
        events.append(
            {
                "op": "add_evidence",
                "id": entry_id,
                "t": None,
                "session": None,
                "text": content,
                "sources": sources,
            }
        )
        if struck:
            struck_evidence.append(entry_id)

    parsed_thoughts: list[dict[str, Any]] = []
    replaced_links: dict[str, list[str]] = {}
    orphan_struck_thoughts: list[tuple[str, str | None]] = []
    raw_thoughts = _split_legacy_blocks(thoughts_path, _THOUGHT_ENTRY)
    thought_ids = {entry_id for entry_id, _ in raw_thoughts}

    for entry_id, raw_body in raw_thoughts:
        content, struck, tail = _unwrap_legacy_body(raw_body)
        rep_match = _REPLACED_BY.search(tail) if struck else None
        replaced_by = rep_match.group(1) if rep_match else None
        content, raw_replaces = _split_trailing(content, "Replaces")
        replaces = list(dict.fromkeys(_THOUGHT_ID.findall(raw_replaces)))
        normalized = _RENDERED_LINK.sub(r"[\1]", content)
        parsed_thoughts.append(
            {
                "id": entry_id,
                "text": normalized,
                "citations": list(dict.fromkeys(CITATION.findall(normalized))),
                "replaces": replaces,
            }
        )
        if struck:
            if (
                replaced_by
                and replaced_by in thought_ids
                and _entry_num(replaced_by) > _entry_num(entry_id)
            ):
                replaced_links.setdefault(replaced_by, []).append(entry_id)
            else:
                orphan_struck_thoughts.append((entry_id, replaced_by))

    for thought in parsed_thoughts:
        replaces = list(
            dict.fromkeys(thought["replaces"] + replaced_links.get(thought["id"], []))
        )
        events.append(
            {
                "op": "add_thought",
                "id": thought["id"],
                "t": None,
                "session": None,
                "text": thought["text"],
                "citations": thought["citations"],
                "replaces": replaces,
            }
        )

    for entry_id in struck_evidence:
        events.append(
            {"op": "strike_evidence", "id": entry_id, "t": None, "session": None}
        )
    for entry_id, replaced_by in orphan_struck_thoughts:
        event: dict[str, Any] = {
            "op": "strike_thought",
            "id": entry_id,
            "t": None,
            "session": None,
        }
        if replaced_by:
            event["replaced_by"] = replaced_by
        events.append(event)

    return events


class ResearchNotes:
    """Append-only JSONL scientific record with rendered Markdown views."""

    def __init__(self, directory: Path):
        self._directory = Path(directory)
        self._lock = _lock_for(self._directory)
        self.record_path = self._directory / "record.jsonl"
        self.evidence_path = self._directory / "evidence.md"
        self.thoughts_path = self._directory / "thoughts.md"
        self._cache: (
            tuple[int, int, dict[str, EvidenceRecord], dict[str, ThoughtRecord]] | None
        ) = None
        self.migrated = False
        with self._lock:
            self._directory.mkdir(parents=True, exist_ok=True)
            if not self.record_path.exists():
                self.migrated = self._initialize_from_legacy_or_empty()
            else:
                self._sync_markdown()

    def _initialize_from_legacy_or_empty(self) -> bool:
        events = _parse_legacy_markdown(self.evidence_path, self.thoughts_path)
        payload = "".join(
            json.dumps(event, ensure_ascii=False) + "\n" for event in events
        )
        write_text(self.record_path, payload)
        self._cache = None
        self._sync_markdown()
        return bool(events)

    def _sync_markdown(self) -> None:
        evidence, thoughts = self._load()
        for path, rendered in (
            (self.evidence_path, self._render_md("Evidence", evidence)),
            (self.thoughts_path, self._render_md("Thoughts", thoughts)),
        ):
            if not path.exists() or path.read_text(encoding="utf-8") != rendered:
                write_text(path, rendered)

    def _load(self) -> tuple[dict[str, EvidenceRecord], dict[str, ThoughtRecord]]:
        stat = self.record_path.stat()
        cached = self._cache
        if (
            cached is not None
            and cached[0] == stat.st_mtime_ns
            and cached[1] == stat.st_size
        ):
            return cached[2], cached[3]

        evidence: dict[str, EvidenceRecord] = {}
        thoughts: dict[str, ThoughtRecord] = {}
        with self.record_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if stripped := line.strip():
                    _apply_event(json.loads(stripped), evidence, thoughts)
        self._cache = (stat.st_mtime_ns, stat.st_size, evidence, thoughts)
        return evidence, thoughts

    def _append_and_apply(
        self,
        event: dict[str, Any],
        evidence: dict[str, EvidenceRecord],
        thoughts: dict[str, ThoughtRecord],
    ) -> None:
        needs_newline = False
        if self.record_path.stat().st_size > 0:
            with self.record_path.open("rb") as probe:
                probe.seek(-1, os.SEEK_END)
                needs_newline = probe.read(1) != b"\n"
        line = (
            ("\n" if needs_newline else "")
            + json.dumps(event, ensure_ascii=False)
            + "\n"
        )
        self._cache = None
        with self.record_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
        _apply_event(event, evidence, thoughts)
        stat = self.record_path.stat()
        self._cache = (stat.st_mtime_ns, stat.st_size, evidence, thoughts)

    def _is_evidence_path(self, path: Path) -> bool:
        return Path(path).resolve() == self.evidence_path.resolve()

    def path_for(self, entry_id: str) -> Path:
        if not re.fullmatch(r"[ET]\d{3,}", entry_id):
            raise ValueError("Use an entry ID such as E001 or T001.")
        return self.evidence_path if entry_id.startswith("E") else self.thoughts_path

    @staticmethod
    def _next_entry_id(entries: dict[str, Any], prefix: str) -> str:
        return f"{prefix}{max((_entry_num(k) for k in entries), default=0) + 1:03d}"

    @classmethod
    def _render_body(cls, rec: EvidenceRecord | ThoughtRecord) -> str:
        if isinstance(rec, EvidenceRecord):
            inner = rec.text + (
                "\n\nSources: " + " · ".join(rec.sources) if rec.sources else ""
            )
            tail = ""
        else:
            inner = CITATION.sub(lambda match: cls.link(match[1]), rec.text)
            if rec.replaces:
                inner += "\n\nReplaces: " + " · ".join(
                    cls.link(key) for key in rec.replaces
                )
            tail = (
                f"Replaced by {cls.link(rec.replaced_by)}.\n\n"
                if rec.struck and rec.replaced_by
                else ""
            )
        if rec.struck:
            return f'\n<a id="{rec.id}"></a>\n\n<del>\n\n{inner}\n\n</del>\n\n{tail}'
        return f'\n<a id="{rec.id}"></a>\n\n{inner}\n\n'

    @classmethod
    def _render_md(
        cls,
        title: str,
        pool: dict[str, EvidenceRecord] | dict[str, ThoughtRecord],
        *,
        active_only: bool = False,
    ) -> str:
        return f"# {title}\n\n" + "".join(
            f"## {rec.id}\n{cls._render_body(rec)}"
            for rec in pool.values()
            if not (active_only and rec.struck)
        )

    def records(
        self,
    ) -> tuple[dict[str, EvidenceRecord], dict[str, ThoughtRecord]]:
        """Return defensive copies of structured evidence and thought records."""
        with self._lock:
            evidence, thoughts = self._load()
            return (
                {k: v.copy() for k, v in evidence.items()},
                {k: v.copy() for k, v in thoughts.items()},
            )

    def dependent_thoughts(self, entry_id: str) -> list[str]:
        """Return active thought IDs that cite ``entry_id``."""
        with self._lock:
            _, thoughts = self._load()
            return [
                rec.id
                for rec in thoughts.values()
                if not rec.struck and entry_id in rec.citations
            ]

    def entries(self, path: Path) -> dict[str, str]:
        with self._lock:
            evidence, thoughts = self._load()
            pool = evidence if self._is_evidence_path(path) else thoughts
            return {rec.id: self._render_body(rec) for rec in pool.values()}

    def entry(self, entry_id: str) -> str:
        entries = self.entries(self.path_for(entry_id))
        if entry_id not in entries:
            raise ValueError(f"Unknown entry: {entry_id}.")
        return entries[entry_id]

    def next_id(self, path: Path, prefix: str) -> str:
        with self._lock:
            evidence, thoughts = self._load()
            pool = evidence if self._is_evidence_path(path) else thoughts
            return self._next_entry_id(pool, prefix)

    def add_evidence(
        self, text: str, sources: list[str], *, session: str | None = None
    ) -> str:
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
            if not sources or any(
                not isinstance(s, str) or not s.strip() or "\n" in s or "\r" in s
                for s in sources
            ):
                raise ValueError(
                    "Evidence needs at least one saved probe output or plot."
                )
            evidence, thoughts = self._load()
            entry_id = self._next_entry_id(evidence, "E")
            self._append_and_apply(
                {
                    "op": "add_evidence",
                    "id": entry_id,
                    "t": time.time(),
                    "session": session,
                    "text": text,
                    "sources": [s.strip() for s in sources],
                },
                evidence,
                thoughts,
            )
            write_text(self.evidence_path, self._render_md("Evidence", evidence))
            return entry_id

    def add_thought(
        self, text: str, replaces: list[str], *, session: str | None = None
    ) -> str:
        with self._lock:
            text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
            if not text or ENTRY.search(text) or re.search(r"</?(?:del|a)\b|~~", text):
                raise ValueError(
                    "Thoughts need text; entry headers, anchors, and strike markup are managed automatically."
                )
            references = list(dict.fromkeys(CITATION.findall(text)))
            if not references:
                raise ValueError(
                    "Cite supporting evidence or thoughts using [E001] or [T001]."
                )
            evidence, thoughts = self._load()
            for reference in references:
                pool = (
                    evidence
                    if self.path_for(reference) == self.evidence_path
                    else thoughts
                )
                if reference not in pool:
                    raise ValueError(f"Unknown entry: {reference}.")
            replaces = list(dict.fromkeys(replaces))
            for old_id in replaces:
                if not old_id.startswith("T"):
                    raise ValueError(
                        "A replacement thought can only supersede thoughts. Use strike_evidence for evidence."
                    )
                self.path_for(old_id)
                if old_id not in thoughts:
                    raise ValueError(f"Unknown entry: {old_id}.")
                if thoughts[old_id].struck:
                    raise ValueError(f"{old_id} is already struck.")
            entry_id = self._next_entry_id(thoughts, "T")
            self._append_and_apply(
                {
                    "op": "add_thought",
                    "id": entry_id,
                    "t": time.time(),
                    "session": session,
                    "text": text,
                    "citations": references,
                    "replaces": replaces,
                },
                evidence,
                thoughts,
            )
            write_text(self.thoughts_path, self._render_md("Thoughts", thoughts))
            return entry_id

    def strike_evidence(self, entry_id: str, *, session: str | None = None):
        with self._lock:
            if not entry_id.startswith("E"):
                raise ValueError(
                    "Only evidence can be struck without a replacement thought."
                )
            self.path_for(entry_id)
            evidence, thoughts = self._load()
            if entry_id not in evidence:
                raise ValueError(f"Unknown entry: {entry_id}.")
            if evidence[entry_id].struck:
                raise ValueError(f"{entry_id} is already struck.")
            self._append_and_apply(
                {
                    "op": "strike_evidence",
                    "id": entry_id,
                    "t": time.time(),
                    "session": session,
                },
                evidence,
                thoughts,
            )
            write_text(self.evidence_path, self._render_md("Evidence", evidence))

    @staticmethod
    def link(entry_id: str) -> str:
        filename = "evidence.md" if entry_id.startswith("E") else "thoughts.md"
        return f"[{entry_id}]({filename}#{entry_id})"

    def read(self, *, active_only: bool = False) -> str:
        with self._lock:
            evidence, thoughts = self._load()
            return (
                self._render_md("Evidence", evidence, active_only=active_only)
                + "\n"
                + self._render_md("Thoughts", thoughts, active_only=active_only)
            )


def migrate_workspace_notes(workspace_or_kb: Path) -> bool:
    """Migrate an existing workspace (or kb directory) from Markdown-only notes to record.jsonl."""
    path = Path(workspace_or_kb)
    kb_dir = (
        path / ".nocturnomath" / "kb" if (path / ".nocturnomath").is_dir() else path
    )
    if not kb_dir.is_dir():
        return False
    return ResearchNotes(kb_dir).migrated
