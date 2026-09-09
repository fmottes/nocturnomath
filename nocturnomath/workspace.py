"""Workspace files, notes, plots, and persisted chat history."""

import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from .notes import ResearchNotes, write_text

logger = logging.getLogger("nocturnomath")


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " […]"


class Workspace:
    """Own all filesystem state belonging to one exploration directory."""

    def __init__(self, path: Path | str):
        self.set_path(path)

    def set_path(self, path: Path | str):
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.nocturnomath_path = self.path / ".nocturnomath"
        self.notes_dir_path = self.nocturnomath_path / "notes"
        self.sessions_path = self.nocturnomath_path / "sessions"
        self.scratch_path = self.nocturnomath_path / "scratch"
        for directory in (
            self.notes_dir_path,
            self.sessions_path,
            self.scratch_path,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.notes = ResearchNotes(self.notes_dir_path)
        self.start_new_transcript()

    def start_new_transcript(self):
        numbers = [
            int(path.name[1:])
            for path in self.sessions_path.iterdir()
            if re.fullmatch(r"S\d+", path.name)
        ]
        number = max(numbers, default=0) + 1
        self.transcript_path = (
            self.sessions_path / f"S{number:03d}" / "transcript.jsonl"
        )
        self._session_pending = True

    def ensure_session(self):
        """Allocate the prepared session folder when the chat first has activity."""
        if not self._session_pending:
            return
        number = int(self.transcript_path.parent.name[1:])
        while True:
            session_path = self.sessions_path / f"S{number:03d}"
            try:
                session_path.mkdir()
                break
            except FileExistsError:
                number += 1
        self.transcript_path = session_path / "transcript.jsonl"
        self.probes_path.mkdir()
        self._session_pending = False

    @property
    def session_pending(self) -> bool:
        return self._session_pending

    def use_transcript(self, path: Path):
        """Select an existing transcript rather than a prepared new session."""
        self.transcript_path = path
        self._session_pending = False

    def session_state(self) -> tuple[Path, bool]:
        return self.transcript_path, self._session_pending

    def restore_session_state(self, state: tuple[Path, bool]):
        self.transcript_path, self._session_pending = state

    @property
    def session_id(self) -> str:
        return self.transcript_path.parent.name

    @property
    def probes_path(self) -> Path:
        return self.transcript_path.parent / "probes"

    def log_transcript(
        self, kind: str, *, transcript_path: Path | None = None, **fields
    ):
        if transcript_path is None:
            self.ensure_session()
        record = {
            "t": time.strftime("%H:%M:%S"),
            "timestamp": time.time(),
            "kind": kind,
            **fields,
        }
        try:
            with (transcript_path or self.transcript_path).open("a") as transcript:
                transcript.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning(f"Failed to log transcript: {exc}")

    def opening_notes(self) -> str:
        return (
            "Scientific record (struck entries are invalid; thoughts are interpretations, "
            "not observations). Relative links are based in .nocturnomath/notes/.\n\n"
            + self.notes.read()
            + "\n---\n\n"
        )

    def start_probe(self, code: str, expected: str, model: str, **context) -> Path:
        self.ensure_session()
        numbers = [
            int(path.name[1:])
            for path in self.probes_path.iterdir()
            if re.fullmatch(r"P\d+", path.name)
        ]
        number = max(numbers, default=0) + 1
        while True:
            path = self.probes_path / f"P{number:03d}"
            try:
                path.mkdir()
                break
            except FileExistsError:
                number += 1
        write_text(path / "code.py", code)
        write_text(
            path / "probe.json",
            json.dumps(
                {
                    "id": path.name,
                    "expected": expected,
                    **context,
                    "model": model,
                    "session": self.session_id,
                    "started": time.time(),
                    "status": "started",
                },
                indent=2,
            )
            + "\n",
        )
        return path

    @staticmethod
    def finish_probe(
        path: Path, output: str, images: list[bytes], status: str, note: str | None
    ):
        # The path is captured before execution; changing UI state cannot redirect a result.
        write_text(path / "output.txt", output)
        for index, image in enumerate(images, 1):
            (path / f"plot-{index}.png").write_bytes(image)
        metadata = json.loads((path / "probe.json").read_text())
        metadata.update(status=status, finished=time.time(), execution_note=note)
        write_text(path / "probe.json", json.dumps(metadata, indent=2) + "\n")

    def evidence_sources(self, sources: list[str]) -> list[str]:
        """Accept only artifacts belonging to a persisted probe outcome."""
        links = []
        for source in dict.fromkeys(sources):
            if not re.fullmatch(
                r"S\d{3,}/P\d{3,}/(?:output\.txt|plot-[1-9]\d*\.png)", source
            ):
                raise ValueError(
                    "Sources must look like S001/P001/output.txt or S001/P001/plot-1.png."
                )
            session_id, probe_id, filename = source.split("/")
            artifact = self.sessions_path / session_id / "probes" / probe_id / filename
            if (
                not artifact.resolve().is_relative_to(self.sessions_path.resolve())
                or not artifact.is_file()
            ):
                raise ValueError(f"Source does not exist: {source}.")
            metadata = json.loads((artifact.parent / "probe.json").read_text())
            if metadata["status"] == "started":
                raise ValueError(
                    f"Probe {artifact.parent.name} has no recorded outcome yet."
                )
            relative = f"../sessions/{session_id}/probes/{probe_id}"
            links.append(
                f"[{source}]({relative}/{filename}) "
                f"([code]({relative}/code.py), [run]({relative}/probe.json))"
            )
        return links

    def list_markdown_files(self) -> list[dict[str, Any]]:
        files = []
        try:
            for path in self.path.glob("**/*.md"):
                if any(
                    ignored in path.parts
                    for ignored in (
                        ".git",
                        "node_modules",
                        ".venv",
                        "venv",
                        ".pytest_cache",
                    )
                ):
                    continue
                try:
                    relative = path.relative_to(self.path)
                    stat = path.stat()
                    files.append(
                        {
                            "name": str(relative),
                            "path": str(relative),
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                            "is_notes": path
                            in (self.notes.evidence_path, self.notes.thoughts_path),
                        }
                    )
                except Exception as exc:
                    logger.debug(f"Skipped {path}: {exc}")
        except Exception as exc:
            logger.warning(f"Error listing markdown files: {exc}")
        files.sort(key=lambda item: (not item["is_notes"], item["name"].lower()))
        return files

    def read_file(self, relative_path: str) -> dict[str, Any]:
        target = (self.path / relative_path).resolve()
        # Preserve the prototype's current path check exactly.
        if not str(target).startswith(str(self.path)):
            raise ValueError("File path outside workspace")
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File {relative_path} not found")
        content = target.read_text(encoding="utf-8", errors="replace")
        stat = target.stat()
        return {
            "path": relative_path,
            "content": content,
            "modified": stat.st_mtime,
            "size": stat.st_size,
        }

    def asset_file(self, relative_path: str) -> Path:
        """Return a workspace-local asset path for a rendered Markdown document."""
        target = (self.path / relative_path).resolve()
        if not target.is_relative_to(self.path):
            raise ValueError("Asset path outside workspace")
        if not target.is_file():
            raise FileNotFoundError(f"Asset {relative_path} not found")
        return target

    def list_plots(self) -> list[dict[str, Any]]:
        plots = []
        for path in self.sessions_path.glob("S*/probes/P*/plot-*.png"):
            stat = path.stat()
            relative = path.relative_to(self.path).as_posix()
            plots.append(
                {
                    "filename": f"{path.parents[2].name}/{path.parent.name}/{path.name}",
                    "session": path.parents[2].name,
                    "url": f"/api/asset?path={relative}",
                    "modified": stat.st_mtime,
                    "size": stat.st_size,
                }
            )
        plots.sort(key=lambda item: item["modified"], reverse=True)
        return plots

    def transcript_file(self, session_id: str) -> Path:
        if not re.fullmatch(r"S\d{3,}", session_id):
            raise ValueError("Use a session ID such as S001.")
        path = (self.sessions_path / session_id / "transcript.jsonl").resolve()
        if not path.is_relative_to(self.sessions_path.resolve()):
            raise ValueError("Transcript path outside workspace")
        if not path.is_file():
            raise FileNotFoundError(f"No transcript for {session_id}")
        return path

    @staticmethod
    def read_transcript(path: Path) -> list[dict[str, Any]]:
        records = []
        try:
            with path.open() as transcript:
                for line in transcript:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as exc:
            logger.warning(f"Failed to read transcript {path}: {exc}")
        return records

    @staticmethod
    def sdk_id(records: list[dict[str, Any]]) -> str | None:
        for record in reversed(records):
            if record.get("kind") == "meta" and record.get("sdk_session_id"):
                return record["sdk_session_id"]
        return None

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = []
        if not self.sessions_path.exists():
            return sessions
        for path in self.sessions_path.glob("S*/transcript.jsonl"):
            records = self.read_transcript(path)
            if not records:
                continue
            user_texts = [
                record.get("text", "")
                for record in records
                if record.get("kind") == "user"
            ]
            if not user_texts:
                continue
            title = user_texts[0]
            if len(title) > 120:
                title = title[:120] + "…"
            started = records[0].get("timestamp", path.stat().st_mtime)
            sessions.append(
                {
                    "id": path.parent.name,
                    "title": title,
                    "started": started,
                    "modified": path.stat().st_mtime,
                    "turns": len(user_texts),
                    "probes": sum(
                        1 for r in records if r.get("kind") == "probe_started"
                    ),
                    "notes": sum(
                        1 for r in records if r.get("kind") in ("evidence", "thought")
                    ),
                    "context_restorable": self.sdk_id(records) is not None,
                    "is_current": path == self.transcript_path,
                }
            )
        sessions.sort(key=lambda item: item["modified"], reverse=True)
        return sessions

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        return self.read_transcript(self.transcript_file(session_id))

    def export_notebook(
        self, session_id: str, python_version: str | None = None
    ) -> dict[str, Any]:
        """Build a notebook from one saved session without rerunning its probes."""
        records = self.load_session(session_id)
        if not any(record.get("kind") == "user" for record in records):
            raise ValueError("The current session has no messages to download.")

        outcomes = {
            record.get("probe_id"): record
            for record in records
            if record.get("kind") in ("run", "probe_failed")
        }
        cells: list[dict[str, Any]] = []
        turn = 0
        idea = 0
        probe = 0
        execution_count = 0
        pending_agent: list[str] = []

        def markdown(source: str, kind: str, **metadata):
            cells.append(
                {
                    "cell_type": "markdown",
                    "id": f"cell-{len(cells) + 1}",
                    "metadata": {"nocturnomath": {"kind": kind, **metadata}},
                    "source": source,
                }
            )

        def flush_agent():
            nonlocal idea
            if not pending_agent:
                return
            idea += 1
            body = "\n\n".join(pending_agent)
            pending_agent.clear()
            markdown(f"## Idea {turn}.{idea}\n\n{body}", "agent", turn=turn)

        for record in records:
            kind = record.get("kind")
            if kind == "agent":
                text = record.get("text")
                if isinstance(text, str) and text.strip():
                    pending_agent.append(text)
                continue
            if kind in ("meta", "kernel_started", "resumed"):
                continue
            flush_agent()

            if kind == "user":
                turn += 1
                idea = 0
                probe = 0
                markdown(
                    f"# User message {turn}\n\n{record.get('text', '')}",
                    "user",
                    turn=turn,
                    time=record.get("t"),
                )
            elif kind == "probe_started":
                probe += 1
                probe_id = record.get("probe_id", f"probe-{probe}")
                expected = record.get("expected", "")
                markdown(
                    f"### Probe {turn}.{probe}\n\n**Prediction:** {expected}",
                    "probe",
                    probe_id=probe_id,
                )
                outcome = outcomes.get(probe_id, {})
                outputs = self._notebook_outputs(outcome)
                execution_count += 1
                cells.append(
                    {
                        "cell_type": "code",
                        "execution_count": execution_count,
                        "id": f"cell-{len(cells) + 1}",
                        "metadata": {
                            "nocturnomath": {
                                "kind": "probe_code",
                                "probe_id": probe_id,
                                "status": outcome.get("status")
                                or (
                                    "completed"
                                    if outcome.get("kind") == "run"
                                    else "no recorded outcome"
                                ),
                            }
                        },
                        "outputs": outputs,
                        "source": record.get("code", ""),
                    }
                )
            elif kind == "verdict":
                markdown(
                    f"**Verdict:** {record.get('text', '')}",
                    "verdict",
                    probe_id=record.get("probe_id"),
                )
            elif kind in ("evidence", "thought", "evidence_struck"):
                label = kind.replace("_", " ").title()
                entry_id = record.get("entry_id", "")
                markdown(
                    f"> **{label} {entry_id}:** {record.get('text', '')}",
                    kind,
                    entry_id=entry_id,
                )

        flush_agent()
        language_info = {"name": "python"}
        if python_version:
            language_info["version"] = python_version
        return {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": language_info,
                "nocturnomath": {"session_id": session_id},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

    def _notebook_outputs(self, outcome: dict[str, Any]) -> list[dict[str, Any]]:
        outputs = []
        text = outcome.get("output") or outcome.get("text")
        if text:
            outputs.append(
                {
                    "name": "stderr"
                    if outcome.get("kind") == "probe_failed"
                    else "stdout",
                    "output_type": "stream",
                    "text": text,
                }
            )
        for relative in outcome.get("images", []):
            try:
                image = self.asset_file(relative).read_bytes()
            except (FileNotFoundError, ValueError):
                outputs.append(
                    {
                        "name": "stderr",
                        "output_type": "stream",
                        "text": f"[Recorded figure missing: {relative}]",
                    }
                )
                continue
            outputs.append(
                {
                    "data": {"image/png": base64.b64encode(image).decode("ascii")},
                    "metadata": {},
                    "output_type": "display_data",
                }
            )
        note = outcome.get("execution_note")
        if note:
            outputs.append({"name": "stderr", "output_type": "stream", "text": note})
        return outputs

    @staticmethod
    def recap(records: list[dict[str, Any]], max_chars: int = 8000) -> str:
        lines = []
        for record in records:
            kind = record.get("kind")
            if kind == "user":
                lines.append("Me: " + record.get("text", ""))
            elif kind == "agent":
                lines.append("You: " + _clip(record.get("text", ""), 800))
            elif kind == "run":
                lines.append(
                    "You ran a probe expecting: "
                    + record.get("expected", "")
                    + "\nIt printed: "
                    + _clip(record.get("output", ""), 400)
                )
            elif kind in ("evidence", "thought", "evidence_struck", "verdict"):
                lines.append(
                    f"Recorded {kind} {record.get('entry_id', '')}: "
                    + record.get("text", "")
                )
        body = "\n\n".join(lines)
        if len(body) > max_chars:
            body = "[earlier turns omitted]\n\n" + body[-max_chars:]
        return (
            "This continues an earlier exploration session. Its full context could not be "
            "restored, so here is the transcript so far. The kernel may no longer hold the "
            "state it had then; re-run whatever you need.\n\n" + body + "\n\n---\n\n"
        )
