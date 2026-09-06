"""Workspace files, notes, plots, and persisted chat history."""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from .prompt import NOTES_TEMPLATE

logger = logging.getLogger("xprober")


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " […]"


class Workspace:
    """Own all filesystem state belonging to one exploration directory."""

    def __init__(self, path: Path | str):
        self.set_path(path)

    def set_path(self, path: Path | str):
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.xprober_path = self.path / "xprober"
        self.notes_dir_path = self.xprober_path / "notes"
        self.transcripts_path = self.xprober_path / "transcripts"
        self.figures_path = self.xprober_path / "figures"
        self.scratch_path = self.xprober_path / "scratch"
        for directory in (
            self.notes_dir_path,
            self.transcripts_path,
            self.figures_path,
            self.scratch_path,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.notes_path = self.notes_dir_path / "notes.md"
        if not self.notes_path.exists():
            self.notes_path.write_text(NOTES_TEMPLATE)
        self.start_new_transcript()

    def start_new_transcript(self):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.transcript_path = (
            self.transcripts_path
            / f"transcript-{stamp}-{uuid.uuid4().hex[:8]}.jsonl"
        )

    def log_transcript(self, kind: str, **fields):
        record = {"t": time.strftime("%H:%M:%S"), "kind": kind, **fields}
        try:
            with self.transcript_path.open("a") as transcript:
                transcript.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning(f"Failed to log transcript: {exc}")

    def opening_notes(self) -> str:
        if not self.notes_path.exists():
            return ""
        body = self.notes_path.read_text().strip()
        if not any(line.startswith("- ") for line in body.splitlines()):
            return ""
        return "What is already known about this system:\n\n" + body + "\n\n---\n\n"

    def add_note(self, kind: str, text: str) -> str:
        heading = "## Facts" if kind == "fact" else "## Dead ends"
        if not self.notes_path.exists():
            self.notes_path.write_text(NOTES_TEMPLATE)
        lines = self.notes_path.read_text().splitlines()
        if heading not in lines:
            lines.extend(["", heading, ""])
        start = lines.index(heading)
        end = start + 1
        while end < len(lines) and not lines[end].startswith("## "):
            end += 1
        while end > start + 1 and not lines[end - 1].strip():
            end -= 1
        lines.insert(end, "- " + text)
        self.notes_path.write_text("\n".join(lines) + "\n")
        return heading

    def save_plots(self, images: list[bytes]) -> list[str]:
        stamp = f"{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:8]}"
        filenames = []
        for index, image in enumerate(images):
            filename = f"plot-{stamp}-{index}.png"
            (self.figures_path / filename).write_bytes(image)
            filenames.append(filename)
        return filenames

    def list_markdown_files(self) -> list[dict[str, Any]]:
        files = []
        try:
            for path in self.path.glob("**/*.md"):
                if any(
                    ignored in path.parts
                    for ignored in (".git", "node_modules", ".venv", ".pytest_cache")
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
                            "is_notes": path == self.notes_path,
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

    def list_plots(self) -> list[dict[str, Any]]:
        plots = []
        if self.figures_path.exists():
            paths = sorted(
                self.figures_path.glob("plot-*.png"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for path in paths:
                stat = path.stat()
                plots.append(
                    {
                        "filename": path.name,
                        "url": f"/figures/{path.name}",
                        "modified": stat.st_mtime,
                        "size": stat.st_size,
                    }
                )
        return plots

    def transcript_file(self, session_id: str) -> Path:
        name = Path(session_id).name
        if not name.startswith("transcript-") or not name.endswith(".jsonl"):
            raise ValueError("Not a transcript file")
        path = (self.transcripts_path / name).resolve()
        if not str(path).startswith(str(self.transcripts_path.resolve())):
            raise ValueError("Transcript path outside workspace")
        if not path.is_file():
            raise FileNotFoundError(f"No transcript {name}")
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
        if not self.transcripts_path.exists():
            return sessions
        for path in self.transcripts_path.glob("transcript-*.jsonl"):
            records = self.read_transcript(path)
            if not records:
                continue
            user_texts = [
                record.get("text", "")
                for record in records
                if record.get("kind") == "user"
            ]
            title = user_texts[0] if user_texts else "(no prompt)"
            if len(title) > 120:
                title = title[:120] + "…"
            stamp = path.stem.removeprefix("transcript-")
            try:
                started = time.mktime(time.strptime(stamp, "%Y%m%d-%H%M%S"))
            except ValueError:
                started = path.stat().st_mtime
            sessions.append(
                {
                    "id": path.name,
                    "title": title,
                    "started": started,
                    "modified": path.stat().st_mtime,
                    "turns": len(user_texts),
                    "probes": sum(1 for r in records if r.get("kind") == "run"),
                    "notes": sum(1 for r in records if r.get("kind") == "note"),
                    "context_restorable": self.sdk_id(records) is not None,
                    "is_current": path.name == self.transcript_path.name,
                }
            )
        sessions.sort(key=lambda item: item["modified"], reverse=True)
        return sessions

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        return self.read_transcript(self.transcript_file(session_id))

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
            elif kind == "note":
                lines.append(
                    f"You noted ({record.get('note_kind', 'fact')}): "
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
