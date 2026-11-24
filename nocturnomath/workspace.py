"""Workspace files, knowledge base, documents, plots, and persisted chat history."""

import base64
import json
import logging
import os
import re
import time
from pathlib import Path, PurePosixPath
from typing import Any

from .auth import PROVIDERS
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
        self.kb_path = self.nocturnomath_path / "kb"
        self.documents_path = self.nocturnomath_path / "documents"
        self.sessions_path = self.nocturnomath_path / "sessions"
        self.scratch_path = self.nocturnomath_path / "scratch"
        self.config_path = self.nocturnomath_path / "config.json"
        for directory in (
            self.kb_path,
            self.documents_path,
            self.sessions_path,
            self.scratch_path,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.notes = ResearchNotes(self.kb_path)
        self._session_summary_cache: dict[
            Path, tuple[int, int, dict[str, Any] | None]
        ] = {}
        self.start_new_transcript()

    def read_config(self) -> dict[str, Any]:
        """Per-workspace settings shared with the research environment choice."""
        if not self.config_path.exists():
            return {}
        try:
            return json.loads(self.config_path.read_text())
        except json.JSONDecodeError:
            return {}

    def update_config(self, **fields):
        data = self.read_config()
        data.update(fields)
        write_text(self.config_path, json.dumps(data, indent=2) + "\n")

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

    def current_records(self) -> list[dict[str, Any]]:
        """Transcript of the active session, empty until it has activity."""
        if self._session_pending or not self.transcript_path.is_file():
            return []
        return self.read_transcript(self.transcript_path)

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
            "not observations). Relative links are based in .nocturnomath/kb/.\n\n"
            + self.notes.read()
            + "\n---\n\n"
        )

    def probe_path(self, probe_id: str) -> Path:
        """Resolve a persisted probe or retry directory such as ``S001/P001``."""
        match = re.fullmatch(r"(S\d{3,})/(P\d{3,})(?:/(A\d{3,}))?", probe_id)
        if not match:
            raise ValueError("probe_id must look like S001/P001 or S001/P001/A002.")
        session_id, probe_name, attempt = match.groups()
        path = self.sessions_path / session_id / "probes" / probe_name
        if attempt:
            path = path / attempt
        if not path.resolve().is_relative_to(self.sessions_path.resolve()):
            raise ValueError("Probe path outside workspace")
        return path

    @staticmethod
    def update_probe_meta(path: Path, **fields: Any) -> dict[str, Any]:
        """Atomically merge ``fields`` into a probe's ``probe.json``."""
        file = path / "probe.json"
        metadata = json.loads(file.read_text())
        metadata.update(fields)
        write_text(file, json.dumps(metadata, indent=2) + "\n")
        return metadata

    def _write_probe_start(
        self,
        path: Path,
        code: str,
        expected: str,
        model: str,
        *,
        retry_of: str | None = None,
        **context: Any,
    ) -> None:
        write_text(path / "code.py", code)
        metadata: dict[str, Any] = {"id": path.name}
        if retry_of is not None:
            metadata["retry_of"] = retry_of
        metadata.update(
            expected=expected,
            **context,
            model=model,
            session=self.session_id,
            started=time.time(),
            status="started",
        )
        write_text(path / "probe.json", json.dumps(metadata, indent=2) + "\n")

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
        self._write_probe_start(path, code, expected, model, **context)
        return path

    def retry_probe(
        self, probe_id: str, code: str, expected: str | None, model: str, **context
    ) -> Path:
        """Save a corrected execution under the same probe without changing old artifacts."""
        match = re.fullmatch(rf"{re.escape(self.session_id)}/(P\d{{3,}})", probe_id)
        if not match:
            raise ValueError(
                "retry_of must name a probe in the current session, such as S001/P001."
            )
        probe = self.probes_path / match.group(1)
        if not (probe / "probe.json").is_file():
            raise ValueError(f"Probe {probe_id} does not exist.")
        probe_meta = json.loads((probe / "probe.json").read_text())
        if probe_meta.get("backend") == "slurm":
            raise ValueError(
                "A Slurm probe cannot run in the persistent kernel. "
                "Correct the script and call submit_slurm for a new probe."
            )
        if not expected:
            expected = probe_meta["expected"]
        attempts = sorted(
            (path for path in probe.iterdir() if re.fullmatch(r"A\d{3,}", path.name)),
            key=lambda path: int(path.name[1:]),
        )
        status = (
            json.loads((attempts[-1] / "probe.json").read_text())["status"]
            if attempts
            else probe_meta["status"]
        )
        if status not in ("error", "failed"):
            raise ValueError(
                f"Probe {probe_id} cannot be retried after status {status}."
            )
        number = int(attempts[-1].name[1:]) + 1 if attempts else 2
        path = probe / f"A{number:03d}"
        path.mkdir()
        self._write_probe_start(
            path, code, expected, model, retry_of=probe_id, **context
        )
        return path

    @classmethod
    def finish_probe(
        cls,
        path: Path,
        output: str,
        images: list[bytes],
        status: str,
        note: str | None,
        **extra_meta: Any,
    ) -> dict[str, Any]:
        # The path is captured before execution; changing UI state cannot redirect a result.
        write_text(path / "output.txt", output)
        for index, image in enumerate(images, 1):
            (path / f"plot-{index}.png").write_bytes(image)
        return cls.update_probe_meta(
            path,
            status=status,
            finished=time.time(),
            execution_note=note,
            **extra_meta,
        )

    def evidence_sources(self, sources: list[str]) -> list[str]:
        """Accept only artifacts belonging to a persisted probe outcome."""
        links = []
        for source in dict.fromkeys(sources):
            if not re.fullmatch(
                r"S\d{3,}/P\d{3,}/(?:A\d{3,}/)?(?:output\.txt|plot-[1-9]\d*\.png)",
                source,
            ):
                raise ValueError(
                    "Sources must look like S001/P001/output.txt or S001/P001/A002/plot-1.png."
                )
            session_id, probe_id, *parts = source.split("/")
            filename = parts[-1]
            artifact = (
                self.sessions_path / session_id / "probes" / probe_id / Path(*parts)
            )
            if (
                not artifact.resolve().is_relative_to(self.sessions_path.resolve())
                or not artifact.is_file()
            ):
                raise ValueError(f"Source does not exist: {source}.")
            metadata = json.loads((artifact.parent / "probe.json").read_text())
            if metadata["status"] not in (
                "completed",
                "error",
                "failed",
                "incomplete",
                "interrupted",
            ):
                raise ValueError(
                    f"Probe {artifact.parent.name} has no recorded outcome yet."
                )
            relative = f"../sessions/{session_id}/probes/{probe_id}"
            if len(parts) == 2:
                relative += f"/{parts[0]}"
            links.append(
                f"[{source}]({relative}/{filename}) "
                f"([code]({relative}/code.py), [run]({relative}/probe.json))"
            )
        return links

    def list_documents_and_folders(self) -> tuple[list[dict[str, Any]], list[str]]:
        """Return Markdown documents and valid subfolders in one directory walk."""
        self.document_folder("")
        documents: list[dict[str, Any]] = []
        folders: list[str] = []
        for root, dirnames, filenames in os.walk(
            self.documents_path, followlinks=False
        ):
            root_path = Path(root)
            rel_root = root_path.relative_to(self.documents_path).as_posix()
            prefix = "" if rel_root == "." else f"{rel_root}/"
            kept_dirs: list[str] = []
            for dirname in dirnames:
                dir_path = root_path / dirname
                if not self._valid_document_part(dirname) or dir_path.is_symlink():
                    continue
                kept_dirs.append(dirname)
                folders.append(f"{prefix}{dirname}")
            dirnames[:] = kept_dirs
            for filename in filenames:
                if not filename.endswith(".md") or not self._valid_document_part(
                    filename
                ):
                    continue
                file_path = root_path / filename
                if file_path.is_symlink() or not file_path.is_file():
                    continue
                stat = file_path.stat()
                documents.append(
                    {
                        "name": f"{prefix}{filename}",
                        "path": str(file_path.relative_to(self.path)),
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                    }
                )
        return (
            sorted(documents, key=lambda document: document["name"].lower()),
            sorted(folders, key=str.lower),
        )

    def list_documents(self) -> list[dict[str, Any]]:
        """Markdown documents kept as extra context under .nocturnomath/documents/."""
        documents, _ = self.list_documents_and_folders()
        return documents

    def list_document_folders(self) -> list[str]:
        """Return valid subfolders, including empty ones."""
        _, folders = self.list_documents_and_folders()
        return folders

    @staticmethod
    def _valid_document_part(part: str) -> bool:
        return (
            bool(part)
            and part == part.strip()
            and not part.startswith(".")
            and "\\" not in part
            and "\0" not in part
        )

    def document_folder(self, name: str) -> Path:
        if not isinstance(name, str):
            raise TypeError("Use a folder path under documents.")
        if self.nocturnomath_path.is_symlink() or self.documents_path.is_symlink():
            raise ValueError("Document symlinks are not supported.")
        if not name:
            return self.documents_path
        parts = name.split("/")
        if any(not self._valid_document_part(part) for part in parts):
            raise ValueError("Use a folder path under documents.")
        path = self.documents_path
        for part in parts:
            path /= part
            if path.is_symlink():
                raise ValueError("Document symlinks are not supported.")
            if path.exists() and not path.is_dir():
                raise ValueError("A document path component is not a folder.")
        return path

    def document_file(self, name: str) -> Path:
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or name.startswith("/")
            or not name.endswith(".md")
        ):
            raise ValueError("Use a document name such as report.md.")
        folder, _, filename = name.rpartition("/")
        if not self._valid_document_part(filename):
            raise ValueError("Use a document name such as report.md.")
        path = self.document_folder(folder) / filename
        if path.is_symlink():
            raise ValueError("Document symlinks are not supported.")
        return path

    def read_document(self, name: str) -> dict[str, Any]:
        path = self.document_file(name)
        if not path.is_file():
            raise FileNotFoundError(f"Document {name} not found")
        stat = path.stat()
        return {
            "name": name,
            "path": str(path.relative_to(self.path)),
            "content": path.read_text(encoding="utf-8", errors="replace"),
            "modified": stat.st_mtime,
            "size": stat.st_size,
        }

    @staticmethod
    def document_name(title: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", title.strip()).strip("-.")
        if not slug:
            raise ValueError("Give the document a title.")
        return slug if slug.endswith(".md") else slug + ".md"

    def create_document(self, title: str, text: str, folder: str = "") -> str:
        directory = self.document_folder(folder)
        if not directory.is_dir():
            raise FileNotFoundError(f"Folder {folder} not found")
        name = "/".join(part for part in (folder, self.document_name(title)) if part)
        self.save_document(name, text)
        return name

    def save_document(self, name: str, text: str, replace_existing: bool = False):
        path = self.document_file(name)
        if replace_existing:
            if not path.is_file():
                raise FileNotFoundError(f"Document {name} not found")
        elif path.exists():
            raise FileExistsError(f"Document {name} already exists")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        write_text(path, text if text.endswith("\n") else text + "\n")

    def write_document(self, name: str, text: str):
        self.save_document(name, text, replace_existing=True)

    def create_document_folder(self, name: str) -> str:
        if not name:
            raise ValueError("Give the folder a name.")
        path = self.document_folder(name)
        if path.exists():
            raise FileExistsError(f"Folder {name} already exists")
        path.mkdir(parents=True)
        return name

    def delete_document(self, name: str):
        path = self.document_file(name)
        if not path.is_file():
            raise FileNotFoundError(f"Document {name} not found")
        path.unlink()

    def _workspace_file(self, relative_path: str, label: str) -> Path:
        target = (self.path / relative_path).resolve()
        if not target.is_relative_to(self.path):
            raise ValueError(f"{label} path outside workspace")
        if not target.is_file():
            raise FileNotFoundError(f"{label} {relative_path} not found")
        return target

    def read_file(self, relative_path: str) -> dict[str, Any]:
        target = self._workspace_file(relative_path, "File")
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
        return self._workspace_file(relative_path, "Asset")

    def list_plots(self) -> list[dict[str, Any]]:
        plots = []
        for path in self.sessions_path.glob("S*/probes/P*/**/plot-*.png"):
            stat = path.stat()
            relative = path.relative_to(self.path).as_posix()
            source = path.relative_to(self.sessions_path).parts
            plots.append(
                {
                    "filename": "/".join((source[0], *source[2:])),
                    "session": source[0],
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
    def session_meta(
        records: list[dict[str, Any]],
    ) -> tuple[str | None, str | None, str | None, str | None]:
        """Return ``(owner_provider, sdk_session_id, model, effort)`` from meta records."""
        owner_provider: str | None = None
        sdk_id: str | None = None
        sdk_id_allowed = True
        model: str | None = None
        effort: str | None = None
        for record in reversed(records):
            if record.get("kind") != "meta":
                continue
            provider = record.get("provider")
            has_provider = provider in PROVIDERS
            raw_sdk_id = record.get("sdk_session_id")
            has_sdk_id = isinstance(raw_sdk_id, str) and bool(raw_sdk_id)
            has_rebuilt = bool(record.get("context_rebuilt"))
            if not (has_provider or has_sdk_id or has_rebuilt):
                continue
            rec_provider = (
                provider if has_provider else ("anthropic" if has_sdk_id else None)
            )
            if owner_provider is None:
                if rec_provider is None:
                    sdk_id_allowed = False
                    continue
                owner_provider = rec_provider
            elif rec_provider is not None and rec_provider != owner_provider:
                break
            if sdk_id is None and sdk_id_allowed:
                if not has_rebuilt and has_sdk_id:
                    sdk_id = raw_sdk_id
                sdk_id_allowed = False
            if model is None:
                candidate_model = record.get("raw_model") or record.get("model")
                if isinstance(candidate_model, str) and candidate_model:
                    model = candidate_model
                    rec_effort = record.get("effort")
                    effort = (
                        rec_effort
                        if isinstance(rec_effort, str) and rec_effort
                        else None
                    )
            if (sdk_id is not None or not sdk_id_allowed) and model is not None:
                break
        return owner_provider, sdk_id, model, effort

    @classmethod
    def sdk_session_info(
        cls,
        records: list[dict[str, Any]],
    ) -> tuple[str | None, str | None]:
        owner_provider, sdk_id, _, _ = cls.session_meta(records)
        if sdk_id is not None and owner_provider is not None:
            return owner_provider, sdk_id
        return None, None

    @classmethod
    def sdk_id(cls, records: list[dict[str, Any]]) -> str | None:
        return cls.sdk_session_info(records)[1]

    @classmethod
    def session_provider_and_model(
        cls,
        records: list[dict[str, Any]],
    ) -> tuple[str | None, str | None, str | None]:
        owner_provider, _, model, effort = cls.session_meta(records)
        return owner_provider, model, effort

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        if not self.sessions_path.exists():
            self._session_summary_cache.clear()
            return sessions
        seen_paths: set[Path] = set()
        for path in self.sessions_path.glob("S*/transcript.jsonl"):
            try:
                stat = path.stat()
            except OSError:
                continue
            seen_paths.add(path)
            cached = self._session_summary_cache.get(path)
            if (
                cached is not None
                and cached[0] == stat.st_mtime_ns
                and cached[1] == stat.st_size
            ):
                summary = cached[2]
            else:
                records = self.read_transcript(path)
                first_user_text: str | None = None
                turns = 0
                probes = 0
                notes = 0
                for record in records:
                    kind = record.get("kind")
                    if kind == "user":
                        if first_user_text is None:
                            first_user_text = record.get("text", "")
                        turns += 1
                    elif kind == "probe_started" and not record.get("retry_of"):
                        probes += 1
                    elif kind in ("evidence", "thought"):
                        notes += 1
                if first_user_text is None:
                    summary = None
                else:
                    title = (
                        first_user_text[:120] + "…"
                        if len(first_user_text) > 120
                        else first_user_text
                    )
                    summary = {
                        "id": path.parent.name,
                        "title": title,
                        "started": records[0].get("timestamp", stat.st_mtime),
                        "modified": stat.st_mtime,
                        "turns": turns,
                        "probes": probes,
                        "notes": notes,
                        "context_restorable": self.sdk_id(records) is not None,
                    }
                self._session_summary_cache[path] = (
                    stat.st_mtime_ns,
                    stat.st_size,
                    summary,
                )
            if summary is not None:
                sessions.append({**summary, "is_current": path == self.transcript_path})
        for stale in list(self._session_summary_cache):
            if stale not in seen_paths:
                del self._session_summary_cache[stale]
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
        started_probes = {
            record.get("probe_id")
            for record in records
            if record.get("kind") == "probe_started"
        }
        cells: list[dict[str, Any]] = []
        turn = 0
        idea = 0
        probe = 0
        execution_count = 0
        pending_agent: list[str] = []

        def markdown(source: str, kind: str, **metadata):
            cell = {
                "cell_type": "markdown",
                "id": f"cell-{len(cells) + 1}",
                "metadata": {"nocturnomath": {"kind": kind, **metadata}},
                "source": source,
            }
            cells.append(cell)
            return cell

        def code_cell(
            source: str,
            *,
            count: int | None,
            outputs: list[dict[str, Any]],
            **metadata: Any,
        ) -> None:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": count,
                    "id": f"cell-{len(cells) + 1}",
                    "metadata": {"nocturnomath": {"kind": "probe_code", **metadata}},
                    "outputs": outputs,
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

        def slurm_cells(probe_id: str, code: str, outcome: dict[str, Any]):
            commented = "\n".join(f"# {line}" for line in code.splitlines())
            code_cell(
                "# Slurm probe saved for inspection; Run All does not rerun it.\n"
                + commented,
                count=None,
                outputs=[],
                backend="slurm",
                probe_id=probe_id,
                status=outcome.get("status", "no recorded outcome"),
            )
            if not outcome:
                markdown(
                    "Slurm result has not been collected.",
                    "slurm_result",
                    probe_id=probe_id,
                )
                return
            result_lines = [f"**Saved Slurm result:** `{probe_id}/output.txt`"]
            attachments = {}
            for item in self._notebook_outputs(outcome):
                if item["output_type"] == "stream":
                    result_lines.append(
                        "\n".join(f"    {line}" for line in item["text"].splitlines())
                    )
                elif item["output_type"] == "display_data":
                    name = f"plot-{len(attachments) + 1}.png"
                    attachments[name] = item["data"]
                    result_lines.append(f"![{name}](attachment:{name})")
            result = markdown(
                "\n\n".join(result_lines), "slurm_result", probe_id=probe_id
            )
            if attachments:
                result["attachments"] = attachments

        for record in records:
            kind = record.get("kind")
            if kind == "agent":
                text = record.get("text")
                if isinstance(text, str) and text.strip():
                    pending_agent.append(text)
                continue
            if kind in ("meta", "kernel_started", "resumed", "message_usage", "usage"):
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
                retry_of = record.get("retry_of")
                if not retry_of:
                    probe += 1
                probe_id = record.get("probe_id", f"probe-{probe}")
                expected = record.get("expected", "")
                heading = (
                    f"### Probe {turn}.{probe}, attempt {probe_id.rsplit('/', 1)[-1][1:]}"
                    if retry_of
                    else f"### Probe {turn}.{probe}"
                )
                markdown(
                    f"{heading}\n\n**Prediction:** {expected}",
                    "probe",
                    probe_id=probe_id,
                )
                outcome = outcomes.get(probe_id, {})
                if record.get("backend") == "slurm":
                    slurm_cells(probe_id, record.get("code", ""), outcome)
                    continue
                outputs = self._notebook_outputs(outcome)
                execution_count += 1
                code_cell(
                    record.get("code", ""),
                    count=execution_count,
                    outputs=outputs,
                    probe_id=probe_id,
                    status=outcome.get("status")
                    or (
                        "completed"
                        if outcome.get("kind") == "run"
                        else "no recorded outcome"
                    ),
                )
            elif (
                kind == "run"
                and record.get("backend") == "slurm"
                and record.get("probe_id") not in started_probes
            ):
                probe_id = record["probe_id"]
                markdown(
                    f"### Slurm probe {probe_id} checked in this session\n\n"
                    f"**Prediction:** {record.get('expected', '')}",
                    "probe",
                    probe_id=probe_id,
                )
                slurm_cells(probe_id, record.get("code", ""), record)
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
    def recap(
        records: list[dict[str, Any]],
        max_chars: int = 8000,
        *,
        kernel_preserved: bool = False,
    ) -> str:
        expected_by_probe = {
            record["probe_id"]: record["expected"]
            for record in records
            if record.get("kind") == "probe_started"
            and record.get("probe_id")
            and record.get("expected")
        }
        lines = []
        for record in records:
            kind = record.get("kind")
            if kind == "user":
                lines.append("Me: " + record.get("text", ""))
            elif kind == "agent":
                lines.append("You: " + _clip(record.get("text", ""), 800))
            elif kind == "run":
                probe_id = record.get("probe_id")
                status = record.get("status", "completed")
                header = (
                    f"You ran probe {probe_id} ({status}), expecting: "
                    if probe_id
                    else "You ran a probe expecting: "
                ) + record.get("expected", "")
                if probe_id:
                    plot_sources = [
                        f"{probe_id}/{PurePosixPath(img).name}"
                        for img in (record.get("images") or [])
                        if isinstance(img, str) and img
                    ]
                    sources = [f"{probe_id}/output.txt", *plot_sources]
                    header += "\nSources: " + ", ".join(sources)
                lines.append(
                    header + "\nIt printed: " + _clip(record.get("output", ""), 400)
                )
            elif kind == "probe_failed":
                lines.append(
                    f"Probe {record.get('probe_id', '')} failed: "
                    + _clip(record.get("text", ""), 300)
                )
            elif kind == "slurm_submitted":
                probe_id = record.get("probe_id", "")
                line = (
                    f"Submitted Slurm probe {probe_id} (job {record.get('job_id', '')})"
                )
                expected = record.get("expected") or expected_by_probe.get(probe_id, "")
                if expected:
                    line += f", expecting: {expected}"
                lines.append(line)
            elif kind == "verdict":
                probe_id = record.get("probe_id")
                prefix = (
                    f"Recorded verdict for {probe_id}: "
                    if probe_id
                    else "Recorded verdict: "
                )
                lines.append(prefix + record.get("text", ""))
            elif kind in ("evidence", "thought", "evidence_struck"):
                lines.append(
                    f"Recorded {kind} {record.get('entry_id', '')}: "
                    + record.get("text", "")
                )
        body = "\n\n".join(lines)
        if len(body) > max_chars:
            body = "[earlier turns omitted]\n\n" + body[-max_chars:]
        if kernel_preserved:
            intro = (
                "This continues the current exploration session on a new model provider, "
                "or after rebuilding provider context. Here is the transcript so far. "
                "The live Jupyter kernel and its in-memory variables are still intact; "
                "do not re-run earlier probes unless needed."
            )
        else:
            intro = (
                "This continues an earlier exploration session. Its full context could not be "
                "restored, so here is the transcript so far. The kernel may no longer hold the "
                "state it had then; re-run whatever you need."
            )
        return intro + "\n\n" + body + "\n\n---\n\n"
