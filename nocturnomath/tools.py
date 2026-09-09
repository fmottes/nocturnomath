"""Claude MCP tools bound to an exploration session."""

import ast
import asyncio
import base64
import re
import threading
from typing import TYPE_CHECKING

from claude_agent_sdk import tool

if TYPE_CHECKING:
    from .session import ExplorationSession

BACKEND_SWITCH = re.compile(
    r"matplotlib\.use\s*\(|switch_backend\s*\(|%matplotlib\s+(?!inline\b)\S"
)


def switches_backend(code: str) -> bool:
    """Return whether code would change matplotlib away from the inline backend."""
    return bool(BACKEND_SWITCH.search(code))


def builds_machinery(code: str) -> bool:
    """Return whether code defines a class or more than two functions."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    nodes = list(ast.walk(tree))
    classes = [node for node in nodes if isinstance(node, ast.ClassDef)]
    functions = [
        node
        for node in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return bool(classes) or len(functions) > 2


def build_tools(session: "ExplorationSession"):
    """Create the exploration and scientific record tools for a session."""

    @tool(
        "run",
        "Run Python in the persistent Jupyter kernel. This is the only way to execute anything. "
        "`expected` is your one-line prediction of what this will show; it is required. "
        "Use install_packages to install missing libraries into the selected research environment.",
        {"code": str, "expected": str},
    )
    async def run(args):
        code = args["code"]
        expected = args.get("expected", "").strip()

        if session._pending_verdict is not None:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Nothing ran. The last probe has no verdict yet; it "
                        f"expected: {session._pending_verdict}. Call `verdict` with "
                        "what the result actually showed against that, then run this "
                        "again. A probe that only loaded or inspected data still needs "
                        "one, and there a single line is enough.",
                    }
                ],
                "is_error": True,
            }

        if not expected:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Nothing ran. `expected` is required: state in one line what you "
                        "predict this code will show, then call run again.",
                    }
                ],
                "is_error": True,
            }

        if switches_backend(code):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Nothing ran. This changes the matplotlib backend, which stops "
                        "plots from being captured for the rest of the session — the kernel is "
                        "persistent, so it would not recover on its own. Plots are captured and "
                        "saved automatically; just build the figure. Remove the backend switch "
                        "and call run again.",
                    }
                ],
                "is_error": True,
            }

        preface = []
        if not session.kernel.is_alive():
            await asyncio.to_thread(session.kernel.restart)
            preface.append(
                "Note: the kernel was dead and has been restarted. Everything in memory is gone; "
                "reload what you need."
            )
        if builds_machinery(code):
            preface.append(
                "Note: this is building machinery. Is there a flatter version that answers the question?"
            )

        if session.kernel.busy or (
            session._probe_task and not session._probe_task.done()
        ):
            return failure("A probe is still executing or saving its artifacts.")
        session.activate_session()
        workspace = session.workspace
        transcript_path = session.transcript_path
        probe_path = workspace.start_probe(
            code,
            expected,
            session.model,
            kernel_id=session.kernel_id,
            environment=session.environment_record,
        )
        source_id = f"{workspace.session_id}/{probe_path.name}"
        session.log_transcript(
            "probe_started", probe_id=source_id, expected=expected, code=code
        )
        cancelled = threading.Event()

        def execute_and_save():
            try:
                text, images, kernel_note = session.kernel.execute(
                    code, session.timeout_s
                )
            except Exception as exc:
                workspace.finish_probe(probe_path, str(exc), [], "failed", str(exc))
                workspace.log_transcript(
                    "probe_failed",
                    transcript_path=transcript_path,
                    probe_id=source_id,
                    text=str(exc),
                )
                raise
            status = "completed"
            if cancelled.is_set():
                status = "interrupted"
            elif session.kernel.errored:
                status = "error"
            elif kernel_note:
                status = "incomplete"
            workspace.finish_probe(probe_path, text, images, status, kernel_note)
            paths = [
                f"{probe_path.relative_to(workspace.path).as_posix()}/plot-{i}.png"
                for i in range(1, len(images) + 1)
            ]
            workspace.log_transcript(
                "run",
                transcript_path=transcript_path,
                probe_id=source_id,
                expected=expected,
                code=code,
                output=text,
                images=paths,
                execution_note=kernel_note,
                status=status,
            )
            return text, images, kernel_note, paths

        await session.emit(
            "probe_start",
            probe_id=source_id,
            expected=expected,
            code=code,
            kernel_alive=session.kernel.is_alive(),
        )
        execution = asyncio.create_task(asyncio.to_thread(execute_and_save))
        session._probe_task = execution
        try:
            text, images, kernel_note, plot_paths = await asyncio.shield(execution)
        except asyncio.CancelledError:
            cancelled.set()
            # Preserve the worker until its output is saved, even when its caller is cancelled.
            try:
                await asyncio.shield(execution)
            except asyncio.CancelledError:
                pass
            raise

        encoded_images = [base64.b64encode(image).decode() for image in images]
        shown = images[: session.image_cap]
        if len(images) > session.image_cap:
            preface.insert(
                0,
                f"Note: {len(images)} plots, showing {session.image_cap}. "
                "Pick the one that answers the question.",
            )

        if not session.kernel.errored:
            session._pending_verdict = expected

        if not session.kernel.errored and (images or text.strip()):
            session._results_since_note += 1
            if session._results_since_note == 1:
                preface.append(
                    "Note: this result is saved, but has no evidence entry yet. If it settled something "
                    "about the system, or ruled something out, call `evidence` with the saved source now: "
                    "one finding, one line."
                )
            else:
                preface.append(
                    f"Note: {session._results_since_note} probes have produced a "
                    "result since the last `evidence`. Record them as separate `evidence` "
                    "calls, one finding each; do not pack several findings into "
                    "one line. If one of them was not worth keeping, say so."
                )

        body = f"Probe {source_id}\nSource: {source_id}/output.txt\n" + (
            text.strip() or "(no text output)"
        )
        if plot_paths:
            body += "\n\n" + "\n".join(
                f"Source: {source_id}/plot-{i}.png" for i in range(1, len(images) + 1)
            )
        if kernel_note:
            body += "\n" + kernel_note
        output = "\n".join(preface) + "\n\n" + body if preface else body

        await session.emit(
            "probe_finish",
            probe_id=source_id,
            expected=expected,
            code=code,
            output=output,
            plot_urls=[f"/api/asset?path={path}" for path in plot_paths],
            plot_images=encoded_images,
        )

        content = [{"type": "text", "text": output}]
        for image in shown:
            content.append(
                {
                    "type": "image",
                    "data": base64.b64encode(image).decode(),
                    "mimeType": "image/png",
                }
            )
        return {"content": content}

    def failure(message):
        return {"content": [{"type": "text", "text": message}], "is_error": True}

    async def record_changed(kind, entry_id, text, **fields):
        session.log_transcript(kind, entry_id=entry_id, text=text, **fields)
        await session.emit(
            "record_changed", kind=kind, entry_id=entry_id, text=text, **fields
        )
        return {"content": [{"type": "text", "text": f"Recorded {entry_id}."}]}

    @tool(
        "evidence",
        "Append a factual observation to .nocturnomath/notes/evidence.md. Maximum 250 characters "
        "of plain text, with conditions; no interpretation or citations to entries. "
        "sources is a list of saved artifacts such as S001/P001/output.txt or "
        "S001/P001/plot-1.png returned by run. Returns a stable E ID.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "sources": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["text", "sources"],
        },
    )
    async def evidence(args):
        try:
            links = session.workspace.evidence_sources(args["sources"])
            entry_id = session.workspace.notes.add_evidence(args["text"], links)
        except (ValueError, FileNotFoundError) as exc:
            return failure(str(exc))
        session._results_since_note = 0
        return await record_changed(
            "evidence", entry_id, args["text"], sources=args["sources"]
        )

    @tool(
        "thought",
        "Append an interpretation, conjecture, explanation, or question to "
        ".nocturnomath/notes/thoughts.md. Aim for one direct paragraph (80–150 words); equations "
        "are welcome. Cite supporting entries as [E001] or [T001]; links are generated. "
        "replaces lists old T IDs to strike in full, or [] for a new thought. A replacement "
        "must explain the correction and retain any still-valid reasoning. Returns a stable T ID.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "replaces": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["text", "replaces"],
        },
    )
    async def thought(args):
        try:
            entry_id = session.workspace.notes.add_thought(
                args["text"], args["replaces"]
            )
        except ValueError as exc:
            return failure(str(exc))
        return await record_changed(
            "thought", entry_id, args["text"], replaces=args["replaces"]
        )

    @tool(
        "strike_evidence",
        "Strike an invalid evidence entry in full, preserving its text and sources. "
        "No explanation or replacement link is added to evidence.md. Record a corrected "
        "observation separately with evidence; explanations belong in thoughts.",
        {"entry_id": str},
    )
    async def strike_evidence(args):
        try:
            session.workspace.notes.strike_evidence(args["entry_id"])
        except ValueError as exc:
            return failure(str(exc))
        return await record_changed(
            "evidence_struck", args["entry_id"], "Evidence struck as invalid."
        )

    @tool(
        "verdict",
        "Say what the last probe's result meant: how it came out against `expected`, "
        "and what it rules in or out. Every probe that produced a result needs one "
        "before the next `run`. Two to four sentences; for a probe that only loaded "
        "or inspected data, one line such as 'loaded as expected' is enough.",
        {"text": str},
    )
    async def verdict(args):
        text = args.get("text", "").strip()
        if not text:
            return {
                "content": [{"type": "text", "text": "text must be non-empty."}],
                "is_error": True,
            }
        if session._pending_verdict is None:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "No probe is waiting for a verdict. Run one first.",
                    }
                ],
                "is_error": True,
            }

        expected = session._pending_verdict
        session._pending_verdict = None
        session.log_transcript("verdict", expected=expected, text=text)
        await session.emit("probe_verdict", expected=expected, text=text)
        return {"content": [{"type": "text", "text": "Verdict recorded."}]}

    @tool(
        "restart_kernel",
        "Restart the Jupyter kernel. All in-memory state is lost.",
        {},
    )
    async def restart_kernel(args):
        if session.kernel.busy or (
            session._probe_task and not session._probe_task.done()
        ):
            return failure("An operation is still running.")
        task = asyncio.create_task(asyncio.to_thread(session.kernel.restart))
        session._probe_task = task
        try:
            await asyncio.shield(task)
        finally:
            if task.done():
                session._probe_task = None
        await session.emit("kernel_restarted")
        return {
            "content": [
                {
                    "type": "text",
                    "text": "Kernel restarted. In-memory state is gone.",
                }
            ]
        }

    @tool(
        "install_packages",
        "Install missing packages into the selected research environment. "
        "State deliberate upgrades before calling. This records installation separately from probes. "
        "Already imported modules retain their loaded versions until restart_kernel.",
        {
            "type": "object",
            "properties": {
                "packages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                }
            },
            "required": ["packages"],
        },
    )
    async def install_packages(args):
        if session.kernel.busy or (
            session._probe_task and not session._probe_task.done()
        ):
            return failure("An operation is still running.")
        task = asyncio.create_task(
            asyncio.to_thread(session.install_packages, args["packages"])
        )
        session._probe_task = task
        try:
            status, output = await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return failure(str(exc))
        finally:
            if task.done():
                session._probe_task = None
        await session.emit(
            "system_message",
            text=f"Package installation {status}. Installer output saved in this session’s environment_changes folder.",
        )
        return {
            "content": [
                {
                    "type": "text",
                    "text": output
                    + "\nLoaded modules are unchanged; restart if needed.",
                }
            ],
            "is_error": status != "completed",
        }

    return [
        run,
        evidence,
        thought,
        strike_evidence,
        verdict,
        restart_kernel,
        install_packages,
    ]
