"""Claude MCP tools bound to an exploration session."""

import ast
import asyncio
import base64
import re
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
    """Create the four SDK tools that operate on a session."""

    @tool(
        "run",
        "Run Python in the persistent Jupyter kernel. This is the only way to execute anything. "
        "`expected` is your one-line prediction of what this will show; it is required. "
        "To install a missing library, run `!uv pip install <name>` here (there is no pip).",
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
            session.kernel.restart()
            preface.append(
                "Note: the kernel was dead and has been restarted. Everything in memory is gone; "
                "reload what you need."
            )
        if builds_machinery(code):
            preface.append(
                "Note: this is building machinery. Is there a flatter version that answers the question?"
            )

        await session.emit(
            "probe_start",
            expected=expected,
            code=code,
            kernel_alive=session.kernel.is_alive(),
        )
        text, images, kernel_note = await asyncio.to_thread(
            session.kernel.execute, code, session.timeout_s
        )

        plot_paths = session.workspace.save_plots(images)
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
                    "Note: this result is not recorded. If it settled something "
                    "about the system, or ruled something out, call `note` now: "
                    "one finding, one line."
                )
            else:
                preface.append(
                    f"Note: {session._results_since_note} probes have produced a "
                    "result since the last `note`. Record them as separate `note` "
                    "calls, one finding each; do not pack several findings into "
                    "one line. If one of them was not worth keeping, say so."
                )

        body = text.strip() or "(no text output)"
        if plot_paths:
            body += "\n\n" + "\n".join("saved: " + path for path in plot_paths)
        if kernel_note:
            body += "\n" + kernel_note
        output = "\n".join(preface) + "\n\n" + body if preface else body

        session.log_transcript(
            "run", expected=expected, code=code, output=output, images=plot_paths
        )
        await session.emit(
            "probe_finish",
            expected=expected,
            code=code,
            output=output,
            plot_urls=[f"/scratch/{path}" for path in plot_paths],
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

    @tool(
        "note",
        "Append one line to notes.md. `kind` is 'fact' (something learned about the system) "
        "or 'dead_end' (something tried that gave nothing).",
        {"kind": str, "text": str},
    )
    async def note(args):
        kind = args["kind"]
        text = args.get("text", "").strip()
        if kind not in ("fact", "dead_end"):
            return {
                "content": [
                    {"type": "text", "text": "kind must be 'fact' or 'dead_end'."}
                ],
                "is_error": True,
            }
        if not text or "\n" in text or "\r" in text:
            return {
                "content": [
                    {"type": "text", "text": "text must be one non-empty line."}
                ],
                "is_error": True,
            }

        heading = session.workspace.add_note(kind, text)
        session._results_since_note = 0
        session.log_transcript("note", note_kind=kind, text=text)
        await session.emit(
            "note_added",
            kind=kind,
            text=text,
            notes_content=session.notes_path.read_text(),
        )
        return {"content": [{"type": "text", "text": f"Noted under {heading}."}]}

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
        session.kernel.restart()
        await session.emit("kernel_restarted")
        return {
            "content": [
                {
                    "type": "text",
                    "text": "Kernel restarted. In-memory state is gone.",
                }
            ]
        }

    return [run, note, verdict, restart_kernel]
