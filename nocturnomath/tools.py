"""Session-bound exploration and scientific record tool definitions."""

import ast
import asyncio
import base64
import json
import re
import threading
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any

from . import slurm

if TYPE_CHECKING:
    from .session import ExplorationSession

_PYTHON_TYPE_SCHEMA: dict[type[Any], str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


@dataclass(frozen=True)
class ToolSpec:
    """Provider-neutral definition of a session-bound tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def _normalize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert shorthand Python-type mappings into JSON Schema objects."""
    if schema.get("type") == "object" and "properties" in schema:
        return dict(schema)
    if not schema:
        return {"type": "object", "properties": {}, "additionalProperties": False}
    properties = {
        key: {"type": _PYTHON_TYPE_SCHEMA.get(value, "string")}
        for key, value in schema.items()
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(schema.keys()),
    }


def tool(name: str, description: str, schema: dict[str, Any]):
    """Bind an async handler into a provider-neutral ``ToolSpec``."""
    normalized = _normalize_schema(schema)

    def decorator(
        handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> ToolSpec:
        return ToolSpec(
            name=name,
            description=description,
            input_schema=normalized,
            handler=handler,
        )

    return decorator


def _check_json_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return True


def _validate_property_value(name: str, value: Any, prop: dict[str, Any]) -> None:
    expected = prop.get("type")
    if isinstance(expected, str):
        if not _check_json_type(value, expected):
            raise ValueError(f"Argument '{name}' must be of type {expected}.")
    elif (
        isinstance(expected, list)
        and expected
        and not any(isinstance(t, str) and _check_json_type(value, t) for t in expected)
    ):
        joined = " | ".join(str(t) for t in expected)
        raise ValueError(f"Argument '{name}' must be of type {joined}.")
    if "enum" in prop and isinstance(prop["enum"], list) and value not in prop["enum"]:
        raise ValueError(f"Argument '{name}' must be one of {prop['enum']}.")
    if isinstance(value, list):
        min_items = prop.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            raise ValueError(
                f"Argument '{name}' must contain at least {min_items} item(s)."
            )
        items_schema = prop.get("items")
        if isinstance(items_schema, dict):
            for index, item in enumerate(value):
                _validate_property_value(f"{name}[{index}]", item, items_schema)


def validate_tool_arguments(
    schema: dict[str, Any], arguments: dict[str, Any]
) -> dict[str, Any]:
    """Validate and clean tool call arguments against a JSON Schema object."""
    properties = (
        schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    )
    required_list = (
        schema.get("required") if isinstance(schema.get("required"), list) else []
    )
    disallow_extra = schema.get("additionalProperties") is False
    required = {item for item in required_list if isinstance(item, str)}
    cleaned: dict[str, Any] = {}
    for key, value in arguments.items():
        if disallow_extra and key not in properties:
            raise ValueError(f"Unexpected argument: '{key}'.")
        prop = properties.get(key)
        if value is None:
            prop_type = prop.get("type") if isinstance(prop, dict) else None
            allows_null = prop_type == "null" or (
                isinstance(prop_type, list) and "null" in prop_type
            )
            if allows_null:
                cleaned[key] = None
                continue
            if key not in required:
                continue
            raise ValueError(f"Argument '{key}' is required and cannot be null.")
        if isinstance(prop, dict):
            _validate_property_value(key, value, prop)
        cleaned[key] = value
    for key in required_list:
        if isinstance(key, str) and key not in cleaned:
            raise ValueError(f"Missing required argument: '{key}'.")
    return cleaned


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


def _text_result(text: str, *, is_error: bool | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if is_error is not None:
        result["is_error"] = is_error
    return result


def _failure(message: str) -> dict[str, Any]:
    return _text_result(message, is_error=True)


def _image_blocks(encoded_images: list[str]) -> list[dict[str, Any]]:
    return [
        {"type": "image", "data": encoded, "mimeType": "image/png"}
        for encoded in encoded_images
    ]


def _probe_payload(
    *,
    probe_id: str,
    expected: str,
    code: str,
    code_path: str,
    output: str,
    text: str,
    status: str,
    execution_note: str | None = None,
    plot_paths: list[str] | None = None,
    plot_images: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    paths = list(plot_paths) if plot_paths else []
    return {
        "probe_id": probe_id,
        **extra,
        "expected": expected,
        "code": code,
        "code_path": code_path,
        "output": output,
        "text": text,
        "status": status,
        "execution_note": execution_note,
        "plot_paths": paths,
        "plot_urls": [f"/api/asset?path={path}" for path in paths],
        "plot_images": list(plot_images) if plot_images else [],
    }


def build_tools(session: "ExplorationSession"):
    """Create the exploration and scientific record tools for a session."""

    def track_probe_task(task: asyncio.Task[Any]) -> asyncio.Task[Any]:
        session._probe_task = task

        def clear(done: asyncio.Task[Any]) -> None:
            if session._probe_task is done:
                session._probe_task = None

        task.add_done_callback(clear)
        return task

    def probe_busy() -> bool:
        return session.kernel.busy or (
            session._probe_task is not None and not session._probe_task.done()
        )

    async def run_probe_task(
        fn: Callable[..., Any],
        *args: Any,
        wait_on_cancel: bool = False,
        **kwargs: Any,
    ) -> Any:
        task = track_probe_task(
            asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if wait_on_cancel:
                with suppress(asyncio.CancelledError, Exception):
                    await asyncio.shield(task)
            raise
        finally:
            if session._probe_task is task and task.done():
                session._probe_task = None

    def serialize_probe_operation(handler):
        @wraps(handler)
        async def guarded(args):
            if session._probe_operation_lock.locked():
                return _failure("Another probe operation is still running.")
            async with session._probe_operation_lock:
                return await handler(args)

        return guarded

    @tool(
        "run",
        "Run Python in the persistent Jupyter kernel. "
        "`expected` is your one-line prediction of what this will show; it is required for a new probe. "
        "After a kernel probe error, set `retry_of` to the original probe ID (for example S001/P001) "
        "and provide corrected code to rerun that probe while preserving each attempt. "
        "A retry keeps the original prediction unless you supply a new `expected`. "
        "Use install_packages to install missing libraries into the selected research environment.",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "expected": {"type": "string"},
                "retry_of": {"type": "string"},
            },
            "required": ["code"],
        },
    )
    @serialize_probe_operation
    async def run(args):
        code = args["code"]
        expected = (args.get("expected") or "").strip()
        retry_of = args.get("retry_of") or None

        if session._pending_verdict is not None:
            return _failure(
                "Nothing ran. The last probe has no verdict yet; it "
                f"expected: {session._pending_verdict}. Call `verdict` with "
                "what the result actually showed against that, then run this "
                "again. A probe that only loaded or inspected data still needs "
                "one, and there a single line is enough."
            )

        steered = await session.consume_steer_message()
        if steered is not None:
            return steered

        if not expected and not retry_of:
            return _failure(
                "Nothing ran. `expected` is required: state in one line what you "
                "predict this code will show, then call run again."
            )

        if switches_backend(code):
            return _failure(
                "Nothing ran. This changes the matplotlib backend, which stops "
                "plots from being captured for the rest of the session — the kernel is "
                "persistent, so it would not recover on its own. Plots are captured and "
                "saved automatically; just build the figure. Remove the backend switch "
                "and call run again."
            )

        if probe_busy():
            return _failure("A probe is still executing or saving its artifacts.")

        preface = []
        if not session.kernel.is_alive():
            await run_probe_task(session.kernel.restart)
            preface.append(
                "Note: the kernel was dead and has been restarted. Everything in memory is gone; "
                "reload what you need."
            )
        if builds_machinery(code):
            preface.append(
                "Note: this is building machinery. Is there a flatter version that answers the question?"
            )

        if probe_busy():
            return _failure("A probe is still executing or saving its artifacts.")
        session.activate_session()
        workspace = session.workspace
        transcript_path = session.transcript_path
        try:
            active_model = session.active_model
            probe_path = (
                workspace.retry_probe(
                    retry_of,
                    code,
                    expected,
                    active_model,
                    kernel_id=session.kernel_id,
                    environment=session.environment_record,
                )
                if retry_of
                else workspace.start_probe(
                    code,
                    expected,
                    active_model,
                    kernel_id=session.kernel_id,
                    environment=session.environment_record,
                )
            )
        except ValueError as exc:
            return _failure(str(exc))
        if retry_of and not expected:
            expected = json.loads((probe_path / "probe.json").read_text())["expected"]
        source_id = f"{workspace.session_id}/{probe_path.relative_to(workspace.probes_path).as_posix()}"
        probe_dir = probe_path.relative_to(workspace.path).as_posix()
        code_path = f"{probe_dir}/code.py"
        session.log_transcript(
            "probe_started",
            probe_id=source_id,
            retry_of=retry_of,
            expected=expected,
            code=code,
        )
        cancelled = threading.Event()
        probe_epoch = session._interrupt_epoch
        query_task = session._query_task
        session._kernel_interrupt_requested = False

        def execute_and_save():
            try:
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
                if cancelled.is_set() or session._kernel_interrupt_requested:
                    status = "interrupted"
                elif session.kernel.errored:
                    status = "error"
                elif kernel_note:
                    status = "incomplete"
                workspace.finish_probe(probe_path, text, images, status, kernel_note)
                paths = [f"{probe_dir}/plot-{i}.png" for i in range(1, len(images) + 1)]
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
                return text, images, kernel_note, paths, status
            finally:
                session._kernel_interrupt_requested = False

        await session.emit(
            "probe_start",
            probe_id=source_id,
            expected=expected,
            code=code,
            code_path=code_path,
            kernel_alive=session.kernel.is_alive(),
        )
        execution = track_probe_task(
            asyncio.create_task(asyncio.to_thread(execute_and_save))
        )
        try:
            text, images, kernel_note, plot_paths, status = await asyncio.shield(
                execution
            )
            if session._kernel_notice and (kernel_note or len(preface) > 0):
                session._kernel_notice = ""
        except asyncio.CancelledError:
            turn_interrupted = (
                session._interrupt_epoch != probe_epoch
                or query_task is None
                or bool(query_task.cancelling())
            )
            if turn_interrupted:
                cancelled.set()
                if session.kernel.busy:
                    session._kernel_interrupt_requested = True
                    session.kernel.interrupt()

            async def finish_cancelled_probe():
                # The worker owns artifact saving; report its outcome only after it finishes.
                try:
                    text, images, kernel_note, plot_paths, status = await execution
                except Exception as exc:
                    text, images, kernel_note, plot_paths, status = (
                        str(exc),
                        [],
                        None,
                        [],
                        "failed",
                    )
                if (
                    session._interrupt_epoch == probe_epoch
                    and query_task is not None
                    and not query_task.cancelling()
                    and status not in ("error", "interrupted", "failed")
                ):
                    session._pending_verdict = expected
                    session._pending_verdict_probe_id = source_id
                await session.emit(
                    "probe_finish",
                    **_probe_payload(
                        probe_id=source_id,
                        expected=expected,
                        code=code,
                        code_path=code_path,
                        output=f"Probe {source_id}\nSource: {source_id}/output.txt\n{text}",
                        text=text,
                        status=status,
                        execution_note=kernel_note,
                        plot_paths=plot_paths,
                        plot_images=[
                            base64.b64encode(image).decode() for image in images
                        ],
                    ),
                )

            finish_task = asyncio.create_task(finish_cancelled_probe())
            # Keep waiting for saved output; a second cancellation leaves the notifier alive.
            try:
                await asyncio.shield(finish_task)
            except asyncio.CancelledError:
                pass
            raise
        except Exception as exc:
            original = retry_of or source_id
            message = (
                f"Probe {source_id} failed: {exc}\n"
                f'Correct the code and call `run` with `retry_of`: "{original}".'
            )
            await session.emit(
                "probe_finish",
                **_probe_payload(
                    probe_id=source_id,
                    expected=expected,
                    code=code,
                    code_path=code_path,
                    output=message,
                    text=str(exc),
                    status="failed",
                ),
            )
            return _failure(message)

        encoded_images = [base64.b64encode(image).decode() for image in images]
        shown_encoded = encoded_images[: session.image_cap]
        if len(images) > session.image_cap:
            preface.insert(
                0,
                f"Note: {len(images)} plots, showing {session.image_cap}. "
                "Pick the one that answers the question.",
            )

        if status not in ("error", "interrupted"):
            session._pending_verdict = expected
            session._pending_verdict_probe_id = source_id

        if status not in ("error", "interrupted") and (images or text.strip()):
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
        if status == "error":
            original = retry_of or source_id
            body += (
                f"\nCorrect the code and call `run` with `retry_of`: "
                f'"{original}" to retry this probe.'
            )
        output = "\n".join(preface) + "\n\n" + body if preface else body

        await session.emit(
            "probe_finish",
            **_probe_payload(
                probe_id=source_id,
                expected=expected,
                code=code,
                code_path=code_path,
                output=output,
                text=text,
                status=status,
                execution_note=kernel_note,
                plot_paths=plot_paths,
                plot_images=encoded_images,
            ),
        )

        return {
            "content": [{"type": "text", "text": output}, *_image_blocks(shown_encoded)]
        }

    async def record_changed(kind, entry_id, text, **fields):
        session.log_transcript(kind, entry_id=entry_id, text=text, **fields)
        await session.emit(
            "record_changed", kind=kind, entry_id=entry_id, text=text, **fields
        )
        return _text_result(f"Recorded {entry_id}.")

    @tool(
        "evidence",
        "Append a factual observation to .nocturnomath/kb/evidence.md. Maximum 250 characters "
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
            return _failure(str(exc))
        session._results_since_note = 0
        return await record_changed(
            "evidence", entry_id, args["text"], sources=args["sources"]
        )

    @tool(
        "thought",
        "Append an interpretation, conjecture, explanation, or question to "
        ".nocturnomath/kb/thoughts.md. Aim for one direct paragraph (80–150 words); equations "
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
            return _failure(str(exc))
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
            return _failure(str(exc))
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
        text = (args.get("text") or "").strip()
        if not text:
            return _failure("text must be non-empty.")
        if session._pending_verdict is None:
            return _failure("No probe is waiting for a verdict. Run one first.")

        expected = session._pending_verdict
        probe_id = session._pending_verdict_probe_id
        is_slurm = False
        if probe_id:
            try:
                probe_meta = session.workspace.probe_path(probe_id) / "probe.json"
                if probe_meta.is_file():
                    is_slurm = (
                        json.loads(probe_meta.read_text()).get("backend") == "slurm"
                    )
            except (ValueError, OSError, json.JSONDecodeError):
                is_slurm = False
        recorded = True
        if is_slurm:
            session.activate_session()
            recorded = await asyncio.to_thread(
                slurm.record_verdict,
                session.workspace,
                probe_id,
                expected,
                text,
                session.transcript_path,
            )
        else:
            session.log_transcript(
                "verdict", expected=expected, text=text, probe_id=probe_id
            )
        # Batch results remain queued by the workspace monitor. Recover any kernel
        # verdict that a newer batch result displaced when this chat was resumed.
        session._restore_pending_verdict(
            session.workspace.current_records(), include_slurm=False
        )
        if not recorded:
            return _failure(f"Probe {probe_id} already has a verdict.")
        await session.emit(
            "probe_verdict", probe_id=probe_id, expected=expected, text=text
        )
        message = "Verdict recorded."
        if session._pending_verdict is not None:
            pending_id = session._pending_verdict_probe_id
            message += (
                f" Probe {pending_id} still needs a verdict against: "
                f"{session._pending_verdict}. Inspect {pending_id}/output.txt "
                "before giving its verdict or starting another probe."
            )
        return _text_result(message)

    @tool(
        "restart_kernel",
        "Restart the Jupyter kernel. All in-memory state is lost.",
        {},
    )
    @serialize_probe_operation
    async def restart_kernel(args):
        if probe_busy():
            return _failure("An operation is still running.")
        await run_probe_task(session.kernel.restart)
        session._kernel_notice = ""
        await session.emit("kernel_restarted")
        return _text_result("Kernel restarted. In-memory state is gone.")

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
    @serialize_probe_operation
    async def install_packages(args):
        if probe_busy():
            return _failure("An operation is still running.")
        try:
            status, output, message = await run_probe_task(
                session.install_packages, args["packages"]
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return _failure(str(exc))
        await session.emit("system_message", text=message)
        return _text_result(
            output + "\nLoaded modules are unchanged; restart if needed.",
            is_error=status != "completed",
        )

    @tool(
        "submit_slurm",
        "Submit a self-contained Python probe to Slurm for long or specialized compute. "
        "Each probe must submit exactly one job on the current cluster: no job arrays "
        "or additional submissions from the script. Scheduler defaults must also select "
        "one job on the current cluster. "
        "The script cannot use the interactive kernel's variables. State a prediction, "
        "the runtime monitors it and automatically resumes this chat to review the result. "
        "Other work can continue while it runs. check_slurm_job is also available manually. "
        "The selected research Python must be available at the same path on compute nodes. "
        "Write plots as plot-1.png, plot-2.png under NOCTURNOMATH_PROBE_DIR. "
        "Declare expected plots and other outputs in result_files so collection waits for them. "
        "For a correction, set correction_of to the failed probe ID; automatic reviews supply it.",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "expected": {"type": "string"},
                "time_minutes": {"type": "integer"},
                "cpus": {"type": "integer"},
                "memory_gb": {"type": "integer"},
                "partition": {"type": "string"},
                "gpus": {"type": "integer"},
                "account": {"type": "string"},
                "qos": {"type": "string"},
                "result_files": {"type": "array", "items": {"type": "string"}},
                "correction_of": {"type": "string"},
            },
            "required": ["code", "expected", "time_minutes", "cpus", "memory_gb"],
        },
    )
    @serialize_probe_operation
    async def submit_slurm(args):
        if session._pending_verdict is not None:
            return _failure(
                "Give the last probe its verdict before submitting another."
            )
        steered = await session.consume_steer_message()
        if steered is not None:
            return steered
        if probe_busy():
            return _failure("An operation is still running.")
        session.activate_session()
        try:
            probe_id, job_id = await run_probe_task(
                slurm.submit,
                session,
                args["code"],
                args["expected"],
                args["time_minutes"],
                args["cpus"],
                args["memory_gb"],
                args.get("partition") or "",
                args.get("gpus") if args.get("gpus") is not None else 0,
                args.get("account") or "",
                args.get("qos") or "",
                args.get("result_files") or [],
                wait_on_cancel=True,
                correction_of=args.get("correction_of")
                or session._slurm_review_probe_id,
            )
        except slurm.SubmissionError as exc:
            probe_id = exc.probe_id
            probe_dir = f".nocturnomath/sessions/{probe_id.split('/')[0]}/probes/{probe_id.split('/')[1]}"
            code_path = f"{probe_dir}/code.py"
            await session.emit(
                "probe_start",
                probe_id=probe_id,
                expected=args["expected"],
                code=args["code"],
                code_path=code_path,
            )
            await session.emit(
                "probe_finish",
                **_probe_payload(
                    probe_id=probe_id,
                    expected=args["expected"],
                    code=args["code"],
                    code_path=code_path,
                    output=str(exc),
                    text=str(exc),
                    status="slurm-error",
                ),
            )
            return _failure(str(exc))
        except ValueError as exc:
            return _failure(str(exc))
        probe_dir = f".nocturnomath/sessions/{probe_id.split('/')[0]}/probes/{probe_id.split('/')[1]}"
        code_path = f"{probe_dir}/code.py"
        await session.emit(
            "probe_start",
            probe_id=probe_id,
            backend="slurm",
            expected=args["expected"],
            code=args["code"],
            code_path=code_path,
        )
        await session.emit(
            "probe_update",
            **_probe_payload(
                probe_id=probe_id,
                backend="slurm",
                job_id=job_id,
                expected=args["expected"],
                code=args["code"],
                code_path=code_path,
                output=f"Slurm job {job_id} submitted. Monitoring automatically; verdict pending.",
                text=f"Slurm job {job_id} submitted.",
                status="submitted",
            ),
        )
        return _text_result(
            f"Probe {probe_id}: Slurm job {job_id} submitted. "
            f"Script: {probe_id}/code.py. Monitoring is armed; this chat will "
            "automatically resume to inspect the output and plots when ready. "
            "You may continue other work while this probe remains unresolved. "
            "Do not give a verdict until its output is collected."
        )

    @tool(
        "check_slurm_job",
        "Check a saved Slurm probe by ID (for example S001/P001), or omit the ID "
        "to list saved jobs. This does not use the Python kernel or start a new probe. "
        "A finished job's stdout, stderr, and exit status are saved as probe output. "
        "If final files are still appearing, check again after the finishing status.",
        {
            "type": "object",
            "properties": {"probe_id": {"type": "string"}},
        },
    )
    @serialize_probe_operation
    async def check_slurm_job(args):
        probe_id = args.get("probe_id")
        if not probe_id:
            return _text_result(json.dumps(slurm.list_jobs(session.workspace)))
        if (
            session._pending_verdict is not None
            and session._pending_verdict_probe_id != probe_id
        ):
            return _failure(
                "Give the pending probe "
                f"{session._pending_verdict_probe_id} its verdict before checking "
                f"{probe_id}."
            )
        session.activate_session()
        try:
            metadata, _ = await run_probe_task(
                slurm.check,
                session.workspace,
                probe_id,
                session.transcript_path,
                wait_on_cancel=True,
            )
        except ValueError as exc:
            return _failure(str(exc))
        status = metadata["status"]
        state = metadata.get("slurm_state", "UNKNOWN")
        if status not in ("completed", "error"):
            return _text_result(
                f"Probe {probe_id}: {state} ({status}), Slurm job {metadata['job_id']}."
            )
        payload = slurm.result_payload(
            session.workspace, probe_id, metadata, session.image_cap
        )
        if session._slurm_review_probe_id != probe_id:
            await session.emit("probe_finish", **payload)
        records = session.workspace.read_transcript(
            session.workspace.transcript_file(probe_id.split("/")[0])
        )
        existing_verdict = next(
            (
                record
                for record in records
                if record.get("kind") == "verdict"
                and record.get("probe_id") == probe_id
            ),
            None,
        )
        reviewed = existing_verdict is not None
        if not reviewed and session._pending_verdict is None:
            session._pending_verdict = metadata["expected"]
            session._pending_verdict_probe_id = probe_id
        message = f"Probe {probe_id}: {status}\nSource: {probe_id}/output.txt\n{payload['text']}"
        plot_paths = payload["plot_paths"]
        if plot_paths:
            message += "\n" + "\n".join(
                f"Source: {probe_id}/{path.rsplit('/', 1)[-1]}" for path in plot_paths
            )
        if not reviewed:
            message += f"\nGive a verdict against: {metadata['expected']}"
        else:
            message += f"\nExisting verdict (preserve it): {existing_verdict['text']}"
        return {
            "content": [
                {"type": "text", "text": message},
                *_image_blocks(payload["plot_images"]),
            ]
        }

    @tool(
        "save_document",
        "Create or revise a Markdown document under .nocturnomath/documents/ without a probe. "
        "Use a relative path ending in .md, such as reports/summary.md. "
        "Set replace_existing to true only when the user clearly identifies an existing document; "
        "false creates a new document and refuses to overwrite one.",
        {"path": str, "text": str, "replace_existing": bool},
    )
    async def save_document(args):
        path = args["path"]
        replace_existing = args["replace_existing"]
        if not isinstance(replace_existing, bool):
            return _failure("replace_existing must be a boolean.")
        try:
            session.workspace.save_document(path, args["text"], replace_existing)
        except (ValueError, OSError) as exc:
            return _failure(str(exc))
        action = "revised" if replace_existing else "created"
        if not session.workspace.session_pending:
            session.log_transcript("document_saved", name=path, action=action)
            await session.emit("document_saved", name=path, action=action)
        return _text_result(f"{action.capitalize()} .nocturnomath/documents/{path}.")

    @tool(
        "list_documents",
        "List Markdown documents and folders under .nocturnomath/documents/, including "
        "documents excluded from automatic message context. Paths are relative to that folder.",
        {},
    )
    async def list_documents(args):
        try:
            raw_documents, folders = session.documents_and_folders()
            documents = [
                {"name": document["name"], "included": document["included"]}
                for document in raw_documents
            ]
        except (ValueError, OSError) as exc:
            return _failure(str(exc))
        return _text_result(json.dumps({"documents": documents, "folders": folders}))

    @tool(
        "read_document",
        "Read one Markdown document by its path relative to .nocturnomath/documents/, "
        "including documents excluded from automatic message context.",
        {"path": str},
    )
    async def read_document(args):
        try:
            document = session.workspace.read_document(args["path"])
        except (ValueError, OSError) as exc:
            return _failure(str(exc))
        return _text_result(f"{document['path']}\n\n{document['content']}")

    return [
        run,
        evidence,
        thought,
        strike_evidence,
        verdict,
        restart_kernel,
        install_packages,
        submit_slurm,
        check_slurm_job,
        save_document,
        list_documents,
        read_document,
    ]
