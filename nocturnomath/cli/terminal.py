"""Run Nocturnomath in an interactive terminal."""

import argparse
import asyncio
import getpass
import json
import logging
import signal
import time
import warnings
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.segment import Segment
from rich.table import Table
from rich.text import Text

from ..runtime import Runtime
from .common import add_session_arguments
from .completion import CommandCompleter

logger = logging.getLogger("nocturnomath.cli")

ASSET_PREFIX = "/api/asset?path="
OUTPUT_LINE_CAP = 60

COMMANDS = [
    ("/help", "", "list these commands"),
    ("/new", "", "start a new chat on a fresh kernel"),
    ("/restart", "", "restart the kernel; in-memory state is gone"),
    ("/notes", "", "show evidence.md and thoughts.md"),
    ("/history", "", "list the chats recorded in this workspace"),
    ("/resume", "<S001> [--kernel]", "reopen a chat, optionally replaying its probes"),
    ("/export", "[path]", "save the current chat as a Jupyter notebook"),
    ("/model", "[name]", "show the catalogue, or use a model from the next message"),
    (
        "/auth",
        "[status|subscription|api-key|claude-code]",
        "show or change Claude login",
    ),
    ("/context", "[on|off]", "show or set whether the agent carries chat context"),
    ("/env", "<python>|managed", "switch the research environment; starts a new chat"),
    ("/workspace", "<path>", "open a different workspace folder"),
    ("/docs", "[name]", "list the Markdown documents, or render one"),
    ("/plots", "", "list the plots saved by probes"),
    ("/exit", "", "stop the agent and leave"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_session_arguments(parser)
    return parser


def format_time(epoch_seconds: float) -> str:
    return time.strftime("%b %d %H:%M", time.localtime(epoch_seconds))


def read_credential(prompt: str) -> str:
    """Refuse getpass's echoed-input fallback when no secure terminal exists."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        try:
            return getpass.getpass(prompt)
        except getpass.GetPassWarning:
            raise ValueError(
                "Secure input is unavailable. Use an interactive terminal or the web auth form."
            ) from None


class Tail:
    """Show only the last lines of a renderable that fit on the screen.

    A live region taller than the terminal cannot be redrawn in place, so the
    streamed block would repeat itself; following its tail keeps it bounded.
    """

    def __init__(self, renderable: Any):
        self.renderable = renderable

    def __rich_console__(self, console, options):
        lines = console.render_lines(self.renderable, options, pad=False)
        keep = max(options.size.height - 2, 1)
        for line in lines[-keep:]:
            yield from line
            yield Segment.line()


class TerminalApp:
    """Render runtime events and run slash commands against the shared runtime."""

    def __init__(
        self,
        runtime: Runtime,
        console: Console | None = None,
        error_console: Console | None = None,
        python: str | None = None,
        credential_reader: Callable[[str], str] = read_credential,
    ):
        self.runtime = runtime
        self.console = console or Console()
        self.error_console = error_console or Console(stderr=True)
        self.initial_python = python
        self.credential_reader = credential_reader
        self.pending_model: str | None = None
        self.status = "idle"
        self.stopped = False
        self._interrupt_task: asyncio.Task | None = None
        self._spinner = None
        self._spinner_message: str | None = None
        self._live: Live | None = None
        self._stream_text = ""

    # ------------------------------------------------------------------ output

    def show(self, renderable: Any, console: Console | None = None, style=None):
        """Print to the scrollback, pausing the spinner and any streamed block."""
        self.hide_spinner()
        self.close_stream()
        (console or self.console).print(
            renderable, style=style, highlight=False, markup=False
        )
        self.resume_spinner()

    def write(self, text: str, style: str | None = None):
        self.show(text, style=style)

    def error(self, message: str):
        self.show(f"error: {message}", console=self.error_console, style="bold red")

    def warn(self, message: str):
        self.show(message, console=self.error_console, style="yellow")

    def markdown(self, text: str):
        self.show(Markdown(text))

    # ----------------------------------------------------------------- spinner

    def show_spinner(self, message: str):
        """Report progress transiently; the spinner never reaches the scrollback."""
        if self._stream_text:
            return
        self._spinner_message = message
        if self._spinner is None:
            self._spinner = self.console.status(message, spinner="dots")
            self._spinner.start()
        else:
            self._spinner.update(message)

    def hide_spinner(self):
        if self._spinner is not None:
            self._spinner.stop()
            self._spinner = None

    def resume_spinner(self):
        if self._spinner_message:
            self.show_spinner(self._spinner_message)

    def clear_spinner(self):
        self.hide_spinner()
        self._spinner_message = None

    # --------------------------------------------------------------- streaming

    def stream_delta(self, chunk: str):
        """Grow the block the assistant is writing; on a terminal, show its tail live."""
        if not self._stream_text:
            self.hide_spinner()
        self._stream_text += chunk
        if not self.console.is_terminal:
            return
        if self._live is None:
            self._live = Live(
                console=self.console,
                transient=True,
                auto_refresh=False,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._live.start()
        self._live.update(Tail(Markdown(self._stream_text)), refresh=True)

    def close_stream(self, final: str | None = None):
        """Replace the live tail with the block's final text, or what streamed so far."""
        streamed, self._stream_text = self._stream_text, ""
        if self._live is not None:
            live, self._live = self._live, None
            live.stop()
        text = streamed if final is None else final
        if text:
            self.console.print(Markdown(text))

    def banner(self):
        session = self.runtime.session
        self.write("Nocturnomath terminal", style="bold")
        if session:
            self.write(
                f"research Python: {session.environment.python} "
                f"({session.environment.version})",
                style="dim",
            )
            self.write(f"workspace: {session.workspace_path}", style="dim")
            self.write(f"model: {session.model}", style="dim")
            auth = self.runtime.auth.public()
            self.write(f"Claude auth: {auth['label']}", style="dim")
            if auth["note"]:
                self.write(auth["note"], style="dim")
            self.write(
                f"next transcript: {session.transcript_path} "
                "(created with first message)",
                style="dim",
            )
        self.write(
            "Enter sends; Esc then Enter (or Alt+Enter) adds a newline. "
            "Type /help for commands.",
            style="dim",
        )

    def bottom_toolbar(self) -> str:
        session = self.runtime.session
        if session is None:
            return f"no workspace · {self.status}"
        if session.kernel.busy:
            kernel = "busy"
        else:
            kernel = "ready" if session.kernel.is_alive() else "dead"
        model = session.model
        if self.pending_model:
            model += f" → {self.pending_model}"
        context = "on" if session.carry_chat_context else "off"
        workspace = session.workspace
        chat = "new" if workspace.session_pending else workspace.session_id
        return (
            f"kernel {kernel} · model {model} · auth {self.runtime.auth.method} · context {context} · "
            f"chat {chat} · {self.status}"
        )

    def show_help(self):
        table = Table(box=None, pad_edge=False, show_header=False)
        table.add_column(style="bold")
        table.add_column(style="dim")
        table.add_column()
        for name, argument, description in COMMANDS:
            table.add_row(name, argument, description)
        self.show(table)

    # ------------------------------------------------------------------ events

    def render_event(self, event_type: str, payload: dict[str, Any]):
        handler = getattr(self, f"_render_{event_type}", None)
        if handler is None:
            self.write(f"[{event_type}] {payload}", style="dim")
            return
        handler(payload)

    def _render_assistant_delta(self, payload):
        chunk = payload.get("text", "")
        if chunk:
            self.stream_delta(chunk)

    def _render_assistant_text(self, payload):
        self.hide_spinner()
        self.close_stream(final=payload.get("text", ""))
        self.resume_spinner()

    def _render_probe_start(self, payload):
        lines = len(payload.get("code", "").splitlines())
        body = Text(payload.get("expected", "") or "(no prediction)")
        body.append(
            f"\ncode: {lines} line{'' if lines == 1 else 's'} · "
            f"{self.probe_file(payload, 'code.py')}",
            style="dim",
        )
        if self._spinner_message:
            self._spinner_message = "running probe…"
        self.show(
            Panel(
                body,
                title=f"probe {payload.get('probe_id', '')}",
                title_align="left",
                border_style="cyan",
            )
        )

    def _render_probe_finish(self, payload):
        status = payload.get("status", "completed")
        text = payload.get("text")
        if text is None:
            text = payload.get("output", "")
        lines = text.rstrip().splitlines()
        body = Text("\n".join(lines[:OUTPUT_LINE_CAP]) or "(no text output)")
        hidden = len(lines) - OUTPUT_LINE_CAP
        if hidden > 0:
            body.append(
                f"\n… {hidden} more lines in {self.probe_file(payload, 'output.txt')}",
                style="dim",
            )
        note = payload.get("execution_note")
        if note:
            body.append(f"\n{note}", style="yellow")
        for plot in self.plot_paths(payload):
            body.append(f"\nsaved plot: {self.workspace_file(plot)}", style="dim")
        if self._spinner_message:
            self._spinner_message = "thinking…"
        self.show(
            Panel(
                body,
                title=f"probe {payload.get('probe_id', '')} · {status}",
                title_align="left",
                border_style="red"
                if status in ("error", "interrupted", "failed")
                else "green",
            )
        )

    def _render_record_changed(self, payload):
        self.write(
            f"[{payload.get('entry_id', '')}] {payload.get('kind', '')}: "
            f"{payload.get('text', '')}",
            style="green",
        )

    def _render_probe_verdict(self, payload):
        self.write(f"verdict: {payload.get('text', '')}", style="magenta")

    def _render_kernel_restarted(self, payload):
        self.write("Kernel restarted. In-memory variables and state cleared.")

    def _render_kernel_interrupted(self, payload):
        self.write("Kernel interrupted by user.")

    def _render_query_cancelled(self, payload):
        self.write("Query cancelled.")

    def _render_status_change(self, payload):
        self.status = payload.get("status", "idle")
        if self.status == "thinking":
            self.show_spinner("thinking…")
        else:
            self.close_stream()
            self.clear_spinner()

    def _render_turn_complete(self, payload):
        self.close_stream()

    def _render_system_message(self, payload):
        self.markdown(payload.get("text", ""))

    def _render_error(self, payload):
        self.error(payload.get("message", ""))

    def _render_user_message(self, payload):
        self.write(f"> {payload.get('text', '')}", style="dim")

    def _render_session_reset(self, payload):
        self.write("New chat on a fresh kernel; previous variables are gone.")

    def _render_session_resumed(self, payload):
        if not payload.get("carry_chat_context"):
            note = (
                "Reopened this chat. New replies are appended to it, but the agent "
                "still starts every query fresh from its prompt and evidence.md and "
                "thoughts.md."
            )
        elif payload.get("context_restored"):
            note = "Resumed this chat. The agent still has its original conversation context."
        else:
            note = (
                "Resumed this chat. The agent's original context was not available, "
                "so it will pick up from a recap of the transcript."
            )
        replayed = payload.get("probes_replayed", 0)
        if payload.get("kernel_restored"):
            note += (
                f" The kernel was reconstructed by replaying {replayed} stored "
                f"{'probe' if replayed == 1 else 'probes'} in order."
            )
        elif payload.get("kernel_reset"):
            note += " A fresh kernel is ready; previous variables are gone."
        else:
            note += " The current kernel is unchanged."
        self.write(f"{payload.get('id', '')}: {note}")

    def _render_workspace_updated(self, payload):
        workspace = payload.get("workspace") or {}
        environment = workspace.get("environment") or {}
        self.write(f"workspace: {workspace.get('path')}")
        if environment:
            self.write(
                f"research Python: {environment.get('python')} "
                f"({environment.get('version')})",
                style="dim",
            )

    def _render_carry_context_changed(self, payload):
        if payload.get("carry_chat_context"):
            self.write(
                "Keeping context from here on. The agent sees this conversation as it "
                "grows; earlier messages are not recovered."
            )
        else:
            self.write(
                "Discarding context. Every message starts fresh from the system prompt "
                "plus evidence.md and thoughts.md."
            )

    # ----------------------------------------------------------------- helpers

    def workspace_file(self, relative_path: str) -> Path:
        session = self.runtime.session
        return (
            session.workspace_path / relative_path if session else Path(relative_path)
        )

    def probe_file(self, payload: dict[str, Any], filename: str) -> Path:
        code_path = payload.get("code_path")
        if code_path:
            return self.workspace_file(str(PurePosixPath(code_path).parent / filename))
        session_id, _, probe = str(payload.get("probe_id", "")).partition("/")
        return self.workspace_file(
            f".nocturnomath/sessions/{session_id}/probes/{probe}/{filename}"
        )

    def plot_paths(self, payload: dict[str, Any]) -> list[str]:
        paths = payload.get("plot_paths")
        if paths is None:
            paths = [
                url.removeprefix(ASSET_PREFIX) for url in payload.get("plot_urls") or []
            ]
        return paths

    def history_path(self) -> Path:
        session = self.runtime.session
        root = session.workspace_path if session else Path.cwd()
        return root / ".nocturnomath" / "terminal_history"

    def require_idle(self, action: str):
        session = self.runtime.require_session()
        if self.runtime.changing or session.has_active_query():
            raise RuntimeError(f"Cannot {action} while the agent is running a query.")

    def check_model(self, model: str):
        if not self.runtime.models:
            self.write(
                f"Model catalogue unavailable; /model cannot switch from {model}.",
                style="dim",
            )

    def interrupt(self) -> bool:
        """Interrupt the kernel or cancel the query; False when nothing is running."""
        session = self.runtime.session
        if session is None:
            return False
        task = self.runtime.current_task
        if not session.kernel.busy and (task is None or task.done()):
            return False
        self._interrupt_task = asyncio.create_task(session.interrupt())
        return True

    def on_interrupt(self):
        if not self.interrupt():
            self.write("Nothing is running. Type /exit to leave.", style="dim")

    async def stop(self):
        """Drain the running query and probe, then release the kernel."""
        if self.stopped:
            return
        self.stopped = True
        self.close_stream()
        self.clear_spinner()
        await self.runtime.drain()
        self.runtime.shutdown()

    # ---------------------------------------------------------------- dispatch

    async def handle_line(self, text: str) -> bool:
        """Run one line of input. Returns False when the terminal should exit."""
        text = text.strip()
        if not text:
            return True
        if not text.startswith("/"):
            await self.run_query(text)
            return True
        name, _, argument = text.partition(" ")
        handler = getattr(self, f"_command_{name[1:]}", None)
        if handler is None:
            self.error(f"Unknown command {name}.")
            self.show_help()
            return True
        try:
            return await handler(argument.strip())
        except (RuntimeError, ValueError, FileNotFoundError, OSError) as exc:
            self.error(str(exc))
            return True

    async def run_query(self, text: str):
        try:
            self.runtime.start_query(text, self.pending_model)
        except (RuntimeError, ValueError) as exc:
            self.error(str(exc))
            return
        self.pending_model = None
        await self.runtime.wait_for_query()
        self.close_stream()
        self.clear_spinner()

    async def _command_help(self, argument: str) -> bool:
        self.show_help()
        return True

    async def _command_new(self, argument: str) -> bool:
        self.require_idle("start a new session")
        session = self.runtime.session
        await self.runtime.transition(session.reset_client_session)
        await self.runtime.emit("session_reset", {})
        return True

    async def _command_restart(self, argument: str) -> bool:
        self.require_idle("restart the kernel")
        session = self.runtime.session
        await self.runtime.transition(session.kernel.restart)
        await self.runtime.emit(
            "kernel_restarted", {"kernel_alive": session.kernel.is_alive()}
        )
        return True

    async def _command_notes(self, argument: str) -> bool:
        session = self.runtime.require_session()
        self.markdown(session.workspace.notes.read())
        return True

    async def _command_history(self, argument: str) -> bool:
        session = self.runtime.require_session()
        sessions = session.list_sessions()
        if not sessions:
            self.write("No past chats recorded in this workspace yet.", style="dim")
            return True
        table = Table(box=None, pad_edge=False)
        for column in ("id", "started", "modified", "turns", "probes", "notes", ""):
            table.add_column(column)
        for record in sessions:
            flags = []
            if record["is_current"]:
                flags.append("current")
            if session.carry_chat_context and not record["context_restorable"]:
                flags.append("context rebuilt from transcript")
            table.add_row(
                record["id"],
                format_time(record["started"]),
                format_time(record["modified"]),
                str(record["turns"]),
                str(record["probes"]),
                str(record["notes"]),
                ", ".join(flags),
            )
        self.show(table)
        for record in sessions:
            self.write(f"{record['id']}: {record['title']}", style="dim")
        return True

    async def _command_resume(self, argument: str) -> bool:
        parts = argument.split()
        restore_kernel = "--kernel" in parts
        ids = [part for part in parts if not part.startswith("-")]
        if not ids:
            self.error("Name a chat to resume, such as /resume S001.")
            return True
        self.require_idle("resume")
        session = self.runtime.session
        data = await self.runtime.transition(
            session.resume_session, ids[0], restore_kernel
        )
        await self.runtime.emit("session_resumed", data)
        return True

    async def _command_export(self, argument: str) -> bool:
        self.require_idle("download the current session")
        filename, notebook = self.runtime.export_notebook()
        target = Path(argument).expanduser() if argument else Path.cwd()
        if target.is_dir():
            target = target / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
        self.write(f"Saved {target}")
        return True

    async def _command_model(self, argument: str) -> bool:
        runtime = self.runtime
        if not argument:
            current = runtime.session.model if runtime.session else runtime.model
            self.write(f"model: {current}")
            if self.pending_model:
                self.write(
                    f"{self.pending_model} applies from the next message.", style="dim"
                )
            for name in runtime.models:
                self.write(f"  {name} ({runtime.model_labels.get(name, name)})")
            if not runtime.models:
                self.write("The model catalogue is unavailable.", style="dim")
            return True
        if not runtime.models:
            self.error(
                "The model catalogue is unavailable, so the model cannot be changed."
            )
            return True
        if argument not in runtime.models:
            self.error(
                f"Choose a model from the catalogue: {', '.join(runtime.models)}."
            )
            return True
        self.pending_model = argument
        self.write(f"Using {argument} from the next message.")
        return True

    async def _command_auth(self, argument: str) -> bool:
        choice = argument or "status"
        if choice == "status":
            auth = self.runtime.auth.public()
            self.write(f"Claude authentication: {auth['label']}")
            if auth["note"]:
                self.write(auth["note"], style="dim")
            self.write(
                "Use /auth subscription, /auth api-key, or /auth claude-code to change it.",
                style="dim",
            )
            return True
        methods = {
            "subscription": "subscription",
            "api-key": "api_key",
            "claude-code": "claude_code",
        }
        if choice not in methods:
            self.error("Use /auth subscription, /auth api-key, or /auth claude-code.")
            return True
        self.require_idle("change Claude authentication")
        method = methods[choice]
        credential = None
        if method != "claude_code":
            prompt = (
                "Claude subscription token (from `claude setup-token`): "
                if method == "subscription"
                else "Claude API key: "
            )
            try:
                credential = await asyncio.to_thread(self.credential_reader, prompt)
            except EOFError:
                self.write("Authentication unchanged.", style="dim")
                return True
        auth = await self.runtime.authenticate(method, credential)
        self.write(f"Selected {auth['label']}.")
        self.write("Credentials will be checked on your next message.", style="dim")
        models = self.runtime.models
        if self.pending_model and models and self.pending_model not in models:
            self.write(
                f"Dropped pending model {self.pending_model}: not in the new catalogue.",
                style="dim",
            )
            self.pending_model = None
        self.check_model(self.runtime.model)
        return True

    async def _command_context(self, argument: str) -> bool:
        session = self.runtime.require_session()
        if not argument:
            state = "on" if session.carry_chat_context else "off"
            self.write(f"carry chat context: {state}")
            return True
        if argument not in ("on", "off"):
            self.error("Use /context on or /context off.")
            return True
        enabled = session.set_carry_chat_context(argument == "on")
        await self.runtime.emit(
            "carry_context_changed", {"carry_chat_context": enabled}
        )
        return True

    async def _command_env(self, argument: str) -> bool:
        if not argument:
            self.error("Name a Python executable, or managed.")
            return True
        session = self.runtime.require_session()
        await self.runtime.transition(
            self.runtime.open_workspace, session.workspace_path, argument
        )
        self.initial_python = None
        self.write("New research environment; this starts a new chat.")
        await self.runtime.emit(
            "workspace_updated", {"workspace": self.runtime.workspace_snapshot()}
        )
        return True

    async def _command_workspace(self, argument: str) -> bool:
        if not argument:
            self.error("Name the workspace folder to open.")
            return True
        await self.runtime.transition(
            self.runtime.open_workspace, argument, self.initial_python
        )
        self.initial_python = None
        await self.runtime.emit(
            "workspace_updated", {"workspace": self.runtime.workspace_snapshot()}
        )
        return True

    async def _command_docs(self, argument: str) -> bool:
        session = self.runtime.require_session()
        if not argument:
            files = session.list_markdown_files()
            if not files:
                self.write("No Markdown documents in this workspace yet.", style="dim")
                return True
            for document in files:
                self.write(
                    f"{document['path']} ({document['size']} bytes, "
                    f"{format_time(document['modified'])})"
                )
            return True
        document = session.read_file(argument)
        self.write(document["path"], style="bold")
        self.markdown(document["content"])
        return True

    async def _command_plots(self, argument: str) -> bool:
        session = self.runtime.require_session()
        plots = session.list_plots()
        if not plots:
            self.write("No plots saved in this workspace yet.", style="dim")
            return True
        for plot in plots:
            relative = plot["url"].removeprefix(ASSET_PREFIX)
            self.write(f"{plot['filename']}  {self.workspace_file(relative)}")
        return True

    async def _command_exit(self, argument: str) -> bool:
        await self.stop()
        self.write("Stopped.", style="dim")
        return False


def build_prompt_session(app: TerminalApp) -> PromptSession:
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def insert_newline(event):
        event.current_buffer.insert_text("\n")

    @bindings.add("enter")
    def submit(event):
        event.current_buffer.validate_and_handle()

    history_path = app.history_path()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    return PromptSession(
        history=FileHistory(str(history_path)),
        multiline=True,
        key_bindings=bindings,
        completer=CommandCompleter(app.runtime, COMMANDS),
        complete_while_typing=True,
        bottom_toolbar=app.bottom_toolbar,
    )


async def input_loop(app: TerminalApp):
    prompt_session = build_prompt_session(app)
    while True:
        try:
            with patch_stdout():
                line = await prompt_session.prompt_async("\n> ")
        except KeyboardInterrupt:
            app.write("Type /exit to leave.", style="dim")
            continue
        except EOFError:
            return
        if not await app.handle_line(line):
            return


async def run_terminal(args: argparse.Namespace):
    console = Console()
    app = TerminalApp(
        Runtime(
            model=args.model,
            timeout_s=args.timeout,
            image_cap=args.images,
            navigator_root=args.path,
        ),
        console=console,
        python=args.python,
    )
    app.runtime.start()
    app.runtime.subscribe(app.render_event)
    try:
        app.runtime.open_workspace(args.path, args.python)
    except Exception as exc:
        app.error(str(exc))
        return
    app.initial_python = None
    await app.runtime.discover_models()
    app.check_model(args.model)
    app.banner()

    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, app.on_interrupt)
    except NotImplementedError:
        logger.debug("This platform has no asyncio SIGINT handler.")
    try:
        await input_loop(app)
    finally:
        try:
            loop.remove_signal_handler(signal.SIGINT)
        except NotImplementedError:
            pass
        await app.stop()


def main():
    args = build_parser().parse_args()
    try:
        asyncio.run(run_terminal(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
