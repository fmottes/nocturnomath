import json
from io import StringIO
from unittest.mock import patch

import pytest
from conftest import FakeEnvironment, FakeKernel
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from rich.console import Console

from nocturnomath.cli.completion import CommandCompleter
from nocturnomath.cli.terminal import COMMANDS, TerminalApp
from nocturnomath.cli.terminal import build_parser as terminal_parser
from nocturnomath.cli.web import build_parser as web_parser
from nocturnomath.runtime import Runtime


def test_web_command_options():
    args = web_parser().parse_args(
        ["--path", "/tmp/project", "--timeout", "10", "--images", "1", "--port", "9000"]
    )
    assert args.path == "/tmp/project"
    assert args.timeout == 10
    assert args.images == 1
    assert args.port == 9000


def test_terminal_command_uses_shared_options():
    args = terminal_parser().parse_args(
        ["--path", "/tmp/project", "--model", "test", "--timeout", "20"]
    )
    assert args.path == "/tmp/project"
    assert args.model == "test"
    assert args.timeout == 20


class Harness:
    """Drive a TerminalApp and read back everything it printed."""

    def __init__(self, app: TerminalApp, buffer: StringIO):
        self.app = app
        self.buffer = buffer
        self.runtime = app.runtime
        self.session = app.runtime.session

    async def run(self, line: str) -> bool:
        return await self.app.handle_line(line)

    def output(self) -> str:
        text = self.buffer.getvalue()
        self.buffer.seek(0)
        self.buffer.truncate(0)
        return text


@pytest.fixture
def terminal(session):
    with (
        patch("nocturnomath.session.Kernel", FakeKernel),
        patch("nocturnomath.session.ResearchEnvironment", FakeEnvironment),
    ):
        buffer = StringIO()
        console = Console(file=buffer, force_terminal=False, width=300)
        runtime = Runtime(session)
        runtime.start()
        app = TerminalApp(runtime, console=console, error_console=console)
        runtime.subscribe(app.render_event)
        yield Harness(app, buffer)
        app.close_stream()
        app.clear_spinner()


CODE_PATH = ".nocturnomath/sessions/S001/probes/P001/code.py"
PLOT_PATH = ".nocturnomath/sessions/S001/probes/P001/plot-1.png"

EVENT_PAYLOAD = {
    "probe_id": "S001/P001",
    "expected": "the mean is 3",
    "code": "print(mean)",
    "code_path": CODE_PATH,
    "output": f"Note: nagging\nProbe S001/P001\nSource: {CODE_PATH}\n3.0\n",
    "plot_paths": [PLOT_PATH],
    "plot_urls": [f"/api/asset?path={PLOT_PATH}"],
    "plot_images": ["ignored-base64"],
    "entry_id": "E001",
    "kind": "evidence",
    "text": "3.0\n",
    "message": "something failed",
    "status": "thinking",
    "id": "S001",
    "carry_chat_context": True,
    "context_restored": True,
    "kernel_restored": True,
    "kernel_reset": True,
    "probes_replayed": 2,
    "workspace": {
        "path": "/tmp/workspace",
        "environment": {"python": "/tmp/venv/bin/python", "version": "3.12"},
    },
}

EVENT_TYPES = [
    "assistant_delta",
    "assistant_text",
    "probe_start",
    "probe_finish",
    "record_changed",
    "probe_verdict",
    "kernel_restarted",
    "kernel_interrupted",
    "query_cancelled",
    "status_change",
    "turn_complete",
    "system_message",
    "error",
    "user_message",
    "session_reset",
    "session_resumed",
    "workspace_updated",
    "carry_context_changed",
]


SILENT_EVENTS = ["assistant_delta", "status_change", "turn_complete"]


@pytest.mark.parametrize(
    "event_type",
    [*[name for name in EVENT_TYPES if name not in SILENT_EVENTS], "mystery_event"],
)
def test_render_event_handles_every_event_type(terminal, event_type):
    terminal.app.render_event(event_type, dict(EVENT_PAYLOAD))
    assert terminal.output().strip()


@pytest.mark.parametrize("event_type", SILENT_EVENTS)
def test_progress_events_leave_nothing_in_the_scrollback(terminal, event_type):
    terminal.app.render_event(event_type, dict(EVENT_PAYLOAD))
    assert terminal.output() == ""


def test_every_renderer_is_exercised_by_the_event_contract():
    handlers = {
        name.removeprefix("_render_")
        for name in dir(TerminalApp)
        if name.startswith("_render_")
    }
    assert handlers == set(EVENT_TYPES)


def test_render_event_shows_the_details_a_reader_needs(terminal):
    terminal.app.render_event("system_message", {"text": "The mean is 3."})
    assert "The mean is 3." in terminal.output()

    terminal.app.render_event("probe_start", dict(EVENT_PAYLOAD))
    started = terminal.output()
    assert "S001/P001" in started and "the mean is 3" in started
    assert "print(mean)" not in started
    assert "1 line" in started
    assert str(terminal.session.workspace_path / CODE_PATH) in started

    terminal.app.render_event("probe_finish", dict(EVENT_PAYLOAD))
    finished = terminal.output()
    assert "3.0" in finished
    assert "Note:" not in finished and "Source:" not in finished
    assert str(terminal.session.workspace_path / PLOT_PATH) in finished
    assert "ignored-base64" not in finished

    terminal.app.render_event(
        "record_changed",
        {"entry_id": "E001", "kind": "evidence", "text": "The mean is 3."},
    )
    assert "[E001] evidence: The mean is 3." in terminal.output()

    terminal.app.render_event("session_resumed", dict(EVENT_PAYLOAD))
    resumed = terminal.output()
    assert "S001:" in resumed and "replaying 2 stored probes" in resumed

    terminal.app.render_event("status_change", {"status": "running"})
    assert terminal.output() == ""
    assert terminal.app.status == "running"

    terminal.app.render_event("mystery_event", {"detail": "unknown"})
    assert "mystery_event" in terminal.output()


@pytest.mark.asyncio
async def test_help_lists_every_command_and_unknown_commands_show_it(terminal):
    assert await terminal.run("/help") is True
    listed = terminal.output()
    for name, _, description in COMMANDS:
        assert name in listed and description in listed

    assert await terminal.run("/nope") is True
    unknown = terminal.output()
    assert "Unknown command /nope." in unknown
    assert "/resume" in unknown


@pytest.mark.asyncio
async def test_blank_input_does_nothing(terminal):
    assert await terminal.run("   ") is True
    assert terminal.output() == ""


@pytest.mark.asyncio
async def test_notes_renders_the_record(terminal):
    workspace = terminal.session.workspace
    probe = workspace.start_probe("print(3)", "3", "test")
    workspace.finish_probe(probe, "3\n", [], "completed", None)
    sources = workspace.evidence_sources(["S001/P001/output.txt"])
    workspace.notes.add_evidence("The mean is 3.", sources)
    workspace.notes.add_thought("[E001] looks stable.", [])

    await terminal.run("/notes")
    printed = terminal.output()
    assert "The mean is 3." in printed
    assert "looks stable." in printed


@pytest.mark.asyncio
async def test_new_and_restart_rebuild_the_kernel(terminal):
    await terminal.run("/new")
    assert terminal.session.kernel.restarts == 1
    assert "New chat on a fresh kernel" in terminal.output()

    await terminal.run("/restart")
    assert terminal.session.kernel.restarts == 2
    assert "Kernel restarted." in terminal.output()


@pytest.mark.asyncio
async def test_history_lists_the_recorded_chats(terminal):
    await terminal.run("/history")
    assert "No past chats" in terminal.output()

    terminal.session.log_transcript("user", text="how large is the effect?")
    await terminal.run("/history")
    listed = terminal.output()
    assert "S001" in listed
    assert "how large is the effect?" in listed
    assert "current" in listed


@pytest.mark.asyncio
async def test_resume_reopens_a_chat_and_can_replay_its_probes(terminal):
    session = terminal.session
    session.log_transcript("user", text="question")
    session.log_transcript("probe_started", probe_id="S001/P001", code="value = 3")
    session.reset_client_session()
    terminal.output()

    await terminal.run("/resume")
    assert "Name a chat to resume" in terminal.output()

    await terminal.run("/resume S001")
    resumed = terminal.output()
    assert "S001:" in resumed
    assert session.workspace.session_id == "S001"
    assert session.kernel.executed_codes == []

    await terminal.run("/resume S001 --kernel")
    assert session.kernel.executed_codes == ["value = 3"]
    assert "replaying 1 stored probe" in terminal.output()


@pytest.mark.asyncio
async def test_export_writes_a_notebook_and_reports_an_empty_chat(terminal, tmp_path):
    await terminal.run("/export")
    assert "no messages to download" in terminal.output()

    terminal.session.log_transcript("user", text="question")
    destination = tmp_path / "exports"
    destination.mkdir()
    await terminal.run(f"/export {destination}")

    expected = destination / f"nocturnomath-{tmp_path.name}-S001.ipynb"
    assert str(expected) in terminal.output()
    assert json.loads(expected.read_text())["cells"]


@pytest.mark.asyncio
async def test_model_needs_a_catalogue_and_applies_to_the_next_message(terminal):
    await terminal.run("/model sonnet")
    assert "catalogue is unavailable" in terminal.output()

    terminal.runtime.models = ["sonnet"]
    terminal.runtime.model_labels = {"sonnet": "claude-sonnet-5"}
    await terminal.run("/model")
    assert "claude-sonnet-5" in terminal.output()

    await terminal.run("/model haiku")
    assert "Choose a model from the catalogue" in terminal.output()

    await terminal.run("/model sonnet")
    assert terminal.app.pending_model == "sonnet"
    assert "Using sonnet from the next message." in terminal.output()

    async def query(text):
        await terminal.session.emit("assistant_text", text=f"answer to {text}")

    terminal.session.query = query
    await terminal.run("how large is the effect?")

    assert "answer to how large is the effect?" in terminal.output()
    assert terminal.session.model == "sonnet"
    assert terminal.app.pending_model is None
    records = terminal.session.workspace.read_transcript(
        terminal.session.transcript_path
    )
    assert any(
        record["kind"] == "meta" and record.get("model") == "sonnet"
        for record in records
    )


@pytest.mark.asyncio
async def test_context_toggles_and_refuses_while_busy(terminal):
    await terminal.run("/context")
    assert "carry chat context: on" in terminal.output()

    await terminal.run("/context maybe")
    assert "/context on or /context off" in terminal.output()

    await terminal.run("/context off")
    assert terminal.session.carry_chat_context is False
    assert "Discarding context." in terminal.output()

    await terminal.run("/context on")
    assert terminal.session.carry_chat_context is True
    assert "Keeping context from here on." in terminal.output()

    terminal.session._is_busy = True
    await terminal.run("/context off")
    assert terminal.session.carry_chat_context is True
    assert "while the agent is running a query." in terminal.output()


@pytest.mark.asyncio
async def test_workspace_and_environment_switch_in_place(terminal, tmp_path):
    other = tmp_path / "other-workspace"
    other.mkdir()

    await terminal.run("/workspace")
    assert "Name the workspace folder" in terminal.output()

    await terminal.run(f"/workspace {other}")
    assert terminal.session.workspace_path == other
    assert str(other) in terminal.output()

    await terminal.run("/env")
    assert "Name a Python executable" in terminal.output()

    await terminal.run("/env managed")
    switched = terminal.output()
    assert "New research environment" in switched
    assert terminal.session.workspace_path == other


@pytest.mark.asyncio
async def test_docs_lists_and_renders_markdown(terminal, tmp_path):
    (tmp_path / "report.md").write_text("# Report\n\nThe effect is small.\n")

    await terminal.run("/docs")
    assert "report.md" in terminal.output()

    await terminal.run("/docs report.md")
    rendered = terminal.output()
    assert "Report" in rendered
    assert "The effect is small." in rendered

    await terminal.run("/docs missing.md")
    assert "missing.md not found" in terminal.output()


@pytest.mark.asyncio
async def test_plots_lists_saved_figures(terminal):
    await terminal.run("/plots")
    assert "No plots saved" in terminal.output()

    workspace = terminal.session.workspace
    probe = workspace.start_probe("plot()", "a plot", "test")
    workspace.finish_probe(probe, "", [b"png-data"], "completed", None)

    await terminal.run("/plots")
    listed = terminal.output()
    assert "S001/P001/plot-1.png" in listed
    assert str(workspace.path / ".nocturnomath/sessions/S001") in listed


@pytest.mark.asyncio
async def test_exit_drains_the_runtime_and_leaves(terminal):
    assert await terminal.run("/exit") is False
    assert terminal.app.stopped is True
    assert terminal.session.kernel.alive is False
    assert "Stopped." in terminal.output()

    assert await terminal.run("/exit") is False


@pytest.mark.asyncio
async def test_mutations_are_refused_while_a_query_runs(terminal):
    terminal.session.log_transcript("user", text="question")
    terminal.session.reset_client_session()
    terminal.output()
    terminal.session._is_busy = True

    expected = {
        "/new": "Cannot start a new session while the agent is running a query.",
        "/restart": "Cannot restart the kernel while the agent is running a query.",
        "/resume S001": "Cannot resume while the agent is running a query.",
        "/export": "Cannot download the current session while the agent is running a query.",
        "/context off": "Cannot change context settings while the agent is running a query.",
    }
    for command, message in expected.items():
        await terminal.run(command)
        assert message in terminal.output(), command

    await terminal.run("/restart")
    assert (
        "Cannot restart the kernel while the agent is running a query."
        in terminal.output()
    )
    assert terminal.session.kernel.restarts == 1


def test_streamed_text_replaces_the_deltas_it_was_built_from(terminal):
    for chunk in ("Hello ", "world"):
        terminal.app.render_event("assistant_delta", {"text": chunk})
    assert terminal.output() == ""

    terminal.app.render_event("assistant_text", {"text": "Hello world"})
    streamed = terminal.output()
    assert streamed.count("Hello world") == 1

    terminal.app.render_event("assistant_text", {"text": "Second block"})
    assert terminal.output().count("Second block") == 1


def test_a_cancelled_turn_keeps_the_text_streamed_so_far(terminal):
    terminal.app.render_event("status_change", {"status": "thinking"})
    terminal.app.render_event("assistant_delta", {"text": "Partial ans"})
    assert terminal.output() == ""
    terminal.app.render_event("status_change", {"status": "cancelled"})
    assert terminal.output().count("Partial ans") == 1
    terminal.app.render_event("turn_complete", {})
    assert terminal.output() == ""


def test_probe_panels_collapse_the_code_and_cap_the_output(terminal):
    payload = dict(EVENT_PAYLOAD)
    payload["code"] = "a = 1\nb = 2\n"
    terminal.app.render_event("probe_start", payload)
    started = terminal.output()
    assert "2 lines" in started
    assert "a = 1" not in started

    payload["text"] = "\n".join(f"line {number}" for number in range(1, 101))
    payload["status"] = "error"
    payload["execution_note"] = "Execution stopped after 600 seconds."
    terminal.app.render_event("probe_finish", payload)
    finished = terminal.output()
    assert "error" in finished
    assert "Execution stopped after 600 seconds." in finished
    assert "line 60" in finished and "line 61" not in finished
    output_path = terminal.session.workspace_path / (
        ".nocturnomath/sessions/S001/probes/P001/output.txt"
    )
    assert f"… 40 more lines in {output_path}" in finished


def test_probe_finish_falls_back_to_the_agent_facing_output(terminal):
    payload = dict(EVENT_PAYLOAD)
    payload.pop("text")
    terminal.app.render_event("probe_finish", payload)
    assert "Note: nagging" in terminal.output()


def test_bottom_toolbar_reports_the_live_state(terminal):
    resting = terminal.app.bottom_toolbar()
    assert "kernel ready" in resting
    assert f"model {terminal.session.model}" in resting
    assert "context on" in resting
    assert "chat new" in resting
    assert resting.endswith("idle")

    terminal.session.activate_session()
    terminal.session.set_carry_chat_context(False)
    terminal.app.pending_model = "sonnet"
    terminal.session.kernel.busy = True
    terminal.app.render_event("status_change", {"status": "thinking"})
    busy = terminal.app.bottom_toolbar()
    assert "kernel busy" in busy
    assert "→ sonnet" in busy
    assert "context off" in busy
    assert "chat S001" in busy
    assert busy.endswith("thinking")

    terminal.app.render_event("status_change", {"status": "idle"})
    terminal.session.kernel.busy = False
    terminal.session.kernel.alive = False
    assert "kernel dead" in terminal.app.bottom_toolbar()

    terminal.runtime.session = None
    assert terminal.app.bottom_toolbar() == "no workspace · idle"


@pytest.fixture
def completer(terminal):
    return CommandCompleter(terminal.runtime, COMMANDS)


def complete(completer, text: str) -> list[tuple[str, str]]:
    document = Document(text, len(text))
    return [
        (completion.text, completion.display_meta_text)
        for completion in completer.get_completions(document, CompleteEvent())
    ]


def test_completer_suggests_commands_with_their_description(completer):
    suggestions = dict(complete(completer, "/re"))
    assert set(suggestions) == {"/restart", "/resume"}
    assert suggestions["/restart"] == "restart the kernel; in-memory state is gone"
    assert complete(completer, "how large is the effect?") == []


def test_completer_suggests_command_arguments(completer, terminal):
    assert [text for text, _ in complete(completer, "/context ")] == ["on", "off"]
    assert [text for text, _ in complete(completer, "/context o")] == ["on", "off"]

    terminal.runtime.models = ["sonnet", "opus"]
    terminal.runtime.model_labels = {"sonnet": "claude-sonnet-5"}
    assert complete(completer, "/model son") == [("sonnet", "claude-sonnet-5")]

    terminal.session.log_transcript("user", text="how large is the effect?")
    resumable = dict(complete(completer, "/resume "))
    assert resumable["S001"] == "how large is the effect?"
    assert "--kernel" in resumable

    (terminal.session.workspace_path / "report.md").write_text("# Report\n")
    assert "report.md" in [text for text, _ in complete(completer, "/docs report")]


def test_completer_suggests_paths_for_env_and_workspace(completer, terminal):
    (terminal.session.workspace_path / "other-workspace").mkdir()
    (terminal.session.workspace_path / "workspace with spaces").mkdir()
    (terminal.session.workspace_path / "report with spaces.md").write_text("# Report\n")
    assert "managed" in [text for text, _ in complete(completer, "/env man")]

    root = terminal.session.workspace_path
    document = Document(f"/workspace {root}/other")
    completions = list(completer.get_completions(document, CompleteEvent()))
    assert [completion.text for completion in completions] == ["-workspace"]
    assert completions[0].display_text.startswith("other-workspace")

    document = Document(f"/workspace {root}/workspace with")
    completions = list(completer.get_completions(document, CompleteEvent()))
    assert [completion.text for completion in completions] == [" spaces"]
    assert completions[0].display_text.startswith("workspace with spaces")

    assert "report with spaces.md" in [
        text for text, _ in complete(completer, "/docs report with")
    ]
