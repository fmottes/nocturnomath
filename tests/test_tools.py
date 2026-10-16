import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from nocturnomath import slurm
from nocturnomath.tools import builds_machinery, switches_backend
from nocturnomath.workspace import Workspace


def tools_by_name(session):
    return {tool.name: tool for tool in session.tools}


async def run_inline(function, *args):
    return function(*args)


def test_code_policy_checks():
    assert switches_backend("matplotlib.use('TkAgg')")
    assert not switches_backend("%matplotlib inline")
    assert builds_machinery("class Model:\n    pass")
    assert not builds_machinery("def load(): pass\ndef plot(): pass")


@pytest.mark.asyncio
async def test_document_tools_use_nested_paths_without_running_a_probe(session):
    tools = tools_by_name(session)
    save = tools["save_document"].handler
    result = await save(
        {
            "path": "reports/2026/result.md",
            "text": "# Result",
            "replace_existing": False,
        }
    )
    assert (
        "Created .nocturnomath/documents/reports/2026/result.md"
        in result["content"][0]["text"]
    )
    assert not session.workspace.probes_path.exists()
    assert session.kernel.executed_codes == []
    assert session._pending_verdict is None

    session.set_document_included("reports/2026/result.md", False)
    listed = await tools["list_documents"].handler({})
    data = json.loads(listed["content"][0]["text"])
    assert data["folders"] == ["reports", "reports/2026"]
    assert data["documents"][0]["name"] == "reports/2026/result.md"
    assert data["documents"][0]["included"] is False
    read = await tools["read_document"].handler({"path": "reports/2026/result.md"})
    assert "# Result\n" in read["content"][0]["text"]

    duplicate = await save(
        {"path": "reports/2026/result.md", "text": "No", "replace_existing": False}
    )
    assert duplicate["is_error"] is True
    revised = await save(
        {
            "path": "reports/2026/result.md",
            "text": "# Revised",
            "replace_existing": True,
        }
    )
    assert "Revised" in revised["content"][0]["text"]
    assert (
        session.workspace.read_document("reports/2026/result.md")["content"]
        == "# Revised\n"
    )
    missing = await tools["read_document"].handler({"path": "missing.md"})
    assert missing["is_error"] is True
    for path in ("../kb/evidence.md", ".nocturnomath/kb/evidence.md"):
        assert (await tools["read_document"].handler({"path": path}))["is_error"]
        assert (
            await save({"path": path, "text": "unsafe", "replace_existing": False})
        )["is_error"]
    assert session.workspace.notes.evidence_path.read_text().startswith("# Evidence")


@pytest.mark.asyncio
async def test_agent_document_save_emits_and_records_chat_card_event(session):
    session.log_transcript("user", text="Save a report")
    events = []
    session.subscribe(lambda kind, payload: events.append((kind, payload)))
    save = tools_by_name(session)["save_document"].handler

    await save(
        {"path": "reports/result.md", "text": "# Result", "replace_existing": False}
    )
    await save(
        {"path": "reports/result.md", "text": "# Updated", "replace_existing": True}
    )
    failed = await save(
        {"path": "missing.md", "text": "No file", "replace_existing": True}
    )

    assert failed["is_error"] is True
    assert [
        (payload["name"], payload["action"])
        for kind, payload in events
        if kind == "document_saved"
    ] == [
        ("reports/result.md", "created"),
        ("reports/result.md", "revised"),
    ]
    records = session.workspace.read_transcript(session.transcript_path)
    assert [
        (record["name"], record["action"])
        for record in records
        if record["kind"] == "document_saved"
    ] == [
        ("reports/result.md", "created"),
        ("reports/result.md", "revised"),
    ]
    assert list(session.workspace.probes_path.iterdir()) == []
    assert session.kernel.executed_codes == []


@pytest.mark.asyncio
async def test_run_verdict_and_evidence_contract(session, monkeypatch):
    monkeypatch.setattr("nocturnomath.tools.asyncio.to_thread", run_inline)
    events = []
    session.subscribe(lambda event_type, payload: events.append((event_type, payload)))
    tools = tools_by_name(session)

    missing = await tools["run"].handler({"code": "1 + 1", "expected": ""})
    assert missing["is_error"] is True

    result = await tools["run"].handler(
        {"code": "print(2)", "expected": "the output is 2"}
    )
    assert "result" in result["content"][0]["text"]
    assert session._pending_verdict == "the output is 2"
    blocked = await tools["run"].handler(
        {"code": "print(3)", "expected": "the output is 3"}
    )
    assert blocked["is_error"] is True

    verdict = await tools["verdict"].handler({"text": "It matched."})
    assert verdict["content"][0]["text"] == "Verdict recorded."
    assert session._pending_verdict is None

    await tools["evidence"].handler(
        {"text": "the output is 2", "sources": ["S001/P001/output.txt"]}
    )
    assert "the output is 2" in session.workspace.notes.evidence_path.read_text()
    assert [event[0] for event in events] == [
        "probe_start",
        "probe_finish",
        "probe_verdict",
        "record_changed",
    ]
    start = next(payload for name, payload in events if name == "probe_start")
    finish = next(payload for name, payload in events if name == "probe_finish")
    assert start["code_path"] == ".nocturnomath/sessions/S001/probes/P001/code.py"
    assert finish["code_path"] == start["code_path"]
    assert finish["status"] == "completed"
    assert finish["text"] == "result"
    assert finish["execution_note"] is None
    assert finish["plot_paths"] == []
    assert "Source:" in finish["output"]


@pytest.mark.asyncio
async def test_slurm_job_is_saved_checked_and_cited_without_kernel(
    session, monkeypatch
):
    commands = []
    submission_env = []
    state = "RUNNING"

    def scheduler(argv, workspace, env=None):
        commands.append(argv)
        if argv[0] == "sbatch":
            submission_env.append(env)
            return SimpleNamespace(returncode=0, stdout="12345\n", stderr="")
        if argv[0] == "squeue":
            return SimpleNamespace(
                returncode=0,
                stdout="RUNNING\n" if state == "RUNNING" else "",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="12345|COMPLETED|0:0\n", stderr="")

    monkeypatch.setattr("nocturnomath.slurm._command", scheduler)
    tools = tools_by_name(session)
    submit = await tools["submit_slurm"].handler(
        {
            "code": "print(42)",
            "expected": "stdout contains 42",
            "time_minutes": 30,
            "cpus": 2,
            "memory_gb": 4,
            "account": "project",
            "qos": "normal",
            "result_files": ["results.csv"],
        }
    )
    assert "12345" in submit["content"][0]["text"]
    probe = session.workspace.probes_path / "P001"
    assert (probe / "code.py").read_text() == "print(42)"
    assert "NOCTURNOMATH_PROBE_DIR" in (probe / "job.sh").read_text()
    assert json.loads((probe / "probe.json").read_text())["job_id"] == "12345"
    assert session.kernel.executed_codes == []

    pending = await tools["check_slurm_job"].handler({"probe_id": "S001/P001"})
    assert "RUNNING" in pending["content"][0]["text"]
    assert not (probe / "output.txt").exists()

    state = "COMPLETED"
    (probe / "stdout.txt").write_text("42\n")
    (probe / "stderr.txt").write_text("")
    (probe / "results.csv").write_text("value\n42\n")
    (probe / "plot-1.png").write_bytes(b"plot")
    finished = await tools["check_slurm_job"].handler({"probe_id": "S001/P001"})
    assert "S001/P001/output.txt" in finished["content"][0]["text"]
    assert finished["content"][1]["type"] == "image"
    assert "42" in (probe / "output.txt").read_text()
    assert "S001/P001/results.csv" in (probe / "output.txt").read_text()
    assert json.loads((probe / "probe.json").read_text())["status"] == "completed"
    assert session._pending_verdict == "stdout contains 42"
    await tools["verdict"].handler({"text": "The output matched."})
    again = await tools["check_slurm_job"].handler({"probe_id": "S001/P001"})
    assert "42" in again["content"][0]["text"]
    assert session._pending_verdict is None
    listed = await tools["check_slurm_job"].handler({})
    assert json.loads(listed["content"][0]["text"])[0]["probe_id"] == "S001/P001"
    evidence = await tools["evidence"].handler(
        {"text": "The job printed 42.", "sources": ["S001/P001/output.txt"]}
    )
    assert not evidence.get("is_error")
    assert session.kernel.executed_codes == []
    records = session.workspace.read_transcript(session.transcript_path)
    assert session._replay_probe_code(records) == 0
    assert session.kernel.executed_codes == []
    assert [command[0] for command in commands] == [
        "sbatch",
        "squeue",
        "squeue",
        "sacct",
    ]
    assert commands[0][commands[0].index("--account") + 1] == "project"
    assert commands[0][commands[0].index("--qos") + 1] == "normal"
    assert "--export=ALL" in commands[0]
    assert submission_env == [session.environment.process_env()]


@pytest.mark.asyncio
async def test_slurm_waits_for_delayed_files_before_saving_outcome(
    session, monkeypatch
):
    def scheduler(argv, workspace, env=None):
        if argv[0] == "sbatch":
            return SimpleNamespace(returncode=0, stdout="91\n", stderr="")
        if argv[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="91|COMPLETED|0:0\n", stderr="")

    monkeypatch.setattr("nocturnomath.slurm._command", scheduler)
    tools = tools_by_name(session)
    await tools["submit_slurm"].handler(
        {
            "code": "print(42)",
            "expected": "42",
            "time_minutes": 10,
            "cpus": 1,
            "memory_gb": 1,
            "result_files": ["result.csv"],
        }
    )
    probe = session.workspace.probes_path / "P001"
    waiting = await tools["check_slurm_job"].handler({"probe_id": "S001/P001"})
    assert "finishing" in waiting["content"][0]["text"]
    assert not (probe / "output.txt").exists()
    assert not any(
        record["kind"] == "run" for record in session.workspace.current_records()
    )

    (probe / "stdout.txt").write_text("42\n")
    (probe / "stderr.txt").write_text("")
    (probe / "result.csv").write_text("value\n42\n")
    finished = await tools["check_slurm_job"].handler({"probe_id": "S001/P001"})
    assert "42" in finished["content"][0]["text"]
    assert json.loads((probe / "probe.json").read_text())["status"] == "completed"


@pytest.mark.asyncio
async def test_slurm_missing_files_eventually_save_an_error(session, monkeypatch):
    def scheduler(argv, workspace, env=None):
        if argv[0] == "sbatch":
            return SimpleNamespace(returncode=0, stdout="91\n", stderr="")
        if argv[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="91|COMPLETED|0:0\n", stderr="")

    monkeypatch.setattr("nocturnomath.slurm._command", scheduler)
    tools = tools_by_name(session)
    await tools["submit_slurm"].handler(
        {
            "code": "print(42)",
            "expected": "42",
            "time_minutes": 10,
            "cpus": 1,
            "memory_gb": 1,
            "result_files": ["result.csv"],
        }
    )
    probe = session.workspace.probes_path / "P001"
    (probe / "stdout.txt").write_text("42\n")
    (probe / "stderr.txt").write_text("")
    await tools["check_slurm_job"].handler({"probe_id": "S001/P001"})
    monkeypatch.setattr(slurm, "FILE_SETTLE_SECONDS", 0)
    result = await tools["check_slurm_job"].handler({"probe_id": "S001/P001"})
    assert "Missing declared result files" in result["content"][0]["text"]
    assert json.loads((probe / "probe.json").read_text())["status"] == "error"


@pytest.mark.asyncio
async def test_two_explorers_collect_one_slurm_result_and_verdict(session, monkeypatch):
    def scheduler(argv, workspace, env=None):
        if argv[0] == "sbatch":
            return SimpleNamespace(returncode=0, stdout="91\n", stderr="")
        if argv[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="91|COMPLETED|0:0\n", stderr="")

    monkeypatch.setattr("nocturnomath.slurm._command", scheduler)
    await tools_by_name(session)["submit_slurm"].handler(
        {
            "code": "print(42)",
            "expected": "42",
            "time_minutes": 10,
            "cpus": 1,
            "memory_gb": 1,
        }
    )
    probe = session.workspace.probes_path / "P001"
    (probe / "stdout.txt").write_text("42\n")
    (probe / "stderr.txt").write_text("")
    other = Workspace(session.workspace.path)
    session.start_new_transcript()
    session.log_transcript("user", text="Check the job")
    current = session.transcript_path
    await asyncio.gather(
        asyncio.to_thread(slurm.check, session.workspace, "S001/P001", current),
        asyncio.to_thread(slurm.check, other, "S001/P001", current),
    )
    original = session.workspace.transcript_file("S001")
    for transcript in (original, current):
        runs = [
            record
            for record in session.workspace.read_transcript(transcript)
            if record["kind"] == "run" and record["probe_id"] == "S001/P001"
        ]
        assert len(runs) == 1
    verdicts = await asyncio.gather(
        asyncio.to_thread(
            slurm.record_verdict,
            session.workspace,
            "S001/P001",
            "42",
            "Matched",
            original,
        ),
        asyncio.to_thread(
            slurm.record_verdict, other, "S001/P001", "42", "Matched", original
        ),
    )
    assert sorted(verdicts) == [False, True]
    assert (
        len(
            [
                record
                for record in session.workspace.read_transcript(original)
                if record["kind"] == "verdict" and record["probe_id"] == "S001/P001"
            ]
        )
        == 1
    )


@pytest.mark.asyncio
async def test_slurm_submission_failure_keeps_saved_script(session, monkeypatch):
    events = []
    session.subscribe(lambda kind, payload: events.append((kind, payload)))
    monkeypatch.setattr(
        "nocturnomath.slurm._command",
        lambda argv, workspace, env=None: SimpleNamespace(
            returncode=1, stdout="", stderr="unknown partition"
        ),
    )
    result = await tools_by_name(session)["submit_slurm"].handler(
        {
            "code": "print(1)",
            "expected": "1",
            "time_minutes": 10,
            "cpus": 1,
            "memory_gb": 1,
        }
    )
    assert result["is_error"]
    probe = session.workspace.probes_path / "P001"
    assert (probe / "code.py").read_text() == "print(1)"
    assert (probe / "output.txt").read_text() == "unknown partition"
    assert json.loads((probe / "probe.json").read_text())["status"] == "failed"
    assert [kind for kind, _ in events] == ["probe_start", "probe_finish"]
    assert events[-1][1]["status"] == "slurm-error"


@pytest.mark.asyncio
async def test_slurm_failed_job_keeps_partial_output(session, monkeypatch):
    def scheduler(argv, workspace, env=None):
        if argv[0] == "sbatch":
            return SimpleNamespace(returncode=0, stdout="91\n", stderr="")
        if argv[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="91|FAILED|1:0\n", stderr="")

    monkeypatch.setattr("nocturnomath.slurm._command", scheduler)
    tools = tools_by_name(session)
    await tools["submit_slurm"].handler(
        {
            "code": "print('partial')\nraise RuntimeError('stop')",
            "expected": "a result",
            "time_minutes": 10,
            "cpus": 1,
            "memory_gb": 1,
            "result_files": ["result.csv"],
        }
    )
    probe = session.workspace.probes_path / "P001"
    (probe / "stdout.txt").write_text("partial\n")
    (probe / "stderr.txt").write_text("RuntimeError: stop\n")
    result = await tools["check_slurm_job"].handler({"probe_id": "S001/P001"})
    assert "partial" in result["content"][0]["text"]
    assert "Missing declared result files" in result["content"][0]["text"]
    assert json.loads((probe / "probe.json").read_text())["status"] == "error"
    assert session._pending_verdict == "a result"
    session._pending_verdict = None  # A resumed chat has no in-memory pending verdict.
    again = await tools["check_slurm_job"].handler({"probe_id": "S001/P001"})
    assert "partial" in again["content"][0]["text"]
    assert session._pending_verdict == "a result"


@pytest.mark.asyncio
async def test_failed_slurm_probe_cannot_be_retried_in_kernel(session):
    session.activate_session()
    probe = session.workspace.start_probe(
        "print('batch')", "batch output", "test", backend="slurm"
    )
    session.workspace.finish_probe(probe, "failed\n", [], "error", "FAILED")

    result = await tools_by_name(session)["run"].handler(
        {"code": "print('retry')", "retry_of": "S001/P001"}
    )

    assert result["is_error"]
    assert "submit_slurm" in result["content"][0]["text"]
    assert not (probe / "A002").exists()
    assert session.kernel.executed_codes == []


@pytest.mark.asyncio
async def test_slurm_check_waits_for_another_probe_verdict(session, monkeypatch):
    def scheduler(argv, workspace, env=None):
        if argv[0] == "sbatch":
            return SimpleNamespace(returncode=0, stdout="91\n", stderr="")
        if argv[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="91|COMPLETED|0:0\n", stderr="")

    monkeypatch.setattr("nocturnomath.slurm._command", scheduler)
    tools = tools_by_name(session)
    await tools["submit_slurm"].handler(
        {
            "code": "print('batch')",
            "expected": "batch output",
            "time_minutes": 10,
            "cpus": 1,
            "memory_gb": 1,
        }
    )
    await tools["run"].handler({"code": "print('local')", "expected": "local output"})
    assert session._pending_verdict_probe_id == "S001/P002"
    result = await tools["check_slurm_job"].handler({"probe_id": "S001/P001"})
    assert result["is_error"]
    assert "S001/P002" in result["content"][0]["text"]
    assert session._pending_verdict_probe_id == "S001/P002"
    await tools["verdict"].handler({"text": "Local output matched."})

    (session.workspace.probes_path / "P001" / "stdout.txt").write_text("batch\n")
    (session.workspace.probes_path / "P001" / "stderr.txt").write_text("")
    result = await tools["check_slurm_job"].handler({"probe_id": "S001/P001"})
    assert not result.get("is_error")
    assert session._pending_verdict_probe_id == "S001/P001"
    await tools["verdict"].handler({"text": "Batch output matched."})
    verdicts = [
        record
        for record in session.workspace.current_records()
        if record["kind"] == "verdict"
    ]
    assert [record["probe_id"] for record in verdicts] == ["S001/P002", "S001/P001"]


@pytest.mark.asyncio
async def test_slurm_check_cannot_overlap_a_kernel_probe(session, monkeypatch):
    def scheduler(argv, workspace, env=None):
        if argv[0] == "sbatch":
            return SimpleNamespace(returncode=0, stdout="91\n", stderr="")
        if argv[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="91|COMPLETED|0:0\n", stderr="")

    monkeypatch.setattr("nocturnomath.slurm._command", scheduler)
    tools = tools_by_name(session)
    await tools["submit_slurm"].handler(
        {
            "code": "print('batch')",
            "expected": "batch output",
            "time_minutes": 10,
            "cpus": 1,
            "memory_gb": 1,
        }
    )
    (session.workspace.probes_path / "P001" / "stdout.txt").write_text("batch\n")
    (session.workspace.probes_path / "P001" / "stderr.txt").write_text("")
    started = threading.Event()
    release = threading.Event()

    def execute(code, timeout):
        started.set()
        release.wait(timeout=2)
        return "local\n", [], None

    session.kernel.execute = execute
    local = asyncio.create_task(
        tools["run"].handler({"code": "print('local')", "expected": "local output"})
    )
    try:
        assert await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=2)
        checked = await tools["check_slurm_job"].handler({"probe_id": "S001/P001"})
        assert checked["is_error"]
        assert "still running" in checked["content"][0]["text"]
    finally:
        release.set()
    await local
    assert session._pending_verdict_probe_id == "S001/P002"
    await tools["verdict"].handler({"text": "Local output matched."})
    await tools["check_slurm_job"].handler({"probe_id": "S001/P001"})
    assert session._pending_verdict_probe_id == "S001/P001"


@pytest.mark.asyncio
async def test_slurm_check_records_result_in_each_checking_session(
    session, monkeypatch
):
    def scheduler(argv, workspace, env=None):
        if argv[0] == "sbatch":
            return SimpleNamespace(returncode=0, stdout="91\n", stderr="")
        if argv[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="91|COMPLETED|0:0\n", stderr="")

    monkeypatch.setattr("nocturnomath.slurm._command", scheduler)
    tools = tools_by_name(session)
    await tools["submit_slurm"].handler(
        {
            "code": "print('batch')",
            "expected": "batch output",
            "time_minutes": 10,
            "cpus": 1,
            "memory_gb": 1,
        }
    )
    (session.workspace.probes_path / "P001" / "stdout.txt").write_text("batch\n")
    (session.workspace.probes_path / "P001" / "stderr.txt").write_text("")
    session.start_new_transcript()
    session.log_transcript("user", text="Check the batch job")
    assert session.workspace.session_id == "S002"
    await tools["check_slurm_job"].handler({"probe_id": "S001/P001"})
    await tools["verdict"].handler({"text": "Batch output matched."})
    for session_id in ("S001", "S002"):
        records = session.workspace.load_session(session_id)
        assert (
            len(
                [
                    record
                    for record in records
                    if record["kind"] == "run" and record["probe_id"] == "S001/P001"
                ]
            )
            == 1
        )
        assert (
            len(
                [
                    record
                    for record in records
                    if record["kind"] == "verdict" and record["probe_id"] == "S001/P001"
                ]
            )
            == 1
        )
    checked_notebook = session.workspace.export_notebook("S002")
    checked_code = next(
        cell for cell in checked_notebook["cells"] if cell["cell_type"] == "code"
    )
    assert checked_code["source"].startswith("# Slurm probe saved for inspection")
    assert "# print('batch')" in checked_code["source"]
    assert checked_code["execution_count"] is None

    session.start_new_transcript()
    session.log_transcript("user", text="Recheck the finished job")
    await tools["check_slurm_job"].handler({"probe_id": "S001/P001"})
    assert [
        record
        for record in session.workspace.current_records()
        if record["kind"] == "run" and record["probe_id"] == "S001/P001"
    ]
    await tools["check_slurm_job"].handler({"probe_id": "S001/P001"})
    assert (
        len(
            [
                record
                for record in session.workspace.current_records()
                if record["kind"] == "run" and record["probe_id"] == "S001/P001"
            ]
        )
        == 1
    )


@pytest.mark.asyncio
async def test_run_saves_all_plots_but_caps_sdk_images(session, monkeypatch):
    monkeypatch.setattr("nocturnomath.tools.asyncio.to_thread", run_inline)
    session.image_cap = 1
    session.kernel.result = ("", [b"one", b"two"], None)
    events = []
    session.subscribe(lambda event_type, payload: events.append((event_type, payload)))
    result = await tools_by_name(session)["run"].handler(
        {"code": "plot()", "expected": "two plots"}
    )

    assert len(session.list_plots()) == 2
    assert [item["type"] for item in result["content"]] == ["text", "image"]
    finish = next(payload for name, payload in events if name == "probe_finish")
    assert finish["plot_paths"] == [
        ".nocturnomath/sessions/S001/probes/P001/plot-1.png",
        ".nocturnomath/sessions/S001/probes/P001/plot-2.png",
    ]
    assert finish["plot_urls"] == [
        f"/api/asset?path={path}" for path in finish["plot_paths"]
    ]
    assert finish["text"] == ""


@pytest.mark.asyncio
async def test_cancelled_probe_keeps_output_and_blocks_workspace_switch(session):
    import asyncio
    import json
    import threading

    started = threading.Event()
    release = threading.Event()
    events = []
    session.subscribe(
        lambda kind, payload: events.append(
            (
                kind,
                payload,
                (session.workspace.probes_path / "P001" / "output.txt").is_file(),
            )
        )
    )

    def execute(code, timeout):
        started.set()
        assert release.wait(5)
        return "partial observation\n", [b"partial plot"], "interrupted"

    session.kernel.execute = execute
    task = asyncio.create_task(
        tools_by_name(session)["run"].handler(
            {
                "code": "long_probe()",
                "expected": "a curve",
            }
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 2)
        probe = session.workspace.probes_path / "P001"
        assert (probe / "code.py").read_text() == "long_probe()"
        assert json.loads((probe / "probe.json").read_text())["status"] == "started"
        task.cancel()
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError):
            session.set_workspace(session.workspace_path / "other")
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (probe / "output.txt").read_text() == "partial observation\n"
    assert (probe / "plot-1.png").read_bytes() == b"partial plot"
    assert json.loads((probe / "probe.json").read_text())["status"] == "interrupted"
    records = session.workspace.read_transcript(session.transcript_path)
    assert [record["kind"] for record in records] == [
        "kernel_started",
        "probe_started",
        "run",
    ]
    assert records[-1]["probe_id"] == "S001/P001"
    assert records[-1]["status"] == "interrupted"
    finishes = [event for event in events if event[0] == "probe_finish"]
    assert len(finishes) == 1
    assert finishes[0][1]["status"] == "interrupted"
    assert finishes[0][2] is True
    assert not session.has_active_query()


@pytest.mark.asyncio
async def test_probe_failure_is_preserved_without_becoming_success(session):
    import json

    def execute(code, timeout):
        raise RuntimeError("kernel connection lost")

    session.kernel.execute = execute
    result = await tools_by_name(session)["run"].handler(
        {"code": "probe()", "expected": "3"}
    )
    assert result["is_error"] is True
    assert '"S001/P001"' in result["content"][0]["text"]
    probe = session.workspace.probes_path / "P001"
    assert json.loads((probe / "probe.json").read_text())["status"] == "failed"
    assert (probe / "output.txt").read_text() == "kernel connection lost"
    assert session._pending_verdict is None


@pytest.mark.asyncio
async def test_corrected_code_retries_error_in_same_probe(session, monkeypatch):
    import json

    monkeypatch.setattr("nocturnomath.tools.asyncio.to_thread", run_inline)

    def execute(code, timeout):
        session.kernel.errored = code == "print(missing)"
        if session.kernel.errored:
            return "NameError: missing", [], None
        return "3\n", [b"plot"], None

    session.kernel.execute = execute
    run = tools_by_name(session)["run"].handler
    first = await run({"code": "print(missing)", "expected": "3"})
    assert "retry_of" in first["content"][0]["text"]
    retry = await run({"code": "print(3)", "retry_of": "S001/P001"})
    assert "S001/P001/A002/output.txt" in retry["content"][0]["text"]
    original = session.workspace.probes_path / "P001"
    attempt = original / "A002"
    assert (original / "code.py").read_text() == "print(missing)"
    assert (original / "output.txt").read_text() == "NameError: missing"
    assert json.loads((original / "probe.json").read_text())["status"] == "error"
    assert (attempt / "code.py").read_text() == "print(3)"
    assert (attempt / "output.txt").read_text() == "3\n"
    assert json.loads((attempt / "probe.json").read_text())["expected"] == "3"
    assert session.workspace.evidence_sources(["S001/P001/A002/plot-1.png"])
    assert session.workspace.list_plots()[0]["filename"] == "S001/P001/A002/plot-1.png"
    assert len(list(session.workspace.probes_path.iterdir())) == 1
    assert session._pending_verdict == "3"

    blocked = await run({"code": "print(4)", "expected": "4", "retry_of": "S001/P001"})
    assert blocked["is_error"] is True


@pytest.mark.asyncio
async def test_evidence_and_thought_tools_report_corrections(session):
    tools = tools_by_name(session)
    await tools["run"].handler({"code": "print(3)", "expected": "3"})
    assert not (
        await tools["evidence"].handler(
            {
                "text": "Value = 3.",
                "sources": ["S001/P001/output.txt"],
            }
        )
    ).get("is_error")
    await tools["thought"].handler(
        {"text": "[E001] is consistent with a positive offset.", "replaces": []}
    )
    await tools["strike_evidence"].handler({"entry_id": "E001"})
    await tools["thought"].handler(
        {
            "text": "[E001] is invalid, so the offset interpretation in [T001] lacks support.",
            "replaces": ["T001"],
        }
    )
    assert "<del>" in session.workspace.notes.entry("E001")
    assert "<del>" in session.workspace.notes.entry("T001")
    assert "<del>" not in session.workspace.notes.entry("T002")
    records = session.workspace.read_transcript(session.transcript_path)
    assert records[-1]["replaces"] == ["T001"]
    assert records[-1]["entry_id"] == "T002"
