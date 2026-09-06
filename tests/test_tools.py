import pytest

from xprober.tools import builds_machinery, switches_backend


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
async def test_run_verdict_and_evidence_contract(session, monkeypatch):
    monkeypatch.setattr("xprober.tools.asyncio.to_thread", run_inline)
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


@pytest.mark.asyncio
async def test_run_saves_all_plots_but_caps_sdk_images(session, monkeypatch):
    monkeypatch.setattr("xprober.tools.asyncio.to_thread", run_inline)
    session.image_cap = 1
    session.kernel.result = ("", [b"one", b"two"], None)
    result = await tools_by_name(session)["run"].handler(
        {"code": "plot()", "expected": "two plots"}
    )

    assert len(session.list_plots()) == 2
    assert [item["type"] for item in result["content"]] == ["text", "image"]


@pytest.mark.asyncio
async def test_cancelled_probe_keeps_output_and_blocks_workspace_switch(session):
    import asyncio
    import json
    import threading

    started = threading.Event()
    release = threading.Event()

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
    assert [record["kind"] for record in records] == ["probe_started", "run"]
    assert records[-1]["probe_id"] == "S001/P001"
    assert records[-1]["status"] == "interrupted"
    assert not session.has_active_query()


@pytest.mark.asyncio
async def test_probe_failure_is_preserved_without_becoming_success(session):
    import json

    def execute(code, timeout):
        raise RuntimeError("kernel connection lost")

    session.kernel.execute = execute
    with pytest.raises(RuntimeError, match="connection lost"):
        await tools_by_name(session)["run"].handler(
            {"code": "probe()", "expected": "3"}
        )
    probe = session.workspace.probes_path / "P001"
    assert json.loads((probe / "probe.json").read_text())["status"] == "failed"
    assert (probe / "output.txt").read_text() == "kernel connection lost"
    assert session._pending_verdict is None


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
