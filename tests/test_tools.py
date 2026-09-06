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
async def test_run_verdict_and_note_contract(session, monkeypatch):
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

    await tools["note"].handler({"kind": "fact", "text": "the output is 2"})
    assert "- the output is 2" in session.notes_path.read_text()
    assert [event[0] for event in events] == [
        "probe_start",
        "probe_finish",
        "probe_verdict",
        "note_added",
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
