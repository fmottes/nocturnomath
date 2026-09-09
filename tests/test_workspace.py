import base64

import pytest

from nocturnomath.workspace import Workspace


def test_workspace_defers_session_folder_until_first_activity(tmp_path):
    workspace = Workspace(tmp_path)
    first = workspace.transcript_path
    assert workspace.session_id == "S001"
    assert not first.parent.exists()
    workspace.start_new_transcript()

    assert workspace.notes.evidence_path.read_text() == "# Evidence\n\n"
    assert workspace.notes.thoughts_path.read_text() == "# Thoughts\n\n"
    assert workspace.transcript_path == first
    assert not workspace.probes_path.exists()
    assert workspace.scratch_path.is_dir()
    assert not (workspace.notes_dir_path / "notes.md").exists()

    workspace.log_transcript("user", text="first question")
    assert workspace.transcript_path == first
    assert workspace.probes_path.is_dir()


def test_sources_remain_citable_across_sessions_and_reopening(tmp_path):
    workspace = Workspace(tmp_path)
    (tmp_path / "report.md").write_text("# Report")
    vendored = tmp_path / ".nocturnomath/venv/lib/site-packages/numpy"
    vendored.mkdir(parents=True)
    (vendored / "LICENSE.md").write_text("vendored")
    probe = workspace.start_probe("print(3)", "positive", "test")
    workspace.finish_probe(probe, "3\n", [b"png"], "completed", None)
    workspace.log_transcript("user", text="inspect the data")
    workspace.log_transcript("meta", sdk_session_id="sdk-1")
    workspace = Workspace(tmp_path)
    assert workspace.session_id == "S002"
    sources = workspace.evidence_sources(
        ["S001/P001/output.txt", "S001/P001/plot-1.png"]
    )
    assert workspace.notes.add_evidence("The value is 3.", sources) == "E001"
    assert (probe / "output.txt").read_text() == "3\n"
    assert (
        "../sessions/S001/probes/P001/code.py"
        in workspace.notes.evidence_path.read_text()
    )
    assert [item["path"] for item in workspace.list_markdown_files()] == [
        ".nocturnomath/notes/evidence.md",
        ".nocturnomath/notes/thoughts.md",
        "report.md",
    ]
    [plot] = workspace.list_plots()
    assert plot["filename"] == "S001/P001/plot-1.png"
    assert plot["session"] == "S001"
    [history] = workspace.list_sessions()
    assert history["id"] == "S001"
    assert history["context_restorable"] is True
    assert not history["is_current"]
    workspace.use_transcript(workspace.transcript_file("S001"))
    assert workspace.start_probe("print(4)", "4", "test").name == "P002"
    assert workspace.load_session("S001")[0]["text"] == "inspect the data"


def test_missing_and_non_probe_sources_are_rejected(tmp_path):
    workspace = Workspace(tmp_path)
    for source in ["S001/P001/output.txt", "../report.md", "S001/P001/code.py"]:
        with pytest.raises(ValueError):
            workspace.evidence_sources([source])
    probe = workspace.start_probe("print(3)", "3", "test")
    (probe / "output.txt").write_text("partly written")
    with pytest.raises(ValueError, match="no recorded outcome"):
        workspace.evidence_sources(["S001/P001/output.txt"])


def test_recap_preserves_interpretation_and_corrections():
    recap = Workspace.recap(
        [
            {"kind": "user", "text": "question"},
            {"kind": "agent", "text": "idea"},
            {"kind": "run", "expected": "positive", "output": "3"},
            {"kind": "verdict", "text": "Does not distinguish the mechanisms."},
            {"kind": "evidence", "entry_id": "E001", "text": "The value is 3."},
            {"kind": "evidence_struck", "entry_id": "E001", "text": "Invalid."},
        ]
    )
    assert "Me: question" in recap
    assert "expecting: positive" in recap
    assert "Does not distinguish" in recap
    assert "evidence_struck E001" in recap


def test_notebook_export_structures_turns_and_embeds_probe_results(tmp_path):
    workspace = Workspace(tmp_path)
    workspace.log_transcript("user", text="What does the system do?")
    workspace.log_transcript(
        "agent", text="A direct measurement should distinguish it."
    )
    probe = workspace.start_probe("print(3)", "The value is 3.", "test")
    workspace.finish_probe(probe, "3\n", [b"png-data"], "completed", None)
    workspace.log_transcript(
        "probe_started",
        probe_id="S001/P001",
        expected="The value is 3.",
        code="print(3)",
    )
    workspace.log_transcript(
        "run",
        probe_id="S001/P001",
        output="3\n",
        images=[".nocturnomath/sessions/S001/probes/P001/plot-1.png"],
        status="completed",
    )
    workspace.log_transcript("verdict", text="The prediction held.")

    notebook = workspace.export_notebook("S001", "3.12")

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["language_info"]["version"] == "3.12"
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    assert notebook["cells"][0]["source"].startswith("# User message 1")
    assert notebook["cells"][1]["source"].startswith("## Idea 1.1")
    assert notebook["cells"][2]["source"].startswith("### Probe 1.1")
    code = notebook["cells"][3]
    assert code["source"] == "print(3)"
    assert code["outputs"][0]["text"] == "3\n"
    assert (
        code["outputs"][1]["data"]["image/png"]
        == base64.b64encode(b"png-data").decode()
    )
    assert notebook["cells"][4]["source"] == "**Verdict:** The prediction held."
