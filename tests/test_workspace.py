from xprober.prompt import NOTES_TEMPLATE
from xprober.workspace import Workspace


def test_workspace_initializes_notes_and_unique_transcripts(tmp_path):
    workspace = Workspace(tmp_path)
    first = workspace.transcript_path
    workspace.start_new_transcript()

    assert workspace.notes_path.read_text() == NOTES_TEMPLATE
    assert workspace.transcript_path != first
    assert workspace.notes_path == tmp_path / "xprober" / "notes" / "notes.md"
    assert workspace.transcript_path.parent == tmp_path / "xprober" / "transcripts"
    assert workspace.figures_path == tmp_path / "xprober" / "figures"
    assert workspace.scratch_path == tmp_path / "xprober" / "scratch"
    assert workspace.scratch_path.is_dir()


def test_notes_files_plots_and_transcript_history(tmp_path):
    workspace = Workspace(tmp_path)
    workspace.add_note("fact", "the value is positive")
    (tmp_path / "report.md").write_text("# Report")
    plot_names = workspace.save_plots([b"one", b"two"])
    workspace.log_transcript("user", text="inspect the data")
    workspace.log_transcript("meta", sdk_session_id="sdk-1")

    assert "- the value is positive" in workspace.notes_path.read_text()
    assert [item["path"] for item in workspace.list_markdown_files()] == [
        "xprober/notes/notes.md",
        "report.md",
    ]
    assert {item["filename"] for item in workspace.list_plots()} == set(plot_names)
    assert all((workspace.figures_path / name).is_file() for name in plot_names)
    [history] = workspace.list_sessions()
    assert history["title"] == "inspect the data"
    assert history["context_restorable"] is True
    assert workspace.load_session(history["id"])[0]["kind"] == "user"


def test_recap_preserves_relevant_record_types():
    recap = Workspace.recap(
        [
            {"kind": "user", "text": "question"},
            {"kind": "agent", "text": "idea"},
            {"kind": "run", "expected": "positive", "output": "3"},
            {"kind": "note", "note_kind": "fact", "text": "it is positive"},
        ]
    )

    assert "Me: question" in recap
    assert "You: idea" in recap
    assert "expecting: positive" in recap
    assert "You noted (fact): it is positive" in recap
