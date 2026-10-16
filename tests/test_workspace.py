import ast
import base64
import json

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
    assert not (workspace.kb_path / "notes.md").exists()
    assert workspace.documents_path.is_dir()

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
    assert workspace.list_documents() == []
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


def test_unfinished_slurm_plot_cannot_support_evidence(tmp_path):
    workspace = Workspace(tmp_path)
    probe = workspace.start_probe("print(1)", "1", "test", backend="slurm")
    (probe / "plot-1.png").write_bytes(b"partial")
    for status in ("submitted", "running"):
        metadata = json.loads((probe / "probe.json").read_text())
        metadata["status"] = status
        (probe / "probe.json").write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="no recorded outcome"):
            workspace.evidence_sources(["S001/P001/plot-1.png"])
    workspace.finish_probe(probe, "1\n", [], "completed", None)
    assert workspace.evidence_sources(["S001/P001/plot-1.png"])


def test_file_reader_rejects_sibling_with_workspace_name_prefix(tmp_path):
    workspace = Workspace(tmp_path / "work")
    sibling = tmp_path / "work-secret"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("secret")

    with pytest.raises(ValueError, match="outside workspace"):
        workspace.read_file("../work-secret/secret.txt")


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


def test_notebook_export_comments_slurm_code_and_keeps_saved_result(tmp_path):
    workspace = Workspace(tmp_path)
    workspace.log_transcript("user", text="Run the batch probe")
    code = "import os\nprint(os.environ['NOCTURNOMATH_PROBE_DIR'])"
    probe = workspace.start_probe(
        code, "prints the probe directory", "test", backend="slurm"
    )
    workspace.finish_probe(probe, "directory\n", [b"png-data"], "completed", None)
    workspace.log_transcript(
        "probe_started",
        probe_id="S001/P001",
        expected="prints the probe directory",
        code=code,
        backend="slurm",
    )
    workspace.log_transcript(
        "run",
        probe_id="S001/P001",
        output="directory\n",
        images=[".nocturnomath/sessions/S001/probes/P001/plot-1.png"],
        status="completed",
        backend="slurm",
    )
    workspace.log_transcript("probe_started", probe_id="S001/P002", code="print(3)")
    workspace.log_transcript("run", probe_id="S001/P002", output="3\n")

    notebook = workspace.export_notebook("S001")

    slurm_code = notebook["cells"][2]
    assert slurm_code["cell_type"] == "code"
    assert slurm_code["execution_count"] is None
    assert slurm_code["outputs"] == []
    assert ast.parse(slurm_code["source"]).body == []
    assert "# print(os.environ['NOCTURNOMATH_PROBE_DIR'])" in slurm_code["source"]
    saved_result = notebook["cells"][3]
    assert saved_result["cell_type"] == "markdown"
    assert "directory" in saved_result["source"]
    assert (
        saved_result["attachments"]["plot-1.png"]["image/png"]
        == base64.b64encode(b"png-data").decode()
    )
    local_code = notebook["cells"][5]
    assert local_code["source"] == "print(3)"
    assert local_code["execution_count"] == 1


def test_notebook_export_identifies_uncollected_slurm_result(tmp_path):
    workspace = Workspace(tmp_path)
    workspace.log_transcript("user", text="Run the batch probe")
    workspace.log_transcript(
        "probe_started",
        probe_id="S001/P001",
        code="print('batch')",
        backend="slurm",
    )
    notebook = workspace.export_notebook("S001")
    assert notebook["cells"][2]["source"].endswith("# print('batch')")
    assert notebook["cells"][3]["source"] == "Slurm result has not been collected."


def test_documents_are_created_listed_and_edited(tmp_path):
    workspace = Workspace(tmp_path)
    (tmp_path / "README.md").write_text("# Not a document\n")

    name = workspace.create_document("Growth rates: first pass", "# Growth\n\nk = 0.3")
    assert name == "Growth-rates-first-pass.md"
    assert (workspace.documents_path / name).read_text() == "# Growth\n\nk = 0.3\n"
    with pytest.raises(FileExistsError):
        workspace.create_document("Growth rates: first pass", "again")
    with pytest.raises(ValueError):
        workspace.create_document("   ", "empty title")

    workspace.write_document(name, "# Growth\n\nk = 0.4\n")
    document = workspace.read_document(name)
    assert document["content"] == "# Growth\n\nk = 0.4\n"
    assert document["path"] == f".nocturnomath/documents/{name}"

    listed = workspace.list_documents()
    assert [item["name"] for item in listed] == [name]
    assert listed[0]["size"] == len("# Growth\n\nk = 0.4\n")

    spaced = workspace.documents_path / "field notes.md"
    spaced.write_text("# Field notes\n")
    assert workspace.read_document(spaced.name)["content"] == "# Field notes\n"

    with pytest.raises(FileNotFoundError):
        workspace.read_document("missing.md")
    for bad in ("../config.json", "notes.txt", ".hidden.md"):
        with pytest.raises(ValueError):
            workspace.document_file(bad)

    outside = tmp_path / "outside.md"
    outside.write_text("outside\n")
    linked = workspace.documents_path / "linked.md"
    linked.symlink_to(outside)
    assert "linked.md" not in [item["name"] for item in workspace.list_documents()]
    with pytest.raises(ValueError, match="symlinks"):
        workspace.write_document("linked.md", "changed")
    assert outside.read_text() == "outside\n"


def test_nested_documents_and_folders(tmp_path):
    workspace = Workspace(tmp_path)
    assert workspace.create_document_folder("reports") == "reports"
    assert workspace.create_document_folder("reports/2026") == "reports/2026"
    assert workspace.list_document_folders() == ["reports", "reports/2026"]

    name = workspace.create_document("Field notes", "First", "reports/2026")
    assert name == "reports/2026/Field-notes.md"
    assert workspace.read_document(name)["content"] == "First\n"
    assert [item["name"] for item in workspace.list_documents()] == [name]

    workspace.save_document("drafts/topic/summary.md", "Draft")
    assert workspace.list_document_folders() == [
        "drafts",
        "drafts/topic",
        "reports",
        "reports/2026",
    ]
    assert workspace.read_document("drafts/topic/summary.md")["content"] == "Draft\n"
    with pytest.raises(FileExistsError):
        workspace.save_document("drafts/topic/summary.md", "Accidental overwrite")
    workspace.save_document("drafts/topic/summary.md", "Revised", replace_existing=True)
    assert workspace.read_document("drafts/topic/summary.md")["content"] == "Revised\n"
    with pytest.raises(FileNotFoundError):
        workspace.save_document("missing.md", "No document", replace_existing=True)


def test_document_paths_reject_traversal_hidden_paths_and_symlinks(tmp_path):
    workspace = Workspace(tmp_path)
    for bad in (
        "/absolute.md",
        "../escape.md",
        "a/../escape.md",
        "a//file.md",
        ".hidden/file.md",
        "a/.hidden.md",
        "a\\file.md",
        "a/ file.md",
    ):
        with pytest.raises(ValueError):
            workspace.document_file(bad)

    outside = tmp_path / "outside"
    outside.mkdir()
    linked = workspace.documents_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        workspace.save_document("linked/file.md", "unsafe")
    assert not (outside / "file.md").exists()
    assert "linked" not in workspace.list_document_folders()

    (workspace.documents_path / "regular.md").write_text("content")
    with pytest.raises(ValueError, match="not a folder"):
        workspace.create_document_folder("regular.md/child")


def test_document_root_symlink_cannot_redirect_document_access(tmp_path):
    workspace = Workspace(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    workspace.documents_path.rmdir()
    workspace.documents_path.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        workspace.save_document("report.md", "unsafe")
    with pytest.raises(ValueError, match="symlinks"):
        workspace.list_documents()
    with pytest.raises(ValueError, match="symlinks"):
        workspace.list_document_folders()
    assert list(elsewhere.iterdir()) == []


def test_document_write_ignores_symlink_at_old_temporary_name(tmp_path):
    workspace = Workspace(tmp_path)
    evidence = workspace.notes.evidence_path
    original_evidence = evidence.read_text()
    temporary = workspace.documents_path / "report.md.tmp"
    temporary.symlink_to(evidence)

    workspace.save_document("report.md", "First")
    workspace.save_document("report.md", "Revised", replace_existing=True)

    assert workspace.read_document("report.md")["content"] == "Revised\n"
    assert evidence.read_text() == original_evidence
    assert temporary.is_symlink()
