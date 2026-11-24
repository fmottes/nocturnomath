import json

import pytest

from nocturnomath.notes import ResearchNotes, migrate_workspace_notes


@pytest.fixture
def notes(tmp_path):
    return ResearchNotes(tmp_path)


def test_evidence_strike_preserves_original_and_numbering(notes):
    sources = ["[output](../sessions/S001/probes/P001/output.txt)"]
    notes.add_evidence("At x < 5, measured y = 3.", sources)
    original = notes.entry("E001")
    notes.strike_evidence("E001")
    struck = notes.entry("E001")
    assert "<del>" in struck and "</del>" in struck
    assert "At x < 5, measured y = 3." in struck
    assert sources[0] in struck
    assert "Replaced" not in struck
    assert notes.add_evidence("At x < 5, corrected y = 4.", sources) == "E002"
    assert original.strip().split("</a>")[1].strip() in struck
    reopened = ResearchNotes(notes.evidence_path.parent)
    assert reopened.add_evidence("At x = 6, y = 5.", sources) == "E003"


def test_thought_replacement_is_atomic_preserves_equations_and_links(notes):
    notes.add_evidence("The value is 3.", ["[output](output.txt)"])
    thought = "A linear relation could explain [E001].\n\n$$y = ax + b$$"
    assert notes.add_thought(thought, []) == "T001"
    before = notes.thoughts_path.read_text()
    with pytest.raises(ValueError):
        notes.add_thought("A correction to [T001].", ["T001", "T999"])
    assert notes.thoughts_path.read_text() == before
    assert (
        notes.add_thought(
            "[T001] assumed zero offset; [E001] does not establish that.", ["T001"]
        )
        == "T002"
    )
    old = notes.entry("T001")
    assert "<del>" in old and "$$y = ax + b$$" in old
    assert "[E001](evidence.md#E001)" in old
    assert "Replaced by [T002](thoughts.md#T002)" in old
    assert "Replaces: [T001](thoughts.md#T001)" in notes.entry("T002")
    assert "<del>" not in notes.entry("T002")
    # Historical references remain legal when discussing invalid evidence.
    notes.strike_evidence("E001")
    notes.add_thought("[E001] was invalid; [T002] therefore lacks support.", ["T002"])


def test_record_constraints_do_not_consume_ids(notes):
    for text, sources in [
        ("a" * 251, ["source"]),
        ("", ["source"]),
        ("See [T001].", ["source"]),
        ("Value = 3.", []),
    ]:
        with pytest.raises(ValueError):
            notes.add_evidence(text, sources)
    assert notes.add_evidence("a" * 250, ["source"]) == "E001"
    with pytest.raises(ValueError, match="Unknown entry"):
        notes.add_thought("Explains [E999].", [])
    assert notes.add_thought("[E001] needs interpretation.", []) == "T001"


def test_jsonl_is_append_only_tracks_provenance_citations_and_heals_markdown(notes):
    notes.add_evidence("Variance = 1.0.", ["[out](out.txt)"], session="S001")
    notes.add_thought("Consistent with unit noise [E001].", [], session="S001")
    lines_before = notes.record_path.read_text(encoding="utf-8").splitlines()
    assert len(lines_before) == 2

    notes.strike_evidence("E001", session="S002")
    lines_after = notes.record_path.read_text(encoding="utf-8").splitlines()
    assert lines_after[:2] == lines_before
    assert json.loads(lines_after[2])["op"] == "strike_evidence"
    assert json.loads(lines_after[2])["session"] == "S002"

    assert notes.dependent_thoughts("E001") == ["T001"]
    notes.add_thought(
        "Supersedes [T001] after [E001] was struck.", ["T001"], session="S002"
    )
    assert notes.dependent_thoughts("E001") == ["T002"]

    evidence, thoughts = notes.records()
    assert evidence["E001"].struck is True
    assert evidence["E001"].session == "S001"
    assert isinstance(evidence["E001"].t, float)
    assert thoughts["T001"].struck is True
    assert thoughts["T001"].replaced_by == "T002"
    assert thoughts["T002"].citations == ["T001", "E001"]
    assert thoughts["T002"].replaces == ["T001"]

    active_view = notes.read(active_only=True)
    assert "## E001" not in active_view
    assert "## T001" not in active_view
    assert "## T002" in active_view

    # Deleting rendered Markdown self-heals from record.jsonl on reopen.
    expected_evidence = notes.evidence_path.read_text(encoding="utf-8")
    expected_thoughts = notes.thoughts_path.read_text(encoding="utf-8")
    notes.evidence_path.unlink()
    notes.thoughts_path.unlink()
    reopened = ResearchNotes(notes.record_path.parent)
    assert reopened.evidence_path.read_text(encoding="utf-8") == expected_evidence
    assert reopened.thoughts_path.read_text(encoding="utf-8") == expected_thoughts


def test_legacy_markdown_workspace_migrates_losslessly_and_idempotently(tmp_path):
    kb = tmp_path / ".nocturnomath" / "kb"
    kb.mkdir(parents=True)

    legacy_evidence = (
        "# Evidence\n\n"
        "## E001\n\n"
        '<a id="E001"></a>\n\n'
        "<del>\n\n"
        "At x < 5, measured y = 3.\n\n"
        "Sources: [output](../sessions/S001/probes/P001/output.txt) · "
        "[plot](../sessions/S001/probes/P001/plot-1.png)\n\n"
        "</del>\n\n"
        "## E002\n\n"
        '<a id="E002"></a>\n\n'
        "At x < 5, corrected y = 4.\n\n"
        "Sources: [output](../sessions/S001/probes/P002/output.txt)\n\n"
    )
    legacy_thoughts = (
        "# Thoughts\n\n"
        "## T001\n\n"
        '<a id="T001"></a>\n\n'
        "<del>\n\n"
        "A linear relation explains [E001](evidence.md#E001).\n\n"
        "$$y = ax + b$$\n\n"
        "</del>\n\n"
        "Replaced by [T002](thoughts.md#T002).\n\n"
        "## T002\n\n"
        '<a id="T002"></a>\n\n'
        "<del>\n\n"
        "Revised using [E002](evidence.md#E002) after [T001](thoughts.md#T001).\n\n"
        "Replaces: [T001](thoughts.md#T001)\n\n"
        "</del>\n\n"
        "Replaced by [T003](thoughts.md#T003).\n\n"
        "## T003\n\n"
        '<a id="T003"></a>\n\n'
        "Final model grounded in [E002](evidence.md#E002).\n\n"
        "Replaces: [T002](thoughts.md#T002)\n\n"
    )
    (kb / "evidence.md").write_text(legacy_evidence, encoding="utf-8")
    (kb / "thoughts.md").write_text(legacy_thoughts, encoding="utf-8")

    assert migrate_workspace_notes(tmp_path) is True
    # Second call is a no-op because record.jsonl already exists.
    assert migrate_workspace_notes(tmp_path) is False

    migrated = ResearchNotes(kb)
    assert migrated.migrated is False
    assert migrated.record_path.is_file()
    # Rendered Markdown round-trips byte-for-byte.
    assert migrated.evidence_path.read_text(encoding="utf-8") == legacy_evidence
    assert migrated.thoughts_path.read_text(encoding="utf-8") == legacy_thoughts

    evidence, thoughts = migrated.records()
    assert evidence["E001"].struck is True
    assert len(evidence["E001"].sources) == 2
    assert evidence["E002"].struck is False
    assert thoughts["T001"].struck is True and thoughts["T001"].replaced_by == "T002"
    assert thoughts["T002"].struck is True and thoughts["T002"].replaced_by == "T003"
    assert thoughts["T003"].struck is False and thoughts["T003"].replaces == ["T002"]
    assert migrated.dependent_thoughts("E002") == ["T003"]
    assert migrated.dependent_thoughts("E001") == []

    # Numbering continues cleanly after migration.
    assert migrated.add_evidence("New measurement.", ["[out](out.txt)"]) == "E003"
    assert migrated.add_thought("Follow-up on [E003].", []) == "T004"


def test_edge_cases_defensive_copies_crlf_orphan_strike_and_missing_newline(tmp_path):
    # Non-existent target does not create directories.
    missing = tmp_path / "does-not-exist"
    assert migrate_workspace_notes(missing) is False
    assert not missing.exists()

    # CRLF legacy files + struck thought referencing a missing replacer stay struck.
    kb = tmp_path / "crlf_kb"
    kb.mkdir()
    (kb / "evidence.md").write_bytes(
        b'# Evidence\r\n\r\n## E001\r\n\r\n<a id="E001"></a>\r\n\r\n'
        b"Measured 7.\r\n\r\nSources: [out](out.txt)\r\n\r\n"
    )
    (kb / "thoughts.md").write_bytes(
        b'# Thoughts\r\n\r\n## T001\r\n\r\n<a id="T001"></a>\r\n\r\n'
        b"<del>\r\n\r\nEarly guess [E001](evidence.md#E001).\r\n\r\n</del>\r\n\r\n"
        b"Replaced by [T999](thoughts.md#T999).\r\n\r\n"
    )
    notes = ResearchNotes(kb)
    assert notes.migrated is True
    evidence, thoughts = notes.records()
    assert evidence["E001"].text == "Measured 7."
    assert thoughts["T001"].struck is True
    assert thoughts["T001"].replaced_by == "T999"

    # Mutating returned records does not corrupt internal cache.
    evidence["E001"].sources.append("MUTATED")
    thoughts["T001"].citations.append("MUTATED")
    fresh_evidence, fresh_thoughts = notes.records()
    assert "MUTATED" not in fresh_evidence["E001"].sources
    assert "MUTATED" not in fresh_thoughts["T001"].citations

    # Appending when record.jsonl lacks a trailing newline stays valid JSONL.
    raw = notes.record_path.read_text(encoding="utf-8").rstrip("\n")
    notes.record_path.write_text(raw, encoding="utf-8")
    assert notes.add_evidence("Second measurement.", ["[out](out.txt)"]) == "E002"
    lines = notes.record_path.read_text(encoding="utf-8").splitlines()
    assert all(json.loads(line)["op"] for line in lines)

    # CRLF fake headings in thoughts and blank/multiline sources in evidence are rejected.
    with pytest.raises(ValueError):
        notes.add_thought("Look at [E001]\r\n## T099\r\nFake header", [])
    with pytest.raises(ValueError):
        notes.add_evidence("Valid line.", ["ok", "bad\nline"])

    # Stale .md content heals on reopen even when mtime >= record.jsonl mtime.
    notes.evidence_path.write_text("# Evidence\n\ncorrupted\n", encoding="utf-8")
    reopened = ResearchNotes(kb)
    assert "## E002" in reopened.evidence_path.read_text(encoding="utf-8")
