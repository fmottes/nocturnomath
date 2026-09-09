import pytest

from nocturnomath.notes import ResearchNotes


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
