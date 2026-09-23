"""Restore the scientific record immediately after Claude compacts a chat."""

import json
import os
import sys
from pathlib import Path


def scientific_record_context(workspace_path: Path) -> str:
    kb_path = workspace_path / ".nocturnomath" / "kb"
    evidence = (kb_path / "evidence.md").read_text(encoding="utf-8")
    thoughts = (kb_path / "thoughts.md").read_text(encoding="utf-8")
    return (
        "The scientific record below is the current authoritative copy after "
        "conversation compaction. Struck entries are invalid; thoughts are "
        "interpretations, not observations. Relative links are based in "
        ".nocturnomath/kb/.\n\n" + evidence + "\n" + thoughts
    )


def main() -> None:
    hook = json.load(sys.stdin)
    if hook.get("source") != "compact":
        return
    workspace_path = Path(os.environ["NOCTURNOMATH_WORKSPACE_PATH"])
    context = scientific_record_context(workspace_path)
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": context,
                }
            }
        )
    )


if __name__ == "__main__":
    main()
