# Xprober

Xprober is a lightweight, hypothesis-driven scientific probing tool with a persistent Jupyter kernel, citable evidence and thoughts in Markdown, and a split-screen local web app dashboard.

## Features

- **Split-Screen Web App**:
  - **Left Half**: Interactive chat interface with Claude, rich probe execution cards (showing expected outcomes, collapsible Python code, stdout/stderr), inline plot rendering, and kernel status controls.
  - **Right Half**: Live-rendered Markdown reader (auto-syncing `evidence.md`, `thoughts.md`, reports, and other markdown docs) and figures gallery.
  - **Draggable Gutter**: Easily resize the split panes to focus on chat or document reading.
  - **Workspace Picker**: The web app opens on a landing page with a visual folder navigator. It does not start a kernel or create workspace files until you open a folder.

## Quick Start

### 1. Install Dependencies

Using `uv`:

```bash
uv sync
```

### 2. Launch the Local Web App

Start the web app, then choose a workspace folder in the browser:

```bash
uv run xprober-web
```

Use `--path` / `-p` to set the starting location for the web folder navigator:

```bash
uv run xprober-web --path /path/to/project
```

#### CLI Options:
- `--path`, `-p`: Starting folder for the web navigator (default: current directory `.`). The terminal CLI uses it as its workspace.
- `--port`: Port for web app (default: `8000`)
- `--host`: Host to bind server to (default: `127.0.0.1`)
- `--model`: Claude model (default: `claude-opus-5`)
- `--timeout`: Seconds allowed per kernel run (default: `600`)
- `--images`: Plots returned to Claude per run (default: `2`)
- `--no-browser`: Do not automatically open the browser on startup

### 3. Terminal CLI Mode (Alternative)

If you prefer running in the terminal without a web browser:

```bash
uv run xprober-cli
```

The terminal and web interfaces share the same prompt, tools, persistent kernel,
notes, transcripts, verdict gating, and context behavior.

Xprober keeps its managed artifacts together inside the selected workspace:

```text
xprober/
├── notes/
│   ├── evidence.md
│   └── thoughts.md
├── sessions/
│   └── S001/
│       ├── transcript.jsonl
│       └── probes/
│           └── P001/
│               ├── code.py
│               ├── probe.json
│               ├── output.txt
│               └── plot-1.png
└── scratch/
```

## Scientific record

`evidence.md` is one numbered list of observations (`E001`, `E002`, …), with no
positive/negative categories. Each observation is at most 250 characters, excluding
its automatically generated source links. It states what was observed and under what
conditions; interpretations and references to other entries belong in `thoughts.md`.

`thoughts.md` contains numbered interpretations (`T001`, `T002`, …). Aim for one plain,
direct paragraph, usually 80–150 words; equations and longer explanations are allowed
when useful. Thoughts cite evidence and earlier thoughts using `[E001]` / `[T001]`;
the tool validates the IDs and produces clickable Markdown links.

The agent records entries through `evidence(text, sources)` and `thought(text, replaces)`.
Sources identify saved output or plots, e.g. `S001/P001/output.txt`. Each evidence entry
also links to the producing code and run metadata. The web reader follows citations to
entries, displays code/output, and opens plots. `/notes` prints both records in either interface.

Entries retain their original wording and IDs. `strike_evidence(entry_id)` strikes an
invalid observation in full, without explanation or a replacement link in the evidence
file. A corrected observation is a separate entry. To correct a thought, append another
thought with the old IDs in `replaces`; the tool strikes those thoughts and links both
ways. The new thought explains the correction and carries forward any valid reasoning.
Struck entries remain citable history, not valid support for conclusions.

Evidence and thoughts belong to the workspace and can cite probes from any session.
Starting a new session allocates a new `S` folder; resuming appends to its existing folder.
Probe numbering is local to each session. Code and prediction are saved before execution;
raw text output and all captured plots are saved after execution, including partial
output returned after an interruption. `probe.json` records the model, prediction,
timestamps, and outcome (`completed`, `error`, `incomplete`, `interrupted`, or `failed`).
A record still marked `started` has no recorded outcome, for example after a process crash.
These files preserve execution artifacts, not input snapshots or a reproducible kernel state.
The existing kernel lifetime and verdict-before-next-run behavior are unchanged.

There is no migration or compatibility layer for the previous generic notes/transcript layout.
Old test artifacts are not imported into the new record.

The agent uses `xprober/scratch/` only when a probe strictly requires a temporary
file or artifact; otherwise probe work remains in memory.

## Project structure

The Python package is organized by responsibility:

- `xprober/session.py` coordinates Claude conversation state and events.
- `xprober/tools.py` defines the probing tools.
- `xprober/workspace.py` owns session folders, probe artifacts, files, and transcripts.
- `xprober/notes.py` manages numbered evidence, thoughts, citations, and corrections.
- `xprober/kernel.py` manages the persistent Jupyter kernel.
- `xprober/web/` contains the FastAPI dashboard and browser assets.
- `xprober/cli/` contains the web and terminal entry points.
