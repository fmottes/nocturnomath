# Xprober

Xprober is a lightweight, hypothesis-driven scientific probing tool with a persistent Jupyter kernel, live markdown memory (`notes.md`), and a split-screen local web app dashboard.

## Features

- **Split-Screen Web App**:
  - **Left Half**: Interactive chat interface with Claude, rich probe execution cards (showing expected outcomes, collapsible Python code, stdout/stderr), inline plot rendering, and kernel status controls.
  - **Right Half**: Live-rendered Markdown reader (auto-syncing `notes.md`, reports, and other markdown docs) and scratch plots gallery.
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

## Project structure

The Python package is organized by responsibility:

- `xprober/session.py` coordinates Claude conversation state and events.
- `xprober/tools.py` defines the probing tools.
- `xprober/workspace.py` owns notes, plots, files, and transcripts.
- `xprober/kernel.py` manages the persistent Jupyter kernel.
- `xprober/web/` contains the FastAPI dashboard and browser assets.
- `xprober/cli/` contains the web and terminal entry points.
