# Nocturnomath

Nocturnomath is a lightweight, hypothesis-driven scientific probing tool with a persistent Jupyter kernel, citable evidence and thoughts in Markdown, and two interfaces on one shared runtime: a split-screen local web app dashboard and a terminal client.

## Features

- **Split-Screen Web App**:
  - **Left Half**: Interactive chat interface with Claude, rich probe execution cards (showing expected outcomes, collapsible Python code, stdout/stderr), inline plot rendering, and kernel status controls.
  - **Right Half**: Live-rendered Markdown reader (auto-syncing `evidence.md`, `thoughts.md`, reports, and other markdown docs) and figures gallery.
  - **Draggable Gutter**: Easily resize the split panes to focus on chat or document reading.
  - **Workspace Picker**: The web app opens on a landing page with a visual folder navigator. It does not start a kernel or create workspace files until you open a folder.
- **Explicit Claude Login**: Choose a Claude subscription token, a Claude API key, or an existing Claude Code login from either interface. Credentials are passed directly to each Agent SDK client and kept only in memory.
- **Terminal Client**: The same runtime without a browser: multi-line input with history and completion, a state toolbar and a progress spinner, replies streamed as Markdown, compact probe cards that link to the saved code and cap the output, and slash commands covering history, resume, notebook export, model switching, context, environment, workspace, documents, and plots.

## Quick Start

### 1. Install Dependencies

Using `uv`:

```bash
uv sync
```

### 2. Launch the Local Web App

Start the web app, then choose a workspace folder in the browser:

```bash
uv run nocturnomath
```

Use `--path` / `-p` to set the starting location for the web folder navigator:

```bash
uv run nocturnomath --path /path/to/project
```

#### Options shared by both commands:
- `--path`, `-p`: Starting folder for the web navigator (default: current directory `.`). The terminal opens it as its workspace.
- `--python`: Research environment Python executable, or `managed` (remembered per workspace)
- `--model`: Claude model (default: `claude-opus-5`)
- `--timeout`: Seconds allowed per kernel run (default: `600`)
- `--images`: Plots returned to Claude per run (default: `2`)

#### Web-only options:
- `--port`: Port for web app (default: `8000`)
- `--host`: Host to bind server to (default: `127.0.0.1`)
- `--no-browser`: Do not automatically open the browser on startup

### Claude authentication

By default, Nocturnomath uses whatever Claude Code is already signed in with,
exactly as the Agent SDK would on its own. If `claude` works in your terminal, no
additional login is needed. When an environment variable such as
`ANTHROPIC_API_KEY` or `CLAUDE_CODE_USE_BEDROCK` is set, Claude Code prefers it
over its saved login; both interfaces say so next to the current method.

The web app's **Claude** button is available on both the landing screen and the
workspace header. In the terminal, use `/auth`. Both interfaces support:

- **Claude subscription**: First run `claude setup-token`, then paste the resulting
  long-lived token. Nocturnomath passes it to the Agent SDK as
  `CLAUDE_CODE_OAUTH_TOKEN`.
- **Claude API key**: Paste a key created in the
  [Claude Console](https://platform.claude.com/settings/keys). Nocturnomath passes it
  to the Agent SDK as `ANTHROPIC_API_KEY`; API usage is billed separately from a
  Claude subscription.
- **Claude Code (automatic)**: Return to the default behavior without changing
  Claude Code's saved login.

A credential entered here replaces any credential set in the environment for the
Claude CLI that Nocturnomath launches. The rest of your Claude Code setup still
applies, including a gateway `ANTHROPIC_BASE_URL`, a cloud provider selection, or an
`apiKeyHelper` in `settings.json`, which the CLI consults before a subscription
token. Selecting a method does not verify the credential or make a billable
request; authentication errors are reported on the next message. Subscription
login uses a token from `claude setup-token`; the installed Agent SDK does not
expose browser login.

A credential entered in either interface is never written to browser storage, shell
history, transcripts, or workspace files. It remains only in the running process,
so enter it again after restarting unless you provide it through the environment.
Changing authentication does not restart the research kernel or interrupt the
conversation.

### 3. Terminal Mode (Alternative)

If you prefer running without a web browser:

```bash
uv run nocturnomath-cli --path /path/to/project
```

The terminal has no landing page: it opens `--path` as the workspace immediately,
preparing the research environment on startup. It then prints the research
Python executable and version, the workspace path, the model, and the transcript path
the next session will use.

Enter sends the message; Esc then Enter (or Alt+Enter) inserts a newline for multi-line
input. Input history persists in `<workspace>/.nocturnomath/terminal_history`.
Ctrl-C interrupts a running kernel run or cancels the running query, and reports that
nothing is running otherwise; it does not quit. Ctrl-D or `/exit` drains the running
query and probe, releases the kernel, and leaves.

Typing `/` completes the command names with their descriptions, and each command
completes its own arguments: chat ids and `--kernel` for `/resume`, workspace documents
for `/docs`, catalogue entries for `/model`, `on` and `off` for `/context`, `managed` or
a Python executable for `/env`, and folders for `/workspace`.

A toolbar under the prompt reports the kernel state (ready, busy, dead), the current
model and any model queued for the next message, whether chat context is carried, the
current chat id (`new` until its first message), and the current status. While a query
runs, a spinner reports thinking or running probe. Neither the toolbar nor the spinner
is left behind in the scrollback.

Assistant replies stream in as they are written and are replaced by the final text of
each block, so nothing is printed twice. Notes, system messages, and Markdown documents
are rendered as Markdown too. Each probe is a card: its prediction and a
`code: <n> lines · <path>` line pointing at the saved `code.py`, never the code itself.
When it finishes, the card shows its status and the raw kernel output, capped at 60
lines with a pointer to the saved `output.txt`, then the path of each saved plot.
Recorded evidence and thoughts, verdicts, kernel restarts and interruptions, and
`install_packages` results are printed as they happen.

| Command | Description |
| --- | --- |
| `/help` | list these commands |
| `/new` | start a new chat on a fresh kernel |
| `/restart` | restart the kernel; in-memory state is gone |
| `/notes` | show `evidence.md` and `thoughts.md` |
| `/history` | list the chats recorded in this workspace |
| `/resume <S001> [--kernel]` | reopen a chat, optionally replaying its probes |
| `/export [path]` | save the current chat as a Jupyter notebook |
| `/model [name]` | show the catalogue, or use a model from the next message |
| `/auth [status\|subscription\|api-key\|claude-code]` | show or securely change Claude authentication |
| `/context [on\|off]` | show or set whether the agent carries chat context |
| `/env <python>\|managed` | switch the research environment; starts a new chat |
| `/workspace <path>` | open a different workspace folder |
| `/docs [name]` | list the Markdown documents, or render one |
| `/plots` | list the plots saved by probes |
| `/exit` | stop the agent and leave |

`/export` writes to the current directory by default; a directory argument keeps the
generated `nocturnomath-<workspace>-<session>.ipynb` name, a file path overrides it.
Commands that change the chat, kernel, context, environment, or workspace (`/new`,
`/restart`, `/resume`, `/export`, `/context`, `/env`, `/workspace`) are refused while the
agent is running a query, with the same messages the web app shows.

The terminal and web interfaces share the same runtime, prompt, tools, persistent
kernel, notes, transcripts, verdict gating, and context behavior.

Nocturnomath keeps its managed artifacts together inside the selected workspace:

```text
.nocturnomath/
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
Starting a new session restarts the kernel and prepares an empty chat (New chat in the
web app, `/new` in the terminal). Its numbered `S` folder is created only when the first
message is accepted. Empty legacy folders are omitted from History and from `/history`.
Resuming a past session appends to its existing folder: History in the web app,
`/resume S001` in the terminal, with `--kernel` (the web app's resume-with-kernel option)
replaying the stored probes in order instead of starting with empty Python memory.
The current session can be exported as a Jupyter notebook with Download .ipynb in the
web app or `/export` in the terminal; both refuse while a query is running and produce
the same `nocturnomath-<workspace>-<session>.ipynb` file.

In the web interface, Evidence and Thoughts open the scientific record directly;
Documents lists workspace files, and Documents refresh automatically. In the terminal,
`/notes` prints the scientific record, `/docs` lists the Markdown documents or renders
one, and `/plots` lists the saved figures with their paths. The model selector above the
composer, and `/model <name>` in the terminal, apply to the next message and keep the
current conversation. Their options come from the Claude SDK at startup, using resolved
model IDs as labels and excluding Default. If discovery is unavailable, the selector
shows No model available and `/model` reports that the catalogue is unavailable.
Discovery does not send a model prompt.
Keep context is on by default and can be changed under Settings → Experimental or with
`/context on|off`. Exit stops the service and attempts to close the tab; browsers that
block tab closing show a message instead. `/exit` and Ctrl-D do the same in the terminal,
draining the running query first.

Probe numbering is local to each session. Code and prediction are saved before execution;
raw text output and all captured plots are saved after execution, including partial
output returned after an interruption. `probe.json` records the model, prediction,
timestamps, and outcome (`completed`, `error`, `incomplete`, `interrupted`, or `failed`).
A record still marked `started` has no recorded outcome, for example after a process crash.
These files preserve execution artifacts, not input snapshots or a reproducible kernel state.
The kernel persists within a session, and the verdict-before-next-run behavior is unchanged.

There is no migration or compatibility layer for the previous generic notes/transcript layout.
Old test artifacts are not imported into the new record.

The agent uses `.nocturnomath/scratch/` only when a probe strictly requires a temporary
file or artifact; otherwise probe work remains in memory.

## Project structure

The Python package is organized by responsibility:

- `nocturnomath/session.py` coordinates Claude conversation state and events.
- `nocturnomath/tools.py` defines the probing tools.
- `nocturnomath/workspace.py` owns session folders, probe artifacts, files, and transcripts.
- `nocturnomath/notes.py` manages numbered evidence, thoughts, citations, and corrections.
- `nocturnomath/kernel.py` manages the persistent Jupyter kernel.
- `nocturnomath/runtime.py` is the transport-agnostic runtime shared by both interfaces:
  it owns the session, the model catalogue, workspace opening, serialised kernel and
  workspace transitions, query start-up, notebook export, and event fan-out to subscribers.
- `nocturnomath/web/` contains the FastAPI routes, the WebSocket runtime subclass that
  forwards runtime events to browsers, and the browser assets.
- `nocturnomath/cli/` contains the entry points and their shared arguments:
  `web.py` (`nocturnomath`) serves the dashboard, `terminal.py` (`nocturnomath-cli`)
  renders runtime events and slash commands in the terminal.

## Research environments

Opening a workspace automatically prepares `.nocturnomath/venv`, even if the folder
already has a `.venv`. The environment starts with ipykernel, matplotlib, numpy, and
pandas. Packages persist across sessions; kernel variables do not. The first opening
may download packages. Later openings reuse the environment. `uv` is a runtime
dependency: the `uv` on `PATH` is used when present, otherwise the binary bundled
with the `uv` Python package.

In Settings → Research environment, or with `/env /path/to/venv/bin/python` in the
terminal, enter another environment's Python executable if you want to use it instead.
It must already contain ipykernel and matplotlib;
Nocturnomath does not automatically modify a manually selected environment during
selection. Subsequent agent package installations target that environment. The choice
is remembered in `.nocturnomath/config.json`. Use “Use managed environment”, or
`/env managed`, to return to the workspace default. Changing environments starts a new
session. The terminal also opens a different workspace folder with `/workspace <path>`,
mirroring the web app's workspace picker.

Both commands accept `--python /path/to/venv/bin/python` (or `--python managed`).
For the web command this applies to the first workspace opened. This also allows
replacing a saved selection whose interpreter is no longer available.

New sessions and workspace changes start fresh kernels. Resuming a different historical
session restores its conversation but starts with empty Python memory; reloading the
current session preserves memory. The agent is told when memory has been cleared.

The `install_packages` tool installs into the exact research interpreter. Its command,
output, and before/after package inventories are saved under the session's
`environment_changes/` and `environments/` folders. It does not restart the kernel;
already imported modules keep their loaded versions until the agent restarts it.
Kernel starts are recorded in the transcript, and each probe records its kernel ID
and environment inventory path. These inventories document installed packages, not
input data or a snapshot of live Python objects. Installs outside this tool are not
tracked until the next environment inventory is captured.
