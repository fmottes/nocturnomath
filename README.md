# Nocturnomath

Nocturnomath is a lightweight, hypothesis-driven scientific probing tool. It combines a persistent Jupyter kernel, citable Markdown evidence and thoughts, and two interfaces on one shared runtime: a split-screen local web app and a terminal client.

> [!WARNING]
> Nocturnomath is an experimental research preview. It executes model-generated Python and can read and modify files in the workspace you select. Run the web interface only on a trusted machine using its default loopback address; it has no authentication and must not be exposed to a network. Expect breaking changes and verify important scientific results independently.

## Features

- **Web dashboard:** Chat with Claude beside a live Markdown reader and a figures gallery that can be filtered by session. Probe cards show predictions, code, output, plots, and kernel status.
- **Terminal client:** Stream Markdown replies, run the same probes and commands, and view compact kernel, model, context, and session status.
- **Persistent research record:** Save session transcripts, probe artifacts, numbered evidence, interpretations, and notebook exports inside each workspace.
- **Flexible environments:** Use an automatically managed Python environment or select an existing interpreter.
- **Explicit authentication:** Use Claude Code's existing login, a subscription token, or an API key. Credentials entered in the app stay in process memory.

## Quick start

Nocturnomath requires Python 3.12 or later, [uv](https://docs.astral.sh/uv/), and access to Claude through Claude Code or an API key.

Install the current version directly from GitHub:

```bash
uv tool install git+https://github.com/fmottes/nocturnomath.git
nocturnomath
```

For a local checkout:

```bash
git clone https://github.com/fmottes/nocturnomath.git
cd nocturnomath
uv sync
uv run nocturnomath
```

Set the folder navigator's starting location with `--path` / `-p`:

```bash
uv run nocturnomath --path /path/to/project
```

Both interfaces accept:

- `--path`, `-p`: starting folder; the terminal opens it immediately
- `--python`: research Python executable, or `managed`
- `--model`: Claude model (default: `claude-opus-5`)
- `--timeout`: kernel-run limit in seconds (default: `600`)
- `--images`: plots returned to Claude per run (default: `2`)

The web app also accepts `--port` (default `8000`), `--host` (default `127.0.0.1`), and `--no-browser`.

### Claude authentication

By default, Nocturnomath uses Claude Code's current authentication. If `claude` works in your terminal, no additional login is needed. Environment settings such as `ANTHROPIC_API_KEY`, cloud-provider configuration, `ANTHROPIC_BASE_URL`, and `apiKeyHelper` continue to follow Claude Code's precedence rules.

Use the web app's **Claude** button or the terminal's `/auth` command to choose:

- **Claude subscription:** Run `claude setup-token`, then enter the resulting token.
- **Claude API key:** Enter a key from the [Claude Console](https://platform.claude.com/settings/keys). API usage is billed separately from a subscription.
- **Claude Code (automatic):** Return to Claude Code's default authentication.

Manually entered credentials override credential environment variables only for the launched Claude client. They are not written to browser storage, shell history, transcripts, or workspace files, and must be re-entered after restarting. Selecting a method does not validate it or make a billable request; errors appear when you send the next message. Authentication changes do not restart the kernel or conversation.

### Terminal mode

Run Nocturnomath without a browser:

```bash
uv run nocturnomath-cli --path /path/to/project
```

Enter sends; Esc then Enter (or Alt+Enter) inserts a newline. Ctrl-C interrupts the active query or kernel run without quitting. Ctrl-D and `/exit` drain active work, release the kernel, and exit. Input history is stored in `.nocturnomath/terminal_history`, and typing `/` opens command completion.

| Command | Description |
| --- | --- |
| `/help` | List commands |
| `/new` | Start a chat on a fresh kernel |
| `/restart` | Restart the kernel and clear memory |
| `/notes` | Show evidence and thoughts |
| `/history` | List workspace chats |
| `/resume <S001> [--kernel]` | Reopen a chat; optionally replay its probes |
| `/export [path]` | Export the chat as a Jupyter notebook |
| `/model [name]` | Show or select a model for the next message |
| `/auth [status\|subscription\|api-key\|claude-code]` | Show or change authentication |
| `/context [on\|off]` | Show or change context carry-over |
| `/env <python>\|managed` | Switch research environment |
| `/workspace <path>` | Open another workspace |
| `/docs [name]` | List the documents folder or render one |
| `/plots` | List saved plots |
| `/exit` | Stop and exit |

`/export` uses the current directory by default. A directory keeps the generated `nocturnomath-<workspace>-<session>.ipynb` name; a file path overrides it. State-changing commands are refused while a query is running.

## Workspace and scientific record

Nocturnomath stores its managed artifacts in the selected workspace:

```text
.nocturnomath/
├── kb/
│   ├── evidence.md
│   └── thoughts.md
├── documents/
│   └── inputs.md
├── sessions/
│   └── S001/
│       ├── transcript.jsonl
│       └── probes/P001/
│           ├── code.py
│           ├── probe.json
│           ├── output.txt
│           └── plot-1.png
└── scratch/
```

`evidence.md` contains numbered observations (`E001`, `E002`, …), each limited to 250 characters plus generated source links. Observations state what happened and under which conditions. `thoughts.md` contains numbered interpretations (`T001`, `T002`, …), usually one 80–150-word paragraph, with citations such as `[E001]` and `[T001]`.

Evidence links to its probe output or plots, code, and run metadata. The web reader follows citations and opens those artifacts; `/notes`, `/docs`, and `/plots` expose them in the terminal.

`documents/` holds free-form Markdown files that give the agent extra context: your inputs, generated reports, calculations. The Documents tab lists them with a checkbox that decides whether each one is sent with the next message, renders a document on click, and lets you create, modify, or delete one in place. Settings chooses whether new documents start checked. With context kept, a document is sent only when it is new or has changed.

Entries keep stable wording and IDs. Invalid evidence is struck in full, and its correction becomes a new observation. Corrected thoughts are also appended and linked to the superseded entries. Struck entries remain part of the citable history but are not valid support for conclusions.

Evidence and thoughts span sessions. A new session starts an empty chat and fresh kernel. Resuming a session appends to its existing record but normally starts with empty Python memory. Choose resume-with-kernel in the web app, or `/resume S001 --kernel`, to replay stored probes. Refreshing the browser reconnects to the same session and restores its chat.

Probe IDs are session-local. Code and predictions are saved before execution; text output and plots are saved afterward, including partial output from interruptions. `probe.json` records the model, timestamps, prediction, kernel, and outcome. These are execution artifacts, not input snapshots or a fully reproducible kernel state.

Both interfaces provide model selection for the next message and context carry-over, which is enabled by default. In the web app, use the model selector and Settings → Experimental; in the terminal, use `/model` and `/context`.

## Research environments

Opening a workspace prepares `.nocturnomath/venv`, separate from any project `.venv`, with ipykernel, matplotlib, NumPy, and pandas. Packages persist across sessions; kernel variables do not. The first setup may download packages, while later openings reuse the environment.

Select another Python executable in Settings → Research environment, with `/env /path/to/venv/bin/python`, or through `--python`. A custom environment must already contain ipykernel and matplotlib and is not modified during selection. The choice is saved in `.nocturnomath/config.json`; choose **Use managed environment**, `/env managed`, or `--python managed` to return to the default. Switching environments starts a new session.

Package installations made through the agent target the selected interpreter and do not restart the kernel. Nocturnomath records the command, output, environment inventories, and kernel identifiers with the session. Already imported modules retain their loaded versions until the kernel restarts.
