# Repository Guidelines

## Product Direction

Nocturnomath is a deliberately constrained agent for scientific discovery and understanding. Optimize for the simplest probe that can resolve the current question, then build later results and interpretations on the saved record. The path from hypothesis to code, output, evidence, and interpretation must stay visible and traceable.

Prefer plain code, explicit state, and small changes over abstractions, framework machinery, speculative extensibility, or software-engineering ceremony. Do not introduce gates, performance work, compatibility layers, or tests merely because they are conventional. Add complexity only when it directly improves scientific understanding, record integrity, or a behavior the user requested.

## Project Structure & Module Organization

Package code lives in `nocturnomath/`. `session.py` coordinates Claude conversations, `runtime.py` handles shared orchestration, `workspace.py` owns artifacts, `notes.py` manages the scientific record, and `kernel.py` controls Jupyter. Interface adapters live in `cli/` and `web/`; browser assets are in `web/static/`. Tests and fixtures live in `tests/`.

Treat the generated `.nocturnomath/` workspace directory as runtime data. Do not commit sessions, probe output, notes, or scratch files.

## Build, Test, and Development Commands

- `uv sync --group dev` installs Python 3.12 dependencies from `uv.lock`.
- `uv run nocturnomath` launches the web interface; add `--path /path/to/workspace` to choose a folder.
- `uv run nocturnomath-cli --path /path/to/workspace` runs the terminal client.
- `uv run pytest tests/ -v` runs the complete test suite.
- `uv run pytest tests/ --cov=nocturnomath --cov-report=term-missing` reports coverage.
- `uv run ruff check .` lints the repository; `uv run ruff format .` formats it.
- `uv run pre-commit run --all-files` runs all commit checks.
- `uv build` creates source and wheel distributions through Hatchling.

## Coding Style & Naming Conventions

Use four-space Python indentation and Ruff's defaults, including the documented `BLE001` exclusion. Use `snake_case` for modules, functions, and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Keep rendering in interface packages and shared behavior in core modules.

## Testing Guidelines

Use pytest and `pytest-asyncio`; mark async tests with `@pytest.mark.asyncio`. Name tests `test_<behavior>` and use fixtures, `tmp_path`, and boundary mocks. Test state transitions, record integrity, and regressions where a test provides meaningful confidence. For reversible presentation changes, use a focused syntax or manual check instead of adding tests that mirror the implementation. Do not add coverage thresholds, performance gates, or other required checks without an explicit request.

## Architecture & Record Invariants

Web and terminal clients must share one runtime contract; keep orchestration and state transitions out of interface adapters. Do not change behavior beyond the requested scope without clearly telling the user. Existing workspaces are development data: prefer a clean design over backward compatibility, migration machinery, or support for obsolete layouts.

### Scientific record

- `.nocturnomath/notes/evidence.md` is an append-only list with stable IDs (`E001`, `E002`, ...). Each entry is a strictly factual observation, at most 250 characters excluding generated source links, and states relevant conditions. It must link to saved output or a plot that directly supports it. It must not interpret, characterize a result as positive or negative, or cite another evidence or thought entry.
- Never revise or delete an evidence entry. If it proves invalid, strike the whole original entry and append the corrected observation under a new ID. Do not add a supersession explanation or replacement link inside the evidence record.
- `.nocturnomath/notes/thoughts.md` is an append-only list with stable IDs (`T001`, `T002`, ...). Thoughts use plain, direct language and may interpret or reconcile evidence, form supported conjectures, develop a unified explanation or theory, and contain equations. They may cite evidence and earlier thoughts.
- Correct a thought by appending a self-contained replacement that carries forward everything still valid, then strike the old thought without removing it from the record. Preserve citations and stable IDs so the history remains traceable.
- Struck entries remain visible historical records but are not valid support for later conclusions.

### Probes, sessions, and kernels

- Probe artifacts belong under `.nocturnomath/sessions/Snnn/probes/Pnnn/`. Save the prediction and readable, flat code before execution. Save raw text output, every captured plot, run metadata, outcome, kernel ID, and partial results after errors or interruption.
- Use the persistent Jupyter kernel to accumulate understanding within a session. Starting a new session or changing environments starts a fresh kernel. Resuming chat alone starts with empty Python memory; restore a historical kernel only when explicitly requested, by replaying its saved probe code in order. Describe replayed state as approximate rather than a snapshot.
- Create a session directory only when its first user message is accepted. Keep empty sessions out of history.
- Use `.nocturnomath/scratch/` only when a probe genuinely needs a temporary file; otherwise keep probe work in memory and its durable artifacts in the probe directory.

### Research environments

The default research interpreter is the workspace-local `.nocturnomath/venv`, even when another `.venv` or environment exists nearby. Reuse it across sessions. Select a different interpreter only through an explicit user choice, remember that choice per workspace, and return to the managed environment only when explicitly selected. Record agent-driven package installations and environment inventories without turning environment capture into a promise of full reproducibility.

### Interface behavior

Keep the interface quiet and focused on the scientific record. Derive the model catalogue from the Claude SDK at startup and show resolved model IDs. Do not invent a default or fallback model entry; when discovery returns nothing, show `No model available`. A selected model applies to the next message while preserving the conversation. Keep context enabled by default and place experimental controls in Settings rather than the primary workflow.

Do not use browser automation in this repository; follow `.agents/rules/no-browser-subagent.md`. Verify web behavior with backend/API checks, JavaScript syntax checks, and the user's manual browser feedback.

## Change and Git Workflow

Work on the `dev` branch unless the user directs otherwise. Preserve unrelated working-tree changes. Do not commit or push unless the user explicitly asks; a request to commit does not also authorize a push. Keep requested changes focused, and report any additional behavior change before making it.

## Commit & Pull Request Guidelines

Use short, imperative, sentence-case subjects, for example `Restore the current chat when the page reconnects`. Keep commits focused and run checks that are proportionate to the change. Pull requests should explain behavior and relevant verification; include screenshots when they materially help review a UI change. Never commit credentials or generated research artifacts.
