"""Run Xprober in an interactive terminal."""

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Any

from ..session import ExplorationSession
from .common import add_session_arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_session_arguments(parser)
    return parser


async def render_event(event_type: str, event: dict[str, Any]):
    if event_type == "assistant_text":
        print("\n" + event["text"])
    elif event_type == "probe_start":
        print(f"\n  [run] expected: {event['expected']}")
    elif event_type == "probe_finish":
        print("\n" + event["output"])
    elif event_type == "record_changed":
        print(f"  [{event['entry_id']}] {event['kind']}: {event['text']}")
    elif event_type == "probe_verdict":
        print(f"  [verdict] {event['text']}")
    elif event_type == "kernel_restarted":
        print("kernel restarted; in-memory state is gone")
    elif event_type == "error":
        print(f"\nerror: {event['message']}", file=sys.stderr)


async def run_terminal(args: argparse.Namespace):
    if os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is set and silently takes precedence over your Claude "
            "subscription. Run `unset ANTHROPIC_API_KEY` and try again."
        )

    session = ExplorationSession(
        workspace_path=Path(args.path).resolve(),
        model=args.model,
        timeout_s=args.timeout,
        image_cap=args.images,
    )
    session.subscribe(render_event)

    def on_sigint(signum, frame):
        if session.kernel.busy:
            session.kernel.interrupt()
            print("\n  [interrupting kernel]")
        elif session._current_task and not session._current_task.done():
            session._current_task.cancel()
            print("\n  [cancelling query]")
        else:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_sigint)
    print(f"transcript: {session.transcript_path}")
    print("commands: /new  /restart  /notes  /exit")

    try:
        while True:
            try:
                line = (await asyncio.to_thread(input, "\n> ")).strip()
            except EOFError:
                return
            if not line:
                continue
            if line == "/exit":
                return
            if line == "/notes":
                print(session.workspace.notes.read())
                continue
            if line == "/restart":
                session.kernel.restart()
                print("kernel restarted; in-memory state is gone")
                continue
            if line == "/new":
                session.reset_client_session()
                print("new session, fresh kernel")
                continue

            task = asyncio.create_task(session.query(line))
            session._current_task = task
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                if session._current_task is task:
                    session._current_task = None
    finally:
        session.unsubscribe(render_event)
        session.shutdown()


def main():
    args = build_parser().parse_args()
    try:
        asyncio.run(run_terminal(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
