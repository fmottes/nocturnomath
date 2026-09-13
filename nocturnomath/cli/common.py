"""Arguments shared by the web and terminal commands."""

import argparse

from ..session import EFFORT_LEVELS


def add_session_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--path",
        "-p",
        default=".",
        help="workspace directory (default: current directory)",
    )
    parser.add_argument(
        "--python",
        help="research environment Python executable (or managed); remembered per workspace",
    )
    parser.add_argument(
        "--default-model",
        default=None,
        help="exact Claude model identifier for new sessions (default: SDK choice)",
    )
    parser.add_argument(
        "--default-effort",
        choices=EFFORT_LEVELS,
        default=None,
        help="reasoning effort for new sessions (default: high)",
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="seconds allowed per kernel run"
    )
    parser.add_argument(
        "--images", type=int, default=2, help="plots returned to Claude per run"
    )
