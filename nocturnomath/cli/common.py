"""Arguments shared by the web and terminal commands."""

import argparse
from importlib.metadata import version

from ..backends.base import EFFORT_LEVELS


def add_session_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('nocturnomath')}",
        help="show the installed Nocturnomath version and exit",
    )
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
        help="exact model identifier for new sessions (default: backend choice)",
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
        "--images", type=int, default=2, help="plots returned to the model per run"
    )
