"""Arguments shared by the web and terminal commands."""

import argparse


def add_session_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--path",
        "-p",
        default=".",
        help="workspace directory (default: current directory)",
    )
    parser.add_argument("--model", default="claude-opus-5", help="Claude model to use")
    parser.add_argument(
        "--timeout", type=int, default=600, help="seconds allowed per kernel run"
    )
    parser.add_argument(
        "--images", type=int, default=2, help="plots returned to Claude per run"
    )
