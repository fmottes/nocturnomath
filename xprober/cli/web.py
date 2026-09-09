"""Launch the local Xprober dashboard."""

import argparse
import logging
import os
from pathlib import Path

import uvicorn

from ..web import create_app
from .common import add_session_arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_session_arguments(parser)
    parser.add_argument("--host", default="127.0.0.1", help="server bind address")
    parser.add_argument("--port", type=int, default=8000, help="server port")
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser on startup"
    )
    return parser


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_parser().parse_args()
    navigator_root = Path(args.path).expanduser().resolve()
    logger = logging.getLogger("xprober.web")
    logger.info("Starting Xprober; choose a workspace in the browser.")
    if os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning(
            "ANTHROPIC_API_KEY is set and silently takes precedence over your Claude subscription."
        )

    url = f"http://{args.host}:{args.port}"
    app = create_app(
        browser_url=None if args.no_browser else url,
        model=args.model,
        timeout_s=args.timeout,
        image_cap=args.images,
        navigator_root=navigator_root,
    )
    logger.info(f"Xprober Web App ready at: {url}")
    server = uvicorn.Server(
        uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    )
    app.state.request_shutdown = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()
