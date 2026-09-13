"""Launch the local Nocturnomath dashboard."""

import argparse
import ipaddress
import logging
from pathlib import Path

import uvicorn

from ..web import create_app
from .common import add_session_arguments


def loopback_host(value: str) -> str:
    if value == "localhost":
        return value
    try:
        if ipaddress.ip_address(value) in (
            ipaddress.ip_address("127.0.0.1"),
            ipaddress.ip_address("::1"),
        ):
            return value
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(
        "the web interface must bind to a loopback address for SSH forwarding"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_session_arguments(parser)
    parser.add_argument(
        "--host",
        type=loopback_host,
        default="127.0.0.1",
        help="loopback server bind address (default: 127.0.0.1)",
    )
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
    logger = logging.getLogger("nocturnomath.web")
    logger.info("Starting Nocturnomath; choose a workspace in the browser.")
    display_host = f"[{args.host}]" if ":" in args.host else args.host
    url = f"http://{display_host}:{args.port}"
    app = create_app(
        browser_url=None if args.no_browser else url,
        model=args.default_model,
        effort=args.default_effort,
        timeout_s=args.timeout,
        image_cap=args.images,
        navigator_root=navigator_root,
        python=args.python,
    )
    logger.info(f"Nocturnomath Web App ready at: {url}")
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            proxy_headers=False,
        )
    )
    app.state.request_shutdown = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()
