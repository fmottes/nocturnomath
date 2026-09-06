from xprober.cli.terminal import build_parser as terminal_parser
from xprober.cli.web import build_parser as web_parser


def test_web_command_options():
    args = web_parser().parse_args(
        ["--path", "/tmp/project", "--timeout", "10", "--images", "1", "--port", "9000"]
    )
    assert args.path == "/tmp/project"
    assert args.timeout == 10
    assert args.images == 1
    assert args.port == 9000


def test_terminal_command_uses_shared_options():
    args = terminal_parser().parse_args(
        ["--path", "/tmp/project", "--model", "test", "--timeout", "20"]
    )
    assert args.path == "/tmp/project"
    assert args.model == "test"
    assert args.timeout == 20
