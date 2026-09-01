from __future__ import annotations

import pytest

from adlearn.cli import build_parser


def test_every_command_is_wired_to_a_handler() -> None:
    """Каждая команда должна что-то делать, а не молча падать при разборе."""

    parser = build_parser()
    commands = [
        ["detect", "prelabel"],
        ["detect", "bundle"],
        ["detect", "build", "--export", "export.zip"],
        ["detect", "check"],
        ["detect", "preview"],
        ["detect", "train"],
        ["detect", "eval", "--weights", "best.pt"],
    ]
    for argv in commands:
        assert callable(parser.parse_args(argv).handler), argv


def test_a_task_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
