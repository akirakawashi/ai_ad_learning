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


def test_review_accepts_shared_vlm_credentials() -> None:
    """Парсер передаёт имя модели и ключ команде VLM-судьи."""

    args = build_parser().parse_args(
        [
            "detect",
            "review",
            "--stage",
            "judge",
            "--model",
            "qwen3-vl",
            "--api-key",
            "secret",
        ]
    )

    assert args.model == "qwen3-vl"
    assert args.api_key == "secret"
