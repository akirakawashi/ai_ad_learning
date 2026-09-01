"""Каркас командной строки: одна точка входа, задачи и команды внутри."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence

Handler = Callable[[argparse.Namespace], int]

Subparsers = argparse._SubParsersAction  # type: ignore[type-arg]


def command(
    commands: Subparsers,
    name: str,
    *,
    help: str,
    handler: Handler,
) -> argparse.ArgumentParser:
    """Заводит команду и привязывает к ней обработчик."""

    parser = commands.add_parser(name, help=help, description=help)
    parser.set_defaults(handler=handler)
    return parser


def dispatch(parser: argparse.ArgumentParser, argv: Sequence[str] | None) -> int:
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(message)s",
    )
    handler: Handler = args.handler
    return handler(args)
