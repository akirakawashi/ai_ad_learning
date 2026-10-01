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
    """Заводит команду и привязывает к ней обработчик.

    Args:
        commands: Набор подкоманд родительского парсера.
        name: Имя новой команды.
        help: Краткое и полное описание команды.
        handler: Функция, которую нужно вызвать после разбора аргументов.

    Returns:
        Парсер созданной команды для добавления её аргументов.
    """

    parser = commands.add_parser(name, help=help, description=help)
    parser.set_defaults(handler=handler)
    return parser


def dispatch(parser: argparse.ArgumentParser, argv: Sequence[str] | None) -> int:
    """Разбирает аргументы, настраивает журнал и вызывает обработчик команды.

    Args:
        parser: Готовый корневой парсер.
        argv: Аргументы без имени программы или `None` для чтения `sys.argv`.

    Returns:
        Код завершения обработчика.
    """
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(message)s",
    )
    handler: Handler = args.handler
    return handler(args)
