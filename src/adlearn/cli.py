"""Единая точка входа: `adlearn <задача> <команда>`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from adlearn.classification import cli as classification_cli
from adlearn.core.cli import dispatch
from adlearn.detection import cli as detection_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adlearn",
        description="Подготовка данных и обучение моделей AI Ad.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный вывод")
    tasks = parser.add_subparsers(dest="task", required=True)
    detection_cli.register(tasks)
    classification_cli.register(tasks)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return dispatch(build_parser(), argv)
