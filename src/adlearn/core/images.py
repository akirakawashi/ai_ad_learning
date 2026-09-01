"""Файлы кадров: поиск, ссылки, чистые папки."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def find_images(directory: Path) -> list[Path]:
    """Кадры из папки, в устойчивом порядке."""

    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(path for path in directory.iterdir() if is_image(path))


def link(*, source: Path, destination: Path) -> None:
    """Кадр попадает в набор ссылкой, а не копией.

    Обучение читает через симлинк так же, как через файл, а два с половиной
    гигабайта на диске остаются в одном экземпляре.

    Ссылка ставится относительной: тогда всё дерево `data/` можно перенести или
    переименовать целиком, и ничего не отвалится. Абсолютная ссылка привязала бы
    набор к одному пути на одной машине.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    target = os.path.relpath(source.resolve(), destination.parent.resolve())
    destination.symlink_to(target)


def reset_dir(path: Path) -> Path:
    """Пустая папка на месте старой.

    Чистить обязательно: при пересборке с другим делением ссылки от прошлого
    раза остались бы лежать рядом с новыми, и один и тот же кадр попал бы сразу
    в две части.
    """

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path
