"""Реестр экстракторов признаков.

Экстрактор — это имя, версия, список имён измерений и функция «пачка кадров →
матрица признаков». Всё остальное в задаче про него ничего не знает.

Такой плоский контракт и есть точка расширения: чтобы позже добавить logo
similarity или сходство с эталонами бренда, достаточно написать ещё один
экстрактор и вписать его имя в состав армы. Ни набор, ни голова, ни абляция при
этом не меняются.

Версия входит в ключ кэша: поправил экстрактор — поднял версию, и пересчитается
только он, а не всё остальное.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Extractor(Protocol):
    name: str
    version: str

    @property
    def dims(self) -> list[str]:
        """Имена измерений — нужны, чтобы читать веса обученной модели."""

    def __call__(self, paths: Sequence[Path]) -> np.ndarray:
        """Матрица (кадров × len(dims)) в том же порядке, что и `paths`."""


REGISTRY: dict[str, Callable[[], Extractor]] = {}


def register(name: str) -> Callable[[Callable[[], Extractor]], Callable[[], Extractor]]:
    def wrap(factory: Callable[[], Extractor]) -> Callable[[], Extractor]:
        REGISTRY[name] = factory
        return factory

    return wrap


def get(name: str) -> Extractor:
    if name not in REGISTRY:
        raise KeyError(f"Экстрактор {name!r} не зарегистрирован. Есть: {sorted(REGISTRY)}")
    return REGISTRY[name]()
