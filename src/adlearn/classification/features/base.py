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
        """Имена измерений — нужны, чтобы читать веса обученной модели.

        Returns:
            Имена измерений признакового блока.
        """

    def __call__(self, paths: Sequence[Path]) -> np.ndarray:
        """Матрица (кадров × len(dims)) в том же порядке, что и `paths`.

        Args:
            paths: Пути к кадрам в требуемом порядке.

        Returns:
            Матрица признаков: по строке на каждый кадр.
        """


REGISTRY: dict[str, Callable[[], Extractor]] = {}


def register(name: str) -> Callable[[Callable[[], Extractor]], Callable[[], Extractor]]:
    """Создаёт декоратор регистрации фабрики экстрактора.

    Args:
        name: Имя экстрактора в реестре.

    Returns:
        Декоратор, который сохранит фабрику под указанным именем.
    """

    def wrap(factory: Callable[[], Extractor]) -> Callable[[], Extractor]:
        """Регистрирует фабрику экстрактора.

        Args:
            factory: Фабрика без аргументов.

        Returns:
            Та же фабрика без обёртки.
        """

        REGISTRY[name] = factory
        return factory

    return wrap


def get(name: str) -> Extractor:
    """Создаёт зарегистрированный экстрактор по имени.

    Args:
        name: Имя экстрактора в реестре.

    Returns:
        Новый экземпляр выбранного экстрактора.
    """

    if name not in REGISTRY:
        raise KeyError(f"Экстрактор {name!r} не зарегистрирован. Есть: {sorted(REGISTRY)}")
    return REGISTRY[name]()
