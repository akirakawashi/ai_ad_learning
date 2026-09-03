"""Набор кадров по брендам и деление на части.

Кадры лежат папками по классам — так же, как их отдал разметчик. Никаких
симлинков и пересборки: набор маленький, читаем прямо из `raw`.

Класс может быть разложен по подпапкам: `other/Магнит`, `other/Сбер` и так далее.
Для обучения это по-прежнему один класс, но имя подпапки запоминается. Без него
нельзя ответить на главный вопрос отчёта — не «сколько ошибок», а «кого именно
модель принимает за МегаФон».
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from adlearn.classification.config import BRANDS
from adlearn.core.images import find_images


@dataclass(frozen=True)
class Sample:
    path: Path
    brand: str
    source: str = ""
    """Подпапка внутри класса: конкретный бренд или вид щита. Пусто, если класс плоский."""

    @property
    def id(self) -> str:
        return (
            f"{self.brand}/{self.source}/{self.path.name}"
            if self.source
            else (f"{self.brand}/{self.path.name}")
        )


def collect(raw: Path, *, brands: tuple[str, ...] = BRANDS) -> list[Sample]:
    """Все кадры набора в устойчивом порядке.

    По умолчанию берутся классы обученной головы. VLM знает больше брендов, и для
    её проверки список передаётся отдельно — папки с теми же именами лежат в том же
    `raw/`.
    """

    samples: list[Sample] = []
    for brand in brands:
        directory = raw / brand
        if not directory.is_dir():
            raise FileNotFoundError(f"Нет папки бренда: {directory}")
        for path in find_images(directory, recursive=True):
            source = path.parent.name if path.parent != directory else ""
            samples.append(Sample(path=path, brand=brand, source=source))
    if not samples:
        raise FileNotFoundError(f"В {raw} нет кадров.")
    return samples


def sources(samples: list[Sample]) -> list[str]:
    """Источник каждого кадра — для разбора ошибок по конкретным брендам."""

    return [item.source or item.brand for item in samples]


def labels(samples: list[Sample]) -> np.ndarray:
    order = {brand: index for index, brand in enumerate(BRANDS)}
    return np.array([order[item.brand] for item in samples], dtype=np.int64)


def duplicate_groups(embeddings: np.ndarray, *, threshold: float) -> np.ndarray:
    """Объединяет почти одинаковые кадры в группы по косинусной близости.

    Набор собран из интернета, и одна и та же картинка приходит в разных
    размерах и пережатиях. Если такие кадры разъедутся между обучением и тестом,
    метрика вырастет на пустом месте — модель просто узнает виденное.

    Хеши здесь не работают: логотип на белом фоне даёт почти одинаковый
    перцептивный хеш у разных картинок, и группы получаются ложные. Эмбеддинг
    смотрит на содержание, а не на раскладку яркости.
    """

    normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    parent = np.arange(len(normalized))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    step = 512
    for start in range(0, len(normalized), step):
        block = normalized[start : start + step] @ normalized.T
        for row, column in zip(*np.where(block >= threshold), strict=True):
            first, second = find(start + int(row)), find(int(column))
            if first != second:
                parent[first] = second
    return np.array([find(index) for index in range(len(normalized))])
