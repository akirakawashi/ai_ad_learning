"""Кэш признаков на диске.

Эмбеддинги считаются один раз и живут в `.npz`; после этого все армы абляции
обучаются за секунды на готовой матрице. Без кэша каждая арма тянула бы за собой
прогон энкодера, и повторная кросс-валидация стала бы неподъёмной.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from adlearn.classification.features.base import Extractor


def cache_path(directory: Path, extractor: Extractor) -> Path:
    return directory / f"{extractor.name}-v{extractor.version}.npz"


def load(path: Path) -> tuple[list[str], np.ndarray, list[str]] | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return list(data["ids"]), data["values"], list(data["dims"])


def save(*, path: Path, ids: Sequence[str], values: np.ndarray, dims: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.array(list(ids)),
        values=values.astype(np.float32),
        dims=np.array(list(dims)),
    )


def compute(
    *,
    extractor: Extractor,
    paths: Sequence[Path],
    ids: Sequence[str],
    directory: Path,
    refresh: bool = False,
) -> np.ndarray:
    """Признаки для набора кадров, по возможности из кэша.

    Кэш пересчитывается целиком, если состав кадров изменился: частичное
    доливание молча смешало бы две версии набора.
    """

    path = cache_path(directory, extractor)
    if not refresh:
        cached = load(path)
        if cached is not None and cached[0] == list(ids):
            return cached[1]
    values = extractor(paths)
    if values.shape != (len(paths), len(extractor.dims)):
        raise ValueError(
            f"{extractor.name} вернул {values.shape}, ожидалось {(len(paths), len(extractor.dims))}"
        )
    save(path=path, ids=ids, values=values, dims=extractor.dims)
    return values
