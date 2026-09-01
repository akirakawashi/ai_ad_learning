"""Настройки классификации брендов. Все числа живут здесь."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from adlearn import paths

BRANDS = ("beeline", "megafon", "tele2")
"""Порядок классов фиксирован: по нему нумеруются метки и строки confusion matrix."""

TASK = paths.CLASSIFICATION

UNSURE = "не уверен"


@dataclass(frozen=True)
class ClassificationConfig:
    """Общие пути и параметры задачи."""

    raw: Path = TASK.root / "raw"
    features: Path = TASK.root / "features"
    model: Path = TASK.root / "model.joblib"
    runs: Path = TASK.root / "runs"
    device: str = "cuda"
    batch_size: int = 32
    seed: int = 0

    test_share: float = 0.2
    folds: int = 5
    repeats: int = 5

    near_duplicate_similarity: float = 0.97
    """Косинус выше этого — считаем кадры одной съёмкой и не разводим по частям."""

    confidence_min: float = 0.55
    """Ниже этого predict отвечает «не уверен», а не выдумывает бренд."""
