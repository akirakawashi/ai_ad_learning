"""Настройки детекции. Все числа живут здесь, как в соседнем ai_ad_ml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from adlearn import paths

CLASS_NAME = "ad_object"

TASK = paths.DETECTION


@dataclass(frozen=True)
class PrelabelConfig:
    """Прогон папки фотографий через детектор щитов.

    `image_size` и `iou` повторяют боевые значения из `DetectionConfig`
    пайплайна: разметка должна получиться такой же, какую увидит рабочая модель,
    иначе правится не то.

    А вот `confidence_min` намеренно ниже боевого порога 0.50. Стереть лишнюю
    рамку в разметчике — одно движение, нарисовать пропущенную — десять. Всё, что
    слабее `confidence_weak`, помечается в отчёте как требующее взгляда: сам
    YOLO-формат уверенность не хранит, и после импорта в CVAT рамка на 0.26
    выглядит ровно так же, как рамка на 0.95.
    """

    weights: Path = TASK.weights
    source: Path = paths.RAW
    output: Path = TASK.prelabel
    class_name: str = CLASS_NAME
    image_size: int = 960
    confidence_min: float = 0.25
    confidence_weak: float = 0.50
    iou: float = 0.50
    batch_size: int = 16
    device: str | None = "0"

    @property
    def labels_dir(self) -> Path:
        return self.output / "labels"

    @property
    def images_dir(self) -> Path:
        return self.output / "images"

    @property
    def report_path(self) -> Path:
        return self.output / "report.csv"

    @property
    def review_path(self) -> Path:
        return self.output / "review.txt"

    @property
    def cvat_archive(self) -> Path:
        return self.output / "cvat_annotations.zip"


@dataclass(frozen=True)
class DatasetConfig:
    """Сборка набора из ручной разметки и уверенной псевдоразметки."""

    export_archive: Path
    prelabel: Path = TASK.prelabel
    output: Path = TASK.dataset
    class_name: str = CLASS_NAME
    validation_share: float = 0.15
    test_share: float = 0.15
    seed: int = 0


@dataclass(frozen=True)
class TrainConfig:
    """Настройки обучения.

    `weights` по умолчанию — предобученная `yolo11m`, а не текущая рабочая
    модель. Дообучать модель на разметке, которую она же и поставила, почти
    бессмысленно: она подтвердит собственные ответы и закрепит собственные
    промахи. Новый набор приносит другие ракурсы и форматы щитов, и им нужен вес,
    а не поправка к старым весам. Продолжить с боевых весов можно, передав их
    явно, — тогда обучение станет коротким дообучением.

    `image_size` повторяет боевое значение: модель, обученная в другом
    разрешении, в пайплайне поведёт себя иначе.
    """

    data: Path = field(default_factory=lambda: TASK.dataset / "data.yaml")
    weights: str = str(paths.PRETRAINED / "yolo11m.pt")
    epochs: int = 120
    image_size: int = 960
    batch: int = 10
    patience: int = 20
    learning_rate: float = 0.001
    device: str = "0"
    project: Path = paths.RUNS
    name: str = "ad_object_v2"
    seed: int = 0
