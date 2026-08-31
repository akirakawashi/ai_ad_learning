"""Настройки псевдоразметки. Все числа живут здесь, как в соседнем ai_ad_ml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


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

    weights: Path = Path("models/detection/best.pt")
    source: Path = Path("raw")
    output: Path = Path("prelabel")
    class_name: str = "ad_object"
    image_size: int = 960
    confidence_min: float = 0.25
    confidence_weak: float = 0.50
    iou: float = 0.50
    batch_size: int = 16
    device: str | None = "0"
