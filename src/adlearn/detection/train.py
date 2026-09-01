"""Обучение детектора щитов и сравнение весов на отложенных частях."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from ultralytics import YOLO

from adlearn.detection.config import TrainConfig

METRIC_KEYS = ("mAP50", "mAP50-95", "precision", "recall")


def train(config: TrainConfig) -> Path:
    """Учит модель и отдаёт путь к лучшим весам."""

    if not config.data.exists():
        raise FileNotFoundError(config.data)
    model = YOLO(config.weights)
    results = cast(Any, model).train(
        data=str(config.data.resolve()),
        epochs=config.epochs,
        imgsz=config.image_size,
        batch=config.batch,
        patience=config.patience,
        lr0=config.learning_rate,
        device=config.device,
        project=str(config.project),
        name=config.name,
        seed=config.seed,
        pretrained=True,
        exist_ok=True,
    )
    return Path(results.save_dir) / "weights" / "best.pt"


def evaluate(
    *, weights: Path, data: Path, image_size: int, device: str, split: str = "test_handmade"
) -> dict[str, float]:
    """Считает метрики на отложенной части.

    По умолчанию берётся `test_handmade` — единственная часть, где разметку
    поставил человек. `test` добавит к ней кадры с машинной разметкой, и там
    метрика измеряет согласие с текущей моделью, а не правоту. `val` покажет ту
    часть, на которую опиралось обучение: по ней выбиралась лучшая эпоха, и цифра
    на ней выше настоящей.
    """

    model = YOLO(str(weights))
    metrics = cast(Any, model).val(
        data=str(data.resolve()),
        imgsz=image_size,
        device=device,
        split=split,
        verbose=False,
    )
    return {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
