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
    *, weights: Path, data: Path, image_size: int, device: str, split: str = "test_dashcam"
) -> dict[str, float]:
    """Считает метрики на отложенной части.

    По умолчанию берётся `test_dashcam` — съёмка другой камерой в другом месте,
    которой в обучении нет ни одного кадра. Обычный `test` состоит из тех же
    записей, что и обучение: соседние кадры, те же щиты, тот же свет, и метрика
    на нём выходит выше настоящей. `val` покажет ту часть, по которой выбиралась
    лучшая эпоха, там цифра завышена сильнее всего. Разброс между ними и есть
    мера того, насколько метрике можно верить.
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
