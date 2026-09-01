"""Запуск обучения детектора щитов на собранном наборе."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ultralytics import YOLO


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

    data: Path = Path("dataset/data.yaml")
    weights: str = "yolo11m.pt"
    epochs: int = 120
    image_size: int = 960
    batch: int = 10
    patience: int = 20
    learning_rate: float = 0.001
    device: str = "0"
    project: Path = Path("runs")
    name: str = "ad_object_v2"
    seed: int = 0


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
    *, weights: Path, data: Path, image_size: int, device: str, split: str = "test"
) -> dict[str, float]:
    """Считает метрики на отложенном тесте — он размечен руками и не виден обучению.

    Проверочная часть тоже доступна (`split="val"`), но её обучение уже видело:
    по ней выбиралась лучшая эпоха, и цифра на ней выше настоящей.
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
