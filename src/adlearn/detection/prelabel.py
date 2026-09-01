"""Прогон папки фотографий через детектор и сборка всего, что нужно разметчику."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from ultralytics import YOLO

from adlearn.core.images import find_images, link
from adlearn.detection.config import PrelabelConfig
from adlearn.detection.labels import (
    pack_for_cvat,
    write_dataset_yaml,
    write_label,
    yolo_line,
)
from adlearn.detection.report import (
    ImageReport,
    classify,
    write_report,
    write_review_list,
)

logger = logging.getLogger("prelabel")


def batched(paths: list[Path], size: int) -> Iterator[list[Path]]:
    for start in range(0, len(paths), size):
        yield paths[start : start + size]


def load_detector(config: PrelabelConfig) -> YOLO:
    if not config.weights.exists():
        raise FileNotFoundError(config.weights)
    return YOLO(str(config.weights))


def run(config: PrelabelConfig) -> list[ImageReport]:
    images = find_images(config.source)
    if not images:
        raise FileNotFoundError(f"В {config.source} нет фотографий.")

    model = load_detector(config)
    reports: list[ImageReport] = []

    predict_kwargs: dict[str, object] = {
        "conf": config.confidence_min,
        "iou": config.iou,
        "imgsz": config.image_size,
        "verbose": False,
    }
    if config.device is not None:
        predict_kwargs["device"] = config.device

    for chunk in batched(images, config.batch_size):
        results = cast(Any, model).predict([str(path) for path in chunk], **predict_kwargs)
        for path, result in zip(chunk, results, strict=True):
            height, width = result.orig_shape
            lines: list[str] = []
            confidences: list[float] = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].detach().cpu().tolist())
                    lines.append(yolo_line(box=(x1, y1, x2, y2), width=width, height=height))
                    confidences.append(float(box.conf[0]))

            write_label(path=config.labels_dir / f"{path.stem}.txt", lines=lines)
            link(source=path, destination=config.images_dir / path.name)
            reports.append(
                ImageReport(
                    file=path.name,
                    width=width,
                    height=height,
                    boxes=len(lines),
                    min_confidence=min(confidences, default=0.0),
                    max_confidence=max(confidences, default=0.0),
                    status=classify(
                        boxes=len(lines),
                        min_confidence=min(confidences, default=0.0),
                        weak_below=config.confidence_weak,
                    ),
                )
            )
        logger.info("размечено %s из %s", len(reports), len(images))

    write_report(reports=reports, path=config.report_path)
    write_review_list(reports=reports, path=config.review_path)
    write_dataset_yaml(
        path=config.output / "data.yaml",
        images_dir=config.images_dir,
        class_name=config.class_name,
    )
    pack_for_cvat(
        archive=config.cvat_archive,
        labels_dir=config.labels_dir,
        image_names=[path.name for path in images],
        class_name=config.class_name,
    )
    return reports
