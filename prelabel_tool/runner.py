"""Прогон папки фотографий через детектор и сборка всего, что нужно разметчику."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from ultralytics import YOLO

from prelabel_tool.config import IMAGE_SUFFIXES, PrelabelConfig
from prelabel_tool.labels import (
    pack_for_cvat,
    write_dataset_yaml,
    write_label,
    yolo_line,
)
from prelabel_tool.report import ImageReport, classify, write_report, write_review_list

logger = logging.getLogger("prelabel")


def find_images(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def batched(paths: list[Path], size: int) -> Iterator[list[Path]]:
    for start in range(0, len(paths), size):
        yield paths[start : start + size]


def load_detector(config: PrelabelConfig) -> YOLO:
    if not config.weights.exists():
        raise FileNotFoundError(config.weights)
    return YOLO(str(config.weights))


def link_image(*, source: Path, images_dir: Path) -> None:
    """Кадр попадает в датасет ссылкой, а не копией.

    Обучение читает через симлинк так же, как через файл, а два с половиной
    гигабайта на диске остаются в одном экземпляре.
    """

    images_dir.mkdir(parents=True, exist_ok=True)
    link = images_dir / source.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(source.resolve())


def run(config: PrelabelConfig) -> list[ImageReport]:
    images = find_images(config.source)
    if not images:
        raise FileNotFoundError(f"В {config.source} нет фотографий.")

    model = load_detector(config)
    labels_dir = config.output / "labels"
    images_dir = config.output / "images"
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
                    x1, y1, x2, y2 = (
                        float(value) for value in box.xyxy[0].detach().cpu().tolist()
                    )
                    lines.append(
                        yolo_line(box=(x1, y1, x2, y2), width=width, height=height)
                    )
                    confidences.append(float(box.conf[0]))

            write_label(path=labels_dir / f"{path.stem}.txt", lines=lines)
            link_image(source=path, images_dir=images_dir)
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

    write_report(reports=reports, path=config.output / "report.csv")
    write_review_list(reports=reports, path=config.output / "review.txt")
    write_dataset_yaml(
        path=config.output / "data.yaml",
        images_dir=images_dir,
        class_name=config.class_name,
    )
    pack_for_cvat(
        archive=config.output / "cvat_annotations.zip",
        labels_dir=labels_dir,
        image_names=[path.name for path in images],
        class_name=config.class_name,
    )
    return reports
