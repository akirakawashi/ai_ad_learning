"""Отчёт по каждой фотографии и порядок ручной проверки."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

STATUS_EMPTY = "пусто"
STATUS_WEAK = "слабая"
STATUS_OK = "ок"

REPORT_FIELDS = (
    "file",
    "width",
    "height",
    "boxes",
    "min_confidence",
    "max_confidence",
    "status",
)


@dataclass(frozen=True)
class ImageReport:
    file: str
    width: int
    height: int
    boxes: int
    min_confidence: float
    max_confidence: float
    status: str


def classify(*, boxes: int, min_confidence: float, weak_below: float) -> str:
    """Кому достанется ручная правка.

    Пустой кадр подозрителен сам по себе: в папке лежат щиты, значит модель либо
    промахнулась, либо щит на фотографии нетипичный — и то и другое интересно.
    """

    if boxes == 0:
        return STATUS_EMPTY
    if min_confidence < weak_below:
        return STATUS_WEAK
    return STATUS_OK


def review_order(reports: list[ImageReport]) -> list[ImageReport]:
    """Сначала пустые кадры, затем слабые от самой сомнительной рамки."""

    empty = [item for item in reports if item.status == STATUS_EMPTY]
    weak = sorted(
        (item for item in reports if item.status == STATUS_WEAK),
        key=lambda item: item.min_confidence,
    )
    return empty + weak


def write_report(*, reports: list[ImageReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REPORT_FIELDS))
        writer.writeheader()
        for item in reports:
            writer.writerow(
                {
                    "file": item.file,
                    "width": item.width,
                    "height": item.height,
                    "boxes": item.boxes,
                    "min_confidence": f"{item.min_confidence:.4f}",
                    "max_confidence": f"{item.max_confidence:.4f}",
                    "status": item.status,
                }
            )


def write_review_list(*, reports: list[ImageReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{item.status}\t{item.min_confidence:.2f}\t{item.file}" for item in review_order(reports)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def summary(reports: list[ImageReport]) -> dict[str, int]:
    counts = {STATUS_EMPTY: 0, STATUS_WEAK: 0, STATUS_OK: 0}
    for item in reports:
        counts[item.status] += 1
    return counts
