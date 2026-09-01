from __future__ import annotations

from adlearn.detection.report import (
    STATUS_EMPTY,
    STATUS_OK,
    STATUS_WEAK,
    ImageReport,
    classify,
    review_order,
)


def report(*, file: str, boxes: int, min_confidence: float, status: str) -> ImageReport:
    return ImageReport(
        file=file,
        width=100,
        height=100,
        boxes=boxes,
        min_confidence=min_confidence,
        max_confidence=0.99,
        status=status,
    )


def test_status_splits_the_work() -> None:
    assert classify(boxes=0, min_confidence=0.0, weak_below=0.5) == STATUS_EMPTY
    assert classify(boxes=2, min_confidence=0.31, weak_below=0.5) == STATUS_WEAK
    assert classify(boxes=2, min_confidence=0.77, weak_below=0.5) == STATUS_OK


def test_review_starts_with_empty_frames_then_the_shakiest() -> None:
    reports = [
        report(file="ok.jpg", boxes=1, min_confidence=0.9, status=STATUS_OK),
        report(file="weak-high.jpg", boxes=1, min_confidence=0.45, status=STATUS_WEAK),
        report(file="empty.jpg", boxes=0, min_confidence=0.0, status=STATUS_EMPTY),
        report(file="weak-low.jpg", boxes=1, min_confidence=0.26, status=STATUS_WEAK),
    ]

    order = [item.file for item in review_order(reports)]

    assert order == ["empty.jpg", "weak-low.jpg", "weak-high.jpg"]
