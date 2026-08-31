from __future__ import annotations

import zipfile
from pathlib import Path

from prelabel_tool.labels import pack_for_cvat, write_label, yolo_line
from prelabel_tool.report import (
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


def test_box_becomes_a_share_of_the_frame() -> None:
    line = yolo_line(box=(10.0, 20.0, 50.0, 60.0), width=100, height=200)
    assert line == "0 0.300000 0.200000 0.400000 0.200000"


def test_box_hanging_over_the_edge_is_trimmed() -> None:
    """Детектор иногда вылезает за край кадра, разметчик такую рамку не примет."""

    line = yolo_line(box=(-30.0, -10.0, 130.0, 90.0), width=100, height=100)
    assert line == "0 0.500000 0.450000 1.000000 0.900000"


def test_empty_label_file_is_still_written(tmp_path: Path) -> None:
    """Пустой файл значит «на кадре ничего нет», отсутствующий — «кадр не размечен».

    Без файла обучение молча пропустит фотографию вместо того, чтобы принять её
    как отрицательный пример.
    """

    path = tmp_path / "labels" / "frame.txt"
    write_label(path=path, lines=[])
    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""


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


def test_cvat_bundle_carries_annotations_without_the_photos(tmp_path: Path) -> None:
    """Фотографии в архив не кладутся: CVAT берёт их из задачи."""

    labels = tmp_path / "labels"
    write_label(path=labels / "one.txt", lines=["0 0.5 0.5 0.2 0.2"])
    archive = tmp_path / "cvat.zip"

    pack_for_cvat(
        archive=archive,
        labels_dir=labels,
        image_names=["one.jpg"],
        class_name="ad_object",
    )

    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        assert names == {"obj.names", "obj.data", "train.txt", "obj_train_data/one.txt"}
        assert bundle.read("train.txt").decode() == "obj_train_data/one.jpg\n"
        assert bundle.read("obj.names").decode() == "ad_object\n"


def test_review_bundle_takes_only_what_needs_a_human(tmp_path: Path) -> None:
    """Уверенные кадры в пачки не попадают, порядок — как в списке проверки."""

    from prelabel_tool.bundle import chunks, files_to_review

    report = tmp_path / "report.csv"
    report.write_text(
        "file,width,height,boxes,min_confidence,max_confidence,status\n"
        "sure.jpg,10,10,1,0.9000,0.9000,ок\n"
        "shaky.jpg,10,10,1,0.4500,0.9000,слабая\n"
        "blank.jpg,10,10,0,0.0000,0.0000,пусто\n"
        "shakier.jpg,10,10,1,0.2600,0.9000,слабая\n",
        encoding="utf-8",
    )

    names = files_to_review(report)

    assert names == ["blank.jpg", "shakier.jpg", "shaky.jpg"]
    assert chunks(names, 2) == [["blank.jpg", "shakier.jpg"], ["shaky.jpg"]]
