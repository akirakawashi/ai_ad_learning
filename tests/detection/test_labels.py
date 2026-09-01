from __future__ import annotations

import zipfile
from pathlib import Path

from adlearn.detection.labels import (
    has_boxes,
    pack_for_cvat,
    read_boxes,
    write_label,
    yolo_line,
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


def test_empty_label_means_no_billboards(tmp_path: Path) -> None:
    write_label(path=tmp_path / "filled.txt", lines=["0 0.5 0.5 0.2 0.2"])
    write_label(path=tmp_path / "blank.txt", lines=[])

    assert has_boxes(tmp_path / "filled.txt")
    assert not has_boxes(tmp_path / "blank.txt")
    assert not has_boxes(tmp_path / "missing.txt")


def test_broken_lines_are_skipped_when_reading(tmp_path: Path) -> None:
    path = tmp_path / "frame.txt"
    path.write_text("0 0.5 0.5 0.2 0.2\nмусор\n0 0.1 0.1\n", encoding="utf-8")

    assert read_boxes(path) == [(0.5, 0.5, 0.2, 0.2)]


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
        assert set(bundle.namelist()) == {
            "obj.names",
            "obj.data",
            "train.txt",
            "obj_train_data/one.txt",
        }
        assert bundle.read("train.txt").decode() == "obj_train_data/one.jpg\n"
        assert bundle.read("obj.names").decode() == "ad_object\n"
