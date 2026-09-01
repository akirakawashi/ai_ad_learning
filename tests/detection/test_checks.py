from __future__ import annotations

from pathlib import Path

from adlearn.detection.checks import Findings, check_label, inspect


def make_split(root: Path, split: str, frames: dict[str, str | None]) -> None:
    images = root / "images" / split
    labels = root / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for name, text in frames.items():
        (images / f"{name}.jpg").write_bytes(name.encode())
        if text is not None:
            (labels / f"{name}.txt").write_text(text, encoding="utf-8")


def test_a_clean_dataset_has_nothing_to_report(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        make_split(tmp_path, split, {f"{split}_frame": "0 0.5 0.5 0.2 0.2\n"})

    findings, boxes = inspect(tmp_path, read_images=False)

    assert findings.clean
    assert boxes == {"train": 1, "val": 1, "test": 1}


def test_the_same_photo_in_two_parts_is_caught(tmp_path: Path) -> None:
    """Кадр в обучении и в тесте сразу поднимает метрику на пустом месте."""

    make_split(tmp_path, "train", {"a": ""})
    make_split(tmp_path, "val", {"b": ""})
    make_split(tmp_path, "test", {"a": ""})

    findings, _ = inspect(tmp_path, read_images=False)

    assert findings.name_in_two_splits == ["a.jpg"]
    assert findings.same_photo_in_two_splits


def test_a_frame_without_a_label_and_a_label_without_a_frame(tmp_path: Path) -> None:
    for split in ("val", "test"):
        make_split(tmp_path, split, {split: ""})
    make_split(tmp_path, "train", {"lonely": None})
    (tmp_path / "labels" / "train" / "orphan.txt").write_text("", encoding="utf-8")

    findings, _ = inspect(tmp_path, read_images=False)

    assert findings.missing_label == ["train/lonely.jpg"]
    assert findings.missing_image == ["train/orphan.txt"]


def test_broken_lines_and_impossible_boxes() -> None:
    findings = Findings()
    check_label(
        "0 0.5 0.5\n1 0.5 0.5 0.2 0.2\n0 1.5 0.5 0.2 0.2\n0 0.5 0.5 0.0 0.2\n",
        source="train/frame.jpg",
        findings=findings,
    )

    assert len(findings.bad_line) == 2
    assert findings.out_of_range == ["train/frame.jpg:3"]
    assert findings.degenerate == ["train/frame.jpg:4"]


def test_an_empty_dataset_is_not_clean(tmp_path: Path) -> None:
    """«Чисто» на пустом наборе прямо перед обучением — худший из ответов."""

    findings, _ = inspect(tmp_path, read_images=False)

    assert not findings.clean
    assert findings.empty_split == ["train", "val", "test"]
