from __future__ import annotations

import csv
import zipfile
from pathlib import Path

from adlearn.core.grouping import group_of
from adlearn.detection.config import DatasetConfig
from adlearn.detection.dataset import (
    Sample,
    build,
    collect_confident,
    split_pool,
)
from adlearn.detection.labels import has_boxes


def sample(tmp_path: Path, name: str, *, boxes: int) -> Sample:
    image = tmp_path / name
    image.write_bytes(b"")
    label = tmp_path / f"{Path(name).stem}.txt"
    label.write_text("\n".join(["0 0.5 0.5 0.2 0.2"] * boxes), encoding="utf-8")
    return Sample(image=image, label=label, handmade=True)


def test_frames_without_boxes_are_spread_across_all_three_parts(tmp_path: Path) -> None:
    """Пустых кадров четверть набора, и они нужны каждой части.

    Собравшись в одной, они перекосили бы всё сразу: обучение недосчитается
    отрицательных примеров, а проверка перестанет ловить выдуманные щиты.
    """

    samples = [sample(tmp_path, f"photo_{i:03d}.jpg", boxes=1) for i in range(30)]
    samples += [sample(tmp_path, f"photo_{i:03d}.jpg", boxes=0) for i in range(30, 40)]

    for part in split_pool(samples):
        empty = sum(1 for item in part if not has_boxes(item.label))
        assert 0.15 < empty / len(part) < 0.35


def test_every_group_reaches_every_part(tmp_path: Path) -> None:
    samples = [sample(tmp_path, f"photo_{i:03d}.jpg", boxes=1) for i in range(20)]
    samples += [sample(tmp_path, f"video_hard_{i:03d}.jpg", boxes=1) for i in range(20)]

    for part in split_pool(samples):
        assert {group_of(item.image.name) for item in part} == {"photo", "video_hard"}


def test_only_confident_frames_are_taken(tmp_path: Path) -> None:
    """Кадры, где модель сомневалась, в набор не идут — они ждут человека."""

    prelabel = tmp_path / "prelabel"
    (prelabel / "images").mkdir(parents=True)
    (prelabel / "labels").mkdir(parents=True)
    for name in ("sure.jpg", "shaky.jpg", "blank.jpg"):
        (prelabel / "images" / name).write_bytes(b"")
        (prelabel / "labels" / f"{Path(name).stem}.txt").write_text("", encoding="utf-8")

    with (prelabel / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "status"])
        writer.writerow(["sure.jpg", "ок"])
        writer.writerow(["shaky.jpg", "слабая"])
        writer.writerow(["blank.jpg", "пусто"])

    chosen = collect_confident(prelabel_dir=prelabel)

    assert [item.image.name for item in chosen] == ["sure.jpg"]


def make_export(tmp_path: Path, *, frames: int) -> Path:
    """Мини-выгрузка CVAT: кадры и разметка в папке obj_train_data."""

    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for index in range(frames):
            bundle.writestr(f"obj_train_data/photo_{index:03d}.jpg", b"")
            bundle.writestr(f"obj_train_data/photo_{index:03d}.txt", "0 0.5 0.5 0.2 0.2\n")
    return archive


def make_prelabel(tmp_path: Path) -> Path:
    prelabel = tmp_path / "prelabel"
    (prelabel / "images").mkdir(parents=True)
    (prelabel / "labels").mkdir(parents=True)
    with (prelabel / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(["file", "status"])
    return prelabel


def test_rebuild_does_not_leave_frames_in_two_parts(tmp_path: Path) -> None:
    """Пересборка с другим жребием не должна оставлять ссылки от прошлого деления.

    Иначе кадр окажется сразу в обучении и в тесте, метрика вырастет на пустом
    месте, а заметить это можно только отдельной проверкой.
    """

    export = make_export(tmp_path, frames=40)
    prelabel = make_prelabel(tmp_path)
    output = tmp_path / "dataset"
    config = DatasetConfig(export_archive=export, prelabel=prelabel, output=output)

    build(config)
    build(DatasetConfig(**{**vars(config), "seed": 99}))

    seen: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for image in (output / "images" / split).iterdir():
            assert image.name not in seen, f"{image.name} и в {seen[image.name]}, и в {split}"
            seen[image.name] = split
    assert len(seen) == 40


def test_dataset_survives_being_moved(tmp_path: Path) -> None:
    """Ссылки внутри набора относительные, значит папку можно перенести целиком."""

    export = make_export(tmp_path, frames=10)
    output = tmp_path / "dataset"
    build(DatasetConfig(export_archive=export, prelabel=make_prelabel(tmp_path), output=output))
    output.rename(tmp_path / "moved")

    images = list((tmp_path / "moved" / "images" / "train").iterdir())
    assert images
    assert all(image.exists() for image in images)
