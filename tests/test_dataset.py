from __future__ import annotations

import csv
from pathlib import Path

from training.dataset import (
    Sample,
    collect_confident,
    group_of,
    has_boxes,
    split_pool,
)


def sample(tmp_path: Path, name: str, *, boxes: int) -> Sample:
    image = tmp_path / name
    image.write_bytes(b"")
    label = tmp_path / f"{Path(name).stem}.txt"
    label.write_text("\n".join(["0 0.5 0.5 0.2 0.2"] * boxes), encoding="utf-8")
    return Sample(image=image, label=label, handmade=True)


def test_group_is_read_from_the_name() -> None:
    assert group_of("video_hard_000187.jpg") == "video_hard"
    assert group_of("billboard_ad_052.jpg") == "billboard_ad"
    assert group_of("3f9c1a2b8e04.jpg") == "raw"
    assert group_of("351.png") == "numbers"


def test_empty_label_means_no_billboards(tmp_path: Path) -> None:
    filled = sample(tmp_path, "a.jpg", boxes=1)
    blank = sample(tmp_path, "b.jpg", boxes=0)

    assert has_boxes(filled.label)
    assert not has_boxes(blank.label)
    assert not has_boxes(tmp_path / "missing.txt")


def test_frames_without_boxes_are_spread_across_all_three_parts(tmp_path: Path) -> None:
    """Пустых кадров четверть набора, и они нужны каждой части.

    Собравшись в одной, они перекосили бы всё сразу: обучение недосчитается
    отрицательных примеров, а проверка перестанет ловить выдуманные щиты.
    """

    samples = [sample(tmp_path, f"photo_{i:03d}.jpg", boxes=1) for i in range(30)]
    samples += [sample(tmp_path, f"photo_{i:03d}.jpg", boxes=0) for i in range(30, 40)]

    train, validation, test = split_pool(samples)

    for part in (train, validation, test):
        empty = sum(1 for item in part if not has_boxes(item.label))
        assert 0.15 < empty / len(part) < 0.35


def test_every_group_reaches_every_part(tmp_path: Path) -> None:
    samples = [sample(tmp_path, f"photo_{i:03d}.jpg", boxes=1) for i in range(20)]
    samples += [sample(tmp_path, f"video_hard_{i:03d}.jpg", boxes=1) for i in range(20)]

    train, validation, test = split_pool(samples)

    for part in (train, validation, test):
        assert {group_of(item.image.name) for item in part} == {"photo", "video_hard"}


def test_split_repeats_itself(tmp_path: Path) -> None:
    samples = [sample(tmp_path, f"photo_{i:03d}.jpg", boxes=1) for i in range(50)]

    assert split_pool(samples, seed=7) == split_pool(samples, seed=7)


def test_a_tiny_stratum_stays_in_training(tmp_path: Path) -> None:
    """Пара кадров одного рода целиком идёт в обучение.

    Тест из одного кадра ничего не измеряет, а обучение теряет и этот кадр.
    """

    samples = [sample(tmp_path, f"rare_{i:03d}.jpg", boxes=1) for i in range(2)]

    train, validation, test = split_pool(samples)

    assert len(train) == 2
    assert validation == []
    assert test == []


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

    import zipfile

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
        writer = csv.writer(handle)
        writer.writerow(["file", "status"])
    return prelabel


def test_rebuild_does_not_leave_frames_in_two_parts(tmp_path: Path) -> None:
    """Пересборка с другим жребием не должна оставлять ссылки от прошлого деления.

    Иначе кадр окажется сразу в обучении и в тесте, метрика вырастет на пустом
    месте, а заметить это можно только отдельной проверкой.
    """

    from training.dataset import build

    export = make_export(tmp_path, frames=40)
    prelabel = make_prelabel(tmp_path)
    output = tmp_path / "dataset"

    build(export_archive=export, prelabel_dir=prelabel, output=output, seed=1)
    build(export_archive=export, prelabel_dir=prelabel, output=output, seed=99)

    seen: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for image in (output / "images" / split).iterdir():
            assert image.name not in seen, f"{image.name} и в {seen.get(image.name)}, и в {split}"
            seen[image.name] = split
    assert len(seen) == 40
