"""Сборка обучающего набора: ручная разметка и уверенная псевдоразметка вместе."""

from __future__ import annotations

import csv
import random
import re
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from prelabel_tool.config import IMAGE_SUFFIXES
from prelabel_tool.report import STATUS_OK

CVAT_DATA_DIR = "obj_train_data"
VALIDATION_SHARE = 0.15
TEST_SHARE = 0.15
SPLIT_SEED = 0
HASH_NAME = re.compile(r"^[0-9a-f]{12,}$")


@dataclass(frozen=True)
class Sample:
    """Кадр, его разметка и то, чьей рукой она поставлена."""

    image: Path
    label: Path
    handmade: bool


@dataclass(frozen=True)
class SplitCounts:
    frames: int
    empty: int

    @property
    def empty_share(self) -> float:
        return self.empty / self.frames if self.frames else 0.0


@dataclass(frozen=True)
class DatasetCounts:
    train: SplitCounts
    validation: SplitCounts
    test: SplitCounts
    test_handmade: SplitCounts


def unpack_export(*, archive: Path, destination: Path) -> Path:
    """Распаковывает выгрузку CVAT и отдаёт папку с кадрами и разметкой."""

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    data_dir = destination / CVAT_DATA_DIR
    if not data_dir.is_dir():
        raise FileNotFoundError(f"В выгрузке нет папки {CVAT_DATA_DIR}.")
    return data_dir


def group_of(name: str) -> str:
    """Откуда кадр: `video_hard_000187.jpg` → `video_hard`, `3f9c1a2b8e04.jpg` → `raw`.

    Группа — это съёмка одного рода: кадры с регистратора, студийные фотографии,
    выгрузка из интернета. Деление на части идёт внутри групп, чтобы рабочий
    домен попал во все три, а не только в обучение.
    """

    stem = Path(name).stem
    if HASH_NAME.match(stem.lower()):
        return "raw"
    prefix = re.sub(r"[0-9].*$", "", stem)
    return prefix.strip("_") or "numbers"


def has_boxes(label: Path) -> bool:
    """Есть ли на кадре хоть одна рамка. Пустой файл значит «щитов нет»."""

    if not label.exists():
        return False
    return any(line.strip() for line in label.read_text(encoding="utf-8").splitlines())


def collect_handmade(directory: Path) -> list[Sample]:
    return [
        Sample(image=path, label=directory / f"{path.stem}.txt", handmade=True)
        for path in sorted(directory.iterdir())
        if path.suffix.lower() in IMAGE_SUFFIXES
        and (directory / f"{path.stem}.txt").exists()
    ]


def collect_confident(*, prelabel_dir: Path) -> list[Sample]:
    """Кадры, где модель не сомневалась. Слабые и пустые ждут ручной правки."""

    images_dir = prelabel_dir / "images"
    labels_dir = prelabel_dir / "labels"
    with (prelabel_dir / "report.csv").open(encoding="utf-8") as handle:
        names = [
            row["file"] for row in csv.DictReader(handle) if row["status"] == STATUS_OK
        ]
    samples = []
    for name in names:
        image = images_dir / name
        label = labels_dir / f"{Path(name).stem}.txt"
        if image.exists() and label.exists():
            samples.append(Sample(image=image, label=label, handmade=False))
    return samples


def split_pool(
    samples: list[Sample],
    *,
    validation_share: float = VALIDATION_SHARE,
    test_share: float = TEST_SHARE,
    seed: int = SPLIT_SEED,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Делит набор на обучение, проверку и отложенный тест.

    Проверка ведёт обучение: по ней выбирается лучшая эпоха и срабатывает ранняя
    остановка. Тест не участвует ни в том, ни в другом — он и отвечает, стала ли
    модель лучше.

    Внутри теста отдельно выделяется часть с ручной разметкой. На остальных
    кадрах разметку поставила текущая модель, и метрика там показывает согласие с
    ней, а не правоту: старая модель получает на них почти единицу просто потому,
    что сравнивается сама с собой.

    Делится внутри пар «группа съёмки и наличие рамок». Первое разводит по всем
    частям кадры с регистратора и снимки из интернета, второе — кадры со щитами и
    пустые. Пустых в наборе заметная доля, и это не мусор: на них модель учится не
    выдумывать щит там, где его нет. Собравшись в одной части, они перекосили бы и
    обучение, и метрику.
    """

    by_stratum: dict[tuple[str, bool], list[Sample]] = defaultdict(list)
    for sample in samples:
        by_stratum[(group_of(sample.image.name), has_boxes(sample.label))].append(sample)

    rng = random.Random(seed)
    train: list[Sample] = []
    validation: list[Sample] = []
    test: list[Sample] = []
    for stratum in sorted(by_stratum):
        members = sorted(by_stratum[stratum], key=lambda item: item.image.name)
        rng.shuffle(members)
        if len(members) < 3:
            train.extend(members)
            continue
        validation_size = max(1, round(len(members) * validation_share))
        test_size = max(1, round(len(members) * test_share))
        while validation_size + test_size >= len(members):
            if test_size >= validation_size:
                test_size -= 1
            else:
                validation_size -= 1
        validation.extend(members[:validation_size])
        test.extend(members[validation_size : validation_size + test_size])
        train.extend(members[validation_size + test_size :])
    return train, validation, test


def link_split(*, samples: list[Sample], output: Path, split: str) -> SplitCounts:
    """Раскладывает часть набора ссылками, начиная с чистой папки.

    Чистить обязательно: при пересборке с другим делением ссылки от прошлого раза
    остались бы лежать рядом с новыми, и один и тот же кадр попал бы сразу в две
    части. Проверка это ловит, но набор к тому моменту уже испорчен.
    """

    images_dir = output / "images" / split
    labels_dir = output / "labels" / split
    for directory in (images_dir, labels_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    empty = 0
    for sample in samples:
        for source, link in (
            (sample.image, images_dir / sample.image.name),
            (sample.label, labels_dir / f"{sample.image.stem}.txt"),
        ):
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(source.resolve())
        if not has_boxes(sample.label):
            empty += 1
    return SplitCounts(frames=len(samples), empty=empty)


def write_data_yaml(*, path: Path, root: Path, class_name: str) -> None:
    """Описание набора для ultralytics.

    Свой ключ `test_handmade` записывается полным путём: имена `train`, `val` и
    `test` библиотека достраивает от `path`, а незнакомый ключ оставляет как есть
    и потом не находит папку.
    """

    path.write_text(
        "\n".join(
            [
                f"path: {root.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"test_handmade: {(root / 'images' / 'test_handmade').resolve()}",
                "names:",
                f"  0: {class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build(
    *,
    export_archive: Path,
    prelabel_dir: Path,
    output: Path,
    class_name: str = "ad_object",
    validation_share: float = VALIDATION_SHARE,
    test_share: float = TEST_SHARE,
    seed: int = SPLIT_SEED,
) -> DatasetCounts:
    handmade_dir = unpack_export(archive=export_archive, destination=output / "handmade")
    samples = collect_handmade(handmade_dir) + collect_confident(prelabel_dir=prelabel_dir)
    if not samples:
        raise FileNotFoundError("Нечего собирать: нет ни одного кадра с разметкой.")

    train, validation, test = split_pool(
        samples,
        validation_share=validation_share,
        test_share=test_share,
        seed=seed,
    )
    counts = DatasetCounts(
        train=link_split(samples=train, output=output, split="train"),
        validation=link_split(samples=validation, output=output, split="val"),
        test=link_split(samples=test, output=output, split="test"),
        test_handmade=link_split(
            samples=[item for item in test if item.handmade],
            output=output,
            split="test_handmade",
        ),
    )
    write_data_yaml(path=output / "data.yaml", root=output, class_name=class_name)
    return counts
