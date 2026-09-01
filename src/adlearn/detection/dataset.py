"""Сборка обучающего набора: ручная разметка и уверенная псевдоразметка вместе."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

from adlearn.core.grouping import group_of, stratified_split
from adlearn.core.images import IMAGE_SUFFIXES, link, reset_dir
from adlearn.detection.config import DatasetConfig
from adlearn.detection.labels import CVAT_DATA_DIR, has_boxes
from adlearn.detection.report import STATUS_OK, read_report

SPLITS = ("train", "val", "test")
HANDMADE_SPLIT = "test_handmade"


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

    reset_dir(destination)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    data_dir = destination / CVAT_DATA_DIR
    if not data_dir.is_dir():
        raise FileNotFoundError(f"В выгрузке нет папки {CVAT_DATA_DIR}.")
    return data_dir


def collect_handmade(directory: Path) -> list[Sample]:
    """Кадры из выгрузки CVAT — те, у которых рядом лежит файл разметки."""

    return [
        Sample(image=path, label=directory / f"{path.stem}.txt", handmade=True)
        for path in sorted(directory.iterdir())
        if path.suffix.lower() in IMAGE_SUFFIXES and (directory / f"{path.stem}.txt").exists()
    ]


def collect_confident(*, prelabel_dir: Path) -> list[Sample]:
    """Кадры, где модель не сомневалась. Слабые и пустые ждут ручной правки."""

    images_dir = prelabel_dir / "images"
    labels_dir = prelabel_dir / "labels"
    rows = read_report(prelabel_dir / "report.csv")
    names = [row["file"] for row in rows if row["status"] == STATUS_OK]
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
    validation_share: float = 0.15,
    test_share: float = 0.15,
    seed: int = 0,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Делит кадры внутри пар «группа съёмки и наличие рамок».

    Первое разводит по всем частям кадры с регистратора и снимки из интернета,
    второе — кадры со щитами и пустые. Пустых в наборе заметная доля, и это не
    мусор: на них модель учится не выдумывать щит там, где его нет. Собравшись в
    одной части, они перекосили бы и обучение, и метрику.

    Внутри теста потом отдельно выделяется часть с ручной разметкой. На остальных
    кадрах разметку поставила текущая модель, и метрика там показывает согласие с
    ней, а не правоту: старая модель получает на них почти единицу просто потому,
    что сравнивается сама с собой.
    """

    return stratified_split(
        samples,
        stratum=lambda item: (group_of(item.image.name), has_boxes(item.label)),
        order=lambda item: item.image.name,
        validation_share=validation_share,
        test_share=test_share,
        seed=seed,
    )


def link_split(*, samples: list[Sample], output: Path, split: str) -> SplitCounts:
    """Раскладывает часть набора ссылками, начиная с чистой папки."""

    images_dir = reset_dir(output / "images" / split)
    labels_dir = reset_dir(output / "labels" / split)
    empty = 0
    for sample in samples:
        link(source=sample.image, destination=images_dir / sample.image.name)
        link(source=sample.label, destination=labels_dir / f"{sample.image.stem}.txt")
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
                f"{HANDMADE_SPLIT}: {(root / 'images' / HANDMADE_SPLIT).resolve()}",
                "names:",
                f"  0: {class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build(config: DatasetConfig) -> DatasetCounts:
    """Собирает набор целиком: распаковка, отбор, деление, ссылки, описание."""

    handmade_dir = unpack_export(
        archive=config.export_archive, destination=config.output / "handmade"
    )
    samples = collect_handmade(handmade_dir) + collect_confident(prelabel_dir=config.prelabel)
    if not samples:
        raise FileNotFoundError("Нечего собирать: нет ни одного кадра с разметкой.")

    train, validation, test = split_pool(
        samples,
        validation_share=config.validation_share,
        test_share=config.test_share,
        seed=config.seed,
    )
    counts = DatasetCounts(
        train=link_split(samples=train, output=config.output, split="train"),
        validation=link_split(samples=validation, output=config.output, split="val"),
        test=link_split(samples=test, output=config.output, split="test"),
        test_handmade=link_split(
            samples=[item for item in test if item.handmade],
            output=config.output,
            split=HANDMADE_SPLIT,
        ),
    )
    write_data_yaml(
        path=config.output / "data.yaml", root=config.output, class_name=config.class_name
    )
    return counts
