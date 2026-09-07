"""Сборка обучающего набора: ручная разметка и уверенная псевдоразметка вместе."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

from adlearn.core.grouping import group_of, stratified_split
from adlearn.core.images import IMAGE_SUFFIXES, find_images, link, reset_dir
from adlearn.detection.config import CLASS_NAME, DatasetConfig
from adlearn.detection.labels import CVAT_DATA_DIR, has_boxes
from adlearn.detection.report import STATUS_OK, read_report

SPLITS = ("train", "val", "test")
HANDMADE_SPLIT = "test_handmade"

DUPLICATE_PROBE_BYTES = 65536
"""Сколько байт кадра хватает, чтобы узнать повтор."""

NO_LABEL = Path("__нет разметки__")
"""Метка «у кадра нет файла разметки»: он идёт в набор пустым, как негатив."""


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
    """Раскладывает часть набора ссылками, начиная с чистой папки.

    Кадр без файла разметки получает пустой `.txt`: для YOLO это «рекламы на
    кадре нет», и такой кадр учит модель молчать.
    """

    images_dir = reset_dir(output / "images" / split)
    labels_dir = reset_dir(output / "labels" / split)
    empty = 0
    for sample in samples:
        link(source=sample.image, destination=images_dir / sample.image.name)
        destination = labels_dir / f"{sample.image.stem}.txt"
        if sample.label == NO_LABEL:
            destination.write_text("", encoding="utf-8")
            empty += 1
            continue
        link(source=sample.label, destination=destination)
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


HOLDOUT_SPLIT = "test_dashcam"

AUGMENTED_PREFIX = "truckaug_"
ORIGINAL_PREFIX = "truck_"


@dataclass(frozen=True)
class Source:
    """Откуда берутся кадры и разметка одного рода.

    `labels` может отсутствовать: тогда кадры считаются негативами, и файл
    разметки создаётся пустым. Так заходят фотографии фур, на которых рекламы
    нет вовсе.
    """

    name: str
    images: Path
    labels: Path | None = None
    handmade: bool = False


def collect_source(source: Source) -> list[Sample]:
    """Кадры одного источника вместе с их разметкой.

    Разметка ищется по имени кадра и в подпапках тоже: в старых выгрузках она
    разложена по частям набора. Кадр без файла разметки пропускается: у
    источника с разметкой это значит «не проверен», и подставлять ему пустую
    разметку нельзя — обучение решит, что рекламы на кадре нет.
    """

    images = {path.stem: path for path in find_images(source.images, recursive=True)}
    if source.labels is None:
        return [
            Sample(image=path, label=NO_LABEL, handmade=source.handmade)
            for path in sorted(images.values())
        ]
    samples = []
    for label in sorted(source.labels.rglob("*.txt")):
        image = images.get(label.stem)
        if image is not None:
            samples.append(Sample(image=image, label=label, handmade=source.handmade))
    return samples


def drop_duplicates(samples: list[Sample]) -> list[Sample]:
    """Один и тот же кадр в наборе только один раз.

    Повторы приходят с двух сторон. Старые выгрузки пересекаются: один кадр
    лежит и в основном наборе, и в дополненном, и с разными файлами разметки.
    А среди фотографий из интернета попадаются одинаковые снимки под разными
    именами. Разъехавшись по частям, такой кадр превращает тест в проверку на
    уже виденном.

    Сравниваются имя, размер файла и начало содержимого: одинаковых снимков
    ровно столько, чтобы читать их целиком было незачем.
    """

    seen_stems: set[str] = set()
    seen_bytes: set[tuple[int, bytes]] = set()
    unique = []
    for sample in samples:
        if sample.image.stem in seen_stems:
            continue
        with sample.image.open("rb") as handle:
            head = handle.read(DUPLICATE_PROBE_BYTES)
        mark = (sample.image.stat().st_size, hashlib.sha256(head).digest())
        if mark in seen_bytes:
            continue
        seen_stems.add(sample.image.stem)
        seen_bytes.add(mark)
        unique.append(sample)
    return unique


def keep_pairs_together(
    train: list[Sample], validation: list[Sample], test: list[Sample]
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Отражённая копия едет туда же, куда оригинал.

    Копия отличается от оригинала зеркалом и яркостью, то есть это тот же кадр.
    Разъехавшись по частям, они превратили бы тест в проверку на уже виденном.
    """

    home = {}
    for split, items in (("train", train), ("val", validation), ("test", test)):
        for item in items:
            if item.image.stem.startswith(ORIGINAL_PREFIX):
                home[item.image.stem[len(ORIGINAL_PREFIX) :]] = split

    moved: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for split, items in (("train", train), ("val", validation), ("test", test)):
        for item in items:
            stem = item.image.stem
            if stem.startswith(AUGMENTED_PREFIX):
                target = home.get(stem[len(AUGMENTED_PREFIX) :], split)
                moved[target].append(item)
            else:
                moved[split].append(item)
    return moved["train"], moved["val"], moved["test"]


def write_multi_yaml(*, path: Path, root: Path, class_name: str) -> None:
    """Описание набора с отложенной частью, снятой отдельной техникой."""

    path.write_text(
        "\n".join(
            [
                f"path: {root.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"{HOLDOUT_SPLIT}: {(root / 'images' / HOLDOUT_SPLIT).resolve()}",
                "names:",
                f"  0: {class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class MultiCounts:
    """Что получилось после сборки из нескольких источников."""

    train: SplitCounts
    validation: SplitCounts
    test: SplitCounts
    holdout: SplitCounts
    by_source: dict[str, int]


def build_multi(
    *,
    sources: list[Source],
    holdout: Source | None,
    output: Path,
    class_name: str = CLASS_NAME,
    validation_share: float = 0.15,
    test_share: float = 0.15,
    seed: int = 0,
) -> MultiCounts:
    """Собирает набор из нескольких источников и откладывает чужую съёмку.

    Отложенная часть не участвует в делении вовсе. Обычный тест состоит из тех
    же записей, что и обучение: соседние кадры, те же щиты, тот же свет, и
    метрика на нём выходит выше настоящей. Съёмка другой камерой в другом месте
    показывает, чего модель стоит на самом деле.
    """

    pool: list[Sample] = []
    by_source: dict[str, int] = {}
    for source in sources:
        found = drop_duplicates(collect_source(source))
        by_source[source.name] = len(found)
        pool.extend(found)
    pool = drop_duplicates(pool)
    if not pool:
        raise FileNotFoundError("Нечего собирать: ни один источник не дал кадров.")

    train, validation, test = split_pool(
        pool,
        validation_share=validation_share,
        test_share=test_share,
        seed=seed,
    )
    train, validation, test = keep_pairs_together(train, validation, test)

    counts = MultiCounts(
        train=link_split(samples=train, output=output, split="train"),
        validation=link_split(samples=validation, output=output, split="val"),
        test=link_split(samples=test, output=output, split="test"),
        holdout=link_split(
            samples=collect_source(holdout) if holdout else [],
            output=output,
            split=HOLDOUT_SPLIT,
        ),
        by_source=by_source,
    )
    if holdout:
        by_source[holdout.name] = counts.holdout.frames
    write_multi_yaml(path=output / "data.yaml", root=output, class_name=class_name)
    return counts


def thin_negatives(*, labels: Path, stride: int, keep_prefix: str, destination: Path) -> Path:
    """Копия разметки, где пустых кадров оставлено каждый `stride`-й.

    Кадры с рамками и кадры, у которых рамки сняла проверка, остаются все: их
    смотрел человек, и именно на них модель училась ошибаться. Прореживаются
    только те, где детектор изначально не нашёл ничего, — их на записи заведомо
    больше, чем всего остального.
    """

    reset_dir(destination)
    kept = 0
    seen = 0
    for label in sorted(labels.glob("*.txt")):
        if not label.stem.startswith(keep_prefix):
            continue
        text = label.read_text(encoding="utf-8")
        if text.strip():
            (destination / label.name).write_text(text, encoding="utf-8")
            kept += 1
            continue
        seen += 1
        if stride <= 1 or seen % stride == 0:
            (destination / label.name).write_text("", encoding="utf-8")
            kept += 1
    return destination


def split_holdout(*, labels: Path, prefix: str, destination: Path) -> Path:
    """Разметка отложенной съёмки, отобранная по имени кадра."""

    reset_dir(destination)
    for label in sorted(labels.glob("*.txt")):
        if label.stem.startswith(prefix):
            (destination / label.name).write_text(
                label.read_text(encoding="utf-8"), encoding="utf-8"
            )
    return destination
