"""Набор целиком из ручной разметки: деление по сценам и плотности рамок."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from adlearn.core.grouping import group_of, stratified_split
from adlearn.core.images import find_images
from adlearn.detection.config import HandmadeConfig
from adlearn.detection.dataset import Sample, SplitCounts, drop_duplicates, link_split
from adlearn.detection.labels import read_boxes
from adlearn.detection.negatives import AUGMENT_SUFFIX

VIDEO_FRAME = re.compile(r"^frame_(\d+)_t(\d+\.\d+)")
"""Кадр из видео: номер кадра и время от начала записи.

Остальные имена в наборе — отдельные снимки. Соседние номера у `video_hard_*`
и `photo_*` принадлежат разным местам: эти кадры отобраны вручную, а не
нарезаны подряд.
"""


@dataclass(frozen=True)
class HandmadeCounts:
    """Что получилось после деления."""

    train: SplitCounts
    validation: SplitCounts
    test: SplitCounts
    scenes: int
    by_stratum: dict[str, tuple[int, int, int]]


def collect(*, images: Path, labels: Path) -> list[Sample]:
    """Кадры с разметкой: у каждого рядом обязан лежать свой `.txt`.

    Кадр без файла разметки пропускается. В выгрузке CVAT файл есть у всех
    кадров задачи, в том числе пустой, и его отсутствие значит не «рекламы нет»,
    а «часть не доразмечена».
    """

    samples = []
    for image in find_images(images, recursive=True):
        label = labels / f"{image.stem}.txt"
        if label.exists():
            samples.append(Sample(image=image, label=label, handmade=True))
    return samples


def _video_time(name: str) -> tuple[int, float] | None:
    """Запись и время кадра: `frame_001336_t00053.440` → `(25, 53.44)`.

    Кадры пришли с двух записей, снятых с разной частотой, и номера у них идут
    каждый свои. Сравнивать времена между записями нельзя: на 53-й секунде обеих
    записей разные места. Частота `номер / время` их и разводит — 30 и 25.
    """

    match = VIDEO_FRAME.match(Path(name).stem)
    if match is None:
        return None
    number, seconds = int(match.group(1)), float(match.group(2))
    if seconds <= 0:
        return None
    return round(number / seconds), seconds


def scenes_of(samples: list[Sample], *, gap_sec: float) -> list[list[Sample]]:
    """Единицы деления: сцена из видео целиком либо отдельный снимок.

    Кадры одной записи, снятые в пределах `gap_sec` друг от друга, показывают
    одно и то же место с одного ракурса. Разъехавшись по частям, они превратили
    бы проверку в проверку на уже виденном: модель отвечала бы на кадре, который
    видела в обучении на полсекунды раньше. Поэтому сцена едет в одну часть
    целиком.

    Снимки делятся поштучно: между ними такой связи нет, а дробить их на куски
    значило бы без нужды огрубить деление на 92% набора. Исключение — снимок и
    его аугментированная копия: это один кадр, и они держатся вместе.
    """

    singles: dict[str, list[Sample]] = defaultdict(list)
    by_video: dict[int, list[tuple[float, Sample]]] = defaultdict(list)
    for sample in samples:
        moment = _video_time(sample.image.name)
        if moment is None:
            singles[sample.image.stem.removesuffix(AUGMENT_SUFFIX)].append(sample)
            continue
        by_video[moment[0]].append((moment[1], sample))

    scenes: list[list[Sample]] = []
    for video in sorted(by_video):
        current: list[Sample] = []
        previous = None
        for seconds, sample in sorted(
            by_video[video], key=lambda item: (item[0], item[1].image.name)
        ):
            if previous is not None and seconds - previous >= gap_sec:
                scenes.append(current)
                current = []
            current.append(sample)
            previous = seconds
        if current:
            scenes.append(current)
    return scenes + [sorted(pair, key=lambda item: item.image.name) for pair in singles.values()]


def density_of(scene: list[Sample]) -> str:
    """Насколько кадр заполнен рекламой: `0`, `1`, `2-3` или `4+`.

    Деления на «есть рамки и нет рамок» мало. Кадров с одним щитом в наборе
    большинство, а плотных — считанные десятки, и при простом перемешивании они
    почти целиком осели бы в обучении. Тогда проверка мерила бы только лёгкие
    кадры и показывала бы больше, чем модель стоит.

    У сцены берётся среднее по кадрам: сцена едет в одну часть целиком, и
    стратой ей служит то, чем она набор наполняет в среднем.
    """

    boxes = sum(len(read_boxes(sample.label)) for sample in scene) / len(scene)
    if boxes < 0.5:
        return "0"
    if boxes < 1.5:
        return "1"
    if boxes < 3.5:
        return "2-3"
    return "4+"


def write_yaml(*, path: Path, root: Path, class_name: str) -> None:
    """Описание набора для ultralytics: три части и один класс."""

    path.write_text(
        "\n".join(
            [
                f"path: {root.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                f"  0: {class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def without_copies(scenes: list[list[Sample]]) -> list[list[Sample]]:
    """Проверка и тест без аугментированных копий.

    Копия — тот же кадр в зеркале и с другой яркостью. В обучении она поднимает
    вес негатива, а в проверке только дублирует оригинал: ошибка на одном кадре
    посчиталась бы дважды.
    """

    kept = (
        [sample for sample in scene if not sample.image.stem.endswith(AUGMENT_SUFFIX)]
        for scene in scenes
    )
    return [scene for scene in kept if scene]


def build(config: HandmadeConfig) -> HandmadeCounts:
    """Собирает набор: отбор, склейка сцен, деление, ссылки, описание."""

    pool = collect(images=config.images, labels=config.labels)
    if config.negatives is not None:
        pool += collect(images=config.negatives / "images", labels=config.negatives / "labels")
    samples = drop_duplicates(pool)
    if not samples:
        raise FileNotFoundError("Нечего собирать: ни одного кадра с разметкой.")

    scenes = scenes_of(samples, gap_sec=config.scene_gap_sec)
    train, validation, test = stratified_split(
        scenes,
        stratum=lambda scene: (group_of(scene[0].image.name), density_of(scene)),
        order=lambda scene: scene[0].image.name,
        validation_share=config.validation_share,
        test_share=config.test_share,
        seed=config.seed,
    )
    validation, test = without_copies(validation), without_copies(test)

    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for part, items in enumerate((train, validation, test)):
        for scene in items:
            tally[f"{group_of(scene[0].image.name)} / {density_of(scene)}"][part] += len(scene)

    flat = [[sample for scene in items for sample in scene] for items in (train, validation, test)]
    counts = HandmadeCounts(
        train=link_split(samples=flat[0], output=config.output, split="train"),
        validation=link_split(samples=flat[1], output=config.output, split="val"),
        test=link_split(samples=flat[2], output=config.output, split="test"),
        scenes=len(scenes),
        by_stratum={key: (value[0], value[1], value[2]) for key, value in sorted(tally.items())},
    )
    write_yaml(path=config.output / "data.yaml", root=config.output, class_name=config.class_name)
    return counts
