"""Пачки для ручной правки: картинки и разметка кусками под задачи CVAT."""

from __future__ import annotations

import zipfile
from pathlib import Path

from adlearn.detection.labels import pack_for_cvat
from adlearn.detection.report import STATUS_EMPTY, STATUS_OK, STATUS_WEAK, read_report

CHUNK_SIZE = 600


def files_to_review(report_path: Path) -> list[str]:
    """Имена кадров, которым нужен человек, в порядке разбора.

    Сначала кадры без единой находки — там модель промахнулась целиком, и правка
    даёт больше всего. За ними слабые, от самой сомнительной рамки.

    Args:
        report_path: CSV-отчёт псевдоразметки.

    Returns:
        Имена пустых и слабых кадров в порядке ручного разбора.
    """

    rows = [row for row in read_report(report_path) if row["status"] != STATUS_OK]
    empty = [row["file"] for row in rows if row["status"] == STATUS_EMPTY]
    weak = [
        row["file"]
        for row in sorted(
            (row for row in rows if row["status"] == STATUS_WEAK),
            key=lambda row: float(row["min_confidence"]),
        )
    ]
    return empty + weak


def chunks(names: list[str], size: int) -> list[list[str]]:
    """Делит последовательность имён на пачки заданного размера.

    Args:
        names: Имена кадров.
        size: Максимальное число имён в пачке.

    Returns:
        Последовательные пачки имён.
    """

    return [names[start : start + size] for start in range(0, len(names), size)]


def pack_images(*, archive: Path, images_dir: Path, names: list[str]) -> None:
    """Упаковывает выбранные изображения в ZIP без повторного сжатия.

    Args:
        archive: Путь к создаваемому архиву.
        images_dir: Каталог исходных изображений.
        names: Имена изображений для архива.
    """

    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        for name in names:
            bundle.write(images_dir / name, name)


def build(
    *,
    report_path: Path,
    images_dir: Path,
    labels_dir: Path,
    output: Path,
    class_name: str,
    chunk_size: int = CHUNK_SIZE,
) -> list[tuple[Path, Path, int]]:
    """Готовит пары «картинки — разметка» под отдельные задачи CVAT.

    Пачками, а не одной кучей: задача на несколько тысяч кадров грузится через
    браузер долго и открывается тяжело, а упавшая загрузка отменяет всю работу
    целиком.

    Картинки складываются без сжатия — jpg и png уже сжаты, а упаковка семисот
    мегабайт впустую стоит минут.

    Args:
        report_path: CSV-отчёт псевдоразметки.
        images_dir: Каталог изображений.
        labels_dir: Каталог YOLO-разметки.
        output: Каталог для архивов.
        class_name: Имя класса в архивах разметки.
        chunk_size: Максимум кадров в одной задаче CVAT.

    Returns:
        Пары архивов изображений и разметки с числом кадров.
    """

    names = files_to_review(report_path)
    parts: list[tuple[Path, Path, int]] = []
    for index, part in enumerate(chunks(names, chunk_size), start=1):
        images_archive = output / f"part_{index:02d}_images.zip"
        annotations_archive = output / f"part_{index:02d}_annotations.zip"
        pack_images(archive=images_archive, images_dir=images_dir, names=part)
        pack_for_cvat(
            archive=annotations_archive,
            labels_dir=labels_dir,
            image_names=part,
            class_name=class_name,
        )
        parts.append((images_archive, annotations_archive, len(part)))
    return parts
