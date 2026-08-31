"""YOLO-разметка и сборка комплекта для импорта в CVAT."""

from __future__ import annotations

import zipfile
from pathlib import Path

CVAT_DATA_DIR = "obj_train_data"

CVAT_OBJ_DATA = """classes = 1
train = data/train.txt
names = data/obj.names
backup = backup/
"""


def yolo_line(
    *,
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    class_id: int = 0,
) -> str:
    """Строка разметки: класс и центр с размерами в долях кадра.

    Координаты подрезаются по границам кадра. Детектор иногда выдаёт рамку,
    выходящую за край на несколько пикселей, а разметчик такую не принимает.
    """

    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    center_x = (x1 + x2) / 2 / width
    center_y = (y1 + y2) / 2 / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return (
        f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
    )


def write_label(*, path: Path, lines: list[str]) -> None:
    """Файл разметки пишется всегда, даже пустой.

    Пустой `.txt` для YOLO означает «на кадре ничего нет», а отсутствующий файл —
    «кадр не размечен». Разница существенная: без файла обучение молча пропустит
    фотографию вместо того, чтобы учиться на отрицательном примере.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_dataset_yaml(*, path: Path, images_dir: Path, class_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"path: {images_dir.parent.resolve()}",
                f"train: {images_dir.name}",
                f"val: {images_dir.name}",
                "names:",
                f"  0: {class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def pack_for_cvat(
    *,
    archive: Path,
    labels_dir: Path,
    image_names: list[str],
    class_name: str,
) -> None:
    """Архив в формате YOLO 1.1 — его CVAT принимает как импорт разметки.

    Внутри лежат имена классов, список кадров и по файлу разметки на кадр.
    Фотографии в архив не кладутся: их CVAT берёт из самой задачи, а дублировать
    два с половиной гигабайта ради разметки незачем.
    """

    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("obj.names", f"{class_name}\n")
        bundle.writestr("obj.data", CVAT_OBJ_DATA)
        listing = [f"{CVAT_DATA_DIR}/{name}" for name in image_names]
        bundle.writestr("train.txt", "\n".join(listing) + "\n")
        for name in image_names:
            label = labels_dir / f"{Path(name).stem}.txt"
            if label.exists():
                bundle.write(label, f"{CVAT_DATA_DIR}/{label.name}")
