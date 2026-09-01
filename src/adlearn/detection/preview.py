"""Разметка поверх кадров: одиночные картинки и контактные листы."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from adlearn.core.images import find_images
from adlearn.core.sheets import save_sheets, tile
from adlearn.detection.labels import read_boxes

BOX_COLOR = (0, 0, 255)


def draw(image: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    """Рисует рамки в долях кадра. Толщина линии зависит от размера кадра."""

    height, width = image.shape[:2]
    thickness = max(2, round(min(width, height) / 250))
    for center_x, center_y, box_width, box_height in boxes:
        x1 = round((center_x - box_width / 2) * width)
        y1 = round((center_y - box_height / 2) * height)
        x2 = round((center_x + box_width / 2) * width)
        y2 = round((center_y + box_height / 2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, thickness)
    return image


def build_sheets(
    *,
    images_dir: Path,
    labels_dir: Path,
    output: Path,
    limit: int,
    columns: int = 4,
    rows: int = 3,
    side: int = 420,
    seed: int = 0,
    only_empty: bool = False,
    save_frames: bool = False,
) -> tuple[int, int]:
    """Собирает контактные листы: кадр с рамками, по `columns × rows` на лист.

    `only_empty` оставляет кадры, на которых модель ничего не нашла: там промах
    виден вернее всего.
    """

    frames_dir = output / "frames"
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(images_dir)
    if only_empty:
        images = [path for path in images if not read_boxes(labels_dir / f"{path.stem}.txt")]
    random.Random(seed).shuffle(images)
    images = images[:limit]

    tiles: list[np.ndarray] = []
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        marked = draw(frame, read_boxes(labels_dir / f"{path.stem}.txt"))
        if save_frames:
            cv2.imwrite(str(frames_dir / f"{path.stem}.jpg"), marked)
        tiles.append(tile(marked, side))

    sheets = save_sheets(tiles, output=output / "sheets", columns=columns, rows=rows, side=side)
    return len(tiles), sheets
