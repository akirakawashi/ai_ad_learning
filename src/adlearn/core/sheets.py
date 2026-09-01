"""Контактные листы: кадры плиткой, чтобы просматривать разметку пачками."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

PADDING_COLOR = 255


def blank(side: int) -> np.ndarray:
    return np.full((side, side, 3), PADDING_COLOR, dtype=np.uint8)


def tile(image: np.ndarray, side: int) -> np.ndarray:
    """Вписывает кадр в квадрат, не растягивая: поля добеливаются."""

    height, width = image.shape[:2]
    scale = side / max(height, width)
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
    canvas = blank(side)
    top = (side - resized.shape[0]) // 2
    left = (side - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def save_sheets(
    tiles: Sequence[np.ndarray],
    *,
    output: Path,
    columns: int,
    rows: int,
    side: int,
) -> int:
    """Складывает плитки в листы `columns × rows` и отдаёт число листов.

    Смотреть семь тысяч кадров по одному невозможно, а лист из дюжины
    просматривается за секунды — и промах видно сразу.
    """

    output.mkdir(parents=True, exist_ok=True)
    per_sheet = columns * rows
    sheets = 0
    for start in range(0, len(tiles), per_sheet):
        chunk = list(tiles[start : start + per_sheet])
        chunk += [blank(side)] * (per_sheet - len(chunk))
        grid = np.vstack(
            [np.hstack(chunk[row * columns : (row + 1) * columns]) for row in range(rows)]
        )
        sheets += 1
        cv2.imwrite(str(output / f"sheet_{sheets:02d}.jpg"), grid)
    return sheets
